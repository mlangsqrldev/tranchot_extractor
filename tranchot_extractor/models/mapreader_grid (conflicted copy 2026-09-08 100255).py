"""
MapReader Spatial Grid & Automated Hybrid Dataset Generator.
Slices large historical GeoTIFFs into georeferenced spatial tiles,
extracts high-precision ground truth polygons via Cartographic Ink Deconvolution & Watershed,
and builds YOLOv8-Seg training datasets.
"""

import os
import glob
import random
import yaml
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from tranchot_extractor.core.ink_deconvolution import CartographicInkDeconvolution


CLASS_MAP = {
    "gebäude": 0,
    "building": 0,
    "house": 0,
    "hofanlage": 1,
    "farm": 1,
    "courtyard": 1,
    "straße": 2,
    "road": 2,
    "path": 2,
    "gewässer": 3,
    "water": 3,
    "wald": 4,
    "forest": 4,
}

CLASS_NAMES = {
    0: "building",
    1: "farm",
    2: "road",
    3: "water",
    4: "forest",
}


class MapReaderGridDatasetBuilder:
    """
    Slices GeoTIFFs using MapReader-style grid patches and auto-labels them
    using mathematical ink deconvolution to build a deep learning dataset.
    """

    def __init__(
        self,
        output_dir: str = "datasets/tranchot_hybrid_yolo",
        tile_size: int = 1024,
        overlap: int = 128,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.val_split = val_split
        self.seed = seed
        random.seed(seed)

        self.images_train_dir = os.path.join(self.output_dir, "images", "train")
        self.images_val_dir = os.path.join(self.output_dir, "images", "val")
        self.labels_train_dir = os.path.join(self.output_dir, "labels", "train")
        self.labels_val_dir = os.path.join(self.output_dir, "labels", "val")
        self._init_dirs()

        self.deconv = CartographicInkDeconvolution()

    def _init_dirs(self):
        for d in [self.images_train_dir, self.images_val_dir, self.labels_train_dir, self.labels_val_dir]:
            os.makedirs(d, exist_ok=True)

    def process_geotiff(
        self,
        geotiff_path: str,
        max_patches: Optional[int] = None,
        min_buildings_per_patch: int = 1,
    ) -> int:
        """
        Slices a single GeoTIFF, extracts building polygons via ink deconvolution,
        and saves training patches to dataset.
        """
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required for GeoTIFF processing.")

        stem = os.path.splitext(os.path.basename(geotiff_path))[0]
        saved_count = 0

        with rasterio.open(geotiff_path) as src:
            w, h = src.width, src.height
            step = self.tile_size - self.overlap

            # Generate grid window coordinates
            windows = []
            for y in range(0, h, step):
                for x in range(0, w, step):
                    win_w = min(self.tile_size, w - x)
                    win_h = min(self.tile_size, h - y)
                    if win_w >= self.tile_size // 2 and win_h >= self.tile_size // 2:
                        windows.append((x, y, win_w, win_h))

            # Shuffle to get diverse spatial coverage
            random.shuffle(windows)

            print(f"🗺️ MapReader Grid: {stem} ({w}x{h} px, {len(windows)} mögliche Kacheln)...")

            for idx, (x, y, win_w, win_h) in enumerate(tqdm(windows, desc=f"Slicing {stem}")):
                if max_patches and saved_count >= max_patches:
                    break

                win = Window(x, y, win_w, win_h)
                data = src.read(window=win)

                if data.shape[0] >= 3:
                    crop_rgb = np.transpose(data[:3, :, :], (1, 2, 0))
                else:
                    crop_rgb = cv2.cvtColor(data[0], cv2.COLOR_GRAY2RGB)

                # Skip completely blank or border areas
                if np.mean(crop_rgb) < 15 or np.mean(crop_rgb) > 250:
                    continue

                # Run Mathematical Ink Deconvolution & Watershed
                res = self.deconv.segment_buildings_watershed(
                    crop_rgb,
                    od_threshold=0.22,
                    min_building_area=20.0,
                )

                if len(res.polygons) < min_buildings_per_patch:
                    continue

                # Assign train vs val split per tile-row (not per tile) so that
                # overlapping neighboring tiles - which can share the same building -
                # always land in the same split instead of leaking across train/val.
                row_idx = y // step
                row_rng = random.Random((self.seed, row_idx))
                is_val = (row_rng.random() < self.val_split)
                img_dir = self.images_val_dir if is_val else self.images_train_dir
                lbl_dir = self.labels_val_dir if is_val else self.labels_train_dir

                patch_name = f"{stem}_p{x}_{y}"

                # 1. Save Image
                img_path = os.path.join(img_dir, f"{patch_name}.jpg")
                bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 2. Write YOLO-seg label txt
                # Format: <class-index> <x1> <y1> <x2> <y2> ... (normalized 0.0 to 1.0)
                lbl_path = os.path.join(lbl_dir, f"{patch_name}.txt")
                lines = []

                for poly, cls_name in zip(res.polygons, res.classes):
                    cls_id = CLASS_MAP.get(cls_name.lower(), 0)
                    coords = list(poly.exterior.coords)
                    if len(coords) < 3:
                        continue

                    norm_pts = []
                    for pt in coords:
                        nx = max(0.0, min(1.0, float(pt[0]) / float(win_w)))
                        ny = max(0.0, min(1.0, float(pt[1]) / float(win_h)))
                        norm_pts.append(f"{nx:.6f}")
                        norm_pts.append(f"{ny:.6f}")

                    lines.append(f"{cls_id} " + " ".join(norm_pts))

                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                saved_count += 1

        print(f"✅ {stem}: {saved_count} Trainings-Patches mit dekonvolvierten Polygonen gespeichert.")
        return saved_count

    def process_all_geotiffs(
        self,
        geotiff_dir: str,
        patches_per_sheet: int = 15,
    ) -> str:
        """
        Processes all GeoTIFFs in a directory and builds the complete YOLO dataset.
        """
        files = glob.glob(os.path.join(geotiff_dir, "**", "*.tif"), recursive=True)
        print(f"📦 Erstelle MapReader-Hybrid-Datensatz aus {len(files)} GeoTIFF-Blättern...")

        total_patches = 0
        for f in files:
            try:
                count = self.process_geotiff(f, max_patches=patches_per_sheet)
                total_patches += count
            except Exception as e:
                print(f"Warnung bei {f}: {e}")

        yaml_path = self.generate_dataset_yaml()
        print(f"🎉 Datensatz erfolgreich erstellt! {total_patches} Gesamt-Patches in {self.output_dir}")
        print(f"📄 Dataset YAML: {yaml_path}")
        return yaml_path

    def generate_dataset_yaml(self) -> str:
        """Generates dataset.yaml for YOLOv8-seg training."""
        yaml_content = {
            "path": self.output_dir.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES,
        }
        yaml_path = os.path.join(self.output_dir, "tranchot_hybrid.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        return yaml_path
