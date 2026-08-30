"""
YOLOv8 Segmentation Dataset Builder for Tranchot Maps.
Converts annotations from Label Studio JSON, LabelMe JSON, or GeoJSON into YOLOv8-seg dataset format.
"""

import os
import json
import random
import shutil
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import yaml

DEFAULT_CLASS_MAP = {
    "gebäude": 0,
    "building": 0,
    "hofanlage": 1,
    "farm": 1,
    "hof": 1,
    "straße": 2,
    "road": 2,
    "weg": 2,
    "feldweg": 2,
    "path": 2,
    "gewässer": 3,
    "water": 3,
    "wald": 4,
    "forest": 4,
    "vegetation": 4,
}

CLASS_NAMES = {
    0: "building",
    1: "farm",
    2: "road",
    3: "water",
    4: "forest",
}


class YOLODatasetBuilder:
    """
    Builds YOLOv8 segmentation datasets from Label Studio JSON or GeoJSON exports.
    """

    def __init__(self, output_dir: str = "datasets/tranchot_yolo", val_split: float = 0.2):
        self.output_dir = os.path.abspath(output_dir)
        self.val_split = val_split
        self.images_train_dir = os.path.join(self.output_dir, "images", "train")
        self.images_val_dir = os.path.join(self.output_dir, "images", "val")
        self.labels_train_dir = os.path.join(self.output_dir, "labels", "train")
        self.labels_val_dir = os.path.join(self.output_dir, "labels", "val")
        self._init_dirs()

    def _init_dirs(self):
        for d in [self.images_train_dir, self.images_val_dir, self.labels_train_dir, self.labels_val_dir]:
            os.makedirs(d, exist_ok=True)

    def add_sample_from_polygons(
        self,
        image_np_or_path: Any,
        polygons_with_classes: List[Dict[str, Any]],
        sample_name: str,
        is_val: Optional[bool] = None,
    ):
        """
        Adds a single annotated image crop with its polygon labels.
        polygons_with_classes format:
        [
            {"class_name": "Gebäude", "points": [[x1, y1], [x2, y2], ...]},
            ...
        ]
        """
        if is_val is None:
            is_val = (random.random() < self.val_split)

        img_dest_dir = self.images_val_dir if is_val else self.images_train_dir
        lbl_dest_dir = self.labels_val_dir if is_val else self.labels_train_dir

        # 1. Save Image
        img_filename = f"{sample_name}.jpg"
        img_out_path = os.path.join(img_dest_dir, img_filename)

        if isinstance(image_np_or_path, str) and os.path.exists(image_np_or_path):
            img = cv2.imread(image_np_or_path)
            h, w = img.shape[:2]
            cv2.imwrite(img_out_path, img)
        elif isinstance(image_np_or_path, np.ndarray):
            h, w = image_np_or_path.shape[:2]
            if len(image_np_or_path.shape) == 3 and image_np_or_path.shape[2] == 3:
                bgr = cv2.cvtColor(image_np_or_path, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_out_path, bgr)
            else:
                cv2.imwrite(img_out_path, image_np_or_path)
        else:
            raise ValueError("Unsupported image format")

        # 2. Write YOLO-seg label file (.txt)
        # Format: <class-index> <x1> <y1> <x2> <y2> ... (normalized 0.0 to 1.0)
        lbl_filename = f"{sample_name}.txt"
        lbl_out_path = os.path.join(lbl_dest_dir, lbl_filename)

        lines = []
        for item in polygons_with_classes:
            raw_class = str(item.get("class_name", item.get("label", "building"))).lower().strip()
            class_id = DEFAULT_CLASS_MAP.get(raw_class, 0)
            pts = item.get("points", [])

            if len(pts) < 3:
                continue

            # Normalize coordinates
            norm_pts = []
            for p in pts:
                nx = max(0.0, min(1.0, float(p[0]) / float(w)))
                ny = max(0.0, min(1.0, float(p[1]) / float(h)))
                norm_pts.append(f"{nx:.6f}")
                norm_pts.append(f"{ny:.6f}")

            line = f"{class_id} " + " ".join(norm_pts)
            lines.append(line)

        with open(lbl_out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def convert_label_studio_export(self, json_path: str, images_dir: str):
        """Converts an exported Label Studio JSON file into the YOLO dataset."""
        with open(json_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        for task in tasks:
            task_id = task.get("id", random.randint(1000, 9999))
            annotations = task.get("annotations", [])
            if not annotations:
                continue

            img_ref = task.get("data", {}).get("image", "")
            img_filename = os.path.basename(img_ref)
            local_img = os.path.join(images_dir, img_filename)
            if not os.path.exists(local_img):
                continue

            img_np = np.array(Image.open(local_img))
            h, w = img_np.shape[:2]

            poly_items = []
            for ann in annotations:
                for res in ann.get("result", []):
                    val = res.get("value", {})
                    labels = val.get("polygonlabels", val.get("labels", ["Gebäude"]))
                    cls_name = labels[0] if labels else "Gebäude"

                    # Label Studio points can be normalized percentages (0..100) or pixel coords
                    points = val.get("points", [])
                    pixel_pts = []
                    for pt in points:
                        px = (pt[0] / 100.0 * w) if pt[0] <= 100.0 and max(p[0] for p in points) <= 100.0 else pt[0]
                        py = (pt[1] / 100.0 * h) if pt[1] <= 100.0 and max(p[1] for p in points) <= 100.0 else pt[1]
                        pixel_pts.append([px, py])

                    if len(pixel_pts) >= 3:
                        poly_items.append({"class_name": cls_name, "points": pixel_pts})

            self.add_sample_from_polygons(img_np, poly_items, sample_name=f"crop_{task_id}")

    def generate_yaml(self) -> str:
        """Generates dataset.yaml for YOLOv8 training."""
        yaml_content = {
            "path": self.output_dir.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES,
        }
        yaml_path = os.path.join(self.output_dir, "tranchot_dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        return yaml_path
