"""
YOLOv8 Segmentation Extractor for Large Historical Map GeoTIFFs.
Performs sliding window instance segmentation with spatial georeferencing.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from tranchot_extractor.models.dataset_builder import CLASS_NAMES
from tranchot_extractor.geo.georeference import GeoReferenceHandler


class YOLOSegExtractor:
    """
    Inference engine using trained YOLOv8-seg weights to extract vector features from whole GeoTIFF sheets.
    """

    def __init__(self, model_path: str = "yolov8n-seg.pt", device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics is not installed.")
        self.model = YOLO(model_path)

    def extract_from_crop(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8-seg on a single image crop and returns segmented polygons.
        """
        results = self.model.predict(
            image_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or results[0].masks is None:
            return detections

        r = results[0]
        masks = r.masks.xy  # List of numpy polygon coordinates [ [ [x, y], ... ], ... ]
        boxes = r.boxes

        for i, mask_pts in enumerate(masks):
            if len(mask_pts) < 3:
                continue

            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            try:
                poly = Polygon(mask_pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "geometry": poly,
                        "area_px": round(poly.area, 2),
                    })
            except Exception:
                continue

        return detections

    def process_large_geotiff(
        self,
        geotiff_path: str,
        output_dir: str,
        tile_size: int = 2048,
        overlap: int = 256,
        conf_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Processes a massive 14,000 x 14,000 GeoTIFF sheet using overlapping sliding windows.
        """
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required for GeoTIFF processing.")

        os.makedirs(output_dir, exist_ok=True)
        geo_handler = GeoReferenceHandler.from_file(geotiff_path)

        all_features = []
        step = tile_size - overlap

        with rasterio.open(geotiff_path) as src:
            w, h = src.width, src.height
            crs = src.crs.to_string() if src.crs else "EPSG:25832"

            x_steps = list(range(0, w, step))
            y_steps = list(range(0, h, step))
            total_tiles = len(x_steps) * len(y_steps)

            print(f"🗺️ Verarbeite GeoTIFF ({w}x{h} px) mit trainiertem YOLOv8-Modell ({total_tiles} Kacheln)...")

            tile_count = 0
            for y in y_steps:
                for x in x_steps:
                    tile_count += 1
                    win_w = min(tile_size, w - x)
                    win_h = min(tile_size, h - y)
                    win = Window(x, y, win_w, win_h)

                    data = src.read(window=win)
                    if data.shape[0] >= 3:
                        crop_rgb = np.transpose(data[:3, :, :], (1, 2, 0))
                    else:
                        crop_rgb = cv2.cvtColor(data[0], cv2.COLOR_GRAY2RGB)

                    # Run YOLO segmentation on crop
                    dets = self.extract_from_crop(crop_rgb, conf_threshold=conf_threshold)

                    for d in dets:
                        poly_local = d["geometry"]
                        # Offset to global GeoTIFF coordinates
                        pts_global = [(p[0] + x, p[1] + y) for p in poly_local.exterior.coords]
                        poly_global = Polygon(pts_global)
                        all_features.append({
                            "class_name": d["class_name"],
                            "confidence": d["confidence"],
                            "geometry": poly_global,
                        })

        print(f"✅ Extraktion abgeschlossen: {len(all_features)} Gesamtobjekte gefunden.")

        # Build GeoDataFrame with spatial coordinates
        records = []
        geoms = []
        for idx, f in enumerate(all_features, 1):
            geo_poly = geo_handler.transform_geometry(f["geometry"])
            records.append({
                "id": idx,
                "class": f["class_name"],
                "confidence": f["confidence"],
                "area_px": round(f["geometry"].area, 2),
            })
            geoms.append(geo_poly)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=crs) if geoms else gpd.GeoDataFrame(geometry=[], crs=crs)

        # Export GIS layers
        exported = {}
        if not gdf.empty:
            geojson_path = os.path.join(output_dir, "yolo_features.geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")
            exported["yolo_features_geojson"] = geojson_path

            gpkg_path = os.path.join(output_dir, "yolo_features.gpkg")
            gdf.to_file(gpkg_path, layer="yolo_features", driver="GPKG")
            exported["yolo_features_gpkg"] = gpkg_path

        return {
            "feature_count": len(all_features),
            "gdf": gdf,
            "exported_files": exported,
        }
