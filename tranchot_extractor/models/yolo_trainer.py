"""
YOLOv8 Segmentation Model Trainer for Tranchot Maps.
Trains instance segmentation models on annotated historical map crops using PyTorch + CUDA (RTX 4090).
"""

import os
import torch
from typing import Optional, Dict, Any

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOSegTrainer:
    """
    Trains YOLOv8-seg models (yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8x-seg) on historical map datasets.
    """

    def __init__(self, base_model: str = "yolov8n-seg.pt", device: Optional[str] = None):
        self.base_model = base_model
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        dataset_yaml_path: str,
        epochs: int = 50,
        imgsz: int = 1024,
        batch_size: int = 8,
        project_name: str = "tranchot_yolo_runs",
        run_name: str = "building_seg",
    ) -> Dict[str, Any]:
        """
        Runs YOLOv8-seg training.
        """
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics is not installed. Please run 'pip install ultralytics'.")

        print(f"🚀 Starte YOLOv8-Seg Training auf Device: {self.device}...")
        print(f"   Modell:       {self.base_model}")
        print(f"   Dataset:      {dataset_yaml_path}")
        print(f"   Epochen:      {epochs}")
        print(f"   Bildgröße:    {imgsz}x{imgsz}")

        model = YOLO(self.base_model)

        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=project_name,
            name=run_name,
            optimizer="AdamW",
            lr0=0.001,
            augment=True,
            degrees=90.0,      # Full 90-degree rotations for orientation-invariant maps
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.02,        # Invariant to parchment ink aging & hue variations
            hsv_s=0.4,
            hsv_v=0.3,
            mosaic=1.0,        # Multi-scale patch context
            verbose=True,
        )

        best_weights = os.path.join(project_name, run_name, "weights", "best.pt")
        return {
            "results": results,
            "best_weights": best_weights,
            "run_dir": os.path.join(project_name, run_name),
        }

