"""
Helper to prepare crops and configurations for official Label Studio.
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GEOTIFFS_DIR = os.path.join(WORKSPACE_ROOT, "output_geotiffs")
OUTPUT_CROPS_DIR = os.path.join(WORKSPACE_ROOT, "label_studio_crops")


def generate_training_crops(crop_size: int = 1024, num_crops_per_sheet: int = 4):
    """
    Extracts representative high-resolution settlement crops from the 10 GeoTIFFs
    for rapid labeling in official Label Studio.
    """
    os.makedirs(OUTPUT_CROPS_DIR, exist_ok=True)
    geotiff_files = glob.glob(os.path.join(GEOTIFFS_DIR, "**", "*.tif"), recursive=True)

    print(f"📦 Generiere Trainings-Ausschnitte aus {len(geotiff_files)} GeoTIFF-Blättern...")
    generated = []

    for f in geotiff_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if not HAS_RASTERIO:
            continue

        try:
            with rasterio.open(f) as src:
                w, h = src.width, src.height
                # Sample center and settlement points
                sample_centers = [
                    (w // 2, h // 2),
                    (w // 3, h // 3),
                    (2 * w // 3, h // 2),
                    (w // 2, 2 * h // 3),
                ]

                for idx, (cx, cy) in enumerate(sample_centers[:num_crops_per_sheet], 1):
                    x1 = max(0, cx - crop_size // 2)
                    y1 = max(0, cy - crop_size // 2)
                    win = Window(x1, y1, min(crop_size, w - x1), min(crop_size, h - y1))

                    data = src.read(window=win)
                    if data.shape[0] >= 3:
                        crop_rgb = np.transpose(data[:3, :, :], (1, 2, 0))
                    else:
                        crop_rgb = cv2.cvtColor(data[0], cv2.COLOR_GRAY2RGB)

                    out_name = f"{stem}_crop_{idx}.jpg"
                    out_path = os.path.join(OUTPUT_CROPS_DIR, out_name)
                    cv2.imwrite(out_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    generated.append(out_path)
        except Exception as e:
            print(f"Warnung bei {stem}: {e}")

    print(f"✅ {len(generated)} hochauflösende Ausschnitte gespeichert in: {OUTPUT_CROPS_DIR}")
    return generated


LABEL_STUDIO_XML_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small">
    <Label value="Gebäude" background="#ef4444"/>
    <Label value="Hofanlage" background="#dc2626"/>
    <Label value="Straße" background="#3b82f6"/>
    <Label value="Feldweg" background="#60a5fa"/>
    <Label value="Toponym" background="#f59e0b"/>
    <Label value="Gewässer" background="#06b6d4"/>
    <Label value="Wald" background="#10b981"/>
  </PolygonLabels>
  <!-- SAM AI Smart Tools (Klicks & Boxen auf GPU) -->
  <KeyPointLabels name="tag" toName="image" smart="true">
    <Label value="Positive" background="#22c55e"/>
    <Label value="Negative" background="#ef4444"/>
  </KeyPointLabels>
  <RectangleLabels name="bbox" toName="image" smart="true">
    <Label value="Gebäude" background="#ef4444"/>
    <Label value="Hofanlage" background="#dc2626"/>
  </RectangleLabels>
</View>"""

