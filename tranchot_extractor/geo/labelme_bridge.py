"""
Bridge module for LabelMe and Label Studio integrations.
Allows exporting and launching native desktop labeling tools for Tranchot maps.
"""

import os
import json
import base64
import subprocess
import numpy as np
import cv2
from PIL import Image

def launch_labelme_on_crop(image_path: str):
    """Launches the native LabelMe desktop GUI on a map image."""
    try:
        cmd = ["labelme", image_path, "--labels", "Gebäude,Hofanlage,Straße,Feldweg,Toponym,Gewässer"]
        print(f"🚀 Starte LabelMe Desktop auf: {image_path}...")
        subprocess.Popen(cmd)
    except Exception as e:
        print(f"Fehler beim Starten von LabelMe: {e}")
