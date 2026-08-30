"""
Image preprocessing, parchment normalization, and tile slicing for historical maps.
"""

import os
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class MapPreprocessor:
    """Preprocesses historical map raster data and handles large-scale tiling."""

    def __init__(self, target_dpi: Optional[int] = None):
        self.target_dpi = target_dpi

    @staticmethod
    def load_image(
        image_input: Any
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Loads an image from file path, numpy array, or PIL Image.
        Returns RGB numpy array and optional GeoTIFF spatial metadata.
        """
        geo_meta = None

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")

            # Check if GeoTIFF
            if HAS_RASTERIO and (image_input.lower().endswith(('.tif', '.tiff', '.geotif', '.geotiff'))):
                try:
                    with rasterio.open(image_input) as src:
                        img_arr = src.read()
                        # rasterio returns (bands, height, width)
                        if img_arr.shape[0] >= 3:
                            img_rgb = np.transpose(img_arr[:3, :, :], (1, 2, 0))
                        elif img_arr.shape[0] == 1:
                            img_rgb = cv2.cvtColor(img_arr[0], cv2.COLOR_GRAY2RGB)
                        else:
                            img_rgb = np.transpose(img_arr, (1, 2, 0))

                        geo_meta = {
                            "crs": src.crs.to_string() if src.crs else "EPSG:25832",
                            "transform": src.transform,
                            "bounds": src.bounds,
                            "width": src.width,
                            "height": src.height,
                        }
                        return img_rgb.astype(np.uint8), geo_meta
                except Exception:
                    pass

            # Fallback to PIL / OpenCV
            pil_img = Image.open(image_input)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return np.array(pil_img, dtype=np.uint8), geo_meta

        elif isinstance(image_input, Image.Image):
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")
            return np.array(image_input, dtype=np.uint8), None

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB), None
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB), None
                return image_input.copy(), None

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def normalize_parchment(image_rgb: np.ndarray) -> np.ndarray:
        """
        Normalizes the aged parchment background using white-balancing
        and localized contrast adjustment (CLAHE).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge and convert back
        merged = cv2.merge((cl, a, b))
        normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return normalized

    @staticmethod
    def generate_tiles(
        image_rgb: np.ndarray,
        tile_size: int = 2048,
        overlap: int = 256
    ) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """
        Yields overlapping sub-tiles (tile_img, (y1, y2, x1, x2))
        for seamless large map processing.
        """
        height, width = image_rgb.shape[:2]

        if height <= tile_size and width <= tile_size:
            yield image_rgb, (0, height, 0, width)
            return

        step = tile_size - overlap
        y_starts = list(range(0, height, step))
        x_starts = list(range(0, width, step))

        for y in y_starts:
            for x in x_starts:
                y1 = y
                y2 = min(y + tile_size, height)
                x1 = x
                x2 = min(x + tile_size, width)
                tile = image_rgb[y1:y2, x1:x2]
                yield tile, (y1, y2, x1, x2)
