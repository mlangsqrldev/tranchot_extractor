"""
Mathematical Ink Deconvolution and Marker-Controlled Watershed for Historical Cartography.
Implements Ruifrok-Johnston Optical Density (OD) matrix separation for 18th/19th-century maps:
- Isolates pure Carmine/Vermilion red (Buildings & Courtyards)
- Isolates Iron-Gall / Carbon black (Hachures, Text, Roads)
- Isolates Indigo blue (Water bodies)
- Uses distance transform and marker-controlled watershed to cleanly split joined building footprints at wall seams.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd


@dataclass
class InkChannels:
    """Optical density separation channels."""
    carmine: np.ndarray    # Red ink (Buildings, Courtyards)
    carbon_black: np.ndarray  # Black/Brown ink (Hachures, Text, Roads)
    indigo_blue: np.ndarray   # Blue wash (Water bodies)
    od_total: np.ndarray      # Total optical density


@dataclass
class DeconvolutionResult:
    """Vector and raster results from ink deconvolution + watershed."""
    polygons: List[Polygon]
    classes: List[str]
    confidences: List[float]
    building_mask: np.ndarray
    carmine_od: np.ndarray
    gdf: gpd.GeoDataFrame


class CartographicInkDeconvolution:
    """
    Mathematical color deconvolution using calibrated optical density vectors for historical maps.
    Based on the Ruifrok-Johnston formulation extended to 18th/19th-century cartographic pigments.
    """

    # Calibrated pigment OD vectors [R, G, B] normalized to unit length
    # 1. Carmine / Vermilion red (low R absorption, high G and B absorption)
    CARMINE_RED_OD = np.array([0.12, 0.70, 0.70], dtype=np.float64)
    # 2. Iron-gall / Carbon black (balanced broad-spectrum absorption across all channels)
    CARBON_BLACK_OD = np.array([0.57, 0.57, 0.57], dtype=np.float64)
    # 3. Indigo / Prussian blue (high R and G absorption, low B absorption)
    INDIGO_BLUE_OD = np.array([0.70, 0.70, 0.12], dtype=np.float64)

    def __init__(
        self,
        carmine_od: Optional[np.ndarray] = None,
        carbon_od: Optional[np.ndarray] = None,
        indigo_od: Optional[np.ndarray] = None,
    ):
        v1 = carmine_od if carmine_od is not None else self.CARMINE_RED_OD
        v2 = carbon_od if carbon_od is not None else self.CARBON_BLACK_OD
        v3 = indigo_od if indigo_od is not None else self.INDIGO_BLUE_OD

        # Normalize to unit vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)

        # Deconvolution matrix M (3x3) where rows are pigment vectors
        self.M = np.vstack([v1, v2, v3])
        # Invert matrix to get separation transform: C = OD * M_inv
        self.M_inv = np.linalg.inv(self.M)

    def separate_inks(self, image_rgb: np.ndarray) -> InkChannels:
        """
        Converts RGB image to Optical Density and projects onto pure ink channels.
        """
        # 1. Clip and normalize RGB to [0.001, 1.0] to prevent log(0)
        img_float = np.clip(image_rgb.astype(np.float64), 1.0, 255.0) / 255.0

        # 2. Beer-Lambert Optical Density: OD = -log10(I / I0)
        od = -np.log10(img_float)  # shape: (H, W, 3)

        h, w = od.shape[:2]
        od_flat = od.reshape(-1, 3)

        # 3. Matrix deconvolution: Stains = OD * M_inv
        concentrations = np.dot(od_flat, self.M_inv)  # shape: (H*W, 3)
        concentrations = np.clip(concentrations, 0.0, None)  # non-negative densities
        concentrations = concentrations.reshape(h, w, 3)

        carmine_channel = concentrations[:, :, 0]
        black_channel = concentrations[:, :, 1]
        blue_channel = concentrations[:, :, 2]
        total_od = np.linalg.norm(od, axis=2)

        return InkChannels(
            carmine=carmine_channel,
            carbon_black=black_channel,
            indigo_blue=blue_channel,
            od_total=total_od,
        )

    def segment_buildings_watershed(
        self,
        image_rgb: np.ndarray,
        od_threshold: float = 0.22,
        min_building_area: float = 18.0,
        max_building_area: float = 8500.0,
        max_aspect_ratio: float = 3.5,
        min_compactness: float = 0.18,
        watershed_footprint: int = 5,
        simplification_factor: float = 0.02,
    ) -> DeconvolutionResult:
        """
        Separates carmine pigment and applies Marker-Controlled Watershed
        to split touching houses cleanly at wall seams.
        """
        h, w = image_rgb.shape[:2]
        inks = self.separate_inks(image_rgb)

        # 1. Carmine density thresholding
        carmine_norm = inks.carmine
        carmine_thresh = (carmine_norm > od_threshold) & (inks.carbon_black < carmine_norm * 1.5)
        binary_carmine = (carmine_thresh.astype(np.uint8)) * 255

        # Morphological opening to eliminate isolated pixel dust
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary_carmine, cv2.MORPH_OPEN, kernel_open)

        # 2. Marker-Controlled Watershed for wall-seam splitting
        dist_transform = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)

        if dist_transform.max() <= 0:
            return DeconvolutionResult([], [], [], np.zeros((h, w), dtype=np.uint8), carmine_norm, gpd.GeoDataFrame(geometry=[]))

        # Sure foreground seeds (peaks inside individual buildings)
        ret, sure_fg = cv2.threshold(dist_transform, 0.30 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Sure background (dilated outside area)
        sure_bg = cv2.dilate(binary_clean, np.ones((3, 3), np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Connected component markers for each building center
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Convert RGB to BGR for cv2.watershed
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.watershed(bgr, markers)

        # 3. Vectorize individual segmented watershed basins
        unique_labels = np.unique(markers)
        polygons: List[Polygon] = []
        classes: List[str] = []
        confidences: List[float] = []

        final_mask = np.zeros((h, w), dtype=np.uint8)

        for label_id in unique_labels:
            if label_id <= 1:  # -1 is watershed boundary, 1 is background
                continue

            single_basin = (markers == label_id).astype(np.uint8) * 255
            # Intersect with binary carmine mask to keep true pigment borders
            single_basin = cv2.bitwise_and(single_basin, binary_clean)

            contours, _ = cv2.findContours(single_basin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_building_area or area > max_building_area:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                if aspect > max_aspect_ratio:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue

                compactness = (4.0 * np.pi * area) / (perimeter ** 2)
                if compactness < min_compactness:
                    continue

                # Simplify contour with Douglas-Peucker
                epsilon = max(1.0, simplification_factor * perimeter)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 3:
                    continue

                pts = approx.reshape(-1, 2)
                try:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty and poly.geom_type == 'Polygon':
                        polygons.append(poly)
                        cls_name = "Hofanlage" if area > 1400 and compactness < 0.45 else "Gebäude"
                        classes.append(cls_name)
                        
                        conf = min(0.99, round(0.70 + (compactness * 0.25), 2))
                        confidences.append(conf)

                        coords = np.array(poly.exterior.coords, dtype=np.int32)
                        cv2.fillPoly(final_mask, [coords], 255)
                except Exception:
                    continue

        # Build GeoDataFrame
        records = []
        for idx, (p, cls_n, conf) in enumerate(zip(polygons, classes, confidences), 1):
            records.append({
                "building_id": idx,
                "class": cls_n,
                "confidence": conf,
                "area_px": round(p.area, 2),
            })

        gdf = gpd.GeoDataFrame(records, geometry=polygons) if polygons else gpd.GeoDataFrame(geometry=[])

        return DeconvolutionResult(
            polygons=polygons,
            classes=classes,
            confidences=confidences,
            building_mask=final_mask,
            carmine_od=carmine_norm,
            gdf=gdf,
        )
