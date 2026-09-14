"""
Boundary extraction module for historical Gemarkungs- and Flurgrenzen in Tranchot maps.
"""

from dataclasses import dataclass
from typing import List, Optional
import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union
from skimage.morphology import skeletonize

from ..config import BoundaryConfig, BuildingConfig


@dataclass
class BoundaryFeature:
    """Represents an individual extracted boundary line segment."""
    id: int
    geometry: LineString
    length_px: float


@dataclass
class BoundaryExtractionResult:
    """Container for boundary extraction output."""
    features: List[BoundaryFeature]
    mask: np.ndarray
    gdf: gpd.GeoDataFrame


class BoundaryExtractor:
    """
    Extracts historical boundaries (Gemarkungsgrenzen, Flurgrenzen, Parzellengrenzen)
    by isolating thin red linear strokes, bridging dashed marks, and vectorizing them as LineStrings.
    """

    def __init__(
        self,
        config: Optional[BoundaryConfig] = None,
        building_config: Optional[BuildingConfig] = None,
    ):
        self.config = config or BoundaryConfig()
        self.building_config = building_config or BuildingConfig()

    def extract(
        self,
        image_rgb: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
    ) -> BoundaryExtractionResult:
        """
        Extracts boundary lines from the RGB image.

        Args:
            image_rgb: RGB image array (H, W, 3).
            building_mask: Optional binary mask of thick buildings to exclude.

        Returns:
            BoundaryExtractionResult with features, mask, and GeoDataFrame.
        """
        h, w = image_rgb.shape[:2]

        # 1. Red ink isolation
        r = image_rgb[:, :, 0].astype(int)
        g = image_rgb[:, :, 1].astype(int)
        b = image_rgb[:, :, 2].astype(int)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        a_ch = lab[:, :, 1]

        is_red = (
            (r > g + 14)
            & (r > b + 18)
            & (a_ch >= 132)
            & (s_ch >= 30)
            & (v_ch >= 55)
            & (v_ch <= 235)
            & ((h_ch <= 20) | (h_ch >= 165))
        )
        binary = (is_red.astype(np.uint8)) * 255

        # 2. Separate thin boundary strokes from thick buildings
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        if building_mask is not None and np.any(building_mask > 0):
            # Exclude dilated building polygons
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            b_dil = cv2.dilate(building_mask, kernel_dil)
            thin_mask = cv2.subtract(binary, b_dil)
        else:
            # Distance transform threshold for thin strokes
            thick_seeds = (dist >= self.config.max_stroke_thickness).astype(np.uint8) * 255
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thick_bldgs = cv2.dilate(thick_seeds, kernel_dil)
            thin_mask = cv2.subtract(binary, thick_bldgs)

        # 3. Bridge dashed and dotted boundary lines
        bridge_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.bridge_kernel_size, self.config.bridge_kernel_size),
        )
        dil = cv2.dilate(thin_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bridged = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, bridge_k)

        # 4. Skeletonize thin boundary lines
        skel = skeletonize(bridged > 0).astype(np.uint8)

        # 5. Extract continuous vector contours & build LineStrings via PCA sorting
        num_cc, labels = cv2.connectedComponents(skel)

        raw_lines = []
        for c_id in range(1, num_cc):
            pts = np.argwhere(labels == c_id)
            if len(pts) < 4:
                continue
            pts_xy = np.fliplr(pts).astype(np.float32)  # (x, y)
            try:
                pca_mean, eigenvectors = cv2.PCACompute(pts_xy, mean=None)
                proj = np.dot(pts_xy - pca_mean, eigenvectors[0])
                sorted_pts = pts_xy[np.argsort(proj)]
                approx = cv2.approxPolyDP(sorted_pts, self.config.simplification_tolerance, False)
                if len(approx) >= 2:
                    line = LineString(approx.reshape(-1, 2))
                    if line.length >= self.config.min_boundary_length_px:
                        raw_lines.append(line)
            except Exception:
                continue

        # Merge connected linear segments
        merged_lines = []
        if raw_lines:
            try:
                merged = linemerge(unary_union(raw_lines))
                if isinstance(merged, LineString):
                    if merged.length >= self.config.min_boundary_length_px:
                        merged_lines.append(merged)
                elif isinstance(merged, MultiLineString):
                    for single_l in merged.geoms:
                        if single_l.length >= self.config.min_boundary_length_px:
                            merged_lines.append(single_l)
                else:
                    merged_lines = raw_lines
            except Exception:
                merged_lines = raw_lines

        features: List[BoundaryFeature] = []
        records = []
        geoms = []

        for idx, line in enumerate(merged_lines, start=1):
            features.append(
                BoundaryFeature(
                    id=idx,
                    geometry=line,
                    length_px=float(line.length),
                )
            )
            records.append({
                "id": idx,
                "feature_type": "boundary",
                "length_px": round(line.length, 2),
            })
            geoms.append(line)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:25832")

        return BoundaryExtractionResult(
            features=features,
            mask=bridged,
            gdf=gdf,
        )
