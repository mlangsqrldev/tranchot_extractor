"""
High-Speed Land-Use Extractor for Tranchot Historical Maps.
Segments Forests (Wald), Meadows/Pastures (Wiesen/Feuchtgrünland), and Water bodies (Gewässer).
Uses parchment-normalized spectral analysis and tree-foliage texture detection
to be robust against yellow paper aging, mountain hachures, and text overlays.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import geopandas as gpd

from tranchot_extractor.config import LandUseConfig


@dataclass
class LandUseFeature:
    """Represents an extracted land-use polygon."""
    id: int
    geometry: Polygon
    category: str  # 'forest', 'meadow', 'water'
    area_px: float


@dataclass
class LandUseExtractionResult:
    """Complete results from land-use extraction."""
    features: List[LandUseFeature]
    forest_polygons: List[Polygon]
    meadow_polygons: List[Polygon]
    water_polygons: List[Polygon]
    gdf: gpd.GeoDataFrame
    execution_time_s: float


class LandUseExtractor:
    """
    Extracts historical land-use categories:
    - Forest (Wald): Olive-green wash + engraved tree foliage crown textures
    - Meadow (Wiesen / Feuchtgrünland): Cyan-pastell valley meadows and gardens
    - Water (Gewässer): Rhine River, streams, ponds, and lakes
    """

    def __init__(self, config: Optional[LandUseConfig] = None):
        self.config = config or LandUseConfig()

    def extract(self, image_rgb: np.ndarray) -> LandUseExtractionResult:
        """
        Fast extraction of land-use polygons with parchment normalization and texture bridging.
        """
        t0 = time.time()
        h, w = image_rgb.shape[:2]

        # 1. Multi-scale Pyramidal Downsampling for 100x Speedup on large GeoTIFFs
        max_dim = max(h, w)
        if max_dim > 1800:
            scale_factor = 1800.0 / max_dim
            small = cv2.resize(image_rgb, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
            small = image_rgb

        inv_scale = 1.0 / scale_factor

        # 2. Color Spaces & Texture Computation
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        r = small[:, :, 0].astype(float)
        g = small[:, :, 1].astype(float)
        b = small[:, :, 2].astype(float)

        # Texture variance (measures tree-stamp roughness vs. smooth water/meadow watercolor)
        blur_g = cv2.GaussianBlur(gray, (13, 13), 0)
        local_var = cv2.GaussianBlur((gray - blur_g) ** 2, (13, 13), 0)
        tex_norm = (local_var - 35.0) / 25.0

        # Relative chrominance against parchment paper
        rel_green = g - (r * 0.88)
        rel_blue = b - (r * 0.68)

        # 3. Competitive Multi-Class Distance Metric
        # Class 0: Water (Blue, smooth)
        d_water_l = (lab[:, :, 0] - 150.0) * 0.4
        d_water_a = (lab[:, :, 1] - 122.0) * 1.8
        d_water_b = (lab[:, :, 2] - 115.0) * 1.8
        dist_water = np.sqrt(d_water_l**2 + d_water_a**2 + d_water_b**2) + (tex_norm * 8.0)
        # Blue bonus
        dist_water -= np.clip(rel_blue * 1.2, 0, 30)

        # Class 1: Forest (Olive-green, tree stamps / high texture)
        d_forest_l = (lab[:, :, 0] - 128.0) * 0.4
        d_forest_a = (lab[:, :, 1] - 122.0) * 1.8
        d_forest_b = (lab[:, :, 2] - 138.0) * 1.8
        dist_forest = np.sqrt(d_forest_l**2 + d_forest_a**2 + d_forest_b**2) - (tex_norm * 10.0)
        dist_forest -= np.clip(rel_green * 0.8, 0, 20)

        # Class 2: Meadow / Pasture (Cyan-Pastell, smooth)
        d_meadow_l = (lab[:, :, 0] - 165.0) * 0.4
        d_meadow_a = (lab[:, :, 1] - 118.0) * 1.8
        d_meadow_b = (lab[:, :, 2] - 130.0) * 1.8
        dist_meadow = np.sqrt(d_meadow_l**2 + d_meadow_a**2 + d_meadow_b**2) + (tex_norm * 6.0)
        dist_meadow -= np.clip(rel_green * 1.4, 0, 25)

        # Class 3: Background / Pergament / Ackerland
        d_paper_l = (lab[:, :, 0] - 190.0) * 0.4
        d_paper_a = (lab[:, :, 1] - 128.0) * 1.8
        d_paper_b = (lab[:, :, 2] - 142.0) * 1.8
        dist_paper = np.sqrt(d_paper_l**2 + d_paper_a**2 + d_paper_b**2)

        # Stack distances and find competitive winner
        dist_stack = np.stack([dist_water, dist_forest, dist_meadow, dist_paper], axis=2)
        winner = np.argmin(dist_stack, axis=2)

        # 4. Clean class binary masks
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        water_mask = cv2.morphologyEx(((winner == 0) & (rel_blue > 4)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        forest_mask = cv2.morphologyEx(((winner == 1) & (dist_forest < 38)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        forest_clean = cv2.morphologyEx(forest_mask, cv2.MORPH_OPEN, k_open)

        meadow_mask = cv2.morphologyEx(((winner == 2) & (rel_green > 2)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        meadow_clean = cv2.morphologyEx(meadow_mask, cv2.MORPH_OPEN, k_open)

        # 5. Vectorize & Scale Back to Full Image Resolution
        def vectorize_mask(mask_u8, min_area_full_px: float) -> List[Polygon]:
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for cnt in cnts:
                if len(cnt) >= 3:
                    approx = cv2.approxPolyDP(cnt, 1.5, True)
                    if len(approx) >= 3:
                        scaled_pts = [(float(p[0][0]) * inv_scale, float(p[0][1]) * inv_scale) for p in approx]
                        p_geom = Polygon(scaled_pts)
                        if p_geom.is_valid and p_geom.area >= min_area_full_px:
                            polys.append(p_geom.simplify(2.0, preserve_topology=True))
            return polys

        forest_polys = vectorize_mask(forest_clean, min_area_full_px=600.0)
        meadow_polys = vectorize_mask(meadow_clean, min_area_full_px=300.0)
        water_polys = vectorize_mask(water_mask, min_area_full_px=150.0)

        features: List[LandUseFeature] = []
        feat_id = 1

        for p in forest_polys:
            features.append(LandUseFeature(id=feat_id, geometry=p, category="forest", area_px=round(p.area, 1)))
            feat_id += 1

        for p in meadow_polys:
            features.append(LandUseFeature(id=feat_id, geometry=p, category="meadow", area_px=round(p.area, 1)))
            feat_id += 1

        for p in water_polys:
            features.append(LandUseFeature(id=feat_id, geometry=p, category="water", area_px=round(p.area, 1)))
            feat_id += 1

        # Build GeoDataFrame
        records = []
        geoms = []
        for f in features:
            records.append({
                "feature_id": f.id,
                "layer": "landuse",
                "category": f.category,
                "area_px": f.area_px,
            })
            geoms.append(f.geometry)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:25832")
        dt = time.time() - t0

        return LandUseExtractionResult(
            features=features,
            forest_polygons=forest_polys,
            meadow_polygons=meadow_polys,
            water_polygons=water_polys,
            gdf=gdf,
            execution_time_s=round(dt, 3),
        )

    def extract_parcel_at(self, image_rgb: np.ndarray, click_x: float, click_y: float) -> Optional[Tuple[Polygon, str]]:
        """
        Extracts the specific continuous land-use parcel around a clicked coordinate.
        """
        res = self.extract(image_rgb)
        pt = Point(click_x, click_y)

        # Check water first, then meadow, then forest
        for p in res.water_polygons:
            if p.contains(pt) or p.distance(pt) < 15.0:
                return (p, "water")

        for p in res.meadow_polygons:
            if p.contains(pt) or p.distance(pt) < 15.0:
                return (p, "meadow")

        for p in res.forest_polygons:
            if p.contains(pt) or p.distance(pt) < 15.0:
                return (p, "forest")

        return None
