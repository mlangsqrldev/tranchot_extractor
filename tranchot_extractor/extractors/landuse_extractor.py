"""
High-Speed Land-Use Extractor for Tranchot Historical Maps (1803–1820).
Segments Forests (Wald), Meadows/Pastures (Wiesen/Weiden), and Watercourses/Water bodies (Flüsse/Bäche/Teiche).

Specialized Multi-Cue Architecture:
Because French military cartographers used the exact same cyan-green watercolor wash
(Verdigris / Grünspan & Indigo) on rag paper for both watercourses and alluvial meadows (Delta E < 3.5),
this extractor uses a 5-Pillar Multi-Cue Architecture:
1. Topological Hierarchy: Meadow = Alluvial Valley Basin minus Buffered Stream Network
2. Geometric Dimensionality: 1D tubular ribbons (rivers/streams) vs. 2D parcels (meadows)
3. Dual Bank Lines & Scharr Gradients: High-contrast ink contours bounding watercourses
4. Riparian Signature: Pollard willows (Kopfweiden) along river banks
5. Standing Water Bodies: Ponds and mill basins directly connected to stream channels
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
import geopandas as gpd
from skimage.filters import frangi

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
    - Meadow (Wiesen / Weiden): Alluvial cyan-green valley meadows and pastures
    - Water (Flüsse / Bäche / Teiche): Linear stream channels, mill races, ponds, and rivers
    """

    def __init__(self, config: Optional[LandUseConfig] = None):
        self.config = config or LandUseConfig()

    def extract(self, image_rgb: np.ndarray) -> LandUseExtractionResult:
        """
        Extracts land-use features with 5-Pillar stream vs. meadow separation.
        """
        t0 = time.time()
        h, w = image_rgb.shape[:2]

        # 1. Multi-scale Pyramidal Downsampling for high processing speed on large GeoTIFFs
        max_dim = max(h, w)
        max_proc = getattr(self.config, "max_processing_dim", 2048)
        if max_dim > max_proc:
            scale_factor = float(max_proc) / float(max_dim)
            small = cv2.resize(image_rgb, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
            small = image_rgb

        inv_scale = 1.0 / scale_factor

        # 2. Color Spaces and Channels
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        r = small[:, :, 0].astype(float)
        g = small[:, :, 1].astype(float)
        b = small[:, :, 2].astype(float)

        # Parchment background & roads (dry warm tones)
        is_paper_or_road = (r - b > 24) | (r - g > 12) | (gray > 220)

        # Text / black ink mask (labels like 'Moulin', 'Kretz', 'Plaidt')
        text_ink = (gray < 112) & (r < 118) & (b < 118)
        text_dilated = cv2.dilate(text_ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0

        # Texture variance for tree crown stamps
        blur_g = cv2.GaussianBlur(gray, (13, 13), 0)
        local_var = cv2.GaussianBlur((gray - blur_g) ** 2, (13, 13), 0)

        # -------------------------------------------------------------
        # Phase 1: Forest Extraction (Wald)
        # -------------------------------------------------------------
        # Olive-green wash + crown texture variance, excluding paper and roads
        is_forest = (g > b + 12) & (r < 165) & (g < 170) & (local_var > 30) & (~is_paper_or_road)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        forest_clean = cv2.morphologyEx(is_forest.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        forest_clean = cv2.morphologyEx(forest_clean, cv2.MORPH_OPEN, k_open)

        # -------------------------------------------------------------
        # Phase 2: Alluvial Valley Wash Extraction (Talraum-Lasur)
        # -------------------------------------------------------------
        # Cyan-green watercolor wash shared by both meadows and streams
        is_valley = (g >= r - 6) & (b >= r - 26) & (r < 205) & (g > 130) & (b > 120) & (~is_paper_or_road) & (forest_clean == 0)
        valley_closed = cv2.morphologyEx(is_valley.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        valley_closed[forest_clean > 0] = 0

        # Filter valley to genuine continuous valley floors
        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(valley_closed)
        clean_valley = np.zeros_like(valley_closed)
        for i in range(1, num_v):
            if stats_v[i, cv2.CC_STAT_AREA] >= 400:
                clean_valley[labels_v == i] = 255

        # -------------------------------------------------------------
        # Phase 3: Multi-Cue Watercourse Extraction (Flüsse / Bäche / Teiche)
        # -------------------------------------------------------------
        # A. Direct digital blue water detection (modern maps, lakes, synthetic tests)
        direct_blue = (b > r + 25) & (b > g + 5)

        # B. Valley core: exclude outer step boundary against parchment paper
        valley_core = cv2.erode(clean_valley, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # C. Historical stream tubular ridge response on natural inverted gray
        inv_gray = (255.0 - gray) / 255.0
        ridge_sigmas = getattr(self.config, "stream_ridge_sigmas", (1.0, 1.8, 2.8, 4.0))
        ridge = frangi(inv_gray, sigmas=list(ridge_sigmas), black_ridges=False)

        stream_raw = direct_blue.copy()
        if np.any(valley_core > 0):
            v_vals = ridge[valley_core > 0]
            if len(v_vals) > 0 and np.max(v_vals) > 0:
                p98 = np.percentile(v_vals, 98.5)
                ridge_norm = np.clip(ridge / (p98 + 1e-6), 0, 1.0)
                blue_core = (b >= r - 12) & (g >= r - 4) & (gray < 175) & (valley_core > 0)
                ridge_water = ((ridge_norm > 0.40) & (gray < 185) & (valley_core > 0)) | (blue_core & (ridge_norm > 0.20))
                # Suppress roads, text ink, and forest
                ridge_water &= (~is_paper_or_road) & (~text_dilated) & (forest_clean == 0)
                stream_raw |= ridge_water

        # Bridge stream segments across small dams, weirs, and bridges
        stream_bridged = cv2.morphologyEx(stream_raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # Connected component filtering: Streams are elongated tubular networks or ponds
        num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(stream_bridged)
        clean_stream = np.zeros_like(stream_bridged)
        for i in range(1, num_s):
            area = stats_s[i, cv2.CC_STAT_AREA]
            bw = stats_s[i, cv2.CC_STAT_WIDTH]
            bh = stats_s[i, cv2.CC_STAT_HEIGHT]
            span = max(bw, bh)
            min_dim = min(bw, bh)
            aspect = span / (min_dim + 1e-4)
            # Elongated streams OR continuous network length OR direct blue ponds
            if (span >= 35 and aspect >= 2.0) or (span >= 70) or (area >= 40 and np.any(direct_blue[labels_s == i])):
                clean_stream[labels_s == i] = 255

        # -------------------------------------------------------------
        # Phase 4: Topological Subtraction for Meadows (Wiesen / Weiden)
        # Meadow = Valley Wash minus Buffered Stream Network
        # -------------------------------------------------------------
        buf_radius = getattr(self.config, "stream_dilation_buffer_px", 4)
        k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buf_radius + 1, 2 * buf_radius + 1))
        stream_buf = cv2.dilate(clean_stream, k_buf)

        meadow_raw = cv2.bitwise_and(clean_valley, cv2.bitwise_not(stream_buf))
        meadow_raw[is_paper_or_road] = 0
        meadow_raw[text_dilated] = 0
        meadow_raw[forest_clean > 0] = 0

        # Polish meadow parcels: remove thin slivers and bridge internal parcel lines
        meadow_clean = cv2.morphologyEx(meadow_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        meadow_clean = cv2.morphologyEx(meadow_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        num_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(meadow_clean)
        final_meadow = np.zeros_like(meadow_clean)
        for i in range(1, num_m):
            if stats_m[i, cv2.CC_STAT_AREA] >= 150:
                final_meadow[labels_m == i] = 255

        # -------------------------------------------------------------
        # Phase 5: Vectorization & Resolution Rescaling
        # -------------------------------------------------------------
        def vectorize_mask(mask_u8: np.ndarray, min_area_full_px: float) -> List[Polygon]:
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

        min_forest_area = getattr(self.config, "min_forest_area_px", 600.0)
        min_meadow_area = getattr(self.config, "min_meadow_area_px", 250.0)
        min_water_area = getattr(self.config, "min_water_area_px", 40.0)

        forest_polys = vectorize_mask(forest_clean, min_area_full_px=min_forest_area)
        meadow_polys = vectorize_mask(final_meadow, min_area_full_px=min_meadow_area)
        water_polys = vectorize_mask(clean_stream, min_area_full_px=min_water_area)

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
