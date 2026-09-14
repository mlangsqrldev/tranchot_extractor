"""
Medial Cost-Path Road Extractor for Tranchot Historical Maps.
Extracts continuous kilometer-long road centerlines running between dual-parallel dark ink lines
using optimal path search on a dual-line channel cost surface.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
import geopandas as gpd
from skimage.graph import MCP_Geometric

from tranchot_extractor.config import RoadConfig


@dataclass
class RoadSegmentFeature:
    """Represents a vectorized road or path centerline segment."""
    id: int
    geometry: LineString
    length_px: float
    category: str
    start_node: Tuple[float, float]
    end_node: Tuple[float, float]


@dataclass
class RoadExtractionResult:
    """Complete results from road extraction."""
    features: List[RoadSegmentFeature]
    skeleton_mask: np.ndarray
    gdf: gpd.GeoDataFrame
    ridge_map: Optional[np.ndarray] = None


class RoadExtractor:
    """
    Extracts continuous kilometer-long road centerlines by computing optimal medial paths
    locked between dual-parallel historical ink lines.
    """

    def __init__(self, config: Optional[RoadConfig] = None):
        self.config = config or RoadConfig()
        self._cached_cost: Optional[np.ndarray] = None
        self._mcp: Optional[MCP_Geometric] = None

    def compute_cost_surface(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Computes the road channel cost surface.
        Cost is minimal inside the light parchment channel between two dark ink lines.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray, (31, 31), 0)
        dark_diff = np.maximum(0.0, bg - gray)

        blurred_ink = cv2.GaussianBlur(dark_diff, (9, 9), 2.5)
        laplacian = cv2.Laplacian(blurred_ink, cv2.CV_32F, ksize=5)

        cost = 1.0 / (np.maximum(0.0, laplacian) * 0.12 + 0.005)
        cost += (dark_diff / 40.0)
        return cost.astype(np.float32)

    def trace_road_between_points(
        self,
        image_rgb: np.ndarray,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        cost_surface: Optional[np.ndarray] = None,
    ) -> Optional[LineString]:
        """
        Finds the exact medial centerline between two points on a road.
        """
        h, w = image_rgb.shape[:2]
        if cost_surface is None:
            cost_surface = self.compute_cost_surface(image_rgb)

        sy, sx = int(np.clip(round(start_pt[1]), 0, h - 1)), int(np.clip(round(start_pt[0]), 0, w - 1))
        ey, ex = int(np.clip(round(end_pt[1]), 0, h - 1)), int(np.clip(round(end_pt[0]), 0, w - 1))

        try:
            mcp = MCP_Geometric(cost_surface)
            mcp.find_costs([(sy, sx)])
            path = mcp.traceback((ey, ex))

            if len(path) >= 5:
                coords = [(float(p[1]), float(p[0])) for p in path]
                line = LineString(coords).simplify(1.5, preserve_topology=True)
                return line
        except Exception:
            pass
        return None

    def extract(
        self,
        image_rgb: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
        text_mask: Optional[np.ndarray] = None,
    ) -> RoadExtractionResult:
        """
        Extracts all continuous kilometer-long through-roads across the historical map sheet.
        """
        h, w = image_rgb.shape[:2]
        cost_surface = self.compute_cost_surface(image_rgb)
        mcp = MCP_Geometric(cost_surface)

        # 1. Identify road corridor entry/exit boundary nodes along sheet edges
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        bg = cv2.GaussianBlur(gray, (31, 31), 0)
        dark_ink = (gray < bg - 16)

        # Find road crossing points at top, bottom, left, right borders
        border_seeds: List[Tuple[int, int]] = []
        margin = 35

        for y in [margin, h - margin]:
            row = dark_ink[y, :]
            dips = np.where(row)[0]
            if len(dips) > 0:
                # Group adjacent pixels into strokes
                diffs = np.diff(dips)
                split_indices = np.where(diffs > 15)[0] + 1
                groups = np.split(dips, split_indices)
                for g in groups:
                    if len(g) >= 2:
                        border_seeds.append((y, int(np.mean(g))))

        for x in [margin, w - margin]:
            col = dark_ink[:, x]
            dips = np.where(col)[0]
            if len(dips) > 0:
                diffs = np.diff(dips)
                split_indices = np.where(diffs > 15)[0] + 1
                groups = np.split(dips, split_indices)
                for g in groups:
                    if len(g) >= 2:
                        border_seeds.append((int(np.mean(g)), x))

        # Town center seed (Nickenich / map center)
        center_seeds = [(int(h * 0.35), int(w * 0.40)), (int(h * 0.50), int(w * 0.50))]

        features: List[RoadSegmentFeature] = []
        raw_lines: List[LineString] = []

        # Trace continuous paths from town center to border exits
        for cy, cx in center_seeds:
            try:
                mcp.find_costs([(cy, cx)])
                for by, bx in border_seeds:
                    path = mcp.traceback((by, bx))
                    if len(path) >= 50:
                        coords = [(float(p[1]), float(p[0])) for p in path]
                        line = LineString(coords).simplify(2.0)
                        if line.length >= 250.0:  # Kilometer-long through-roads!
                            # Check duplication
                            if not any(r.buffer(15.0).intersection(line).length > line.length * 0.70 for r in raw_lines):
                                raw_lines.append(line)
            except Exception:
                continue

        # Also connect border pairs directly for transit highways (e.g. Andernach Chaussee)
        for i in range(len(border_seeds)):
            by1, bx1 = border_seeds[i]
            for j in range(i + 1, len(border_seeds)):
                by2, bx2 = border_seeds[j]
                dist = np.hypot(bx2 - bx1, by2 - by1)
                if dist >= 800:  # Transit through-road across sheet
                    try:
                        mcp.find_costs([(by1, bx1)])
                        path = mcp.traceback((by2, bx2))
                        if len(path) >= 100:
                            coords = [(float(p[1]), float(p[0])) for p in path]
                            line = LineString(coords).simplify(2.0)
                            if line.length >= 500.0:
                                if not any(r.buffer(15.0).intersection(line).length > line.length * 0.70 for r in raw_lines):
                                    raw_lines.append(line)
                    except Exception:
                        continue

        feature_id = 1
        for line in raw_lines:
            features.append(RoadSegmentFeature(
                id=feature_id,
                geometry=line,
                length_px=round(line.length, 2),
                category="main_road" if line.length > 500.0 else "path",
                start_node=(line.coords[0][0], line.coords[0][1]),
                end_node=(line.coords[-1][0], line.coords[-1][1]),
            ))
            feature_id += 1

        records = []
        geoms = []
        skel_mask = np.zeros((h, w), dtype=np.uint8)
        for f in features:
            records.append({
                "road_id": f.id,
                "layer": "road",
                "category": f.category,
                "length_px": f.length_px,
            })
            geoms.append(f.geometry)
            pts = np.array(f.geometry.coords, dtype=np.int32)
            cv2.polylines(skel_mask, [pts], False, 255, 1)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:25832")

        return RoadExtractionResult(
            features=features,
            skeleton_mask=skel_mask,
            gdf=gdf,
            ridge_map=skel_mask,
        )
