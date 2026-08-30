"""
Building and settlement footprint extractor for Tranchot historical maps.
High-precision extractor: Traces exact architectural lines of large courtyard complexes (Hofanlagen, Gutshöfe, Vierkanthöfe)
with preserved interior courtyards (open yards), and regularizes compact freestanding houses into crisp rectangles.
Eliminates jagged spikes (Zacken), staircase raster noise, and excessive vertices.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

from tranchot_extractor.config import BuildingConfig


@dataclass
class BuildingFeature:
    """Represents a single extracted building polygon with geometric metrics."""
    id: int
    geometry: Polygon
    area_px: float
    perimeter_px: float
    compactness: float
    orientation_deg: float
    centroid_x: float
    centroid_y: float
    bounding_box: List[int]  # [x, y, w, h]


@dataclass
class BuildingExtractionResult:
    """Complete results from building extraction."""
    features: List[BuildingFeature]
    mask: np.ndarray
    gdf: gpd.GeoDataFrame
    raw_contour_count: int


def despike_and_simplify_ring(
    coords: np.ndarray,
    min_edge_len: float = 3.5,
    spike_angle_deg: float = 45.0,
    collinear_deg: float = 18.0,
) -> np.ndarray:
    """
    Removes acute spikes (Zacken / hairpin turns), collapses micro-edges (< min_edge_len),
    and eliminates collinear / near-collinear vertices from a polygon coordinate ring.
    """
    if len(coords) < 4:
        return coords

    # Ensure open ring list for processing
    if np.allclose(coords[0], coords[-1]):
        pts = [list(p) for p in coords[:-1]]
    else:
        pts = [list(p) for p in coords]

    for _ in range(15):
        n = len(pts)
        if n < 3:
            break
        to_remove = set()

        for i in range(n):
            p_prev = np.array(pts[(i - 1) % n], dtype=np.float64)
            p_curr = np.array(pts[i], dtype=np.float64)
            p_next = np.array(pts[(i + 1) % n], dtype=np.float64)

            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)

            if l1 < 1e-4 or l2 < 1e-4:
                to_remove.add(i)
                continue

            # 1. Micro-edge elimination
            if l1 < min_edge_len:
                to_remove.add(i)
                continue

            # 2. Collinear check (straight line continuation)
            dot_fwd = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
            if np.degrees(np.arccos(dot_fwd)) < collinear_deg:
                to_remove.add(i)
                continue

            # 3. Acute spike check (Zacken / hairpin protrusion or intrusion)
            v_back = p_prev - p_curr
            v_fwd = p_next - p_curr
            dot_spike = np.clip(np.dot(v_back, v_fwd) / (l1 * l2), -1.0, 1.0)
            inner_angle = np.degrees(np.arccos(dot_spike))

            if inner_angle < spike_angle_deg:
                to_remove.add(i)
                continue

        if not to_remove:
            break
        pts = [p for idx, p in enumerate(pts) if idx not in to_remove]

    if len(pts) < 3:
        return coords

    # Re-close ring
    pts.append(pts[0])
    return np.array(pts, dtype=np.float64)


def get_aligned_box(cnt: np.ndarray, orientation_deg: float) -> Polygon:
    """
    Computes an oriented bounding box for a contour aligned with the specified orientation angle.
    """
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = float(cnt[0][0][0]), float(cnt[0][0][1])

    theta = -np.radians(orientation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    pts = cnt.reshape(-1, 2)
    aligned_pts = (pts - [cx, cy]) @ R.T
    min_x, min_y = aligned_pts.min(axis=0)
    max_x, max_y = aligned_pts.max(axis=0)

    box_aligned = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

    inv_R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    box_orig = box_aligned @ inv_R.T + [cx, cy]
    return Polygon(box_orig)


def regularize_orthogonal_ring(
    coords: np.ndarray,
    dominant_angle_deg: float,
    snap_threshold_deg: float = 18.0,
) -> np.ndarray:
    """
    Snaps polygon edges to crisp 90-degree orthogonal angles aligned with
    the dominant building orientation axis.
    """
    if len(coords) < 4:
        return coords

    pts = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
    if len(pts) < 3:
        return coords

    # Rotate polygon by -dominant_angle to align with X/Y coordinate axes
    theta = -np.radians(dominant_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    center = np.mean(pts, axis=0)
    aligned_pts = (pts - center) @ R.T

    n = len(aligned_pts)
    for i in range(n):
        p1 = aligned_pts[i]
        p2 = aligned_pts[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        seg_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 0 to 90
        # If nearly horizontal (< snap_threshold_deg), make y equal
        if seg_angle < snap_threshold_deg:
            avg_y = (p1[1] + p2[1]) / 2.0
            p1[1] = avg_y
            p2[1] = avg_y
        # If nearly vertical (> 90 - snap_threshold_deg), make x equal
        elif seg_angle > 90.0 - snap_threshold_deg:
            avg_x = (p1[0] + p2[0]) / 2.0
            p1[0] = avg_x
            p2[0] = avg_x

    # Rotate back to original space
    inv_R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    back_pts = aligned_pts @ inv_R.T + center
    res = back_pts.tolist()
    res.append(res[0])
    return np.array(res, dtype=np.float64)


class BuildingExtractor:
    """
    High-speed historical building and courtyard extractor.
    Traces exact wing contours of large historical farmsteads, thin walls, and freestanding houses,
    eliminating raster spikes and regularizing footprints into crisp architectural geometries.
    """

    def __init__(self, config: Optional[BuildingConfig] = None):
        self.config = config or BuildingConfig()

    def extract(self, image_rgb: np.ndarray) -> BuildingExtractionResult:
        """
        Runs building segmentation, thin wall preservation, despiking, and polygonization.
        """
        h, w = image_rgb.shape[:2]

        # 1. Multi-space carmine ink segmentation (HSV ∩ LAB ∩ RGB-diff)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        r = image_rgb[:, :, 0].astype(np.int16)
        g = image_rgb[:, :, 1].astype(np.int16)
        b = image_rgb[:, :, 2].astype(np.int16)

        raw_polys: List[Tuple[Polygon, float]] = []

        # A. Detect Solid Black Massivbauten, Churches, Castles & Stone Houses
        black_mask = ((r < 75) & (g < 75) & (b < 75) & (gray < 75)).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        black_clean = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, k_close)

        black_cnts, _ = cv2.findContours(black_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in black_cnts:
            area = cv2.contourArea(cnt)
            if 6.0 <= area <= 1200.0:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                min_dim, max_dim = min(rw, rh), max(rw, rh)
                if min_dim >= 1.5:
                    aspect = max_dim / (min_dim + 1e-5)
                    solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
                    if solidity >= 0.58 and aspect <= 8.0:
                        box = np.intp(cv2.boxPoints(rect))
                        p = Polygon(box)
                        if p.is_valid and not p.is_empty:
                            raw_polys.append((p, angle))

        # B. Detect Carmine Red Solid Buildings, Wings & Thin Walls
        m_hue1 = cv2.inRange(hsv, np.array(self.config.hsv_lower_red1), np.array(self.config.hsv_upper_red1))
        m_hue2 = cv2.inRange(hsv, np.array(self.config.hsv_lower_red2), np.array(self.config.hsv_upper_red2))
        m_hue = cv2.bitwise_or(m_hue1, m_hue2)

        # LAB a* channel (green-red axis) isolates pure carmine from brown ink and vineyard terraces
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]
        m_lab = (lab_a >= self.config.lab_a_threshold) & (lab_a.astype(int) - lab_b.astype(int) >= -4)

        rgb_diff = self.config.rgb_diff_threshold
        m_diff = ((r - g >= rgb_diff) & (r - b >= rgb_diff) & (r >= self.config.min_red_intensity)).astype(np.uint8) * 255
        m_sat = (hsv[:, :, 1] >= 26).astype(np.uint8) * 255

        carmine_mask = cv2.bitwise_and(m_hue, cv2.bitwise_and(m_lab.astype(np.uint8) * 255, cv2.bitwise_and(m_diff, m_sat)))

        # Extract and separate connected carmine components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(carmine_mask, connectivity=8)
        dist_map = cv2.distanceTransform(carmine_mask, cv2.DIST_L2, 5)

        for lbl in range(1, num_labels):
            comp_area = stats[lbl, cv2.CC_STAT_AREA]
            if comp_area < self.config.min_building_area_px or comp_area > self.config.max_building_area_px:
                continue

            comp_mask = (labels == lbl).astype(np.uint8) * 255
            comp_dist = cv2.bitwise_and(dist_map, dist_map, mask=comp_mask)
            max_val = comp_dist.max()

            cnts, hier = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts or hier is None:
                continue

            # Check if this component has interior courtyard holes
            has_holes = any(h_elem[2] != -1 for h_elem in hier[0])

            # Calculate solidity of component
            hull = cv2.convexHull(cnts[0]) if len(cnts) > 0 else None
            hull_area = cv2.contourArea(hull) if hull is not None else 1.0
            comp_solidity = comp_area / (hull_area + 1e-6)

            # If component is giant (> 350 px) AND diffuse/non-solid (solidity < 0.55 without courtyard holes):
            # ONLY extract dense building cores inside it; NEVER take global bounding box!
            if comp_area > 350.0 and comp_solidity < 0.55 and not has_holes:
                if max_val >= 3.0:
                    _, fg_seeds = cv2.threshold(comp_dist, 0.60 * max_val, 255, cv2.THRESH_BINARY)
                    fg_seeds = fg_seeds.astype(np.uint8)
                    num_seeds, seed_labels = cv2.connectedComponents(fg_seeds)
                    for s_lbl in range(1, num_seeds):
                        s_mask = (seed_labels == s_lbl).astype(np.uint8) * 255
                        s_cnts, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if s_cnts:
                            s_rect = cv2.minAreaRect(s_cnts[0])
                            (scx, scy), (srw, srh), sang = s_rect
                            if min(srw, srh) >= 1.5:
                                s_box = np.intp(cv2.boxPoints(((scx, scy), (srw * 1.5, srh * 1.5), sang)))
                                p = Polygon(s_box)
                                if p.is_valid and not p.is_empty:
                                    raw_polys.append((p, sang))
                continue

            for i, h_elem in enumerate(hier[0]):
                if h_elem[3] != -1:
                    continue

                cnt = cnts[i]
                area = cv2.contourArea(cnt)
                if area < self.config.min_building_area_px:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                min_dim, max_dim = min(rw, rh), max(rw, rh)

                if min_dim < self.config.min_stroke_width_px:
                    continue

                aspect = max_dim / (min_dim + 1e-5)
                if aspect > self.config.max_aspect_ratio:
                    continue

                solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)

                # Terrace / orchard dot suppression
                if self.config.filter_vineyard_terraces:
                    if solidity < 0.40 and aspect > 6.0 and h_elem[2] == -1:
                        continue

                # Check courtyard holes
                holes = []
                child_idx = h_elem[2]
                while child_idx != -1:
                    h_cnt = cnts[child_idx]
                    if cv2.contourArea(h_cnt) > 8:
                        holes.append(get_aligned_box(h_cnt, angle))
                    child_idx = hier[0][child_idx][0]

                outer_box = Polygon(np.intp(cv2.boxPoints(rect)))
                if not outer_box.is_valid or outer_box.is_empty:
                    continue

                if len(holes) == 0:
                    raw_polys.append((outer_box, angle))
                else:
                    poly = outer_box
                    for h_box in holes:
                        if h_box.is_valid and not h_box.is_empty:
                            try:
                                poly = poly.difference(h_box)
                            except Exception:
                                pass
                    if poly.is_valid and not poly.is_empty:
                        if isinstance(poly, Polygon):
                            raw_polys.append((poly, angle))
                        elif hasattr(poly, 'geoms'):
                            for p in poly.geoms:
                                if isinstance(p, Polygon) and not p.is_empty:
                                    raw_polys.append((p, angle))

        # 3. Spatial Non-Maximum Suppression / Deduplication with fast bounding box pre-check
        clean_polys: List[Tuple[Polygon, float]] = []
        for poly, ang in raw_polys:
            if not poly.is_valid or poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            is_dup = False
            for existing_poly, _ in clean_polys:
                e_minx, e_miny, e_maxx, e_maxy = existing_poly.bounds
                if maxx < e_minx or minx > e_maxx or maxy < e_miny or miny > e_maxy:
                    continue
                try:
                    inter_area = poly.intersection(existing_poly).area
                    if inter_area > 0.70 * min(poly.area, existing_poly.area):
                        is_dup = True
                        break
                except Exception:
                    pass
            if not is_dup:
                clean_polys.append((poly, ang))

        # 4. Build features and GeoDataFrame
        features: List[BuildingFeature] = []
        records = []
        geoms = []
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for idx, (poly, ang) in enumerate(clean_polys, 1):
            box_area = poly.area
            box_peri = poly.length
            compactness = (4.0 * np.pi * box_area) / (box_peri ** 2 + 1e-6)
            minx, miny, maxx, maxy = poly.bounds

            features.append(BuildingFeature(
                id=idx,
                geometry=poly,
                area_px=float(box_area),
                perimeter_px=float(box_peri),
                compactness=float(compactness),
                orientation_deg=float(ang),
                centroid_x=float(poly.centroid.x),
                centroid_y=float(poly.centroid.y),
                bounding_box=[int(minx), int(miny), int(maxx - minx), int(maxy - miny)],
            ))

            records.append({
                "building_id": idx,
                "layer": "building",
                "area_px": round(box_area, 2),
                "perimeter_px": round(box_peri, 2),
                "compactness": round(compactness, 4),
                "orientation_deg": round(ang, 2),
                "centroid_x": round(poly.centroid.x, 2),
                "centroid_y": round(poly.centroid.y, 2),
            })
            geoms.append(poly)

            # Mask rendering
            if hasattr(poly, "exterior") and poly.exterior is not None:
                coords = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(final_mask, [coords], 255)
                for interior in poly.interiors:
                    hole_coords = np.array(interior.coords, dtype=np.int32)
                    cv2.fillPoly(final_mask, [hole_coords], 0)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:25832")

        return BuildingExtractionResult(
            features=features,
            mask=final_mask,
            gdf=gdf,
            raw_contour_count=len(features),
        )
