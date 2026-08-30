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
from scipy.ndimage import maximum_filter
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
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


def regularize_manhattan_contour(cnt: np.ndarray, angle_deg: float, simplify_eps: float = 2.0) -> Optional[Polygon]:
    """
    Regularizes ANY historical building contour (L-shape, U-shape, T-shape, or rectangle)
    into a crisp architectural polygon with exact 90-degree and 270-degree right angles,
    preserving open courtyards without filling them with giant bounding boxes.
    """
    if len(cnt) < 4:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    pts = cnt.reshape(-1, 2).astype(np.float64)
    local_pts = (pts - np.array([cx, cy])) @ R.T

    local_cnt = local_pts.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(local_cnt, simplify_eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    n = len(approx)
    ortho_pts = []
    for i in range(n):
        p1 = approx[i]
        p2 = approx[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) >= abs(dy):
            y_snap = (p1[1] + p2[1]) / 2.0
            ortho_pts.append([p1[0], y_snap])
            ortho_pts.append([p2[0], y_snap])
        else:
            x_snap = (p1[0] + p2[0]) / 2.0
            ortho_pts.append([x_snap, p1[1]])
            ortho_pts.append([x_snap, p2[1]])

    cleaned = []
    for pt in ortho_pts:
        if len(cleaned) == 0 or np.linalg.norm(np.array(pt) - np.array(cleaned[-1])) > 2.0:
            cleaned.append(pt)

    if len(cleaned) < 3:
        return None

    cleaned = np.array(cleaned, dtype=np.float64)
    inv_R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    world_pts = cleaned @ inv_R.T + np.array([cx, cy])

    try:
        p = Polygon(world_pts)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_valid and not p.is_empty and isinstance(p, Polygon):
            return p
    except Exception:
        pass
    return None


def separate_building_clusters(comp_mask: np.ndarray, min_area: float = 6.0) -> List[np.ndarray]:
    """
    Separates clusters of touching houses and courtyard wings into individual building bodies
    using distance transform peak watershed segmentation.
    """
    comp_area = cv2.countNonZero(comp_mask)
    if comp_area < 35.0:
        return [comp_mask]

    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    max_d = dist.max()
    if max_d < 2.0:
        return [comp_mask]

    local_max = maximum_filter(dist, size=7) == dist
    local_max[dist < 0.42 * max_d] = False
    local_max[dist < 1.4] = False

    num_peaks, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if num_peaks <= 2:
        return [comp_mask]

    markers[comp_mask == 0] = num_peaks
    grad = cv2.cvtColor(255 - comp_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad, markers)

    separated = []
    for m_id in range(1, num_peaks):
        w_mask = ((markers == m_id) & (comp_mask > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(w_mask) >= min_area:
            separated.append(w_mask)

    return separated if separated else [comp_mask]


def despike_and_regularize(poly: Polygon, simplify_eps: float = 2.5) -> Tuple[Polygon, float]:
    """
    Consolidates building geometry, eliminates single-pixel staircase spikes, and regularizes
    footprints into crisp architectural rectangular/Manhattan shapes.
    """
    if not poly.is_valid or poly.is_empty:
        return poly, 0.0

    coords = np.array(poly.exterior.coords, dtype=np.float32)
    if len(coords) < 4:
        return poly, 0.0

    cnt = coords.reshape(-1, 1, 2).astype(np.int32)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    min_box_area = rw * rh
    area = poly.area
    rect_ratio = area / (min_box_area + 1e-6)

    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    # 1. Simple freestanding rectangular houses (without courtyard holes) -> 4-corner bounding rectangle
    if len(poly.interiors) == 0 and ((rect_ratio >= 0.68 and solidity >= 0.72) or (area <= 220.0 and solidity >= 0.70)):
        box_pts = np.intp(cv2.boxPoints(rect))
        p_rect = Polygon(box_pts)
        if p_rect.is_valid and not p_rect.is_empty:
            return p_rect, angle

    # 2. Complex farmsteads / courtyard wings -> Snap to 90 degree orthogonal Manhattan ring
    p_man = regularize_manhattan_contour(cnt, angle_deg=angle, simplify_eps=simplify_eps)
    if p_man is not None and p_man.is_valid and not p_man.is_empty:
        # Preserve interior courtyards
        for interior in poly.interiors:
            h_pts = np.array(interior.coords)
            if len(h_pts) >= 4:
                h_poly = Polygon(h_pts).buffer(0)
                if h_poly.is_valid and h_poly.area >= 12.0:
                    try:
                        p_man = p_man.difference(h_poly)
                    except Exception:
                        pass
        if isinstance(p_man, Polygon):
            return p_man, angle

    # 3. Fallback Douglas-Peucker simplification
    return poly.simplify(simplify_eps, preserve_topology=True), angle


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

        seg_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if seg_angle < snap_threshold_deg:
            avg_y = (p1[1] + p2[1]) / 2.0
            p1[1] = avg_y
            p2[1] = avg_y
        elif seg_angle > 90.0 - snap_threshold_deg:
            avg_x = (p1[0] + p2[0]) / 2.0
            p1[0] = avg_x
            p2[0] = avg_x

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

    def extract(
        self,
        image_rgb: np.ndarray,
        settlement_boundaries: Optional[List[Polygon]] = None,
    ) -> BuildingExtractionResult:
        """
        Runs building segmentation, thin wall preservation, despiking, and polygonization.
        If settlement_boundaries are provided, restricts extraction exclusively within these polygons.
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
            if 6.0 <= area <= 1500.0:
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
        m_lab = (lab_a >= self.config.lab_a_threshold) & (lab_a.astype(int) - lab_b.astype(int) >= -8)

        rgb_diff = self.config.rgb_diff_threshold
        m_diff = ((r - g >= rgb_diff) & (r - b >= max(rgb_diff, 16)) & (r >= self.config.min_red_intensity)).astype(np.uint8) * 255
        m_sat = (hsv[:, :, 1] >= 18).astype(np.uint8) * 255

        carmine_mask = cv2.bitwise_and(m_hue, cv2.bitwise_and(m_lab.astype(np.uint8) * 255, cv2.bitwise_and(m_diff, m_sat)))

        # Extract and separate connected carmine components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(carmine_mask, connectivity=8)

        for lbl in range(1, num_labels):
            comp_area = stats[lbl, cv2.CC_STAT_AREA]
            if comp_area < self.config.min_building_area_px or comp_area > self.config.max_building_area_px:
                continue

            raw_comp = (labels == lbl).astype(np.uint8) * 255

            cnts_orig, hier_orig = cv2.findContours(raw_comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            has_holes = False
            if hier_orig is not None:
                has_holes = any(h_elem[2] != -1 for h_elem in hier_orig[0])

            # If component has closed interior courtyard holes, preserve as single enclosed courtyard
            if has_holes:
                wing_masks = [raw_comp]
            else:
                wing_masks = separate_building_clusters(raw_comp, min_area=self.config.min_building_area_px)

            for w_mask in wing_masks:
                cnts, hier = cv2.findContours(w_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts or hier is None:
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
                    rect_area = rw * rh
                    rectangularity = area / (rect_area + 1e-6)
                    hull = cv2.convexHull(cnt)
                    solidity = area / (cv2.contourArea(hull) + 1e-6)
                    min_dim, max_dim = min(rw, rh), max(rw, rh)

                    if min_dim < self.config.min_stroke_width_px:
                        continue

                    aspect = max_dim / (min_dim + 1e-5)
                    if aspect > self.config.max_aspect_ratio:
                        continue

                    # Suppress vineyard hillside terrace hatching:
                    # Terraces are thin (min_dim <= 2.2 px), isolated open line slivers with high aspect ratio (aspect >= 4.0)
                    # and low solidity (solidity < 0.68) without interior courtyard holes.
                    if self.config.filter_vineyard_terraces:
                        is_vineyard_terrace = (
                            min_dim <= 2.2 and
                            aspect >= 4.0 and
                            solidity < 0.68 and
                            not has_holes and
                            h_elem[2] == -1
                        )
                        if is_vineyard_terrace:
                            continue

                        if area < 8.0 and (solidity < 0.60 or min_dim < 1.4):
                            continue

                    holes = []
                    child_idx = h_elem[2]
                    while child_idx != -1:
                        h_cnt = cnts[child_idx]
                        if cv2.contourArea(h_cnt) > 8:
                            h_rect = cv2.minAreaRect(h_cnt)
                            holes.append(Polygon(np.intp(cv2.boxPoints(h_rect))))
                        child_idx = hier[0][child_idx][0]

                    is_simple_rectangle = (
                        (rectangularity >= 0.70 and solidity >= 0.78) or
                        (area <= 90.0 and solidity >= 0.72 and aspect <= 5.0) or
                        (min_dim <= 2.8 and aspect >= 2.5)
                    ) and len(holes) == 0

                    if is_simple_rectangle:
                        poly = Polygon(np.intp(cv2.boxPoints(rect)))
                        if poly.is_valid and not poly.is_empty:
                            raw_polys.append((poly, angle))
                    else:
                        # L-shaped, U-shaped, C-shaped courtyard regularizer
                        poly = regularize_manhattan_contour(cnt, angle_deg=angle)
                        if poly is None or not poly.is_valid or poly.is_empty:
                            approx = cv2.approxPolyDP(cnt, 2.0, True).reshape(-1, 2)
                            if len(approx) >= 4:
                                poly = Polygon(approx)

                        if poly and poly.is_valid and not poly.is_empty:
                            for h_poly in holes:
                                if h_poly.is_valid:
                                    try:
                                        poly = poly.difference(h_poly)
                                    except Exception:
                                        pass
                            if isinstance(poly, Polygon) and not poly.is_empty:
                                raw_polys.append((poly, angle))
                            elif hasattr(poly, 'geoms'):
                                for p in poly.geoms:
                                    if isinstance(p, Polygon) and not p.is_empty:
                                        raw_polys.append((p, angle))

        # Filter strictly within settlement boundaries if provided
        if settlement_boundaries and len(settlement_boundaries) > 0:
            scoped_polys = []
            for poly, ang in raw_polys:
                if any(sb.intersects(poly.centroid) or sb.intersects(poly) for sb in settlement_boundaries):
                    scoped_polys.append((poly, ang))
            raw_polys = scoped_polys

        # 3. Consolidate fragmented house pieces and despike polygons into clean blocks
        valid_raw = [p for p, ang in raw_polys if p.is_valid and not p.is_empty]
        if valid_raw:
            try:
                solid_raw = [p for p in valid_raw if len(p.interiors) == 0]
                holed_raw = [p for p in valid_raw if len(p.interiors) > 0]
                merged_list = list(holed_raw)

                if solid_raw:
                    buffered = [p.buffer(1.2, join_style=2) for p in solid_raw]
                    merged_union = unary_union(buffered)
                    unbuffered = merged_union.buffer(-1.2, join_style=2)
                    if isinstance(unbuffered, Polygon):
                        merged_list.append(unbuffered)
                    elif isinstance(unbuffered, MultiPolygon):
                        merged_list.extend(list(unbuffered.geoms))
                if not merged_list:
                    merged_list = valid_raw
            except Exception:
                merged_list = valid_raw
        else:
            merged_list = []

        clean_polys: List[Tuple[Polygon, float]] = []
        for poly in merged_list:
            if not poly.is_valid or poly.is_empty or poly.area < self.config.min_building_area_px:
                continue
            p_clean, ang = despike_and_regularize(poly, simplify_eps=self.config.simplify_tolerance)
            if p_clean.is_valid and not p_clean.is_empty and p_clean.area >= self.config.min_building_area_px:
                clean_polys.append((p_clean, ang))

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
