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
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd

from tranchot_extractor.config import BuildingConfig


def unpack_polygons(geom: Any) -> List[Polygon]:
    """
    Recursively extracts all valid 2D Polygon instances from any geometry,
    including Polygon, MultiPolygon, GeometryCollection, and handles make_valid outputs.
    """
    out: List[Polygon] = []
    if geom is None or geom.is_empty:
        return out
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, Polygon):
        if not geom.is_empty and geom.area > 0:
            out.append(geom)
    elif isinstance(geom, MultiPolygon):
        for sub in geom.geoms:
            if isinstance(sub, Polygon) and not sub.is_empty and sub.area > 0:
                out.append(sub)
    elif hasattr(geom, "geoms"):
        for sub in geom.geoms:
            out.extend(unpack_polygons(sub))
    return out



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

    pts = cv2.findNonZero(comp_mask)
    if pts is None:
        return [comp_mask]
    bx, by, bw, bh = cv2.boundingRect(pts)
    pad = 4
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(comp_mask.shape[1], bx + bw + pad), min(comp_mask.shape[0], by + bh + pad)

    local_comp = comp_mask[y0:y1, x0:x1]
    dist = cv2.distanceTransform(local_comp, cv2.DIST_L2, 5)
    max_d = dist.max()
    if max_d < 2.0:
        return [comp_mask]

    local_max = maximum_filter(dist, size=7) == dist
    local_max[dist < 0.42 * max_d] = False
    local_max[dist < 1.4] = False

    num_peaks, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if num_peaks <= 2:
        return [comp_mask]

    markers[local_comp == 0] = num_peaks
    grad = cv2.cvtColor(255 - local_comp, cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad, markers)

    separated = []
    for m_id in range(1, num_peaks):
        w_sub = ((markers == m_id) & (local_comp > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(w_sub) >= min_area:
            w_full = np.zeros_like(comp_mask)
            w_full[y0:y1, x0:x1] = w_sub
            separated.append(w_full)

    return separated if separated else [comp_mask]


def regularize_orthogonal_ring(
    coords: np.ndarray,
    dominant_angle_deg: float,
    snap_threshold_deg: float = 22.0,
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


def despike_and_regularize(poly: Polygon, simplify_eps: float = 2.0) -> Tuple[Any, float]:
    """
    Consolidates building geometry, eliminates single-pixel staircase spikes, and regularizes
    footprints into crisp architectural rectangular/Manhattan shapes.
    Preserves and orthogonalizes interior courtyards (Hofanlagen).
    """
    if not poly.is_valid:
        poly_valid = make_valid(poly)
        unpacked = unpack_polygons(poly_valid)
        if not unpacked:
            return poly, 0.0
        poly = max(unpacked, key=lambda p: p.area)

    if poly.is_empty:
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
    min_dim, max_dim = min(rw, rh), max(rw, rh)
    aspect = max_dim / (min_dim + 1e-5)

    # 1. Simple freestanding rectangular houses (without courtyard holes) -> 4-corner bounding rectangle
    if len(poly.interiors) == 0:
        is_rect = (
            (rect_ratio >= 0.50 and solidity >= 0.72 and aspect <= 8.0) or
            (solidity >= 0.80 and aspect <= 8.0) or
            (area <= 350.0 and rect_ratio >= 0.45 and solidity >= 0.66) or
            (min_dim <= 3.2 and aspect >= 2.0)
        )
        if is_rect:
            box_pts = np.intp(cv2.boxPoints(rect))
            p_rect = Polygon(box_pts)
            if p_rect.is_valid and not p_rect.is_empty:
                return p_rect, angle

    # 2. Despike and orthogonalize exterior ring
    ext_approx = cv2.approxPolyDP(cnt, simplify_eps, True).reshape(-1, 2)
    if len(ext_approx) >= 3:
        clean_ext = despike_and_simplify_ring(ext_approx, min_edge_len=3.5, spike_angle_deg=45.0)
        clean_ext = regularize_orthogonal_ring(clean_ext, dominant_angle_deg=angle, snap_threshold_deg=22.0)
    else:
        clean_ext = ext_approx

    # 3. Preserve, despike, and orthogonalize interior courtyard holes
    clean_holes = []
    for interior in poly.interiors:
        h_cnt = np.array(interior.coords, dtype=np.int32).reshape(-1, 1, 2)
        h_area = cv2.contourArea(h_cnt)
        if h_area >= 15.0:
            h_approx = cv2.approxPolyDP(h_cnt, simplify_eps, True).reshape(-1, 2)
            if len(h_approx) >= 3:
                h_clean = despike_and_simplify_ring(h_approx, min_edge_len=3.0, spike_angle_deg=45.0)
                h_clean = regularize_orthogonal_ring(h_clean, dominant_angle_deg=angle, snap_threshold_deg=22.0)
                if len(h_clean) >= 4:
                    clean_holes.append(h_clean)

    try:
        p_res = Polygon(shell=clean_ext, holes=clean_holes if clean_holes else None)
        if not p_res.is_valid:
            p_res = p_res.buffer(0)
        if p_res.is_valid and not p_res.is_empty:
            return p_res, angle
    except Exception:
        pass

    # 4. Fallback simplification
    simplified = poly.simplify(simplify_eps, preserve_topology=True)
    if not simplified.is_valid:
        simplified = make_valid(simplified)
    return simplified, angle


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

        # Fast ROI Crop Optimization: if settlement boundaries are provided, crop processing strictly to their bounds
        x_off, y_off = 0, 0
        if settlement_boundaries and len(settlement_boundaries) > 0:
            minx = min(sb.bounds[0] for sb in settlement_boundaries)
            miny = min(sb.bounds[1] for sb in settlement_boundaries)
            maxx = max(sb.bounds[2] for sb in settlement_boundaries)
            maxy = max(sb.bounds[3] for sb in settlement_boundaries)
            pad = 25
            x0 = max(0, int(minx) - pad)
            y0 = max(0, int(miny) - pad)
            x1 = min(w, int(maxx) + pad)
            y1 = min(h, int(maxy) + pad)
            if (x1 - x0) > 20 and (y1 - y0) > 20 and ((x1 - x0) < w or (y1 - y0) < h):
                image_rgb = image_rgb[y0:y1, x0:x1]
                x_off, y_off = x0, y0
                h, w = image_rgb.shape[:2]
                settlement_boundaries = [translate(sb, xoff=-x0, yoff=-y0) for sb in settlement_boundaries]

        # 1. Multi-space carmine ink segmentation (HSV ∩ LAB ∩ RGB-diff)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        r = image_rgb[:, :, 0].astype(np.int16)
        g = image_rgb[:, :, 1].astype(np.int16)
        b = image_rgb[:, :, 2].astype(np.int16)

        raw_polys: List[Tuple[Polygon, float]] = []

        # Boundary mask if settlement boundaries provided
        scope_mask = None
        if settlement_boundaries and len(settlement_boundaries) > 0:
            scope_mask = np.zeros((h, w), dtype=np.uint8)
            for sb in settlement_boundaries:
                if hasattr(sb, "exterior") and sb.exterior:
                    pts = np.array(sb.exterior.coords, dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(scope_mask, [pts], 255)

        # A. Detect Solid Black Massivbauten, Churches, Castles & Stone Houses (strictly optional, disabled by default to prevent lettering & hachures)
        if getattr(self.config, "enable_black_buildings", False):
            black_mask = ((r < 75) & (g < 75) & (b < 75) & (gray < 75)).astype(np.uint8) * 255
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            black_clean = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, k_close)
            if scope_mask is not None:
                black_clean = cv2.bitwise_and(black_clean, scope_mask)

            black_cnts, _ = cv2.findContours(black_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_blk_area = getattr(self.config, "min_black_building_area_px", 45.0)
            min_blk_thick = getattr(self.config, "min_black_building_thickness_px", 4.5)
            max_blk_aspect = getattr(self.config, "max_black_building_aspect_ratio", 3.8)
            min_blk_solidity = getattr(self.config, "min_black_building_solidity", 0.72)

            for cnt in black_cnts:
                area = cv2.contourArea(cnt)
                if min_blk_area <= area <= self.config.max_building_area_px:
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (rw, rh), angle = rect
                    min_dim, max_dim = min(rw, rh), max(rw, rh)
                    if min_dim >= min_blk_thick:
                        aspect = max_dim / (min_dim + 1e-5)
                        solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
                        if solidity >= min_blk_solidity and aspect <= max_blk_aspect:
                            box = np.intp(cv2.boxPoints(rect))
                            p = Polygon(box)
                            if p.is_valid and not p.is_empty:
                                raw_polys.append((p, angle))

        # B. Detect Carmine & Brownish/Terracotta Red Solid Buildings, Wings & Thin Walls
        # 1. Estimate local parchment white point (bright, non-ink pixels)
        paper_mask = (gray > 160) & (np.abs(r - g) < 32) & (np.abs(r - b) < 45)
        if np.count_nonzero(paper_mask) > 100:
            p_r = float(np.percentile(r[paper_mask], 85))
            p_g = float(np.percentile(g[paper_mask], 85))
            p_b = float(np.percentile(b[paper_mask], 85))
        else:
            p_r, p_g, p_b = 230.0, 220.0, 195.0

        # 2. Optical Density (Beer-Lambert law): OD = -log10(I / I_paper)
        od_r = -np.log10(np.clip(r.astype(np.float32) / (p_r + 1e-5), 0.01, 1.0))
        od_g = -np.log10(np.clip(g.astype(np.float32) / (p_g + 1e-5), 0.01, 1.0))
        carmine_od = od_g - od_r  # Pure carmine absorption signal

        # 3. Normalized RGB (Von Kries chromatic adaptation)
        r_norm = np.clip((r.astype(np.float32) / (p_r + 1e-5)) * 255.0, 0, 255).astype(np.int16)
        g_norm = np.clip((g.astype(np.float32) / (p_g + 1e-5)) * 255.0, 0, 255).astype(np.int16)
        b_norm = np.clip((b.astype(np.float32) / (p_b + 1e-5)) * 255.0, 0, 255).astype(np.int16)

        m_hue1 = cv2.inRange(hsv, np.array(self.config.hsv_lower_red1), np.array(self.config.hsv_upper_red1))
        m_hue2 = cv2.inRange(hsv, np.array(self.config.hsv_lower_red2), np.array(self.config.hsv_upper_red2))
        m_hue = cv2.bitwise_or(m_hue1, m_hue2) > 0

        # LAB a* channel (green-red axis) includes weathered terracotta, courtyard walls and carmine
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]
        m_lab = (lab_a >= self.config.lab_a_threshold) & (lab_a.astype(int) - lab_b.astype(int) >= -4)

        rgb_diff = self.config.rgb_diff_threshold
        min_red = self.config.min_red_intensity
        min_norm_red = getattr(self.config, "min_norm_red_intensity", 135)

        # Carmine / terracotta red hue: H in [0, 16] or [165, 180], sat >= 22 (excludes yellow/ochre roads and brown iron-gall ink at H 22..45)
        m_carmine_hue = ((hsv[:, :, 0] <= 16) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 22)
        m_od = (carmine_od >= getattr(self.config, "carmine_od_threshold", 0.035)) & (r_norm > g_norm) & (r_norm > b_norm) & (g_norm - r_norm < 12)
        m_norm_diff = (r_norm - g_norm >= getattr(self.config, "norm_rgb_diff_threshold", 14)) & (r_norm - b_norm >= getattr(self.config, "norm_rgb_diff_threshold", 14))
        m_legacy_diff = ((r.astype(int) - g.astype(int) >= rgb_diff) & (r.astype(int) - b.astype(int) >= max(rgb_diff, 8)))

        # Gated Carmine Mask: every carmine pixel MUST strictly satisfy carmine hue, LAB red threshold, minimum red intensity, and positive normalized red excess
        m_signal = (m_od | m_norm_diff | m_legacy_diff) & (r >= min_red) & (r_norm >= min_norm_red)
        carmine_mask = (m_signal & m_carmine_hue & m_lab & (r_norm > g_norm)).astype(np.uint8) * 255
        if scope_mask is not None:
            carmine_mask = cv2.bitwise_and(carmine_mask, scope_mask)

        # Morphological clean: only apply opening if explicitly requested, preserving fine 1-2 px walls
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        carmine_mask = cv2.morphologyEx(carmine_mask, cv2.MORPH_CLOSE, k_close)
        if getattr(self.config, "morph_kernel_size", 1) > 1 and self.config.min_stroke_width_px > 2.0:
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            carmine_mask = cv2.morphologyEx(carmine_mask, cv2.MORPH_OPEN, k_open)

        # Distance transform across candidate carmine pixels to enforce architectural building thickness
        dist_all = cv2.distanceTransform(carmine_mask, cv2.DIST_L2, 3)

        # Extract and separate connected carmine components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(carmine_mask, connectivity=8)

        for lbl in range(1, num_labels):
            comp_area = stats[lbl, cv2.CC_STAT_AREA]
            if comp_area < self.config.min_building_area_px or comp_area > self.config.max_building_area_px:
                continue

            raw_comp = (labels == lbl).astype(np.uint8) * 255

            cnts_orig, hier_orig = cv2.findContours(raw_comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            num_large_holes = 0
            total_hole_area = 0.0
            if hier_orig is not None:
                for h_elem in hier_orig[0]:
                    if h_elem[2] != -1:
                        child = h_elem[2]
                        while child != -1:
                            child_area = cv2.contourArea(cnts_orig[child])
                            if child_area >= 35.0:
                                num_large_holes += 1
                                total_hole_area += child_area
                            child = hier_orig[0][child][0]

            comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
            comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
            comp_thick = np.max(dist_all[raw_comp > 0]) * 2.0 if np.any(raw_comp > 0) else 0.0
            hof_fill_ratio = comp_area / (comp_area + total_hole_area + 1e-6)

            # A genuine individual Hofanlage has at most 2 internal yards, bounded span (<= 115 px), and substantial wall thickness
            max_hof_span = getattr(self.config, "max_hof_span_px", 115.0)
            max_hof_area = getattr(self.config, "max_unified_hof_area_px", 8500.0)
            is_genuine_hof = (
                (1 <= num_large_holes <= 2) and
                (comp_area <= max_hof_area) and
                (max(comp_w, comp_h) <= max_hof_span) and
                (comp_thick >= 3.2) and
                (hof_fill_ratio >= 0.18)
            )

            # If component is an authentic compact Hofanlage with courtyard, keep unified;
            # otherwise separate touching houses along streets using watershed distance peaks
            if is_genuine_hof:
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
                    allow_thin_walls = self.config.min_stroke_width_px <= 1.8
                    max_aspect = 35.0 if self.config.min_stroke_width_px <= 1.0 else getattr(self.config, "max_aspect_ratio", 5.5)
                    if aspect > max_aspect:
                        continue

                    # Narrow feature filter:
                    # In 1:25000 scale, buildings have a minimum width of 5-7 meters (min_dim >= 4.5 px).
                    # Narrow features (min_dim < 4.5 px) are only kept if compact (aspect <= 2.2), unless thin walls are enabled.
                    if not allow_thin_walls:
                        if min_dim < 4.5 and aspect > 2.2:
                            continue
                        if min_dim < 2.8:
                            continue

                    # Reject thin stone boundary lines, ditches, or garden perimeter fences
                    wing_thick = np.max(dist_all[w_mask > 0]) * 2.0 if np.any(w_mask > 0) else 0.0
                    min_thick = getattr(self.config, "min_building_thickness_px", 1.8)
                    if not allow_thin_walls and wing_thick < min_thick and (aspect > 2.5 or area < 40.0):
                        continue

                    # Reject giant hollow loops enclosing gardens, streets or open parcels
                    if area > 1200.0 and rectangularity < getattr(self.config, "min_courtyard_fill_ratio", 0.18):
                        continue

                    # Suppress vineyard hillside terrace hatching:
                    # Terraces are thin (min_dim < 2.5 px), open jagged slivers with high aspect ratio (aspect > 3.8)
                    # and low rectangularity (rectangularity < 0.65). Crisp rectangular walls are preserved.
                    if self.config.filter_vineyard_terraces:
                        if min_dim < 2.5 and aspect > 3.8 and rectangularity < 0.65:
                            continue
                        if area < 10.0 and (solidity < 0.60 or min_dim < 1.6):
                            continue

                    # Suppress isolated orchard tree stamp dots:
                    # Small area (<= 45 px) and compact (aspect <= 1.8) without courtyard holes
                    if area <= 45.0 and aspect <= 1.8 and h_elem[2] == -1:
                        continue

                    holes = []
                    child_idx = h_elem[2]
                    while child_idx != -1:
                        h_cnt = cnts[child_idx]
                        h_area = cv2.contourArea(h_cnt)
                        if h_area >= 15.0:
                            h_pts = cv2.approxPolyDP(h_cnt, 2.0, True).reshape(-1, 2)
                            if len(h_pts) >= 3:
                                h_clean = despike_and_simplify_ring(h_pts, min_edge_len=3.0, spike_angle_deg=45.0)
                                if len(h_clean) >= 4:
                                    h_clean = regularize_orthogonal_ring(h_clean, dominant_angle_deg=angle, snap_threshold_deg=22.0)
                                    holes.append(h_clean)
                        child_idx = hier[0][child_idx][0]

                    is_simple_rectangle = len(holes) == 0 and (
                        (rectangularity >= 0.50 and solidity >= 0.72 and aspect <= 8.0) or
                        (solidity >= 0.80 and aspect <= 8.0) or
                        (area <= 350.0 and rectangularity >= 0.45 and solidity >= 0.66) or
                        (min_dim <= 3.2 and aspect >= 2.0)
                    )

                    if is_simple_rectangle:
                        poly = Polygon(np.intp(cv2.boxPoints(rect)))
                        if poly.is_valid and not poly.is_empty:
                            raw_polys.append((poly, angle))
                    else:
                        approx = cv2.approxPolyDP(cnt, 2.5, True).reshape(-1, 2)
                        if len(approx) >= 3:
                            clean_ext = despike_and_simplify_ring(approx, min_edge_len=3.5, spike_angle_deg=45.0)
                            if len(clean_ext) >= 4:
                                clean_ext = regularize_orthogonal_ring(clean_ext, dominant_angle_deg=angle, snap_threshold_deg=22.0)
                                try:
                                    poly = Polygon(shell=clean_ext, holes=holes if holes else None)
                                    if not poly.is_valid:
                                        poly = poly.buffer(0)
                                    if poly.is_valid and not poly.is_empty:
                                        for p in unpack_polygons(poly):
                                            if p.area >= self.config.min_building_area_px:
                                                raw_polys.append((p, angle))
                                except Exception:
                                    pass

        # Filter strictly within settlement boundaries if provided
        if settlement_boundaries and len(settlement_boundaries) > 0:
            scoped_polys = []
            for poly, ang in raw_polys:
                if any(sb.intersects(poly.centroid) or sb.intersects(poly) for sb in settlement_boundaries):
                    scoped_polys.append((poly, ang))
            raw_polys = scoped_polys

        # 3. Consolidate and regularize polygons
        valid_raw = [p for p, ang in raw_polys if p.is_valid and not p.is_empty]
        clean_polys: List[Tuple[Polygon, float]] = []
        allow_thin_walls = self.config.min_stroke_width_px <= 1.8
        max_aspect = 35.0 if self.config.min_stroke_width_px <= 1.0 else getattr(self.config, "max_aspect_ratio", 5.5)

        for poly in valid_raw:
            if not poly.is_valid or poly.is_empty or poly.area < self.config.min_building_area_px:
                continue
            p_clean, ang = despike_and_regularize(poly, simplify_eps=self.config.simplify_tolerance)
            for p_sub in unpack_polygons(p_clean):
                if p_sub.area >= self.config.min_building_area_px:
                    sub_coords = np.array(p_sub.exterior.coords)
                    if len(sub_coords) >= 4:
                        sub_rect = cv2.minAreaRect(sub_coords.astype(np.int32))
                        (scx, scy), (srw, srh), _ = sub_rect
                        s_min, s_max = min(srw, srh), max(srw, srh)
                        s_aspect = s_max / (s_min + 1e-5)
                        if s_aspect > max_aspect:
                            continue
                        min_w = self.config.min_stroke_width_px if allow_thin_walls else 2.5
                        if s_min < min_w:
                            continue
                    clean_polys.append((p_sub, ang))

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

        if x_off > 0 or y_off > 0:
            shifted_features = []
            shifted_geoms = []
            for f in features:
                p_glob = translate(f.geometry, xoff=x_off, yoff=y_off)
                minx, miny, maxx, maxy = p_glob.bounds
                shifted_features.append(BuildingFeature(
                    id=f.id,
                    geometry=p_glob,
                    area_px=f.area_px,
                    perimeter_px=f.perimeter_px,
                    compactness=f.compactness,
                    orientation_deg=f.orientation_deg,
                    centroid_x=float(p_glob.centroid.x),
                    centroid_y=float(p_glob.centroid.y),
                    bounding_box=[int(minx), int(miny), int(maxx - minx), int(maxy - miny)],
                ))
                shifted_geoms.append(p_glob)
            features = shifted_features
            geoms = shifted_geoms
            for rec, geom in zip(records, geoms):
                rec["centroid_x"] = round(geom.centroid.x, 2)
                rec["centroid_y"] = round(geom.centroid.y, 2)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:25832") if geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:25832")

        return BuildingExtractionResult(
            features=features,
            mask=final_mask,
            gdf=gdf,
            raw_contour_count=len(features),
        )
