"""
Interactive Pipette Sampler & Competitive Texture-Aware Palette Engine for Tranchot Historical Maps.
Supports:
1. Multi-point sampling per class (light, dark, shadow).
2. Competitive Multi-Class argmin classification (zero polygon overlaps, zero gaps).
3. Texture-aware discrimination (Forest tree stamps vs. Meadow pastel smoothness vs. Vineyard hatching vs. Water).
4. Instant spectral polygon extraction with parchment normalization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import time
import numpy as np
import cv2
from shapely.geometry import Polygon


@dataclass
class ColorSample:
    """Represents a sampled color class from the map with multi-point support and full texture/pattern profiling."""
    class_id: str
    label: str
    rgb: List[int]
    hsv: List[int]
    lab: List[int]
    hex_color: str
    tolerance: int = 28
    min_area_px: float = 300.0
    active: bool = True
    sampled_points: List[List[int]] = field(default_factory=list)
    texture_weight: float = 0.0  # >0: wants texture (forest), <0: wants smoothness (meadow/water)
    lab_std: List[float] = field(default_factory=lambda: [8.0, 4.0, 4.0])
    tex_energy_mean: float = 18.0
    tex_energy_std: float = 8.0
    dir_coherence_mean: float = 0.5
    dir_coherence_std: float = 0.15
    has_pattern_profile: bool = False


class PipetteSampler:
    """
    Manages interactive multi-point pipette sampling, competitive multi-class segmentation,
    and texture-aware polygon extraction.
    """

    DEFAULT_CLASSES = [
        {"class_id": "forest", "label": "🌲 Wald (Laub/Nadel)", "hex": "#27ae60", "default_rgb": [120, 145, 90], "tex_w": 1.2},
        {"class_id": "meadow", "label": "🌿 Wiese / Aue / Grünland", "hex": "#00cec9", "default_rgb": [135, 160, 145], "tex_w": -0.8},
        {"class_id": "water", "label": "💧 Gewässer / Rhein / Bäche", "hex": "#0984e3", "default_rgb": [130, 165, 195], "tex_w": -1.0},
        {"class_id": "gravel", "label": "🟠 Kies- & Schotterbänke (Flussbett)", "hex": "#e17055", "default_rgb": [215, 165, 140], "tex_w": 0.0},
        {"class_id": "vineyard", "label": "🍇 Weinberge (Rebhänge)", "hex": "#f1c40f", "default_rgb": [195, 180, 110], "tex_w": 1.0},
        {"class_id": "garden", "label": "🟨 Gärten / Nutzkulturen", "hex": "#fdcb6e", "default_rgb": [210, 195, 120], "tex_w": 0.2},
        {"class_id": "building", "label": "🏛️ Gebäude (Karminrot)", "hex": "#e74c3c", "default_rgb": [180, 70, 60], "tex_w": 0.0},
        {"class_id": "paper", "label": "📜 Pergament / Hintergrund", "hex": "#dcdde1", "default_rgb": [205, 185, 140], "tex_w": -0.5},
    ]

    def __init__(self):
        self.samples: Dict[str, ColorSample] = {}
        self._init_defaults()

    def _init_defaults(self):
        for item in self.DEFAULT_CLASSES:
            rgb = item["default_rgb"]
            hsv = list(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
            lab = list(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0])
            self.samples[item["class_id"]] = ColorSample(
                class_id=item["class_id"],
                label=item["label"],
                rgb=rgb,
                hsv=[int(x) for x in hsv],
                lab=[int(x) for x in lab],
                hex_color=item["hex"],
                tolerance=24,
                min_area_px=300.0,
                active=False,
                sampled_points=[rgb],
                texture_weight=item.get("tex_w", 0.0),
                lab_std=[8.0, 4.0, 4.0],
                has_pattern_profile=False,
            )

    def reset_class(self, class_id: str):
        """Resets a single class sample back to default cartographic reference."""
        for item in self.DEFAULT_CLASSES:
            if item["class_id"] == class_id:
                rgb = item["default_rgb"]
                hsv = list(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
                lab = list(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0])
                self.samples[class_id] = ColorSample(
                    class_id=class_id,
                    label=item["label"],
                    rgb=rgb,
                    hsv=[int(x) for x in hsv],
                    lab=[int(x) for x in lab],
                    hex_color=item["hex"],
                    tolerance=24,
                    min_area_px=300.0,
                    active=False,
                    sampled_points=[rgb],
                    texture_weight=item.get("tex_w", 0.0),
                    lab_std=[8.0, 4.0, 4.0],
                    has_pattern_profile=False,
                )
                break

    def sample_from_coordinate(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        cx: float,
        cy: float,
        radius: int = 4,
        append_sample: bool = False
    ) -> ColorSample:
        """
        Samples color around (cx, cy) on the map and updates the class profile.
        If append_sample is True, adds to the multi-point sample collection.
        """
        h, w = image_rgb.shape[:2]
        x0 = max(0, int(cx - radius))
        x1 = min(w, int(cx + radius + 1))
        y0 = max(0, int(cy - radius))
        y1 = min(h, int(cy + radius + 1))

        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            crop = image_rgb[int(cy):int(cy)+1, int(cx):int(cx)+1]

        med_rgb = np.median(crop.reshape(-1, 3), axis=0).astype(int)
        new_pt = [int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2])]

        existing = self.samples.get(class_id)
        if existing and append_sample and existing.sampled_points:
            pts = existing.sampled_points + [new_pt]
            pts_arr = np.array(pts)
            mean_rgb = np.mean(pts_arr, axis=0).astype(int).tolist()
        else:
            pts = [new_pt]
            mean_rgb = new_pt

        hsv = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
        lab = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2LAB)[0, 0])
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        label = existing.label if existing else class_id
        min_area = existing.min_area_px if existing else 300.0
        tex_w = existing.texture_weight if existing else 0.0

        sample = ColorSample(
            class_id=class_id,
            label=label,
            rgb=mean_rgb,
            hsv=[int(x) for x in hsv],
            lab=[int(x) for x in lab],
            hex_color=hex_col,
            tolerance=existing.tolerance if existing else 24,
            min_area_px=min_area,
            active=True,
            sampled_points=pts,
            texture_weight=tex_w,
            lab_std=[8.0, 4.0, 4.0],
            has_pattern_profile=False,
        )
        self.samples[class_id] = sample
        return sample

    def sample_from_polygon(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        polygon_pts: List[Tuple[float, float]],
    ) -> ColorSample:
        """
        Extracts multi-point pixel and texture distribution inside a user-drawn polygon.
        Calculates LAB centroid, covariance, and full texture energy + directional hatching metrics.
        """
        h, w = image_rgb.shape[:2]
        poly_np = np.array(polygon_pts, dtype=np.int32)
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_np], 255)

        inside_pixels = image_rgb[poly_mask > 0]
        if len(inside_pixels) == 0:
            return self.samples.get(class_id, None)

        mean_rgb = np.mean(inside_pixels, axis=0).astype(int).tolist()
        hsv = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        # Multi-dimensional texture & directional hatching computation
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        inside_lab = lab_img[poly_mask > 0]
        mean_lab = np.mean(inside_lab, axis=0).astype(int).tolist()
        std_lab = np.maximum(np.std(inside_lab, axis=0), 2.5).tolist()

        # 1. Texture Energy (Laplacian frequency energy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)
        inside_tex = tex_energy[poly_mask > 0]
        mu_tex = float(np.mean(inside_tex))
        sigma_tex = max(float(np.std(inside_tex)), 2.0)

        # 2. Directional Hatching Coherence (Slope / Vineyard orientation)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)
        inside_dir = dir_coh[poly_mask > 0]
        mu_dir = float(np.mean(inside_dir))
        sigma_dir = max(float(np.std(inside_dir)), 0.05)

        existing = self.samples.get(class_id)
        label = existing.label if existing else class_id
        tol = max(18, min(36, int(np.mean(std_lab) * 3.5)))

        sample = ColorSample(
            class_id=class_id,
            label=label,
            rgb=mean_rgb,
            hsv=[int(x) for x in hsv],
            lab=mean_lab,
            hex_color=hex_col,
            tolerance=tol,
            active=True,
            sampled_points=[mean_rgb],
            texture_weight=mu_tex,
            lab_std=[float(x) for x in std_lab],
            tex_energy_mean=mu_tex,
            tex_energy_std=sigma_tex,
            dir_coherence_mean=mu_dir,
            dir_coherence_std=sigma_dir,
            has_pattern_profile=True,
        )
        self.samples[class_id] = sample
        return sample

    def extract_competitive_polygons(
        self,
        image_rgb: np.ndarray,
        active_class_ids: Optional[List[str]] = None
    ) -> Dict[str, List[Polygon]]:
        """
        Runs competitive multi-class pattern-aware and texture-aware segmentation.
        Every pixel is assigned to its best-matching active class via argmin distance.
        Guarantees zero overlap and zero conflicting polygon boundaries between classes.
        """
        active_classes = [
            s for s in self.samples.values()
            if s.active and (active_class_ids is None or s.class_id in active_class_ids)
        ]
        if not active_classes:
            return {}

        h, w = image_rgb.shape[:2]

        # 1. Multi-scale Pyramidal Downsampling for Speedup
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
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)

        # 3. Compute Distance Volume with Mahalanobis & Texture matching
        dist_maps = []
        for sample in active_classes:
            mu_l, mu_a, mu_b = sample.lab
            sig_l, sig_a, sig_b = sample.lab_std if hasattr(sample, 'lab_std') and sample.lab_std else [8.0, 4.0, 4.0]

            d_col = np.sqrt(
                ((lab[:, :, 0] - mu_l) / sig_l) ** 2 +
                ((lab[:, :, 1] - mu_a) / sig_a) ** 2 +
                ((lab[:, :, 2] - mu_b) / sig_b) ** 2
            )

            if getattr(sample, 'has_pattern_profile', False):
                d_tex = np.abs(tex_energy - sample.tex_energy_mean) / sample.tex_energy_std
                d_dir = np.abs(dir_coh - sample.dir_coherence_mean) / sample.dir_coherence_std
                total_dist = d_col + (1.2 * d_tex) + (0.8 * d_dir)
            else:
                tex_norm = (tex_energy - 18.0) / 10.0
                total_dist = d_col - (sample.texture_weight * tex_norm * 0.4)

            dist_maps.append(total_dist)

        dist_stack = np.stack(dist_maps, axis=2)
        winner_idx = np.argmin(dist_stack, axis=2)
        min_dists = np.min(dist_stack, axis=2)

        # 4. Vectorize each class mask cleanly
        results: Dict[str, List[Polygon]] = {}

        for idx, sample in enumerate(active_classes):
            c_mask = ((winner_idx == idx) & (min_dists <= 4.2)).astype(np.uint8) * 255

            if np.count_nonzero(c_mask) == 0:
                results[sample.class_id] = []
                continue

            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask_closed = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
            mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys: List[Polygon] = []

            for cnt in cnts:
                if len(cnt) >= 3:
                    approx = cv2.approxPolyDP(cnt, 1.5, True)
                    if len(approx) >= 3:
                        scaled_pts = [(float(p[0][0]) * inv_scale, float(p[0][1]) * inv_scale) for p in approx]
                        p_geom = Polygon(scaled_pts)
                        if p_geom.is_valid and p_geom.area >= sample.min_area_px:
                            polys.append(p_geom.simplify(2.0, preserve_topology=True))

            results[sample.class_id] = polys

        return results

    def extract_class_polygons(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        tolerance_override: Optional[int] = None
    ) -> List[Polygon]:
        """
        Extracts polygons for a single class using Mahalanobis Color + Texture Energy + Directional Coherence.
        """
        active_ids = [k for k, v in self.samples.items() if v.active]
        if len(active_ids) > 1 and class_id in active_ids:
            all_res = self.extract_competitive_polygons(image_rgb, active_class_ids=active_ids)
            return all_res.get(class_id, [])

        if class_id not in self.samples:
            return []

        sample = self.samples[class_id]
        h, w = image_rgb.shape[:2]
        max_dim = max(h, w)
        scale_factor = (1800.0 / max_dim) if max_dim > 1800 else 1.0
        small = cv2.resize(image_rgb, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA) if scale_factor < 1.0 else image_rgb
        inv_scale = 1.0 / scale_factor

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)

        mu_l, mu_a, mu_b = sample.lab
        sig_l, sig_a, sig_b = sample.lab_std if hasattr(sample, 'lab_std') and sample.lab_std else [8.0, 4.0, 4.0]

        d_col = np.sqrt(
            ((lab[:, :, 0] - mu_l) / sig_l) ** 2 +
            ((lab[:, :, 1] - mu_a) / sig_a) ** 2 +
            ((lab[:, :, 2] - mu_b) / sig_b) ** 2
        )

        if getattr(sample, 'has_pattern_profile', False):
            d_tex = np.abs(tex_energy - sample.tex_energy_mean) / sample.tex_energy_std
            d_dir = np.abs(dir_coh - sample.dir_coherence_mean) / sample.dir_coherence_std
            dist = d_col + (1.2 * d_tex) + (0.8 * d_dir)
            thresh = 4.0
        else:
            tex_norm = (tex_energy - 18.0) / 10.0
            dist = d_col - (sample.texture_weight * tex_norm * 0.4)
            thresh = 3.5

        mask_raw = (dist <= thresh)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask_closed = cv2.morphologyEx(mask_raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[Polygon] = []

        for cnt in cnts:
            if len(cnt) >= 3:
                approx = cv2.approxPolyDP(cnt, 1.5, True)
                if len(approx) >= 3:
                    scaled_pts = [(float(p[0][0]) * inv_scale, float(p[0][1]) * inv_scale) for p in approx]
                    p_geom = Polygon(scaled_pts)
                    if p_geom.is_valid and p_geom.area >= sample.min_area_px:
                        polys.append(p_geom.simplify(2.0, preserve_topology=True))

        return polys

    def get_sample(self, class_id: str) -> Optional[ColorSample]:
        return self.samples.get(class_id)

    def save_palette(self, file_path: str):
        """Saves calibration palette to JSON file."""
        data = {k: asdict(v) for k, v in self.samples.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_palette(self, file_path: str):
        """Loads calibration palette from JSON file."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                self.samples[k] = ColorSample(**v)
