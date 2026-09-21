"""
Color & Texture Pipette Sampler with CIE-Lab Mahalanobis Metric,
Laplacian Texture Energy, and Directional Hatching Coherence.
Provides multi-class competitive land-use extraction with strict pigment gating.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import cv2
from shapely.geometry import Polygon


@dataclass
class StampEntry:
    stamp_id: int
    cx: float
    cy: float
    radius: int
    rgb: List[int]
    hsv: List[int]
    lab: List[int]
    hex_color: str
    lab_std: List[float] = field(default_factory=lambda: [10.0, 4.0, 4.0])
    tex_energy_mean: float = 18.0
    tex_energy_std: float = 5.0
    dir_coherence_mean: float = 0.5
    dir_coherence_std: float = 0.15
    name: str = ""
    raw_pixels: int = 0
    distilled_pixels: int = 0


@dataclass
class ColorSample:
    class_id: str
    label: str
    rgb: List[int]
    hsv: List[int]
    lab: List[int]
    hex_color: str
    tolerance: int = 24
    min_area_px: float = 350.0
    active: bool = True
    sampled_points: List[List[int]] = field(default_factory=list)
    texture_weight: float = 0.5
    lab_std: List[float] = field(default_factory=lambda: [8.0, 4.0, 4.0])
    tex_energy_mean: float = 18.0
    tex_energy_std: float = 5.0
    dir_coherence_mean: float = 0.5
    dir_coherence_std: float = 0.15
    has_pattern_profile: bool = False
    stamps: List[StampEntry] = field(default_factory=list)


class PipetteSampler:
    """
    Manages custom and predefined color/texture profiles for historical map land-use extraction.
    Uses Mahalanobis metric on CIE-Lab + Laplacian Texture + Directional Coherence.
    """

    DEFAULT_CLASSES = [
        {"class_id": "forest", "label": "Wald & Busch / Hecken", "hex": "#27ae60", "rgb": [90, 142, 75], "min_area": 100.0, "tex_w": 0.3},
        {"class_id": "meadow", "label": "Wiese / Aue / Grünland", "hex": "#00cec9", "rgb": [120, 195, 175], "min_area": 200.0, "tex_w": 0.3},
        {"class_id": "water", "label": "Gewässer / Rhein / Bäche", "hex": "#0984e3", "rgb": [95, 155, 200], "min_area": 15.0, "tex_w": 0.2},
        {"class_id": "gravel", "label": "Kies- & Schotterbänke", "hex": "#e17055", "rgb": [215, 160, 120], "min_area": 80.0, "tex_w": 0.5},
        {"class_id": "vineyard", "label": "Weinberge (Rebhänge)", "hex": "#d63031", "rgb": [190, 130, 95], "min_area": 200.0, "tex_w": 0.9},
        {"class_id": "garden", "label": "Gärten / Nutzkulturen", "hex": "#a4b07e", "rgb": [165, 175, 140], "min_area": 100.0, "tex_w": 0.5},
    ]

    def __init__(self):
        self.samples: Dict[str, ColorSample] = {}
        self._init_defaults()

    def _create_default_forest_stamps(self) -> List[StampEntry]:
        """Creates the authentic historical Tranchot forest nuances: green wash, yellow-ochre woodland glaze, and Eifel hill forest."""
        rgb_green = [90, 142, 75]
        lab_green = list(cv2.cvtColor(np.uint8([[rgb_green]]), cv2.COLOR_RGB2LAB)[0, 0])
        s_green = StampEntry(
            stamp_id=1, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_green, hsv=[0, 0, 0], lab=[int(x) for x in lab_green],
            hex_color="#5a8e4b", lab_std=[14.0, 4.5, 5.5],
            tex_energy_mean=50.0, tex_energy_std=15.0,
            dir_coherence_mean=0.5, dir_coherence_std=0.15,
            name="Wald (Grünlasur / Laub- & Nadelwald)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_yellow = [182, 198, 118]
        lab_yellow = list(cv2.cvtColor(np.uint8([[rgb_yellow]]), cv2.COLOR_RGB2LAB)[0, 0])
        s_yellow = StampEntry(
            stamp_id=2, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_yellow, hsv=[0, 0, 0], lab=[int(x) for x in lab_yellow],
            hex_color="#b6c676", lab_std=[14.0, 4.5, 6.0],
            tex_energy_mean=50.0, tex_energy_std=15.0,
            dir_coherence_mean=0.5, dir_coherence_std=0.15,
            name="Hangwald & Buschwerk (Gelbgrün)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_woodland = [224, 208, 138]
        lab_woodland = list(cv2.cvtColor(np.uint8([[rgb_woodland]]), cv2.COLOR_RGB2LAB)[0, 0])
        s_woodland = StampEntry(
            stamp_id=3, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_woodland, hsv=[0, 0, 0], lab=[int(x) for x in lab_woodland],
            hex_color="#e0d08a", lab_std=[16.0, 5.0, 7.0],
            tex_energy_mean=50.0, tex_energy_std=15.0,
            dir_coherence_mean=0.5, dir_coherence_std=0.15,
            name="Bergkuppen-Waldlasur (Mühlberg / Busch & Hecken)",
            raw_pixels=100, distilled_pixels=100
        )
        return [s_green, s_yellow, s_woodland]

    def _create_default_gravel_stamps(self) -> List[StampEntry]:
        """Creates authentic historical Tranchot gravel bank nuances: warm sandbar, grey-beige river gravel, and shore silt."""
        rgb_sand = [215, 175, 140]
        lab_sand = list(cv2.cvtColor(np.uint8([[rgb_sand]]), cv2.COLOR_RGB2LAB)[0, 0])
        s1 = StampEntry(
            stamp_id=1, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_sand, hsv=[0, 0, 0], lab=[int(x) for x in lab_sand],
            hex_color="#d7af8c", lab_std=[14.0, 5.0, 6.0],
            tex_energy_mean=32.0, tex_energy_std=12.0,
            dir_coherence_mean=0.4, dir_coherence_std=0.2,
            name="Kiesbank (Lachs-Sand / Schotter)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_grey = [190, 182, 152]
        lab_grey = list(cv2.cvtColor(np.uint8([[rgb_grey]]), cv2.COLOR_RGB2LAB)[0, 0])
        s2 = StampEntry(
            stamp_id=2, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_grey, hsv=[0, 0, 0], lab=[int(x) for x in lab_grey],
            hex_color="#beb698", lab_std=[14.0, 5.0, 6.0],
            tex_energy_mean=35.0, tex_energy_std=12.0,
            dir_coherence_mean=0.4, dir_coherence_std=0.2,
            name="Flussbett-Schotter (Grau-Beige)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_silt = [208, 196, 165]
        lab_silt = list(cv2.cvtColor(np.uint8([[rgb_silt]]), cv2.COLOR_RGB2LAB)[0, 0])
        s3 = StampEntry(
            stamp_id=3, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_silt, hsv=[0, 0, 0], lab=[int(x) for x in lab_silt],
            hex_color="#d0c4a5", lab_std=[14.0, 5.0, 6.0],
            tex_energy_mean=30.0, tex_energy_std=10.0,
            dir_coherence_mean=0.4, dir_coherence_std=0.2,
            name="Ufersand & Schlick",
            raw_pixels=100, distilled_pixels=100
        )
        return [s1, s2, s3]

    def _create_default_water_stamps(self) -> List[StampEntry]:
        """Creates authentic historical Tranchot water nuances: Prussian blue river wash, deep channel flow, and shallow brooks."""
        rgb_river = [95, 155, 200]
        lab_river = list(cv2.cvtColor(np.uint8([[rgb_river]]), cv2.COLOR_RGB2LAB)[0, 0])
        s1 = StampEntry(
            stamp_id=1, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_river, hsv=[0, 0, 0], lab=[int(x) for x in lab_river],
            hex_color="#5f9bc8", lab_std=[14.0, 4.5, 6.0],
            tex_energy_mean=14.0, tex_energy_std=6.0,
            dir_coherence_mean=0.3, dir_coherence_std=0.15,
            name="Flusslauf & Hauptstrom (Cyan-/Preußischblau)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_deep = [65, 120, 170]
        lab_deep = list(cv2.cvtColor(np.uint8([[rgb_deep]]), cv2.COLOR_RGB2LAB)[0, 0])
        s2 = StampEntry(
            stamp_id=2, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_deep, hsv=[0, 0, 0], lab=[int(x) for x in lab_deep],
            hex_color="#4178aa", lab_std=[14.0, 4.5, 5.5],
            tex_energy_mean=16.0, tex_energy_std=6.0,
            dir_coherence_mean=0.3, dir_coherence_std=0.15,
            name="Tiefe Strömungsrinne (Dunkelblau)",
            raw_pixels=100, distilled_pixels=100
        )
        rgb_brook = [135, 180, 210]
        lab_brook = list(cv2.cvtColor(np.uint8([[rgb_brook]]), cv2.COLOR_RGB2LAB)[0, 0])
        s3 = StampEntry(
            stamp_id=3, cx=0.0, cy=0.0, radius=8,
            rgb=rgb_brook, hsv=[0, 0, 0], lab=[int(x) for x in lab_brook],
            hex_color="#87b4d2", lab_std=[14.0, 4.5, 5.5],
            tex_energy_mean=14.0, tex_energy_std=6.0,
            dir_coherence_mean=0.3, dir_coherence_std=0.15,
            name="Uferzone & Seitenbach (Hellblau / Mühlengraben)",
            raw_pixels=100, distilled_pixels=100
        )
        return [s1, s2, s3]

    def _init_defaults(self):
        self.samples.clear()
        for c in self.DEFAULT_CLASSES:
            rgb = np.uint8([[c["rgb"]]])
            hsv = list(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0])
            lab = list(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0])
            if c["class_id"] == "forest":
                stamps = self._create_default_forest_stamps()
            elif c["class_id"] == "gravel":
                stamps = self._create_default_gravel_stamps()
            elif c["class_id"] == "water":
                stamps = self._create_default_water_stamps()
            else:
                stamps = []
            self.samples[c["class_id"]] = ColorSample(
                class_id=c["class_id"],
                label=c["label"],
                rgb=c["rgb"],
                hsv=[int(x) for x in hsv],
                lab=[int(x) for x in lab],
                hex_color=c["hex"],
                tolerance=24,
                min_area_px=c["min_area"],
                active=True,
                sampled_points=[c["rgb"]],
                texture_weight=c["tex_w"],
                lab_std=[8.0, 4.0, 4.0],
                has_pattern_profile=(len(stamps) > 0),
                stamps=stamps,
            )

    def get_sample(self, class_id: str) -> Optional[ColorSample]:
        return self.samples.get(class_id)

    def set_sample_active(self, class_id: str, active: bool):
        if class_id in self.samples:
            self.samples[class_id].active = active

    def reset_class(self, class_id: str):
        for c in self.DEFAULT_CLASSES:
            if c["class_id"] == class_id:
                rgb = np.uint8([[c["rgb"]]])
                hsv = list(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0])
                lab = list(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0])
                if class_id == "forest":
                    stamps = self._create_default_forest_stamps()
                elif class_id == "gravel":
                    stamps = self._create_default_gravel_stamps()
                elif class_id == "water":
                    stamps = self._create_default_water_stamps()
                else:
                    stamps = []
                self.samples[class_id] = ColorSample(
                    class_id=class_id,
                    label=c["label"],
                    rgb=c["rgb"],
                    hsv=[int(x) for x in hsv],
                    lab=[int(x) for x in lab],
                    hex_color=c["hex"],
                    tolerance=24,
                    min_area_px=c["min_area"],
                    active=True,
                    sampled_points=[c["rgb"]],
                    texture_weight=c["tex_w"],
                    lab_std=[8.0, 4.0, 4.0],
                    has_pattern_profile=(len(stamps) > 0),
                    stamps=stamps,
                )
                break

    def sample_from_point(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        px: float,
        py: float,
        radius: int = 5,
    ) -> ColorSample:
        h, w = image_rgb.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        patch = image_rgb[y0:y1, x0:x1]


        mean_rgb = np.mean(patch, axis=(0, 1)).astype(int).tolist()
        hsv = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
        lab = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2LAB)[0, 0])
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        existing = self.samples.get(class_id)
        label = existing.label if existing else class_id
        min_area = existing.min_area_px if existing else 350.0
        tex_w = existing.texture_weight if existing else 0.5
        pts = (existing.sampled_points + [mean_rgb]) if existing else [mean_rgb]

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

    def sample_from_stamp(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        cx: float,
        cy: float,
        radius: int = 22,
        name: Optional[str] = None
    ) -> Optional[StampEntry]:
        """
        Samples a circular region with automatic ink / hachure deconvolution
        and adds the distilled pigment nuance to the class stamp table.
        """
        h, w = image_rgb.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))

        # Bounding box for crop
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        if (x1 - x0) < 3 or (y1 - y0) < 3:
            return None

        # Create circular mask in crop
        crop_rgb = image_rgb[y0:y1, x0:x1]
        ch, cw = crop_rgb.shape[:2]
        yy, xx = np.ogrid[:ch, :cw]
        local_cx, local_cy = ix - x0, iy - y0
        circle_mask = ((xx - local_cx)**2 + (yy - local_cy)**2) <= (radius**2)

        raw_count = int(np.count_nonzero(circle_mask))
        if raw_count == 0:
            return None

        # 1. Color spaces
        gray_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv_crop[:, :, 1]
        l_chan = lab_crop[:, :, 0]
        a_chan = lab_crop[:, :, 1]
        b_chan = lab_crop[:, :, 2]
        chroma = np.sqrt((a_chan - 128.0)**2 + (b_chan - 128.0)**2)

        # 2. Ink & Paper Masks
        # Exclude only true dense black ink lines (letters, border frames, deep hachure strokes)
        # and completely uncolored stark white/cream margins
        r_crop = crop_rgb[:, :, 0].astype(int)
        g_crop = crop_rgb[:, :, 1].astype(int)
        b_crop = crop_rgb[:, :, 2].astype(int)

        is_hachure_ink = (gray_crop < 95.0) | ((l_chan < 105.0) & (sat < 30.0))
        is_paper = (l_chan > 175.0) & (a_chan >= 125.5) & (a_chan <= 130.5) & (sat < 20.0) & (chroma < 14.0)

        # 3. Class-specific pigment preference (broad & generous so slope/border nuances succeed)
        cid = class_id.lower()
        if "forest" in cid or "wald" in cid:
            is_pigment = (
                (
                    # Classic foliage green wash
                    ((a_chan <= 124.5) & (g_crop >= b_crop + 6) & (g_crop >= r_crop - 4)) |
                    # Authentic historical yellow-green woodland glaze (Mühlberg, hilltops, Busch/Hecken)
                    ((b_chan >= 145.0) & (g_crop >= b_crop + 15) & (a_chan <= 128.0) & (r_crop <= g_crop + 35))
                ) &
                (l_chan >= 50.0) &
                (l_chan <= 235.0) &
                (~is_paper) & (~is_hachure_ink)
            )
        elif "meadow" in cid or "wiese" in cid or "weide" in cid:
            is_pigment = (
                (a_chan <= 124.5) &
                (b_chan <= 136.0) &
                (l_chan >= 80.0) &
                (l_chan <= 245.0) &
                (g_crop >= r_crop - 2) &
                (g_crop >= b_crop - 4) &
                (sat >= 7.0) &
                (~is_paper) & (~is_hachure_ink)
            )
        elif "water" in cid or "gewässer" in cid or "wasser" in cid or "see" in cid or "bach" in cid:
            is_pigment = (
                (b_chan <= 129.0) &
                (l_chan >= 80.0) &
                (l_chan <= 250.0) &
                (b_crop >= r_crop - 8) &
                (~is_paper) & (~is_hachure_ink)
            )
        elif "vineyard" in cid or "wein" in cid:
            is_pigment = (a_chan >= 130.0) & (r_crop >= g_crop + 2) & (r_crop >= b_crop + 6) & (~is_paper)
        elif "gravel" in cid or "kies" in cid:
            is_pigment = (a_chan >= 129.0) & (b_chan <= 146.0) & (r_crop >= b_crop + 4) & (r_crop >= g_crop - 2) & (~is_paper)
        elif "garden" in cid or "garten" in cid or "gärten" in cid:
            is_pigment = (a_chan <= 127.5) & (b_chan >= 127.0) & (b_chan <= 145.0) & (g_crop >= b_crop) & (~is_paper)
        elif "building" in cid or "gebäude" in cid:
            is_pigment = (a_chan > 130.0) & (r_crop > g_crop + 8)
        else:
            is_pigment = (~is_paper) & (~is_hachure_ink)

        # Distill target mask inside circle
        target_mask = circle_mask & is_pigment & (~is_hachure_ink)
        n_target = int(np.count_nonzero(target_mask))

        if n_target < 6:
            # Fallback to non-paper and non-ink inside circle if user clicked directly on hachure ink line
            fallback_mask = circle_mask & (~is_paper) & (~is_hachure_ink)
            if np.count_nonzero(fallback_mask) >= 6:
                target_mask = fallback_mask
            else:
                return None

        # Take the top 70% most saturated pigment pixels to eliminate boundary bleed
        sat_inside = sat[target_mask]
        thresh_sat = float(np.percentile(sat_inside, 30))
        pure_mask = target_mask & (sat >= thresh_sat)
        if np.count_nonzero(pure_mask) >= 6:
            sample_mask = pure_mask
        else:
            sample_mask = target_mask

        distilled_count = int(np.count_nonzero(sample_mask))
        inside_pixels = crop_rgb[sample_mask]
        inside_lab = lab_crop[sample_mask]
        inside_hsv = hsv_crop[sample_mask]

        mean_rgb = np.mean(inside_pixels, axis=0).astype(int).tolist()
        mean_hsv = np.mean(inside_hsv, axis=0).astype(int).tolist()
        mean_lab = np.mean(inside_lab, axis=0).astype(int).tolist()
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        raw_std = np.std(inside_lab, axis=0)
        std_lab = [
            float(max(8.0, min(16.0, raw_std[0]))),
            float(max(3.0, min(5.5, raw_std[1]))),
            float(max(3.0, min(6.0, raw_std[2]))),
        ]

        # Texture and Directional Coherence
        lap = np.abs(cv2.Laplacian(gray_crop, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)
        inside_tex = tex_energy[sample_mask]
        mu_tex = float(np.mean(inside_tex))
        sigma_tex = float(max(2.0, min(12.0, np.std(inside_tex))))

        gx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)
        inside_dir = dir_coh[sample_mask]
        mu_dir = float(np.mean(inside_dir))
        sigma_dir = float(max(0.05, min(0.35, np.std(inside_dir))))

        sample = self.samples.get(class_id)
        if sample is None:
            self.reset_class(class_id)
            sample = self.samples[class_id]

        stamp_id = len(sample.stamps) + 1
        stamp_name = name or f"Nuance #{stamp_id}"

        entry = StampEntry(
            stamp_id=stamp_id,
            cx=float(cx),
            cy=float(cy),
            radius=int(radius),
            rgb=mean_rgb,
            hsv=mean_hsv,
            lab=mean_lab,
            hex_color=hex_col,
            lab_std=std_lab,
            tex_energy_mean=mu_tex,
            tex_energy_std=sigma_tex,
            dir_coherence_mean=mu_dir,
            dir_coherence_std=sigma_dir,
            name=stamp_name,
            raw_pixels=raw_count,
            distilled_pixels=distilled_count,
        )

        sample.stamps.append(entry)
        sample.active = True
        sample.has_pattern_profile = True
        sample.hex_color = hex_col
        sample.rgb = mean_rgb
        sample.lab = mean_lab
        sample.lab_std = std_lab
        sample.tex_energy_mean = mu_tex
        sample.tex_energy_std = sigma_tex
        sample.dir_coherence_mean = mu_dir
        sample.dir_coherence_std = sigma_dir

        return entry

    def remove_stamp(self, class_id: str, stamp_index: int) -> bool:
        if class_id in self.samples:
            sample = self.samples[class_id]
            if 0 <= stamp_index < len(sample.stamps):
                del sample.stamps[stamp_index]
                if sample.stamps:
                    last = sample.stamps[-1]
                    sample.rgb = last.rgb
                    sample.lab = last.lab
                    sample.lab_std = last.lab_std
                    sample.hex_color = last.hex_color
                    sample.tex_energy_mean = last.tex_energy_mean
                    sample.tex_energy_std = last.tex_energy_std
                    sample.dir_coherence_mean = last.dir_coherence_mean
                    sample.dir_coherence_std = last.dir_coherence_std
                else:
                    sample.has_pattern_profile = False
                return True
        return False

    def clear_stamps(self, class_id: str):
        if class_id in self.samples:
            self.samples[class_id].stamps.clear()
            self.samples[class_id].has_pattern_profile = False
            if class_id == "forest":
                self.samples[class_id].stamps = self._create_default_forest_stamps()
                self.samples[class_id].has_pattern_profile = True
            elif class_id == "gravel":
                self.samples[class_id].stamps = self._create_default_gravel_stamps()
                self.samples[class_id].has_pattern_profile = True
            elif class_id == "water":
                self.samples[class_id].stamps = self._create_default_water_stamps()
                self.samples[class_id].has_pattern_profile = True

    def get_stamps(self, class_id: str) -> List[StampEntry]:
        if class_id in self.samples:
            return self.samples[class_id].stamps
        return []

    def sample_from_polygon(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        polygon_pts: List[Tuple[float, float]],
        name: Optional[str] = None
    ) -> Optional[StampEntry]:
        """
        Samples all pixels within a user-drawn polygon, deconvolves black ink strokes,
        and records a new StampEntry into the class's multi-nuance profile.
        """
        return self.sample_from_single_polygon(image_rgb, class_id, polygon_pts, name=name)

    def sample_from_single_polygon(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        polygon_pts: List[Tuple[float, float]],
        name: Optional[str] = None
    ) -> Optional[StampEntry]:
        """
        Extracts color and texture distribution from an irregular sample polygon,
        purges black ink strokes, and records it as an active StampEntry nuance.
        """
        if len(polygon_pts) < 3:
            return None

        h, w = image_rgb.shape[:2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_pts, dtype=np.int32)
        cv2.fillPoly(poly_mask, [pts], 255)

        inside_pixels = image_rgb[poly_mask > 0]
        if len(inside_pixels) < 6:
            return None

        # Bounding box of polygon for texture computation
        bx, by, bw, bh = cv2.boundingRect(pts)
        bx0, bx1 = max(0, bx), min(w, bx + bw)
        by0, by1 = max(0, by), min(h, by + bh)
        crop_rgb = image_rgb[by0:by1, bx0:bx1]
        crop_mask = poly_mask[by0:by1, bx0:bx1] > 0

        gray_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        sat = hsv_crop[:, :, 1]
        l_chan = lab_crop[:, :, 0]
        a_chan = lab_crop[:, :, 1]
        b_chan = lab_crop[:, :, 2]
        chroma = np.sqrt((a_chan - 128.0)**2 + (b_chan - 128.0)**2)

        # Deconvolve black ink strokes & aged paper margins
        r_crop = crop_rgb[:, :, 0].astype(int)
        g_crop = crop_rgb[:, :, 1].astype(int)
        b_crop = crop_rgb[:, :, 2].astype(int)

        is_paper = (l_chan > 195.0) & (a_chan >= 126.0) & (g_crop <= r_crop + 2) & (sat < 35.0)
        is_ink = (gray_crop < 95.0) | ((l_chan < 110.0) & (sat < 30.0))

        # Class-specific pigment wash filter: extracts authentic watercolor pigment even if
        # user drew a loose polygon covering surrounding slopes or bare paper fields.
        cid = class_id.lower()
        if "forest" in cid or "wald" in cid:
            is_pigment = (
                (a_chan <= 127.0) &
                (b_chan >= 126.0) &
                (l_chan <= 230.0) &
                (g_crop >= b_crop + 8) &
                (r_crop <= g_crop + 28) &
                (~is_paper) & (~is_ink)
            )
        elif "meadow" in cid or "wiese" in cid or "weide" in cid:
            is_pigment = (
                (a_chan <= 125.0) &
                (b_chan < 142.0) &
                (l_chan >= 80.0) &
                (l_chan <= 242.0) &
                (g_crop >= r_crop + 1) &
                (g_crop >= b_crop - 6) &
                (~is_paper) & (~is_ink)
            )
        elif "water" in cid or "gewässer" in cid or "wasser" in cid:
            is_pigment = (
                (b_chan <= 129.0) &
                (l_chan >= 80.0) &
                (b_crop >= r_crop - 6) &
                (~is_paper) & (~is_ink)
            )
        elif "garden" in cid or "garten" in cid:
            is_pigment = (a_chan <= 127.0) & (b_chan >= 127.0) & (g_crop >= b_crop + 2) & (~is_paper) & (~is_ink)
        elif "vineyard" in cid or "wein" in cid:
            is_pigment = (a_chan >= 128.0) & (r_crop >= b_crop + 2) & (~is_paper) & (~is_ink)
        else:
            is_pigment = (~is_paper) & (~is_ink)

        target_mask = crop_mask & is_pigment & (~is_ink)
        if np.count_nonzero(target_mask) >= 10:
            valid_mask = target_mask
        else:
            non_paper = crop_mask & (~is_paper) & (~is_ink)
            if np.count_nonzero(non_paper) >= 10:
                valid_mask = non_paper
            else:
                valid_mask = crop_mask & (~is_ink)
                if np.count_nonzero(valid_mask) < 6:
                    valid_mask = crop_mask

        distilled_pixels = crop_rgb[valid_mask]
        distilled_lab = lab_crop[valid_mask]
        distilled_hsv = hsv_crop[valid_mask]

        mean_rgb = np.mean(distilled_pixels, axis=0).astype(int).tolist()
        mean_hsv = np.mean(distilled_hsv, axis=0).astype(int).tolist()
        mean_lab = np.mean(distilled_lab, axis=0).astype(int).tolist()
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        raw_std = np.std(distilled_lab, axis=0)
        std_lab = [
            float(max(6.0, min(14.0, raw_std[0]))),
            float(max(3.0, min(5.5, raw_std[1]))),
            float(max(3.0, min(6.0, raw_std[2]))),
        ]

        # Texture and Directional Coherence
        lap = np.abs(cv2.Laplacian(gray_crop, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)
        inside_tex = tex_energy[valid_mask]
        mu_tex = float(np.mean(inside_tex)) if len(inside_tex) > 0 else 18.0
        sigma_tex = float(max(2.0, min(12.0, np.std(inside_tex)))) if len(inside_tex) > 0 else 5.0

        gx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)
        inside_dir = dir_coh[valid_mask]
        mu_dir = float(np.mean(inside_dir)) if len(inside_dir) > 0 else 0.5
        sigma_dir = float(max(0.05, min(0.35, np.std(inside_dir)))) if len(inside_dir) > 0 else 0.15

        sample = self.samples.get(class_id)
        if sample is None:
            self.reset_class(class_id)
            sample = self.samples[class_id]

        stamp_id = len(sample.stamps) + 1
        stamp_name = name or f"Muster-Polygon #{stamp_id}"
        cx, cy = float(bx + bw / 2.0), float(by + bh / 2.0)
        radius = int(max(bw, bh) / 2.0)

        entry = StampEntry(
            stamp_id=stamp_id,
            cx=cx,
            cy=cy,
            radius=radius,
            rgb=mean_rgb,
            hsv=mean_hsv,
            lab=mean_lab,
            hex_color=hex_col,
            lab_std=std_lab,
            tex_energy_mean=mu_tex,
            tex_energy_std=sigma_tex,
            dir_coherence_mean=mu_dir,
            dir_coherence_std=sigma_dir,
            name=stamp_name,
            raw_pixels=int(np.count_nonzero(crop_mask)),
            distilled_pixels=int(np.count_nonzero(valid_mask)),
        )

        sample.stamps.append(entry)
        sample.active = True
        sample.has_pattern_profile = True
        sample.hex_color = hex_col
        sample.rgb = mean_rgb
        sample.lab = mean_lab
        sample.lab_std = std_lab
        sample.tex_energy_mean = mu_tex
        sample.tex_energy_std = sigma_tex
        sample.dir_coherence_mean = mu_dir
        sample.dir_coherence_std = sigma_dir

        return entry

    def sample_from_polygons(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        polygons: List[Any],
    ) -> Optional[ColorSample]:
        """
        Extracts multi-polygon aggregated pixel and texture distribution.
        Combines ALL user-drawn sample polygons for this class into a single robust profile.
        """
        if not polygons:
            return self.samples.get(class_id, None)

        h, w = image_rgb.shape[:2]
        comb_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            if hasattr(poly, "exterior"):
                pts = np.array(poly.exterior.coords, dtype=np.int32)
            else:
                pts = np.array(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(comb_mask, [pts], 255)

        inside_pixels = image_rgb[comb_mask > 0]
        if len(inside_pixels) == 0:
            return self.samples.get(class_id, None)

        # 1. Color spaces & representations
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv_img[:, :, 1]
        l_chan = lab_img[:, :, 0]
        a_chan = lab_img[:, :, 1]
        b_chan = lab_img[:, :, 2]
        chroma = np.sqrt((a_chan - 128.0)**2 + (b_chan - 128.0)**2)

        # 2. General background paper & dark ink masks
        r_img = image_rgb[:, :, 0].astype(int)
        g_img = image_rgb[:, :, 1].astype(int)
        b_img = image_rgb[:, :, 2].astype(int)

        is_paper = (l_chan > 175.0) & (a_chan >= 125.5) & (a_chan <= 130.5) & (g_img <= r_img + 2) & (sat < 22.0)
        is_ink = (l_chan < 155.0) & (g_img <= r_img + 2)

        # 3. Class-specific pigment wash filter to purge surrounding parchment & slope lines
        cid = class_id.lower()
        if "forest" in cid or "wald" in cid:
            is_pigment = (
                (
                    # Classic foliage green wash
                    ((a_chan <= 124.5) & (g_img >= b_img + 6) & (g_img >= r_img - 4)) |
                    # Authentic historical yellow-green woodland glaze (Mühlberg, hilltops, Busch/Hecken)
                    ((b_chan >= 145.0) & (g_img >= b_img + 15) & (a_chan <= 128.0) & (r_img <= g_img + 35))
                ) &
                (l_chan >= 50.0) &
                (l_chan <= 235.0) &
                (~is_paper) & (~is_ink)
            )
        elif "meadow" in cid or "wiese" in cid or "weide" in cid:
            is_pigment = (
                (a_chan <= 124.5) &
                (b_chan <= 136.0) &
                (l_chan >= 90.0) &
                (l_chan <= 242.0) &
                (sat >= 7.0) &
                (g_img >= r_img - 2) &
                (g_img >= b_img - 4) &
                (~is_paper) & (~is_ink)
            )
        elif "water" in cid or "gewässer" in cid or "wasser" in cid:
            is_pigment = (
                (b_chan <= 128.5) &
                (l_chan >= 90.0) &
                (b_img >= r_img - 4) &
                (~is_paper) & (~is_ink)
            )
        elif "vineyard" in cid or "wein" in cid:
            is_pigment = (a_chan >= 130.0) & (r_img >= g_img + 2) & (r_img >= b_img + 6) & (~is_paper)
        elif "gravel" in cid or "kies" in cid:
            is_pigment = (a_chan >= 129.0) & (b_chan <= 146.0) & (r_img >= b_img + 4) & (r_img >= g_img - 2) & (~is_paper)
        elif "garden" in cid or "garten" in cid or "gärten" in cid:
            is_pigment = (a_chan <= 127.5) & (b_chan >= 127.0) & (b_chan <= 145.0) & (g_img >= b_img) & (~is_paper)
        elif "building" in cid or "gebäude" in cid:
            is_pigment = (a_chan > 132.0) & (r_img > g_img + 15)
        else:
            is_pigment = (~is_paper) & (~is_ink)

        # Distill target mask inside drawn polygons
        target_mask = (comb_mask > 0) & is_pigment & (~is_ink)
        n_target = np.count_nonzero(target_mask)

        # If we have enough target pigment pixels inside the polygon (>= 15 px), use ONLY them!
        # This completely discards the 80% surrounding slope hachures and parchment paper.
        if n_target >= 15:
            sample_mask = target_mask
        else:
            # Fallback to non-paper or entire polygon if no prior pigment matched
            non_paper_mask = (comb_mask > 0) & (~is_paper)
            if np.count_nonzero(non_paper_mask) >= 15:
                sample_mask = non_paper_mask
            else:
                sample_mask = (comb_mask > 0)

        inside_pixels = image_rgb[sample_mask]
        if len(inside_pixels) == 0:
            return self.samples.get(class_id, None)

        mean_rgb = np.mean(inside_pixels, axis=0).astype(int).tolist()
        hsv = list(cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2HSV)[0, 0])
        hex_col = "#{:02x}{:02x}{:02x}".format(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        inside_lab = lab_img[sample_mask]
        mean_lab = np.mean(inside_lab, axis=0).astype(int).tolist()
        raw_std = np.std(inside_lab, axis=0)
        std_lab = [
            float(max(8.0, min(20.0, raw_std[0]))),
            float(max(3.5, min(9.0, raw_std[1]))),
            float(max(3.5, min(9.0, raw_std[2]))),
        ]

        # 1. Texture Energy (Laplacian frequency energy on distilled sample)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)
        inside_tex = tex_energy[sample_mask]
        mu_tex = float(np.mean(inside_tex))
        sigma_tex = float(max(2.0, min(12.0, np.std(inside_tex))))

        # 2. Directional Hatching Coherence (Slope / Vineyard orientation)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)
        inside_dir = dir_coh[sample_mask]
        mu_dir = float(np.mean(inside_dir))
        sigma_dir = float(max(0.05, min(0.35, np.std(inside_dir))))

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
            min_area_px=max(350.0, getattr(existing, 'min_area_px', 350.0)),
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

    def _chaikin_smooth(self, pts: List[Tuple[float, float]], iterations: int = 1) -> List[Tuple[float, float]]:
        """Applies gentle Chaikin corner rounding preserving true boundary perimeter without shrinkage."""
        if len(pts) < 4:
            return pts
        curr = pts[:]
        for _ in range(iterations):
            next_pts = []
            n = len(curr)
            for i in range(n - 1):
                p0 = curr[i]
                p1 = curr[(i + 1) % n]
                q = (0.85 * p0[0] + 0.15 * p1[0], 0.85 * p0[1] + 0.15 * p1[1])
                r = (0.15 * p0[0] + 0.85 * p1[0], 0.15 * p0[1] + 0.85 * p1[1])
                next_pts.extend([q, r])
            if next_pts:
                next_pts.append(next_pts[0])
            curr = next_pts
        return curr

    def extract_competitive_polygons(
        self,
        image_rgb: np.ndarray,
        active_class_ids: Optional[List[str]] = None,
        closing_kernel_px: int = 18,
        fill_holes_area: float = 20000.0,
    ) -> Dict[str, List[Polygon]]:
        """
        Rock-solid competitive multi-class land-use segmentation using CIE-Lab Mahalanobis metric,
        Texture Energy, strict Chroma/Paper gating, and organic Chaikin contour smoothing.
        """
        active_classes = [
            s for s in self.samples.values()
            if s.active and (active_class_ids is None or s.class_id in active_class_ids)
        ]
        if not active_classes:
            return {}

        h, w = image_rgb.shape[:2]

        # 1. Processing resolution
        max_dim = max(h, w)
        if max_dim > 2400:
            scale_factor = 2400.0 / max_dim
            small = cv2.resize(image_rgb, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
            small = image_rgb

        inv_scale = 1.0 / scale_factor

        # 2. Color Spaces & Texture Computation
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex_energy = cv2.GaussianBlur(lap, (15, 15), 0)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        dir_coh = cv2.GaussianBlur(np.abs(gx**2 - gy**2) / (gx**2 + gy**2 + 1e-4), (15, 15), 0)

        # 2b. Cartographic Hachure Deconvolution (Infilling under ink strokes)
        # Hachures (Böschungsschraffuren) and ink labels cover 40-50% of slopes and valleys.
        # We detect dark ink strokes and inpaint/dilate the underlying watercolor from adjacent non-ink pixels.
        bg = cv2.GaussianBlur(gray, (25, 25), 0)
        dark_diff = np.maximum(0.0, bg - gray)
        hsv_raw = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab_raw = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        sat_raw = hsv_raw[:, :, 1]
        chroma_raw = np.sqrt((lab_raw[:, :, 1] - 128.0)**2 + (lab_raw[:, :, 2] - 128.0)**2)
        r_raw = small[:, :, 0].astype(int)
        g_raw = small[:, :, 1].astype(int)
        b_raw = small[:, :, 2].astype(int)
        is_black_ink = ((dark_diff > 14.0) | (gray < 110.0)) & ((sat_raw < 45.0) | (chroma_raw < 18.0))
        is_sepia_ink = (dark_diff > 10.0) & (gray < 155.0) & (sat_raw < 95.0) & (r_raw >= b_raw + 8) & (g_raw <= r_raw + 5)
        is_ink_stroke = is_black_ink | is_sepia_ink

        k_hatch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        r_infill = cv2.dilate(small[:, :, 0], k_hatch)
        g_infill = cv2.dilate(small[:, :, 1], k_hatch)
        b_infill = cv2.dilate(small[:, :, 2], k_hatch)

        pigment_rgb = small.copy()
        pigment_rgb[is_ink_stroke, 0] = r_infill[is_ink_stroke]
        pigment_rgb[is_ink_stroke, 1] = g_infill[is_ink_stroke]
        pigment_rgb[is_ink_stroke, 2] = b_infill[is_ink_stroke]
        hatch_density = cv2.boxFilter(is_ink_stroke.astype(np.float32), -1, (15, 15))

        lab = cv2.cvtColor(pigment_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv = cv2.cvtColor(pigment_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]
        chroma = np.sqrt((a_chan - 128.0)**2 + (b_chan - 128.0)**2)
        r_p = pigment_rgb[:, :, 0].astype(int)
        g_p = pigment_rgb[:, :, 1].astype(int)
        b_p = pigment_rgb[:, :, 2].astype(int)

        # 3. Paper & Mountain Relief Baselines
        # Real forest wash detection (classic green wash OR yellow-green woodland glaze on hilltops)
        has_classic_green = (a_chan <= 124.5) & (g_p >= b_p + 10) & (g_p >= r_p - 4) & (b_chan >= 132.0)
        has_yellow_woodland = (
            (b_chan >= 148.0) & 
            (g_p >= b_p + 22) & 
            (a_chan <= 139.0) & 
            (r_p <= g_p + 50) & 
            (l_chan >= 100.0)
        )
        has_forest_wash = has_classic_green | has_yellow_woodland

        # Real meadow wash detection (cool turquoise/green valley wash)
        has_meadow_wash = (
            (a_chan <= 124.2) &
            (b_chan <= 141.0) &
            (g_p >= r_p) &
            (g_p >= b_p - 4) &
            (sat >= 7.0)
        )

        # Baseline: Mountain Relief / Slope Hachures (Schraffen auf Bergen/Hängen OHNE Farblasur)
        paper_background = (
            (l_chan >= 170.0) &
            (b_chan <= 144.5) &
            (a_chan >= 125.0) &
            (a_chan <= 130.5) &
            (sat < 52.0)
        )

        is_uncolored_slope = (
            ((hatch_density > 0.05) | ((tex_energy > 14.0) & (dir_coh > 0.20))) &
            paper_background &
            (~has_forest_wash) &
            (~has_meadow_wash)
        )
        d_relief_slope = np.where(is_uncolored_slope, 0.05, 5.0)

        # Baseline: Neutral parchment (roads, bare fields, uncolored margins)
        is_pure_paper = (
            paper_background &
            (~has_forest_wash) &
            (~has_meadow_wash) &
            (hatch_density <= 0.08) &
            (tex_energy <= 18.0)
        )
        paper_chroma = np.sqrt((a_chan - 128.0)**2 + (np.maximum(0.0, b_chan - 142.0))**2)
        p_l_dist = (np.maximum(0.0, 172.0 - l_chan) / 16.0) + (np.maximum(0.0, l_chan - 232.0) / 20.0)
        d_paper = (paper_chroma / 2.5) + p_l_dist
        has_gravel_active = any(("gravel" in s.class_id.lower() or "kies" in s.class_id.lower()) for s in active_classes)
        if not has_gravel_active:
            d_paper = np.where(is_pure_paper, 0.05, d_paper)
        else:
            d_paper = np.where(is_pure_paper, np.maximum(0.18, d_paper), d_paper)

        # 4. Compute Distance Volume for Active Classes
        dist_maps = [d_paper, d_relief_slope]  # Index 0: Paper, Index 1: Mountain Relief
        water_grid_idx = None

        for idx, sample in enumerate(active_classes, 2):
            cid = sample.class_id.lower()
            if "water" in cid or "gewässer" in cid or "wasser" in cid:
                water_grid_idx = idx

            is_watercolor_class = ("forest" in cid or "wald" in cid or "meadow" in cid or "wiese" in cid or "weide" in cid or "water" in cid or "gewässer" in cid or "wasser" in cid)
            tol_scale = float(getattr(sample, "tolerance", 24)) / 24.0

            if getattr(sample, "stamps", None) and len(sample.stamps) > 0:
                stamp_dists = []
                for s in sample.stamps:
                    mu_l, mu_a, mu_b = s.lab
                    sig_l, sig_a, sig_b = s.lab_std
                    d_col = np.sqrt(
                        ((l_chan - mu_l) / (sig_l * tol_scale)) ** 2 +
                        ((a_chan - mu_a) / (sig_a * tol_scale)) ** 2 +
                        ((b_chan - mu_b) / (sig_b * tol_scale)) ** 2
                    )
                    if is_watercolor_class:
                        total_dist_stamp = d_col
                    else:
                        d_tex = np.abs(tex_energy - s.tex_energy_mean) / s.tex_energy_std
                        d_dir = np.abs(dir_coh - s.dir_coherence_mean) / s.dir_coherence_std
                        w_tex = float(getattr(sample, "texture_weight", 0.5))
                        total_dist_stamp = d_col + (w_tex * 0.4 * d_tex) + (w_tex * 0.25 * d_dir)
                    stamp_dists.append(total_dist_stamp)

                total_dist = np.minimum.reduce(stamp_dists) if len(stamp_dists) > 1 else stamp_dists[0]
            else:
                mu_l, mu_a, mu_b = sample.lab
                sig_l, sig_a, sig_b = sample.lab_std if hasattr(sample, 'lab_std') and sample.lab_std else [14.0, 4.0, 5.0]

                d_col = np.sqrt(
                    ((l_chan - mu_l) / (sig_l * tol_scale)) ** 2 +
                    ((a_chan - mu_a) / (sig_a * tol_scale)) ** 2 +
                    ((b_chan - mu_b) / (sig_b * tol_scale)) ** 2
                )
                if is_watercolor_class:
                    total_dist = d_col
                else:
                    if getattr(sample, 'has_pattern_profile', False):
                        d_tex = np.abs(tex_energy - sample.tex_energy_mean) / sample.tex_energy_std
                        d_dir = np.abs(dir_coh - sample.dir_coherence_mean) / sample.dir_coherence_std
                        w_tex = float(getattr(sample, "texture_weight", 0.5))
                        total_dist = d_col + (w_tex * 0.4 * d_tex) + (w_tex * 0.25 * d_dir)
                    else:
                        tex_norm = (tex_energy - 18.0) / 10.0
                        total_dist = d_col - (sample.texture_weight * tex_norm * 0.3)

            dist_maps.append(total_dist)

        dist_stack = np.stack(dist_maps, axis=2)
        winner_idx = np.argmin(dist_stack, axis=2)
        min_dists = np.min(dist_stack, axis=2)

        # 5. Broad Spectral Plausibility Gates (prevents false bleed across categories)
        gate_forest = has_forest_wash & (~is_pure_paper) & (~is_uncolored_slope)
        gate_meadow = has_meadow_wash & (~is_pure_paper) & (~is_uncolored_slope)
        gate_water = (b_chan <= 129.5) & (b_p >= r_p - 8) & (~has_forest_wash) & (~is_pure_paper) & (~is_uncolored_slope)
        gate_vineyard = (
            (a_chan >= 130.0) & 
            (r_p >= g_p) & 
            (r_p >= b_p + 4) & 
            (~( (a_chan <= 127.5) & (b_chan >= 148.0) & (g_p >= b_p + 30) )) &
            (~is_pure_paper)
        )
        gate_gravel = (
            (r_p >= b_p - 4) &
            (g_p >= b_p - 10) &
            (~has_forest_wash)
        )

        # Collar filter (ignore scanned map collar and outer border lines)
        sh, sw = small.shape[:2]
        is_black = (small[:, :, 0] < 15) & (small[:, :, 1] < 15) & (small[:, :, 2] < 15)
        is_white = (small[:, :, 0] > 252) & (small[:, :, 1] > 252) & (small[:, :, 2] > 252)
        border_mask = np.zeros((sh, sw), dtype=bool)
        border_mask[6:sh-6, 6:sw-6] = True
        valid_collar = (~(is_black | is_white)) & border_mask

        # 6. Discrete Winner Grid (0 = Paper, 1 = Mountain Relief, 2..N+1 = Classes)
        winner_grid = np.zeros((sh, sw), dtype=np.uint8)

        for idx, sample in enumerate(active_classes, 2):
            cid = sample.class_id.lower()
            tol_scale = float(getattr(sample, "tolerance", 24)) / 24.0
            max_allowed_dist = 3.6 * (tol_scale ** 0.6)
            if "forest" in cid or "wald" in cid:
                gate = gate_forest
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope))
            elif "meadow" in cid or "wiese" in cid or "weide" in cid:
                gate = gate_meadow
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope))
            elif "garden" in cid or "garten" in cid:
                gate = (a_chan <= 127.0) & (b_chan <= 145.0) & (g_p >= b_p + 5) & (~is_pure_paper) & (~is_uncolored_slope)
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope))
            elif "water" in cid or "gewässer" in cid or "wasser" in cid:
                gate = gate_water
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope))
            elif "vineyard" in cid or "wein" in cid:
                gate = gate_vineyard
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope))
            elif "gravel" in cid or "kies" in cid:
                gate = gate_gravel
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar)
            else:
                gate = (~is_pure_paper)
                c_mask = ((winner_idx == idx) & (min_dists <= max_allowed_dist) & gate & valid_collar)

            if "meadow" in cid or "wiese" in cid or "weide" in cid:
                density = cv2.boxFilter(c_mask.astype(np.float32), -1, (15, 15))
                c_mask = (density >= 0.25) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope)
            elif "forest" in cid or "wald" in cid:
                density = cv2.boxFilter(c_mask.astype(np.float32), -1, (15, 15))
                c_mask = (density >= 0.20) & gate & valid_collar & (~is_pure_paper) & (~is_uncolored_slope)

            winner_grid[c_mask] = idx

        # Label filter: preserve delicate water streams while smoothing noise for other classes
        water_raw_mask = (winner_grid == water_grid_idx) if water_grid_idx is not None else None
        winner_grid_clean = cv2.medianBlur(winner_grid, 3)
        if water_raw_mask is not None:
            winner_grid_clean[water_raw_mask] = water_grid_idx

        # 7. Vectorize each class with specialized geometry pipelines
        results: Dict[str, List[Polygon]] = {}

        for idx, sample in enumerate(active_classes, 2):
            c_mask = (winner_grid_clean == idx).astype(np.uint8) * 255

            if np.count_nonzero(c_mask) == 0:
                results[sample.class_id] = []
                continue

            cid = sample.class_id
            user_min_area = float(getattr(sample, "min_area_px", 0.0))
            if cid == "water":
                # Specialized water ribbon & lake pipeline with dynamic gap closing (bridges, dams, text)
                k_size = max(3, int(round(closing_kernel_px * scale_factor)))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_closed = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
                mask_clean = cv2.dilate(mask_closed, k_dilate, iterations=1)
                mask_clean[is_pure_paper] = 0
                mask_clean[is_uncolored_slope] = 0
                min_area = max(5.0, user_min_area) if user_min_area > 0 else 15.0
                approx_eps = 0.5
                min_hole_area = max(50.0, 300.0 * (scale_factor ** 2))
            elif cid == "forest":
                # Dynamic forest canopy closing: bridges foliage crowns without fusing distant hills
                k_size = max(5, int(round(min(14, closing_kernel_px) * scale_factor)))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_clean = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
                mask_clean[is_pure_paper] = 0
                mask_clean[is_uncolored_slope] = 0
                min_area = max(25.0, user_min_area) if user_min_area > 0 else 100.0
                approx_eps = 0.5
                min_hole_area = min(3000.0, fill_holes_area) * (scale_factor ** 2)
            elif cid in ("meadow", "garden"):
                # Clean agricultural parcel blocks without bridging roads
                k_size = max(5, int(round(closing_kernel_px * scale_factor * 0.6)))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_clean = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
                mask_clean[is_pure_paper] = 0
                mask_clean[is_uncolored_slope] = 0
                min_area = max(25.0, user_min_area) if user_min_area > 0 else 200.0
                approx_eps = 0.5
                min_hole_area = 2000.0 * (scale_factor ** 2)
            elif cid in ("gravel", "kies"):
                # Sandbar and gravel bank morphology: bridge stippling dots into contiguous bars
                k_size = max(3, int(round(closing_kernel_px * scale_factor)))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_clean = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
                min_area = max(10.0, user_min_area) if user_min_area > 0 else 50.0
                approx_eps = 0.5
                min_hole_area = 2000.0 * (scale_factor ** 2)
            else:
                k_size = max(5, int(round(closing_kernel_px * scale_factor * 0.6)))
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_closed = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, k_close)
                mask_clean = cv2.dilate(mask_closed, k_dilate, iterations=1)
                mask_clean[is_pure_paper] = 0
                mask_clean[is_uncolored_slope] = 0
                min_area = max(25.0, user_min_area) if user_min_area > 0 else 150.0
                approx_eps = 0.5
                min_hole_area = 5000.0 * (scale_factor ** 2)

            cnts, hierarchy = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts or hierarchy is None:
                results[sample.class_id] = []
                continue

            polys: List[Polygon] = []
            hierarchy = hierarchy[0]
            scaled_min_area = min_area * (scale_factor ** 2)

            for i, cnt in enumerate(cnts):
                if hierarchy[i][3] != -1:
                    continue  # Child hole

                area = cv2.contourArea(cnt)
                if area < scaled_min_area:
                    continue

                approx = cv2.approxPolyDP(cnt, approx_eps, True)
                if len(approx) < 3:
                    continue

                raw_pts = [(float(p[0][0]) * inv_scale, float(p[0][1]) * inv_scale) for p in approx]
                ext_pts = self._chaikin_smooth(raw_pts, iterations=1)

                # Collect only genuine large holes (ignore tiny tree crown voids)
                holes = []
                child_idx = hierarchy[i][2]
                while child_idx != -1:
                    child_cnt = cnts[child_idx]
                    if cv2.contourArea(child_cnt) >= min_hole_area:
                        child_approx = cv2.approxPolyDP(child_cnt, 1.2, True)
                        if len(child_approx) >= 3:
                            raw_h = [(float(p[0][0]) * inv_scale, float(p[0][1]) * inv_scale) for p in child_approx]
                            holes.append(self._chaikin_smooth(raw_h, iterations=1))
                    child_idx = hierarchy[child_idx][0]

                try:
                    poly = Polygon(shell=ext_pts, holes=holes if holes else None)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    
                    def _accept_polygon(p: Polygon) -> bool:
                        if p is None or p.is_empty:
                            return False
                        if not p.is_valid:
                            p = p.buffer(0)
                        if not p.is_valid or p.is_empty or p.area < min_area:
                            return False
                        # Filter out extreme 1-2 pixel raster sliver artifacts
                        try:
                            rect = p.minimum_rotated_rectangle
                            if rect and not rect.is_empty and hasattr(rect, "exterior") and rect.exterior:
                                pts = list(rect.exterior.coords)
                                if len(pts) >= 4:
                                    edge1 = np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
                                    edge2 = np.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
                                    min_thick = min(edge1, edge2)
                                    if min_thick < 3.0:
                                        return False
                        except Exception:
                            pass
                        return True

                    if poly.is_valid and not poly.is_empty:
                        if isinstance(poly, Polygon):
                            if _accept_polygon(poly):
                                polys.append(poly.simplify(1.0, preserve_topology=True))
                        elif hasattr(poly, "geoms"):
                            for sub_p in poly.geoms:
                                if _accept_polygon(sub_p):
                                    polys.append(sub_p.simplify(1.0, preserve_topology=True))
                except Exception:
                    continue

            results[sample.class_id] = polys

        # 8. Strict topological non-overlap enforcement
        # Guarantee mathematically ZERO overlap across all extracted polygons (both internal and cross-class)
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon, GeometryCollection

        priority_order = ["water", "forest", "vineyard", "meadow", "garden", "gravel"]
        ordered_classes = [c for c in priority_order if c in results]
        for c in results:
            if c not in ordered_classes:
                ordered_classes.append(c)

        clean_results: Dict[str, List[Polygon]] = {}
        occupied_union = None

        for cid in ordered_classes:
            plist = results.get(cid, [])
            if not plist:
                clean_results[cid] = []
                continue

            # A. Dissolve internal overlaps within the class
            u = unary_union(plist)
            if not u.is_valid:
                u = u.buffer(0)

            # B. Subtract already occupied areas from higher-priority classes
            if occupied_union is not None and not occupied_union.is_empty:
                try:
                    u = u.difference(occupied_union)
                    if not u.is_valid:
                        u = u.buffer(0)
                except Exception:
                    pass

            # C. Extract individual polygons and filter micro-slivers
            user_min_area = float(getattr(self.samples.get(cid), "min_area_px", 0.0))
            thresh_area = max(25.0, user_min_area) if user_min_area > 0 else 25.0

            sub_polys: List[Polygon] = []
            if isinstance(u, Polygon):
                if u.is_valid and not u.is_empty and u.area >= thresh_area:
                    sub_polys.append(u)
            elif isinstance(u, (MultiPolygon, GeometryCollection)) or hasattr(u, "geoms"):
                for g in u.geoms:
                    if isinstance(g, Polygon) and g.is_valid and not g.is_empty and g.area >= thresh_area:
                        sub_polys.append(g)

            clean_results[cid] = sub_polys

            # D. Update occupied union
            if sub_polys:
                this_u = unary_union(sub_polys)
                if occupied_union is None:
                    occupied_union = this_u
                else:
                    occupied_union = unary_union([occupied_union, this_u])

        return clean_results



    def extract_class_polygons(
        self,
        image_rgb: np.ndarray,
        class_id: str,
        tolerance_override: Optional[int] = None
    ) -> List[Polygon]:
        """Extracts polygons for a specific class using the unified competitive pipeline."""
        if class_id not in self.samples:
            return []
        if tolerance_override is not None:
            self.samples[class_id].tolerance = tolerance_override
        all_res = self.extract_competitive_polygons(image_rgb, active_class_ids=[class_id])
        return all_res.get(class_id, [])

    def save_profile(self, filepath: str):
        data = {}
        for k, v in self.samples.items():
            data[k] = {
                "class_id": v.class_id,
                "label": v.label,
                "rgb": v.rgb,
                "hsv": v.hsv,
                "lab": v.lab,
                "hex_color": v.hex_color,
                "tolerance": v.tolerance,
                "min_area_px": v.min_area_px,
                "active": v.active,
                "texture_weight": v.texture_weight,
                "lab_std": v.lab_std,
                "tex_energy_mean": v.tex_energy_mean,
                "tex_energy_std": v.tex_energy_std,
                "dir_coherence_mean": v.dir_coherence_mean,
                "dir_coherence_std": v.dir_coherence_std,
                "has_pattern_profile": v.has_pattern_profile,
                "stamps": [
                    {
                        "stamp_id": s.stamp_id,
                        "cx": s.cx,
                        "cy": s.cy,
                        "radius": s.radius,
                        "rgb": s.rgb,
                        "hsv": s.hsv,
                        "lab": s.lab,
                        "hex_color": s.hex_color,
                        "lab_std": s.lab_std,
                        "tex_energy_mean": s.tex_energy_mean,
                        "tex_energy_std": s.tex_energy_std,
                        "dir_coherence_mean": s.dir_coherence_mean,
                        "dir_coherence_std": s.dir_coherence_std,
                        "name": s.name,
                        "raw_pixels": s.raw_pixels,
                        "distilled_pixels": s.distilled_pixels,
                    }
                    for s in getattr(v, "stamps", [])
                ],
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_profile(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            stamps_list = []
            for s_data in v.get("stamps", []):
                stamps_list.append(StampEntry(
                    stamp_id=s_data.get("stamp_id", 1),
                    cx=s_data.get("cx", 0.0),
                    cy=s_data.get("cy", 0.0),
                    radius=s_data.get("radius", 22),
                    rgb=s_data.get("rgb", [70, 140, 75]),
                    hsv=s_data.get("hsv", [60, 100, 100]),
                    lab=s_data.get("lab", [140, 120, 135]),
                    hex_color=s_data.get("hex_color", "#27ae60"),
                    lab_std=s_data.get("lab_std", [10.0, 4.0, 4.0]),
                    tex_energy_mean=s_data.get("tex_energy_mean", 18.0),
                    tex_energy_std=s_data.get("tex_energy_std", 5.0),
                    dir_coherence_mean=s_data.get("dir_coherence_mean", 0.5),
                    dir_coherence_std=s_data.get("dir_coherence_std", 0.15),
                    name=s_data.get("name", ""),
                    raw_pixels=s_data.get("raw_pixels", 0),
                    distilled_pixels=s_data.get("distilled_pixels", 0),
                ))
            self.samples[k] = ColorSample(
                class_id=v["class_id"],
                label=v["label"],
                rgb=v["rgb"],
                hsv=v["hsv"],
                lab=v["lab"],
                hex_color=v["hex_color"],
                tolerance=v.get("tolerance", 24),
                min_area_px=v.get("min_area_px", 350.0),
                active=v.get("active", True),
                texture_weight=v.get("texture_weight", 0.5),
                lab_std=v.get("lab_std", [8.0, 4.0, 4.0]),
                tex_energy_mean=v.get("tex_energy_mean", 18.0),
                tex_energy_std=v.get("tex_energy_std", 5.0),
                dir_coherence_mean=v.get("dir_coherence_mean", 0.5),
                dir_coherence_std=v.get("dir_coherence_std", 0.15),
                has_pattern_profile=v.get("has_pattern_profile", False) or len(stamps_list) > 0,
                stamps=stamps_list,
            )

    def sample_from_coordinate(self, image_rgb: np.ndarray, class_id: str, px: int, py: int, radius: int = 5) -> ColorSample:
        return self.sample_from_point(image_rgb, class_id, px, py, radius)

    def save_palette(self, filepath: str):
        self.save_profile(filepath)

    def load_palette(self, filepath: str):
        self.load_profile(filepath)


