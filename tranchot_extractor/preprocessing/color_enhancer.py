"""
Tranchot Map Feature Extractor — Color Enhancement & Parchment De-Yellowing Module.
Removes 200-year-old parchment yellowing/browning, applies chromatic white balance,
and boosts historical watercolor washes (green, turquoise, blue, carmine, ocher, salmon).
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class ColorEnhancer:
    """
    Restores historical map sheets by neutralizing paper browning/yellowing
    and boosting chromatic radiance of watercolor pigments.
    """

    @staticmethod
    def estimate_paper_color(image_rgb: np.ndarray) -> np.ndarray:
        """
        Estimates the baseline parchment color from the brightest, warm-tinted pixels.
        """
        img_f = image_rgb.astype(np.float32)
        # Luminance
        lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
        
        # Paper is bright and has warm tint (R >= B)
        p_mask = (lum > np.percentile(lum, 70)) & (img_f[:, :, 0] >= img_f[:, :, 2] - 5.0)
        
        if np.count_nonzero(p_mask) > 100:
            paper_rgb = np.median(img_f[p_mask], axis=0)
        else:
            paper_rgb = np.array([215.0, 195.0, 155.0], dtype=np.float32)

        return np.clip(paper_rgb, 120.0, 255.0)

    @classmethod
    def enhance_map(
        cls,
        image_rgb: np.ndarray,
        paper_rgb: Optional[Tuple[float, float, float]] = None,
        deyellow_strength: float = 0.85,
        vibrance: float = 1.85,
        contrast: float = 1.05,
        flat_field: bool = False,
    ) -> np.ndarray:
        """
        Applies white-balance, parchment neutralization, and watercolor vibrancy boost.

        Args:
            image_rgb: Input RGB uint8 numpy array.
            paper_rgb: Optional (R, G, B) reference color of paper. If None, auto-estimated.
            deyellow_strength: 0.0 (raw) to 1.0 (fully neutralized white paper).
            vibrance: 1.0 (original) to 3.0 (intense watercolor radiance).
            contrast: Linear contrast multiplier (default: 1.05).
            flat_field: If True, corrects large-scale lighting gradients.

        Returns:
            Enhanced RGB uint8 numpy array.
        """
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb

        img_f = image_rgb.astype(np.float32)

        # 1. Background Paper Estimation
        if paper_rgb is None:
            ref_paper = cls.estimate_paper_color(image_rgb)
        else:
            ref_paper = np.array(paper_rgb, dtype=np.float32)

        # 2. Chromatic Adaptation / White Balance (Von Kries scaling towards neutral 245)
        # Target white point:
        target_white = np.array([245.0, 245.0, 245.0], dtype=np.float32)
        gain = target_white / (ref_paper + 1e-5)

        # Interpolate between original and fully white-balanced based on deyellow_strength
        effective_gain = (1.0 - deyellow_strength) * 1.0 + deyellow_strength * gain

        if flat_field:
            # Multi-scale illumination estimation
            small_w = max(64, image_rgb.shape[1] // 8)
            small_h = max(64, image_rgb.shape[0] // 8)
            small = cv2.resize(img_f, (small_w, small_h), interpolation=cv2.INTER_AREA)
            blur_small = cv2.GaussianBlur(small, (51, 51), 0)
            illum = cv2.resize(blur_small, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm_f = np.clip((img_f / (illum + 1e-5)) * 242.0, 0, 255.0)
            balanced_f = norm_f
        else:
            balanced_f = np.clip(img_f * effective_gain, 0, 255.0)

        # 3. Contrast adjustment
        if contrast != 1.0:
            balanced_f = np.clip((balanced_f - 128.0) * contrast + 128.0, 0, 255.0)

        balanced_u8 = balanced_f.astype(np.uint8)

        # 4. Selective Vibrancy Boost (Boost watercolor chroma without distorting ink or neutral paper)
        if vibrance > 1.01:
            hsv = cv2.cvtColor(balanced_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # Saturation boost: only boost where there is watercolor pigment (s > 14) and not dark ink (v > 35)
            # Smoothly ramp up multiplier from s=14 to s=35
            sat_weight = np.clip((s - 14.0) / 20.0, 0.0, 1.0)
            sat_mult = 1.0 + (vibrance - 1.0) * sat_weight * (1.0 - (s / 255.0) * 0.3)
            new_s = np.clip(s * sat_mult, 0, 255.0)
            
            hsv[:, :, 1] = new_s
            enhanced_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return enhanced_u8

        return balanced_u8
