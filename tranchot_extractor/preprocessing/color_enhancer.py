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
        ink_blackening: bool = False,
    ) -> np.ndarray:
        """
        Applies white-balance, parchment neutralization, and watercolor vibrancy boost.
        """
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb

        img_f = image_rgb.astype(np.float32)

        # 1. Background Paper Estimation
        if paper_rgb is None:
            ref_paper = cls.estimate_paper_color(image_rgb)
        else:
            ref_paper = np.array(paper_rgb, dtype=np.float32)

        # 2. Chromatic Adaptation / White Balance (Von Kries scaling towards neutral 255)
        target_white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        gain = target_white / (ref_paper + 1e-5)

        # Lightness-Gated Chromatic Adaptation:
        # Paper & light watercolor washes (L > 150) get full white-balance.
        # Dark hachure ink (L < 100) is NOT multiplied by blue gain, preventing blue halo artifacts!
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ink_factor = np.clip((gray - 110.0) / 60.0, 0.0, 1.0)[:, :, np.newaxis]
        effective_gain = 1.0 + ink_factor * (gain - 1.0) * deyellow_strength

        if flat_field:
            small_w = max(64, image_rgb.shape[1] // 8)
            small_h = max(64, image_rgb.shape[0] // 8)
            small = cv2.resize(img_f, (small_w, small_h), interpolation=cv2.INTER_AREA)
            blur_small = cv2.GaussianBlur(small, (51, 51), 0)
            illum = cv2.resize(blur_small, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm_f = np.clip((img_f / (illum + 1e-5)) * 255.0, 0, 255.0)
            balanced_f = norm_f
        else:
            balanced_f = np.clip(img_f * effective_gain, 0.0, 255.0)

        # Dark Ink Neutrality (Prevents blue halos on dark hachures without bleaching the lake)
        is_dark_ink = (gray < 100.0)
        orig_is_blue = (image_rgb[:, :, 2].astype(int) > image_rgb[:, :, 0].astype(int) + 6)
        balanced_f[:, :, 2] = np.where(
            is_dark_ink & (~orig_is_blue),
            np.minimum(balanced_f[:, :, 2], balanced_f[:, :, 0]),
            balanced_f[:, :, 2]
        )

        # 3. Contrast adjustment
        if contrast != 1.0:
            balanced_f = np.clip((balanced_f - 128.0) * contrast + 128.0, 0.0, 255.0)

        # 4. Pure Paper Whiten & Ink Blacken (Strictly on blank unpainted paper, NEVER on watercolor washes)
        if deyellow_strength >= 0.8:
            hsv_b = cv2.cvtColor(np.clip(balanced_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            sat = hsv_b[:, :, 1]
            val = hsv_b[:, :, 2]
            lab_b = cv2.cvtColor(np.clip(balanced_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            chroma = np.sqrt((lab_b[:, :, 1] - 128.0)**2 + (lab_b[:, :, 2] - 128.0)**2)
            # Only unpainted white parchment (sat < 8, chroma < 5)
            is_paper = (val > 240.0) & (sat < 8.0) & (chroma < 5.0)
            balanced_f[is_paper] = [255.0, 255.0, 255.0]

        if ink_blackening:
            # Protect genuine blue streams and lake
            is_blue_water = (balanced_f[:, :, 2] > balanced_f[:, :, 0] + 6)
            is_ink = (gray < 85.0) & (~is_blue_water)
            ink_weight = np.clip((85.0 - gray) / 45.0, 0.0, 1.0)[:, :, np.newaxis]
            balanced_f = np.where(is_ink[:, :, np.newaxis], balanced_f * (1.0 - 0.85 * ink_weight), balanced_f)

        balanced_u8 = np.clip(balanced_f, 0, 255).astype(np.uint8)

        # 5. Selective Vibrancy Boost (Radiates watercolor pigments: lake, meadow, forest, gardens)
        if vibrance > 1.01:
            hsv = cv2.cvtColor(balanced_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            s = hsv[:, :, 1]
            sat_boost = 1.0 + (vibrance - 1.0) * np.clip((s - 6.0) / 20.0, 0.0, 1.0)
            hsv[:, :, 1] = np.clip(s * sat_boost, 0.0, 255.0)
            enhanced_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            if deyellow_strength >= 0.8:
                enhanced_u8[is_paper] = [255, 255, 255]
            return enhanced_u8

        return balanced_u8

    @classmethod
    def normalize_paper_and_ink(
        cls,
        image_rgb: np.ndarray,
        paper_rgb: Optional[Tuple[float, float, float]] = None,
        ink_blackening: bool = True,
        vibrance: float = 1.65,
    ) -> np.ndarray:
        """
        Pure Paper White & Ink Black Normalization.
        Maps parchment background to 100% white, dark hachures to deep black,
        and radiates watercolor pigments with maximum clarity.
        """
        return cls.enhance_map(
            image_rgb,
            paper_rgb=paper_rgb,
            deyellow_strength=1.0,
            vibrance=vibrance,
            contrast=1.05,
            ink_blackening=ink_blackening,
        )
