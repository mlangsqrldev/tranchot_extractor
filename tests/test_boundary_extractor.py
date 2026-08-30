"""
Unit tests for the BoundaryExtractor module.
"""

import unittest
import numpy as np
import cv2
from tranchot_extractor.extractors.boundary_extractor import BoundaryExtractor
from tranchot_extractor.config import BoundaryConfig


class TestBoundaryExtractor(unittest.TestCase):
    def setUp(self):
        self.config = BoundaryConfig(min_boundary_length_px=10.0)
        self.extractor = BoundaryExtractor(self.config)

    def test_extract_synthetic_boundary_line(self):
        # Create beige canvas
        img = np.full((100, 100, 3), (230, 220, 200), dtype=np.uint8)
        
        # Draw thin dashed red boundary line (width = 1 px)
        for i in range(10, 80, 6):
            cv2.line(img, (i, 50), (i + 4, 50), (40, 40, 200), 1) # BGR red

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.extractor.extract(img_rgb)

        self.assertIsNotNone(result)
        self.assertTrue(len(result.features) >= 1)
        self.assertFalse(result.gdf.empty)
        self.assertEqual(result.gdf.iloc[0]["feature_type"], "boundary")


if __name__ == "__main__":
    unittest.main()
