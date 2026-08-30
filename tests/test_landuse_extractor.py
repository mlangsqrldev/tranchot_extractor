import unittest
import numpy as np
import cv2
from tranchot_extractor.config import LandUseConfig
from tranchot_extractor.extractors.landuse_extractor import LandUseExtractor


class TestLandUseExtractor(unittest.TestCase):

    def setUp(self):
        self.config = LandUseConfig()
        self.extractor = LandUseExtractor(self.config)

    def test_extract_synthetic_forest_and_water(self):
        # Create light background
        img = np.full((300, 300, 3), 220, dtype=np.uint8)

        # Draw dark olive-green forest polygon (G > R+3, R > B+10)
        forest_color = (95, 120, 75)
        cv2.rectangle(img, (30, 30), (120, 120), forest_color, -1)

        # Draw blue water polygon (B > R+15, B > G)
        water_color = (40, 80, 190)
        cv2.rectangle(img, (160, 160), (260, 260), water_color, -1)

        res = self.extractor.extract(img)

        self.assertGreaterEqual(len(res.features), 1)
        self.assertLess(res.execution_time_s, 1.0)


if __name__ == "__main__":
    unittest.main()
