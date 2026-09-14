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

    def test_stream_and_meadow_separation_same_color(self):
        """
        Tests the core cartographic challenge:
        Stream and Meadow share the identical cyan-green watercolor pigment (Verdigris),
        distinguished by 1D tubular bank geometry vs 2D areal parcel geometry.
        """
        # Historical parchment background (warm ochre paper)
        img = np.full((400, 600, 3), [225, 215, 185], dtype=np.uint8)

        # Identical pigment wash for valley floor: RGB = (180, 188, 168)
        cyan_pigment = (180, 188, 168)
        # Wide meadow corridor (y from 100 to 300)
        cv2.rectangle(img, (50, 100), (550, 300), cyan_pigment, -1)

        # Stream channel running through the center of the meadow:
        # Bounded by dark ink bank lines (y=195 and y=205) and slightly darker watercore
        cv2.line(img, (50, 196), (550, 196), (70, 75, 70), 2)  # Bank line 1
        cv2.line(img, (50, 204), (550, 204), (70, 75, 70), 2)  # Bank line 2
        cv2.rectangle(img, (50, 197), (550, 203), (170, 182, 172), -1)  # Channel watercore

        res = self.extractor.extract(img)

        # Verify that both water and meadow features were extracted separately
        categories = {f.category for f in res.features}
        self.assertIn("meadow", categories, "Meadow parcels should be extracted")
        self.assertIn("water", categories, "Stream channel should be extracted as water")

    def test_extract_parcel_at_click(self):
        """Tests interactive point-click query for stream vs meadow."""
        img = np.full((300, 300, 3), [225, 215, 185], dtype=np.uint8)
        cyan_pigment = (180, 188, 168)
        cv2.rectangle(img, (30, 50), (270, 250), cyan_pigment, -1)

        # Direct blue stream
        cv2.rectangle(img, (30, 140), (270, 160), (40, 80, 190), -1)

        water_click = self.extractor.extract_parcel_at(img, 150.0, 150.0)
        self.assertIsNotNone(water_click)
        self.assertEqual(water_click[1], "water")

        meadow_click = self.extractor.extract_parcel_at(img, 150.0, 80.0)
        self.assertIsNotNone(meadow_click)
        self.assertEqual(meadow_click[1], "meadow")


if __name__ == "__main__":
    unittest.main()
