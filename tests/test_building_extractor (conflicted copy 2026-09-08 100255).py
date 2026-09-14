import unittest
import numpy as np
import cv2
from tranchot_extractor.config import BuildingConfig
from tranchot_extractor.extractors.building_extractor import (
    BuildingExtractor,
    despike_and_simplify_ring,
    regularize_orthogonal_ring,
)


class TestBuildingExtractor(unittest.TestCase):

    def setUp(self):
        self.config = BuildingConfig(min_building_area_px=10.0)
        self.extractor = BuildingExtractor(self.config)

    def test_extract_synthetic_red_buildings(self):
        # Create beige canvas (parchment)
        img = np.full((200, 200, 3), (220, 210, 180), dtype=np.uint8)

        # Draw two distinct red building rectangles (Carmine red: R=210, G=40, B=40)
        cv2.rectangle(img, (30, 30), (70, 60), (210, 40, 40), -1)
        cv2.rectangle(img, (120, 100), (160, 150), (220, 50, 50), -1)

        res = self.extractor.extract(img)

        self.assertGreaterEqual(len(res.features), 2)
        self.assertGreater(np.count_nonzero(res.mask), 0)
        self.assertEqual(len(res.gdf), len(res.features))

        # Check geometry properties
        f1 = res.features[0]
        self.assertGreater(f1.area_px, 10)
        self.assertTrue(f1.geometry.is_valid)

        # Both should be regularized 4-corner rectangles (5 coords in closed ring)
        for f in res.features:
            self.assertEqual(len(f.geometry.exterior.coords), 5)

    def test_noisy_jagged_building_despiked(self):
        """Verify that buildings with 1-pixel jagged raster spikes are regularized cleanly."""
        img = np.full((150, 150, 3), (220, 210, 180), dtype=np.uint8)
        # Base rectangle
        cv2.rectangle(img, (40, 40), (90, 70), (210, 40, 40), -1)
        # Add random 1-2 px jagged spikes to the edge
        img[39, 45:48] = [210, 40, 40]
        img[38, 55] = [210, 40, 40]
        img[71, 60:63] = [210, 40, 40]
        img[50:53, 91] = [210, 40, 40]

        res = self.extractor.extract(img)
        self.assertEqual(len(res.features), 1)
        f = res.features[0]
        self.assertTrue(f.geometry.is_valid)
        # Should be regularized to a clean 4-corner rectangle (5 coords in closed ring)
        self.assertEqual(len(f.geometry.exterior.coords), 5)

    def test_courtyard_with_hole(self):
        """Verify that a large courtyard complex preserves interior yard (hole) without jagged spikes."""
        img = np.full((200, 200, 3), (220, 210, 180), dtype=np.uint8)
        # Outer building complex
        cv2.rectangle(img, (30, 30), (120, 120), (210, 40, 40), -1)
        # Inner courtyard hole (parchment)
        cv2.rectangle(img, (55, 55), (95, 95), (220, 210, 180), -1)

        res = self.extractor.extract(img)
        self.assertGreaterEqual(len(res.features), 1)
        f = res.features[0]
        self.assertTrue(f.geometry.is_valid)
        # Must have an interior hole
        self.assertGreaterEqual(len(f.geometry.interiors), 1)
        # Exterior and interior should have low vertex counts (no staircase noise)
        self.assertLessEqual(len(f.geometry.exterior.coords), 10)

    def test_despike_and_simplify_ring(self):
        """Direct unit test for despike_and_simplify_ring function."""
        # Square with an acute spike on one side and collinear points on another
        raw_ring = np.array([
            [10.0, 10.0],
            [20.0, 10.0],  # collinear
            [30.0, 10.0],
            [30.0, 20.0],
            [32.0, 20.5],  # acute spike forward
            [30.0, 21.0],  # acute spike back
            [30.0, 30.0],
            [10.0, 30.0],
            [10.0, 10.0],
        ])
    def test_thin_walls_extracted(self):
        """Verify that thin walls (1-2 px stroke width) are detected cleanly as 4-corner rectangles."""
        img = np.full((120, 120, 3), (220, 210, 180), dtype=np.uint8)
        # Draw a thin 2-pixel red wall
        cv2.line(img, (30, 20), (30, 70), (210, 40, 40), 2)

        cfg = BuildingConfig(min_building_area_px=6.0, min_stroke_width_px=1.0)
        ext = BuildingExtractor(cfg)
        res = ext.extract(img)

        self.assertGreaterEqual(len(res.features), 1)
        f = res.features[0]
        self.assertTrue(f.geometry.is_valid)
        self.assertEqual(len(f.geometry.exterior.coords), 5)


if __name__ == "__main__":
    unittest.main()
