import unittest
import numpy as np
import cv2
from tranchot_extractor.config import RoadConfig
from tranchot_extractor.extractors.road_extractor import RoadExtractor


class TestRoadExtractor(unittest.TestCase):

    def setUp(self):
        self.config = RoadConfig(min_path_length_px=10.0, ridge_threshold=15.0)
        self.extractor = RoadExtractor(self.config)

    def test_trace_road_between_points(self):
        # Create light parchment background
        img = np.full((200, 200, 3), 220, dtype=np.uint8)

        # Draw dual-parallel dark ink lines (8px gap)
        cv2.line(img, (20, 50), (180, 50), (40, 40, 40), 2)
        cv2.line(img, (20, 58), (180, 58), (40, 40, 40), 2)

        line = self.extractor.trace_road_between_points(img, (25.0, 54.0), (175.0, 54.0))

        self.assertIsNotNone(line)
        self.assertGreaterEqual(line.length, 100.0)
        # Verify centerline is around y=54 (midpoint between 50 and 58)
        mid_y = np.mean([p[1] for p in line.coords])
        self.assertAlmostEqual(mid_y, 54.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
