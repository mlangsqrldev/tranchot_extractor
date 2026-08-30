import unittest
import numpy as np
import cv2
from tranchot_extractor.extractors.sam_extractor import SAMExtractor


class TestSAMExtractor(unittest.TestCase):

    def setUp(self):
        self.sam = SAMExtractor()

    def test_sam_point_prompt(self):
        # Create synthetic canvas with red rectangle
        img = np.full((120, 120, 3), (220, 210, 180), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (80, 80), (210, 30, 30), -1)

        res = self.sam.segment_with_prompts(
            img,
            positive_points=[(50, 50)],
        )

        self.assertGreater(np.count_nonzero(res["mask"]), 100)
        self.assertGreater(len(res["polygons"]), 0)
        self.assertGreaterEqual(res["iou_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
