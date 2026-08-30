import unittest
import numpy as np
import cv2
from tranchot_extractor.config import TextConfig
from tranchot_extractor.extractors.text_extractor import TextExtractor


class TestTextExtractor(unittest.TestCase):

    def setUp(self):
        self.config = TextConfig(confidence_threshold=0.1)
        self.extractor = TextExtractor(self.config)

    def test_extract_text_features(self):
        # Create light image with clean text
        img = np.full((150, 400, 3), 245, dtype=np.uint8)
        cv2.putText(img, "Rommerskirchen", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)

        res = self.extractor.extract(img)

        # Ensure extraction runs and returns valid structures
        self.assertIsNotNone(res.features)
        self.assertIsNotNone(res.mask)
        self.assertEqual(res.mask.shape, (150, 400))


if __name__ == "__main__":
    unittest.main()
