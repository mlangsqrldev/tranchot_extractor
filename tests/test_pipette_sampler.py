import unittest
import numpy as np
import cv2
from tranchot_extractor.extractors.pipette_sampler import PipetteSampler, ColorSample


class TestPipetteSampler(unittest.TestCase):

    def setUp(self):
        self.sampler = PipetteSampler()

    def test_sample_and_extract(self):
        # Create image with specific color patch (e.g. Ahr gravel bank salmon color)
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        gravel_color = [225, 175, 150]
        cv2.rectangle(img, (20, 20), (80, 80), gravel_color, -1)

        # Sample at (50, 50)
        sample = self.sampler.sample_from_coordinate(img, "gravel", 50, 50, radius=3)
        self.assertEqual(sample.rgb, gravel_color)
        self.assertTrue(sample.active)

        # Extract polygons for gravel
        polys = self.sampler.extract_class_polygons(img, "gravel", tolerance_override=25)
        self.assertGreaterEqual(len(polys), 1)

    def test_save_and_load_palette(self):
        tmp_path = "tests_palette.json"
        try:
            self.sampler.save_palette(tmp_path)
            new_sampler = PipetteSampler()
            new_sampler.load_palette(tmp_path)
            self.assertIn("forest", new_sampler.samples)
            self.assertIn("gravel", new_sampler.samples)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
