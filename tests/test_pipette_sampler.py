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

    def test_circular_stamp_with_hachures(self):
        # Create map patch: parchment background with brown slope hachures
        img = np.full((150, 150, 3), [220, 205, 160], dtype=np.uint8)
        for y in range(0, 150, 8):
            cv2.line(img, (0, y), (150, y + 20), (80, 60, 40), 2)
        
        # Add green forest patch at center (75, 75)
        forest_green = [75, 145, 80]
        cv2.circle(img, (75, 75), 25, forest_green, -1)
        # Superimpose dark hachure line across the forest
        cv2.line(img, (50, 75), (100, 75), (50, 40, 30), 2)

        # Stamp at (75, 75) with radius 20
        stamp = self.sampler.sample_from_stamp(img, "forest", 75, 75, radius=20)
        self.assertIsNotNone(stamp)
        self.assertEqual(len(self.sampler.samples["forest"].stamps), 1)
        # Verify ink was stripped: mean RGB must be green (G > R)
        self.assertGreater(stamp.rgb[1], stamp.rgb[0] + 30)

    def test_multi_stamp_nearest_extraction(self):
        img = np.full((200, 200, 3), [220, 205, 160], dtype=np.uint8)
        
        # Shade 1: Light Forest Green
        cv2.rectangle(img, (20, 20), (70, 70), [90, 155, 95], -1)
        # Shade 2: Deep Olive Green
        cv2.rectangle(img, (120, 120), (170, 170), [60, 125, 65], -1)

        # Stamp both shades
        s1 = self.sampler.sample_from_stamp(img, "forest", 45, 45, radius=15)
        s2 = self.sampler.sample_from_stamp(img, "forest", 145, 145, radius=15)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertEqual(len(self.sampler.samples["forest"].stamps), 2)

        # Extract
        res = self.sampler.extract_competitive_polygons(img, active_class_ids=["forest"])
        forest_polys = res.get("forest", [])
        self.assertGreaterEqual(len(forest_polys), 2)

    def test_save_and_load_palette(self):
        tmp_path = "tests_palette.json"
        try:
            img = np.full((100, 100, 3), [70, 140, 75], dtype=np.uint8)
            self.sampler.sample_from_stamp(img, "forest", 50, 50, radius=20, name="Wald-Hell")
            self.sampler.save_palette(tmp_path)

            new_sampler = PipetteSampler()
            new_sampler.load_palette(tmp_path)
            self.assertIn("forest", new_sampler.samples)
            self.assertEqual(len(new_sampler.samples["forest"].stamps), 1)
            self.assertEqual(new_sampler.samples["forest"].stamps[0].name, "Wald-Hell")
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_thin_stream_extraction_and_paper_rejection(self):
        sampler = PipetteSampler()
        # White/cream parchment with slope lines
        img = np.full((300, 300, 3), [235, 230, 205], dtype=np.uint8)
        for y in range(0, 150, 15):
            cv2.line(img, (150, y), (300, y + 30), (100, 80, 60), 2)

        # Forest area (100x150)
        cv2.rectangle(img, (20, 20), (120, 170), [80, 150, 85], -1)

        # 2-pixel stream
        pts = np.array([[150, 200], [200, 230], [250, 270]], np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=[70, 140, 210], thickness=2)

        sampler.sample_from_stamp(img, "forest", 60, 80, radius=15)
        sampler.sample_from_stamp(img, "water", 200, 230, radius=8)

        res = sampler.extract_competitive_polygons(img, active_class_ids=["forest", "water", "meadow"])

        self.assertGreater(len(res.get("forest", [])), 0, "Forest must be extracted")
        self.assertGreater(len(res.get("water", [])), 0, "2-pixel stream must be extracted")
        self.assertEqual(len(res.get("meadow", [])), 0, "Meadow must NOT appear on white/cream paper")

    def test_forest_ribbon_and_road_rejection(self):
        """Verifies that narrow linear corridors (roads, ditches, hedges) are rejected as forest."""
        sampler = PipetteSampler()
        # Parchment background (400x400)
        img = np.full((400, 400, 3), [235, 230, 205], dtype=np.uint8)

        # True compact forest block (80x120)
        cv2.rectangle(img, (30, 30), (150, 110), [75, 145, 80], -1)

        # Long narrow road / ditch strip with similar green tint (12px wide, 250px long)
        cv2.rectangle(img, (30, 200), (280, 214), [75, 145, 80], -1)

        sampler.sample_from_stamp(img, "forest", 80, 70, radius=20)
        res = sampler.extract_competitive_polygons(img, active_class_ids=["forest"])
        forest_polys = res.get("forest", [])

        # Only the compact forest block must survive; the 14px narrow road ribbon must be rejected
        self.assertEqual(len(forest_polys), 1, "Only true forest must survive, road ribbon must be rejected")
        p = forest_polys[0]
        self.assertLess(p.bounds[1], 150, "Extracted polygon must be the true forest block in the upper part")


if __name__ == "__main__":
    unittest.main()
