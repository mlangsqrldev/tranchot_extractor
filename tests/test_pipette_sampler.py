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

        # Clear default stamps to test single custom stamp addition
        self.sampler.samples["forest"].stamps.clear()

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

        # Clear default stamps to test 2 explicit stamps
        self.sampler.samples["forest"].stamps.clear()

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
            self.sampler.samples["forest"].stamps.clear()
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

    def test_meadow_mountain_and_road_rejection(self):
        """Verifies that hachured mountain slopes and unpainted roads are never recognized as meadow."""
        sampler = PipetteSampler()
        # Parchment background (400x400)
        img = np.full((400, 400, 3), [225, 215, 185], dtype=np.uint8)

        # 1. True meadow in valley (emerald/cyan green wash)
        meadow_color = [135, 175, 145]
        cv2.rectangle(img, (30, 30), (160, 140), meadow_color, -1)

        # 2. Hachured mountain slope (brownish background with dense black hachures)
        slope_color = [195, 180, 150]
        cv2.rectangle(img, (220, 220), (370, 370), slope_color, -1)
        for y in range(220, 370, 6):
            cv2.line(img, (220, y), (370, y + 10), (60, 45, 30), 2)

        # 3. Road casing lines
        cv2.line(img, (0, 200), (400, 200), (40, 30, 20), 2)
        cv2.line(img, (0, 215), (400, 215), (40, 30, 20), 2)

        # Sample meadow at center of valley
        sampler.sample_from_stamp(img, "meadow", 95, 85, radius=20)
        res = sampler.extract_competitive_polygons(img, active_class_ids=["meadow"])
        meadow_polys = res.get("meadow", [])

        # Only true meadow in the valley must be extracted; mountain slope and roads must be rejected
        self.assertEqual(len(meadow_polys), 1, "Only true meadow must survive, slope and road must be rejected")
        p = meadow_polys[0]
        self.assertLess(p.bounds[1], 160, "Extracted polygon must be the true valley meadow")

    def test_uncolored_slope_rejection_no_meadow_or_forest(self):
        """Verifies that uncolored mountain relief slopes with hachures produce zero meadow or forest polygons."""
        sampler = PipetteSampler()
        # Create map patch: neutral parchment with dense black/brown hachures
        img = np.full((300, 300, 3), [220, 210, 180], dtype=np.uint8)
        for y in range(20, 280, 5):
            cv2.line(img, (20, y), (280, y + 10), (70, 55, 40), 2)

        res = sampler.extract_competitive_polygons(img, active_class_ids=["meadow", "forest"])
        self.assertEqual(len(res.get("meadow", [])), 0, "Uncolored slope must NOT produce meadow polygons")
        self.assertEqual(len(res.get("forest", [])), 0, "Uncolored slope must NOT produce forest polygons")

    def test_olive_ochre_woodland_recognized(self):
        """Verifies that authentic Eifel Hangwald (olive-ochre glaze with hachures) is recognized as forest."""
        sampler = PipetteSampler()
        # Parchment background (300x300)
        img = np.full((300, 300, 3), [225, 215, 185], dtype=np.uint8)
        
        # Authentic Eifel woodland wash (RGB ~ [168, 160, 120])
        cv2.rectangle(img, (50, 50), (250, 250), [168, 160, 120], -1)
        # Tree and slope hachures over the woodland
        for y in range(60, 240, 7):
            cv2.line(img, (60, y), (240, y + 5), (60, 45, 30), 2)

        # Explicitly sample the woodland nuance with pipette tool
        sampler.sample_from_stamp(img, "forest", 150, 150, radius=20)

        res = sampler.extract_competitive_polygons(img, active_class_ids=["forest", "meadow"])
        forest_polys = res.get("forest", [])
        meadow_polys = res.get("meadow", [])

        self.assertGreaterEqual(len(forest_polys), 1, "Olive-ochre woodland must be extracted as forest")
        self.assertEqual(len(meadow_polys), 0, "Olive-ochre woodland must NOT be classified as meadow")

    def test_gravel_and_vineyard_exclusion_from_forest(self):
        """Verifies that warm gravel sandbars and vineyards are strictly rejected when extracting forest."""
        sampler = PipetteSampler()
        img = np.full((300, 300, 3), [225, 215, 185], dtype=np.uint8)

        # 1. Real forest block (Mühlberg green: R=185, G=196, B=126)
        cv2.rectangle(img, (20, 20), (120, 120), [185, 196, 126], -1)

        # 2. Gravel sandbar in river (Ahr gravel: R=184, G=179, B=142)
        cv2.rectangle(img, (150, 20), (250, 120), [184, 179, 142], -1)

        # 3. Vineyard on slope (Authentic Tranchot vineyard: carmine/reddish-brown wash with hachures)
        cv2.rectangle(img, (50, 160), (250, 260), [180, 135, 120], -1)
        for y in range(165, 255, 6):
            cv2.line(img, (55, y), (245, y + 3), (90, 65, 45), 2)

        # User samples forest on the real forest patch
        sampler.sample_from_stamp(img, "forest", 70, 70, radius=20)

        # Extract only forest
        res = sampler.extract_competitive_polygons(img, active_class_ids=["forest"])
        forest_polys = res.get("forest", [])

        self.assertGreaterEqual(len(forest_polys), 1, "Real forest must be extracted")
        for p in forest_polys:
            cx, cy = p.centroid.x, p.centroid.y
            # Must not be inside gravel sandbar
            self.assertFalse(140 <= cx <= 260 and 10 <= cy <= 130, f"Gravel bank at ({cx}, {cy}) must NOT be forest")
            # Must not be inside vineyard
            self.assertFalse(40 <= cx <= 260 and 150 <= cy <= 270, f"Vineyard at ({cx}, {cy}) must NOT be forest")

    def test_zero_overlap_guarantee(self):
        """Verifies that under no circumstances can extracted polygons overlap each other."""
        sampler = PipetteSampler()
        img = np.full((300, 300, 3), [225, 215, 185], dtype=np.uint8)

        # Adjoining forest and vineyard with mutual border
        cv2.rectangle(img, (20, 20), (140, 150), [185, 196, 126], -1)  # Forest
        cv2.rectangle(img, (140, 20), (260, 150), [180, 135, 120], -1) # Vineyard (shares border at x=140)
        cv2.rectangle(img, (50, 150), (200, 260), [110, 150, 210], -1) # Water stream adjoining both

        # Multi-class extraction
        res = sampler.extract_competitive_polygons(img, active_class_ids=["forest", "vineyard", "water"])
        all_polys = []
        for cid, plist in res.items():
            for p in plist:
                all_polys.append((cid, p))

        self.assertGreater(len(all_polys), 0)
        # Check every pair of polygons for intersection
        for i in range(len(all_polys)):
            for j in range(i + 1, len(all_polys)):
                c1, p1 = all_polys[i]
                c2, p2 = all_polys[j]
                inter_area = p1.intersection(p2).area
                self.assertAlmostEqual(
                    inter_area, 0.0, places=4,
                    msg=f"Polygons of class {c1} and {c2} overlap by {inter_area:.4f} px2! Overlap must be 0.0."
                )


if __name__ == "__main__":
    unittest.main()


