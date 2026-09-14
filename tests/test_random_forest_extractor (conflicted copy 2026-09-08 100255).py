"""
Unit tests for RandomForestLandUseExtractor.
"""

import unittest
import numpy as np
from shapely.geometry import Polygon

from tranchot_extractor.extractors.random_forest_extractor import RandomForestLandUseExtractor, HAS_SKLEARN


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestRandomForestLandUseExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = RandomForestLandUseExtractor()
        # Synthetic map image: 200x200
        # Background: parchment (light yellowish)
        self.image = np.full((200, 200, 3), (230, 220, 190), dtype=np.uint8)
        
        # Forest region (green)
        self.image[20:70, 20:70] = (40, 120, 50)
        # Meadow region (cyan-green)
        self.image[120:170, 120:170] = (80, 190, 160)

    def test_feature_stack_shape(self):
        feats = self.extractor.compute_feature_stack(self.image)
        self.assertEqual(feats.shape, (200, 200, 16))
        self.assertEqual(feats.dtype, np.float32)

    def test_train_and_predict(self):
        forest_poly = Polygon([(25, 25), (65, 25), (65, 65), (25, 65)])
        meadow_poly = Polygon([(125, 125), (165, 125), (165, 165), (125, 165)])

        exemplars = {
            "forest": [forest_poly],
            "meadow": [meadow_poly],
        }

        trained = self.extractor.train(self.image, exemplars)
        self.assertTrue(trained)
        self.assertIsNotNone(self.extractor.model)

        # Predict probability maps
        prob_maps = self.extractor.predict_tile(self.image, min_probability=0.50)
        self.assertIn("forest", prob_maps)
        self.assertIn("meadow", prob_maps)
        self.assertEqual(prob_maps["forest"].shape, (200, 200))

        # Check that forest region has high probability for forest
        self.assertGreater(prob_maps["forest"][45, 45], 0.70)
        self.assertGreater(prob_maps["meadow"][145, 145], 0.70)

    def test_extract_competitive_polygons(self):
        forest_poly = Polygon([(25, 25), (65, 25), (65, 65), (25, 65)])
        meadow_poly = Polygon([(125, 125), (165, 125), (165, 165), (125, 165)])

        exemplars = {
            "forest": [forest_poly],
            "meadow": [meadow_poly],
        }

        self.extractor.train(self.image, exemplars)
        polys = self.extractor.extract_competitive_polygons(self.image, min_probability=0.50, min_area_px=100.0)

        self.assertIn("forest", polys)
        self.assertIn("meadow", polys)
        self.assertGreater(len(polys["forest"]), 0)
        self.assertGreater(len(polys["meadow"]), 0)

        # Ensure no overlap
        f_poly = polys["forest"][0]
        m_poly = polys["meadow"][0]
        self.assertFalse(f_poly.intersects(m_poly))


if __name__ == "__main__":
    unittest.main()
