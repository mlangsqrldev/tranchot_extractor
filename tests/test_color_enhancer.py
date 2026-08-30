import unittest
import numpy as np
from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer


class TestColorEnhancer(unittest.TestCase):

    def test_paper_estimation_and_enhancement(self):
        # Create synthetic yellowed parchment image
        img = np.full((100, 100, 3), [210, 185, 140], dtype=np.uint8) # Aged yellowish paper
        # Add faint blue watercolor river
        img[40:60, :] = [180, 190, 195]
        # Add faint green watercolor forest
        img[:, 40:60] = [170, 195, 150]

        paper = ColorEnhancer.estimate_paper_color(img)
        self.assertGreater(paper[0], paper[2]) # R > B (warm parchment)

        enhanced = ColorEnhancer.enhance_map(img, deyellow_strength=0.9, vibrance=2.0)
        self.assertEqual(enhanced.shape, (100, 100, 3))
        # Paper should be whiter (R, G, B closer together and higher)
        enhanced_paper = enhanced[10, 10]
        self.assertGreater(enhanced_paper[2], 200) # Blue component lifted from 140 to >200


if __name__ == "__main__":
    unittest.main()
