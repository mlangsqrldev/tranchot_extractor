"""
Unit tests for Mathematical Ink Deconvolution and Marker-Controlled Watershed.
"""

import unittest
import numpy as np
import cv2

from tranchot_extractor.core.ink_deconvolution import CartographicInkDeconvolution


class TestInkDeconvolution(unittest.TestCase):

    def setUp(self):
        self.deconv = CartographicInkDeconvolution()

    def test_ink_separation_carmine_vs_black(self):
        # Create synthetic image with:
        # - Carmine red square (RGB: 190, 45, 45)
        # - Black text stroke (RGB: 40, 35, 30)
        # - Blue water area (RGB: 70, 130, 180)
        # - Beige parchment background (RGB: 235, 220, 190)
        img = np.full((120, 120, 3), [235, 220, 190], dtype=np.uint8)

        # Red building
        img[20:50, 20:50] = [190, 45, 45]
        # Black line
        img[70:80, 20:100] = [40, 35, 30]
        # Blue water
        img[90:110, 20:100] = [70, 130, 180]

        inks = self.deconv.separate_inks(img)

        # Carmine density inside red square should be significantly higher than inside black or blue areas
        carmine_in_building = np.mean(inks.carmine[25:45, 25:45])
        carmine_in_black = np.mean(inks.carmine[72:78, 30:70])
        carmine_in_parchment = np.mean(inks.carmine[5:15, 5:15])

        self.assertGreater(carmine_in_building, carmine_in_black)
        self.assertGreater(carmine_in_building, carmine_in_parchment)

    def test_watershed_building_segmentation(self):
        # Create image with two adjacent red houses touching at a seam
        img = np.full((100, 100, 3), [240, 225, 195], dtype=np.uint8)

        # House 1
        img[30:50, 20:40] = [195, 40, 40]
        # House 2 (touching House 1 at x=40)
        img[30:50, 40:60] = [195, 40, 40]

        res = self.deconv.segment_buildings_watershed(
            img,
            od_threshold=0.15,
            min_building_area=15.0,
            watershed_footprint=3,
        )

        self.assertGreater(len(res.polygons), 0)
        self.assertEqual(res.building_mask.shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
