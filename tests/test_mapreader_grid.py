"""
Unit tests for MapReader Grid Dataset Builder.
"""

import os
import shutil
import unittest
import numpy as np
import cv2

from tranchot_extractor.models.mapreader_grid import MapReaderGridDatasetBuilder


class TestMapReaderGrid(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_temp_dataset")
        self.builder = MapReaderGridDatasetBuilder(output_dir=self.test_dir, tile_size=256)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_directory_init_and_yaml(self):
        yaml_path = self.builder.generate_dataset_yaml()
        self.assertTrue(os.path.exists(yaml_path))
        self.assertTrue(os.path.exists(self.builder.images_train_dir))
        self.assertTrue(os.path.exists(self.builder.labels_train_dir))


if __name__ == "__main__":
    unittest.main()
