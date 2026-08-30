import unittest
import os
import shutil
import tempfile
import numpy as np
from tranchot_extractor.config import PipelineConfig
from tranchot_extractor.core.pipeline import TranchotPipeline


class TestPipelineAndExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tranchot_test_")
        self.sample_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_data",
            "rommerskirchen_sample.png"
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_pipeline_on_sample(self):
        if not os.path.exists(self.sample_path):
            self.skipTest("Sample image not found")

        pipeline = TranchotPipeline()
        res = pipeline.process(
            self.sample_path,
            output_dir=self.temp_dir,
            extract_buildings=True,
            extract_roads=True,
            extract_text=True,
        )

        self.assertGreater(len(res.buildings.features), 3)
        self.assertGreater(len(res.roads.features), 1)
        self.assertGreaterEqual(len(res.text.features), 0)

        # Check export files exist
        self.assertTrue(os.path.exists(res.exported_files["buildings_geojson"]))
        self.assertTrue(os.path.exists(res.exported_files["roads_geojson"]))
        if "toponyms_geojson" in res.exported_files:
            self.assertTrue(os.path.exists(res.exported_files["toponyms_geojson"]))
        self.assertTrue(os.path.exists(res.exported_files["geopackage"]))
        self.assertTrue(os.path.exists(res.exported_files["zip_package"]))


if __name__ == "__main__":
    unittest.main()
