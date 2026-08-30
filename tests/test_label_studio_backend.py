"""
Unit tests for the Label Studio ML Backend server.
"""

import os
import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from tranchot_extractor.models.label_studio_backend import app, MODEL_CONFIG


class TestLabelStudioBackend(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")
        self.assertIn("engine", data)

    def test_setup_endpoint(self):
        xml_schema = """<View>
          <Image name="image" value="$image"/>
          <PolygonLabels name="label" toName="image">
            <Label value="Gebäude" background="#ef4444"/>
            <Label value="Hofanlage" background="#dc2626"/>
          </PolygonLabels>
        </View>"""
        response = self.client.post("/setup", json={"schema": xml_schema, "extra_params": {"engine": "building"}})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        self.assertIn("model_version", data)

    def test_predict_with_sample_image(self):
        # Create a synthetic 100x100 RGB image with a red rectangle (simulating a building)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Add red box
        img[30:60, 30:70] = [220, 20, 20]

        temp_img_path = os.path.join(os.path.dirname(__file__), "test_temp_crop.png")
        cv2.imwrite(temp_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        try:
            tasks = [
                {
                    "id": 1,
                    "data": {
                        "image": temp_img_path
                    }
                }
            ]
            response = self.client.post("/predict", json={"tasks": tasks})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("results", data)
            self.assertEqual(len(data["results"]), 1)
            
            res = data["results"][0]
            self.assertIn("result", res)
            self.assertIn("model_version", res)
            
            # Verify prediction structure
            if len(res["result"]) > 0:
                pred = res["result"][0]
                self.assertEqual(pred["type"], "polygonlabels")
                self.assertIn("value", pred)
                self.assertIn("points", pred["value"])
                self.assertIn("polygonlabels", pred["value"])
                # Points must be normalized percentages (0..100)
                for pt in pred["value"]["points"]:
                    self.assertGreaterEqual(pt[0], 0.0)
                    self.assertLessEqual(pt[0], 100.0)
                    self.assertGreaterEqual(pt[1], 0.0)
                    self.assertLessEqual(pt[1], 100.0)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)


if __name__ == "__main__":
    unittest.main()
