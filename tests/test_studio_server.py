import unittest
from fastapi.testclient import TestClient
from tranchot_extractor.studio.server import app


class TestStudioServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_studio_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Tranchot", r.text)

    def test_studio_presets(self):
        r = self.client.get("/api/presets")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("presets", data)
        self.assertGreater(len(data["presets"]), 0)

    def test_studio_export_formats(self):
        sample_ann = [{
            "id": 101,
            "label": "Gebäude",
            "color": "#ef4444",
            "type": "polygon",
            "points": [[50, 50], [120, 50], [120, 120], [50, 120]],
            "properties": {"test": True}
        }]

        # Test GeoJSON export
        r_geojson = self.client.post("/api/export", json={"annotations": sample_ann, "format": "geojson"})
        self.assertEqual(r_geojson.status_code, 200)

        # Test Label Studio JSON export
        r_ls = self.client.post("/api/export", json={"annotations": sample_ann, "format": "label_studio"})
        self.assertEqual(r_ls.status_code, 200)


if __name__ == "__main__":
    unittest.main()
