import unittest
import numpy as np
from PIL import Image
import tkinter as tk
import customtkinter as ctk

from tranchot_extractor.ui.desktop_app import TranchotDesktopApp, MapCanvas


class TestDesktopApp(unittest.TestCase):

    def setUp(self):
        self.app = TranchotDesktopApp()
        self.app.withdraw()

    def tearDown(self):
        try:
            self.app.destroy()
        except Exception:
            pass

    def test_app_initialization(self):
        self.assertIsNotNone(self.app.canvas)
        self.assertIn(self.app.active_tool, ["pipette", "extract"])

    def test_canvas_loading_and_extraction(self):
        synthetic = np.full((200, 200, 3), (230, 220, 190), dtype=np.uint8)
        # Draw red building
        synthetic[40:80, 40:80] = [200, 40, 40]
        pil_img = Image.fromarray(synthetic)

        self.app.current_pil = pil_img
        self.app.current_np = synthetic
        self.app.canvas.load_image(pil_img, synthetic)

        self.assertEqual(self.app.canvas.pil_image.size, (200, 200))

        # Test auto-buildings
        self.app._run_auto_buildings()
        self.assertGreater(len(self.app.extracted_buildings), 0)

        # Test single building click
        self.app.extract_single_building_at(60, 60)
        self.assertGreater(len(self.app.extracted_buildings), 0)

        # Clear all features
        self.app._clear_all_features()
        self.assertEqual(len(self.app.extracted_buildings), 0)
        self.assertEqual(len(self.app.extracted_roads), 0)


if __name__ == "__main__":
    unittest.main()
