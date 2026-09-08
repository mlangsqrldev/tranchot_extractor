import unittest
import os
import sys
from shapely.geometry import Polygon

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint

# Ensure QApplication exists in offscreen mode
os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication(sys.argv)

from tranchot_extractor.ui.qt_studio import TranchotStudioWindow, TranchotMapView, SettlementDialog


class TestQtStudioInteractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.win = TranchotStudioWindow()

    def test_tool_persistence(self):
        """Verify that switching tools stays active and does not reset."""
        self.win._set_active_tool("add_building")
        self.assertEqual(self.win.view.current_tool, "add_building")
        self.assertTrue(self.win.btn_tool_add_building.isChecked())

        self.win._set_active_tool("select_building")
        self.assertEqual(self.win.view.current_tool, "select_building")
        self.assertTrue(self.win.btn_tool_select.isChecked())

    def test_manual_building_creation_and_delete(self):
        """Test manually drawing a building, selecting it, and deleting it."""
        initial_count = len(self.win.extracted_polygons_by_class.get("building", []))
        
        # 1. Manually add building (x1=100, y1=100, x2=150, y2=130)
        self.win._on_manual_building_created(100.0, 100.0, 150.0, 130.0)
        bldgs = self.win.extracted_polygons_by_class.get("building", [])
        self.assertEqual(len(bldgs), initial_count + 1)
        
        new_bldg = bldgs[-1]
        self.assertAlmostEqual(new_bldg.area, 1500.0)
        new_idx = len(bldgs) - 1
        
        # Verify it was highlighted
        self.assertEqual(self.win.selected_building_idx, new_idx)
        self.assertIsNotNone(self.win.selected_highlight_item)

        # 2. Add metadata
        self.win.building_metadata[new_idx] = {
            "settlement": "Testort",
            "geonames_id": "123456"
        }
        self.win._add_or_update_building_badge(new_idx, "Testort", "123456", 125.0, 115.0)
        self.assertIn(new_idx, self.win.building_badges)

        # 3. Delete building
        self.win._on_building_delete_requested()
        bldgs_after = self.win.extracted_polygons_by_class.get("building", [])
        self.assertEqual(len(bldgs_after), initial_count)
        self.assertIsNone(self.win.selected_building_idx)
        self.assertNotIn(new_idx, self.win.building_badges)

    def test_box_selection_and_bulk_delete(self):
        """Test selecting multiple buildings via box selection and deleting them in bulk."""
        self.win._clear_building_selection()
        initial_count = len(self.win.extracted_polygons_by_class.get("building", []))

        # Add 3 buildings
        self.win._on_manual_building_created(200.0, 200.0, 240.0, 230.0)
        self.win._on_manual_building_created(250.0, 200.0, 290.0, 230.0)
        self.win._on_manual_building_created(400.0, 400.0, 440.0, 430.0)  # far away

        bldgs = self.win.extracted_polygons_by_class.get("building", [])
        self.assertEqual(len(bldgs), initial_count + 3)

        # 1. Box select over first two buildings (x=190..300, y=190..240)
        self.win._on_canvas_box_selected_buildings(190.0, 190.0, 300.0, 240.0, is_shift=False)
        self.assertEqual(len(self.win.selected_building_indices), 2)
        idx1 = initial_count
        idx2 = initial_count + 1
        self.assertIn(idx1, self.win.selected_building_indices)
        self.assertIn(idx2, self.win.selected_building_indices)

        # 2. Shift-click to add the third building
        self.win._on_canvas_building_clicked(420.0, 415.0, is_shift=True)
        self.assertEqual(len(self.win.selected_building_indices), 3)

        # 3. Shift-click again to deselect the third building
        self.win._on_canvas_building_clicked(420.0, 415.0, is_shift=True)
        self.assertEqual(len(self.win.selected_building_indices), 2)

        # 4. Bulk delete the 2 selected buildings
        self.win._on_building_delete_requested()
        bldgs_rem = self.win.extracted_polygons_by_class.get("building", [])
        self.assertEqual(len(bldgs_rem), initial_count + 1)
        self.assertEqual(len(self.win.selected_building_indices), 0)

        # Clean up remaining building
        self.win._on_canvas_building_clicked(420.0, 415.0, is_shift=False)
        self.win._on_building_delete_requested()
        self.assertEqual(len(self.win.extracted_polygons_by_class.get("building", [])), initial_count)

    def test_sam_prompt_segmentation(self):
        """Test SAM prompt handling when simulated or run on synthetic data."""
        import numpy as np
        # Create a synthetic canvas with a red building block
        synthetic = np.full((300, 300, 3), 220, dtype=np.uint8)  # parchment
        synthetic[100:140, 100:150] = [180, 80, 75]  # carmine building
        self.win.enhanced_image_rgb = synthetic

        cnt_before = len(self.win.extracted_polygons_by_class.get("building", []))
        # Trigger SAM prompt on the building center (125, 120)
        self.win._on_sam_prompt(125.0, 120.0, 125.0, 120.0, is_point=True)

        cnt_after = len(self.win.extracted_polygons_by_class.get("building", []))
        self.assertGreaterEqual(cnt_after, cnt_before)
        if cnt_after > cnt_before:
            new_poly = self.win.extracted_polygons_by_class["building"][-1]
            self.assertTrue(new_poly.is_valid)
            self.assertGreater(new_poly.area, 50.0)
            self.assertEqual(len(self.win.selected_building_indices), 1)

    def test_spectral_building_click(self):
        """Test clicking a building to extract it spectrally without SAM."""
        import numpy as np
        synthetic = np.full((200, 200, 3), 220, dtype=np.uint8)
        synthetic[50:80, 50:90] = [210, 40, 40]
        self.win.enhanced_image_rgb = synthetic

        cnt_before = len(self.win.extracted_polygons_by_class.get("building", []))
        self.win._on_manual_building_clicked(70.0, 65.0)

        cnt_after = len(self.win.extracted_polygons_by_class.get("building", []))
        self.assertEqual(cnt_after, cnt_before + 1)
        new_poly = self.win.extracted_polygons_by_class["building"][-1]
        self.assertTrue(new_poly.is_valid)
        self.assertGreater(new_poly.area, 50.0)


if __name__ == "__main__":
    unittest.main()
