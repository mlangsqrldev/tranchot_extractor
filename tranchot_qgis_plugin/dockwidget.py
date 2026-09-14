"""
Dockable widget panel for the Tranchot Extractor QGIS Plugin.
Provides full parameter control, interactive ROI selection, and progress monitoring.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
import os
import sys
import shutil
import tempfile
import time

import numpy as np
from osgeo import gdal
from shapely.geometry import Polygon

from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QVariant
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QFrame,
    QScrollArea,
    QComboBox,
)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.gui import QgsMapLayerComboBox, QgsRubberBand
from qgis.core import (
    QgsProject,
    QgsRaster,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsRectangle,
    QgsMapLayerProxyModel,
    QgsApplication,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsWkbTypes,
    QgsSpatialIndex,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsMultiBandColorRenderer,
    QgsPalLayerSettings,
    QgsVectorLayerSimpleLabeling,
    QgsTextFormat,
    QgsTextBufferSettings,
    QgsMarkerSymbol,
    QgsDistanceArea,
    QgsPointXY,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
)

from .map_tools import PolygonRoiMapTool, RoiExtentMapTool, PipetteMapTool, GazetteerPickMapTool, LandUseStampMapTool
from .tasks import BuildingExtractionTask, LandUseExtractionTask, TextExtractionTask


# -----------------------------------------------------------------------------
# Dynamic Backend Resolution
# Ensures the core 'tranchot_extractor' processing engine can be loaded:
# 1. Directly from the Python environment (if installed via `pip install -e .`)
# 2. Relative to the plugin repository directory (symlinked or standalone)
# -----------------------------------------------------------------------------
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_backend_dirs = [
    os.path.abspath(os.path.join(_plugin_dir, "..")),
    os.path.abspath(os.path.join(_plugin_dir, "..", "tranchot_extractor")),
    os.path.abspath(os.path.join(_plugin_dir, "..", "..")),
]
for _candidate in _candidate_backend_dirs:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        if os.path.exists(os.path.join(_candidate, "tranchot_extractor")) or os.path.exists(os.path.join(_candidate, "config.py")):
            sys.path.insert(0, _candidate)

try:
    from tranchot_extractor.config import BuildingConfig, RoadConfig, LandUseConfig, TextConfig
    from tranchot_extractor.extractors.building_extractor import BuildingExtractor
    from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
    from tranchot_extractor.extractors.text_extractor import TextExtractor
    from tranchot_extractor.extractors.pipette_sampler import PipetteSampler, ColorSample, StampEntry
except Exception:
    BuildingConfig = None
    RoadConfig = None
    LandUseConfig = None
    TextConfig = None
    BuildingExtractor = None
    ColorEnhancer = None
    TextExtractor = None
    PipetteSampler = None
    ColorSample = None
    StampEntry = None


class TranchotDockWidget(QDockWidget):
    """
    Main dockable control panel for Tranchot historical map extraction.
    """

    closingPlugin = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__("Tranchot Extractor", parent)
        self.iface = iface
        self.canvas = self.iface.mapCanvas()
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.current_roi: Optional[QgsRectangle] = None
        self.roi_tool: Optional[RoiExtentMapTool] = None
        self.pipette_tool: Optional[PipetteMapTool] = None
        self.cached_roi_rgb = None
        self.cached_roi_crs = None
        self.current_task: Optional[BuildingExtractionTask] = None
        self.gazetteer_layer = None
        self.current_gazetteer_match = None
        self.current_type_match = None
        self.current_type_uri = None
        self.roi_rubber_band: Optional[QgsRubberBand] = None

        # ROI cache for instant live preview calculations
        self.cached_roi_rgb = None
        self.cached_roi_geo = None
        self.cached_roi_crs = "EPSG:25832"
        self.cached_roi_crs_wkt = ""

        # Debounced timer for live updates during slider interaction (150ms)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(150)
        self.preview_timer.timeout.connect(self._update_live_preview)

        # Land Use states & multi-stamp sampler
        self.pipette_sampler = PipetteSampler() if PipetteSampler else None
        self.current_lu_roi: Optional[QgsRectangle] = None
        self.current_lu_roi_geom: Optional[QgsGeometry] = None
        self.cached_lu_roi_rgb = None
        self.cached_lu_sub_origin = (0.0, 0.0)
        self.cached_lu_px_size = (1.0, 1.0)
        self.lu_roi_rubber_band: Optional[QgsRubberBand] = None
        self.lu_stamp_rubber_bands: List[Tuple[str, QgsRubberBand]] = []

        self.lu_preview_timer = QTimer(self)
        self.lu_preview_timer.setSingleShot(True)
        self.lu_preview_timer.setInterval(200)
        self.lu_preview_timer.timeout.connect(self._update_lu_live_preview)

        self._init_ui()
        self._init_map_tools()

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header Title Banner
        header_box = QFrame()
        header_box.setStyleSheet("background-color: #2b2b2b; border-radius: 6px; padding: 6px;")
        header_layout = QVBoxLayout(header_box)
        header_layout.setContentsMargins(6, 4, 6, 4)

        title_lbl = QLabel("🗺️ HistMap Extractor")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("color: #f5eedc;")
        header_layout.addWidget(title_lbl)

        sub_lbl = QLabel("Historical Map Vectorization • BCDH Bonn")
        sub_lbl.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        header_layout.addWidget(sub_lbl)
        main_layout.addWidget(header_box)

        # Raster Layer Selection
        layer_group = QGroupBox("1. Active Historical Map Sheet")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_combo)

        # Global Map View Toggle: Aged (Original) vs Normalized (De-yellowed)
        view_toggle_layout = QHBoxLayout()
        self.btn_view_orig = QPushButton("📜 Original (Aged Paper)")
        self.btn_view_orig.setCheckable(True)
        self.btn_view_orig.setChecked(True)
        self.btn_view_orig.setStyleSheet(
            "QPushButton:checked { background-color: #2e7d32; color: white; font-weight: bold; border: 2px solid #81c784; border-radius: 4px; padding: 5px; } "
            "QPushButton { background-color: #333333; color: #cccccc; border-radius: 4px; padding: 5px; }"
        )
        self.btn_view_orig.setToolTip("Displays the original historical map scan (aged paper).")
        self.btn_view_orig.clicked.connect(lambda: self._set_global_view_mode("original"))
        view_toggle_layout.addWidget(self.btn_view_orig)

        self.btn_view_norm = QPushButton("✨ Normalized (De-yellowed)")
        self.btn_view_norm.setCheckable(True)
        self.btn_view_norm.setChecked(False)
        self.btn_view_norm.setStyleSheet(
            "QPushButton:checked { background-color: #1565c0; color: white; font-weight: bold; border: 2px solid #64b5f6; border-radius: 4px; padding: 5px; } "
            "QPushButton { background-color: #333333; color: #cccccc; border-radius: 4px; padding: 5px; }"
        )
        self.btn_view_norm.setToolTip(
            "Switches map display to the restored, de-yellowed sheet (von Kries chromatic adaptation & contrast enhancement)."
        )
        self.btn_view_norm.clicked.connect(lambda: self._set_global_view_mode("normalized"))
        view_toggle_layout.addWidget(self.btn_view_norm)
        layer_layout.addLayout(view_toggle_layout)

        main_layout.addWidget(layer_group)

        # Scrollable area for tabs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs Widget
        self.tabs = QTabWidget()
        self._create_building_tab()
        self._create_landuse_tab()
        self._create_text_tab()
        self._create_roads_tab()
        self._create_system_tab()
        scroll_layout.addWidget(self.tabs)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # Status & Progress Footer
        footer_box = QFrame()
        footer_box.setStyleSheet("background-color: #232323; border-radius: 4px; padding: 4px;")
        footer_layout = QVBoxLayout(footer_box)
        footer_layout.setContentsMargins(4, 4, 4, 4)

        self.status_lbl = QLabel("Ready. Select a historical map sheet.")
        self.status_lbl.setStyleSheet("color: #d0d0d0; font-size: 10px;")
        footer_layout.addWidget(self.status_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        footer_layout.addWidget(self.progress_bar)
        main_layout.addWidget(footer_box)

        self.setWidget(main_widget)

    def _create_building_tab(self):
        """Creates Tab 1: Building & Courtyard extraction."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ROI Selection Section
        roi_group = QGroupBox("A. Extraction Area (Settlement Polygon)")
        roi_layout = QVBoxLayout(roi_group)

        btn_box = QHBoxLayout()
        self.btn_roi = QPushButton("📐 Draw Settlement Polygon")
        self.btn_roi.setToolTip("Click point-by-point on the map canvas to draw an irregular polygon around the settlement (Right-click / double-click to finish).")
        self.btn_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.btn_roi.clicked.connect(self._activate_roi_tool)
        btn_box.addWidget(self.btn_roi)

        self.btn_clear_roi = QPushButton("❌ Reset")
        self.btn_clear_roi.setToolTip("Reset settlement polygon")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        btn_box.addWidget(self.btn_clear_roi)
        roi_layout.addLayout(btn_box)

        self.roi_info_lbl = QLabel("Area: No settlement polygon drawn yet (use button above)")
        self.roi_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        roi_layout.addWidget(self.roi_info_lbl)

        # Checkbox to toggle Gazetteer places on map
        self.chk_show_gazetteer = QCheckBox("📍 Orte aus Gazetteer auf Karte anzeigen")
        self.chk_show_gazetteer.setChecked(False)
        self.chk_show_gazetteer.setToolTip("Zeigt alle historischen Orte, Dörfer und Landmarken aus dem GOV-Gazetteer mit Beschriftung auf der Karte an.")
        self.chk_show_gazetteer.toggled.connect(self._toggle_gazetteer_map_layer)
        roi_layout.addWidget(self.chk_show_gazetteer)

        # Gazetteer Match Section
        gaz_box = QHBoxLayout()
        gaz_lbl = QLabel("Toponym / Ortsname:")
        self.gazetteer_combo = QComboBox()
        self.gazetteer_combo.setEditable(True)
        self.gazetteer_combo.addItem("None", {"name": None, "type": None})
        self.gazetteer_combo.setToolTip("Wähle den Ortsnamen aus den Gazetteer-Treffern oder tippe einen Namen frei ein.")
        self.gazetteer_combo.currentTextChanged.connect(self._on_gazetteer_match_changed)

        self.btn_pick_gazetteer = QPushButton("🎯 Auf Karte wählen")
        self.btn_pick_gazetteer.setToolTip("Klicke direkt auf einen Gazetteer-Punkt oder ein Dorf auf der Karte, um den Ort auszuwählen.")
        self.btn_pick_gazetteer.setStyleSheet(
            "background-color: #e65100; color: white; font-weight: bold; padding: 5px 8px; border-radius: 4px;"
        )
        self.btn_pick_gazetteer.clicked.connect(self._activate_gazetteer_picker)

        gaz_box.addWidget(gaz_lbl)
        gaz_box.addWidget(self.gazetteer_combo, 1)
        gaz_box.addWidget(self.btn_pick_gazetteer)
        roi_layout.addLayout(gaz_box)
        
        type_box = QHBoxLayout()
        type_lbl = QLabel("Siedlungstyp (LVR):")
        self.type_combo = QComboBox()
        self.type_combo.addItem("Default (Building/Courtyard)", None)
        csv_path = os.path.join(os.path.dirname(__file__), "resources", "siedlungstypen_lvr_vokabular.csv")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                header = next(f)
                for line in f:
                    parts = line.strip().split(";")
                    if len(parts) >= 2:
                        s_type = parts[0]
                        s_uri = parts[1]
                        self.type_combo.addItem(s_type, s_uri)
        except Exception:
            pass
        self.type_combo.currentTextChanged.connect(self._on_type_match_changed)
        type_box.addWidget(type_lbl)
        type_box.addWidget(self.type_combo)
        roi_layout.addLayout(type_box)
        
        self.gazetteer_info_lbl = QLabel("Awaiting polygon...")
        self.gazetteer_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        roi_layout.addWidget(self.gazetteer_info_lbl)

        # Action button to permanently save village and buildings
        self.btn_save_village = QPushButton("💾 Dorf & Gebäude speichern")
        self.btn_save_village.setToolTip("Speichert das aktuelle Dorf (Ortslagen-Polygon) und alle erkannten Gebäude dauerhaft in die Ebenen, damit du das nächste Dorf erfassen kannst.")
        self.btn_save_village.setStyleSheet(
            "background-color: #1b5e20; color: white; font-weight: bold; padding: 7px; border-radius: 4px;"
        )
        self.btn_save_village.clicked.connect(lambda: self._save_current_village(silent=False))
        roi_layout.addWidget(self.btn_save_village)

        # Checkbox for interactive live preview
        self.chk_live_preview = QCheckBox("⚡ Live Preview active (instant response on slider adjustments)")
        self.chk_live_preview.setChecked(True)
        self.chk_live_preview.setToolTip(
            "Immediately updates detected building polygons on the map whenever sliders are moved."
        )
        self.chk_live_preview.toggled.connect(self._on_live_preview_toggled)
        roi_layout.addWidget(self.chk_live_preview)

        layout.addWidget(roi_group)

        # Parameter Section with Sliders and Rich Tooltips
        param_group = QGroupBox("B. Color Space & Filter Parameters")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(10)

        # Helper to create synchronized Label + Slider + SpinBox with rich tooltips
        def add_slider_spin_row(layout, label_text, tooltip_html, min_val, max_val, step, default_val, scale=1.0, decimals=0):
            row_box = QVBoxLayout()
            header_layout = QHBoxLayout()
            
            lbl = QLabel(f"<b>{label_text}</b>")
            lbl.setToolTip(tooltip_html)
            header_layout.addWidget(lbl)
            header_layout.addStretch(1)

            if decimals > 0:
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
            else:
                spin = QSpinBox()

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default_val)
            spin.setToolTip(tooltip_html)
            spin.setFixedWidth(75)
            header_layout.addWidget(spin)
            row_box.addLayout(header_layout)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider_min = int(round(min_val * scale))
            slider_max = int(round(max_val * scale))
            slider.setRange(slider_min, slider_max)
            slider.setValue(int(round(default_val * scale)))
            slider.setToolTip(tooltip_html)
            slider.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; } "
                                 "QSlider::handle:horizontal { width: 14px; margin-top: -5px; margin-bottom: -5px; "
                                 "background: #c83232; border-radius: 7px; }")

            # Two-way sync with proper type casting to avoid TypeError on QSpinBox
            def on_slider_changed(val):
                spin.blockSignals(True)
                if decimals > 0:
                    spin.setValue(val / scale)
                else:
                    spin.setValue(int(round(val / scale)))
                spin.blockSignals(False)

            def on_spin_changed(val):
                slider.blockSignals(True)
                slider.setValue(int(round(val * scale)))
                slider.blockSignals(False)

            slider.valueChanged.connect(on_slider_changed)
            spin.valueChanged.connect(on_spin_changed)

            # Connect live preview trigger
            slider.valueChanged.connect(self._trigger_preview_update)
            spin.valueChanged.connect(self._trigger_preview_update)

            row_box.addWidget(slider)
            layout.addLayout(row_box)
            return spin, slider

        # 1. Lower and Upper Carmine Red Threshold Sliders (Bandpass)
        tt_min_red = (
            "<h3>🔴 Carmine Red: Lower Threshold (Min Intensity)</h3>"
            "<p>Determines the minimum red prominence required to classify a pixel as a building:</p>"
            "<ul>"
            "<li><b>Lower (10–25):</b> Highest sensitivity. Captures faint watercolor washes, weathered wings, and fine outlines.</li>"
            "<li><b>Higher (40–80):</b> Stricter filter. Responds only to solid, opaque carmine red ink.</li>"
            "<li><b>Default:</b> 18</li>"
            "</ul>"
        )
        self.spin_min_red, self.slider_min_red = add_slider_spin_row(
            param_layout, "🔴 Carmine Red: Lower Threshold (Min):", tt_min_red,
            min_val=5, max_val=150, step=1, default_val=18, scale=1.0, decimals=0
        )

        tt_max_red = (
            "<h3>🔴 Carmine Red: Upper Threshold (Max Intensity)</h3>"
            "<p>Sets upper intensity cutoff:</p>"
            "<ul>"
            "<li><b>Default (255):</b> Unlimited (includes all vibrant red inks).</li>"
            "<li><b>Lower (120–200):</b> Masks out over-saturated blotches.</li>"
            "</ul>"
        )
        self.spin_max_red, self.slider_max_red = add_slider_spin_row(
            param_layout, "🔴 Carmine Red: Upper Threshold (Max):", tt_max_red,
            min_val=30, max_val=255, step=2, default_val=255, scale=1.0, decimals=0
        )

        # Cross-link sliders so Min <= Max always
        self.slider_min_red.valueChanged.connect(
            lambda v: self.slider_max_red.setValue(max(v, self.slider_max_red.value()))
        )
        self.slider_max_red.valueChanged.connect(
            lambda v: self.slider_min_red.setValue(min(v, self.slider_min_red.value()))
        )

        # Compatibility aliases
        self.spin_sens = self.spin_min_red
        self.slider_sens = self.slider_min_red
        self.spin_od_min = self.spin_min_red
        self.slider_od_min = self.slider_min_red
        self.spin_od = self.spin_min_red
        self.slider_od = self.slider_min_red
        self.spin_od_max = self.spin_max_red

        # 2. Min Wall Thickness
        tt_thick = (
            "<h3>Minimum Wall Thickness (px)</h3>"
            "<p>Distinguishes fine outline strokes from solid mass structures:</p>"
            "<ul>"
            "<li><b>Lower (0.4 - 0.8 px):</b> Preserves thin outline drawings, garden walls, and narrow wings.</li>"
            "<li><b>Higher (1.5 - 3.5 px):</b> Keeps only solid building blocks.</li>"
            "<li><b>Default:</b> 0.6 px</li>"
            "</ul>"
        )
        self.spin_thickness, self.slider_thickness = add_slider_spin_row(
            param_layout, "Minimum Wall Thickness (px):", tt_thick,
            min_val=0.4, max_val=5.0, step=0.1, default_val=0.6, scale=10.0, decimals=1
        )

        # 3. Min Area
        tt_area = (
            "<h3>Minimum Building Area (px)</h3>"
            "<p>Filters out isolated speckle noise below this total pixel area:</p>"
            "<ul>"
            "<li><b>Lower (2 - 5 px):</b> Detects small outbuildings and sheds.</li>"
            "<li><b>Higher (15 - 30 px):</b> Suppresses scan artifacts and dots.</li>"
            "<li><b>Default:</b> 5 px</li>"
            "</ul>"
        )
        self.spin_min_area, self.slider_min_area = add_slider_spin_row(
            param_layout, "Minimum Area (px):", tt_area,
            min_val=2, max_val=200, step=1, default_val=5, scale=1.0, decimals=0
        )

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        param_layout.addWidget(sep)

        tt_courtyard = (
            "<h3>Courtyard Subtraction (Interior Yards)</h3>"
            "<p>Analyzes enclosed courtyard farmsteads (Dreiseithöfe, Vierkanthöfe, castle yards):</p>"
            "<ul>"
            "<li><b>Checked (Recommended):</b> Subtracts open interior courtyards as geometry holes (donut polygons), preserving authentic wing perimeters.</li>"
            "<li><b>Unchecked:</b> Fills entire estate footprint as a solid block.</li>"
            "</ul>"
        )
        self.chk_courtyard = QCheckBox("Courtyard Subtraction (preserve interior yards as open holes)")
        self.chk_courtyard.setChecked(True)
        self.chk_courtyard.setToolTip(tt_courtyard)
        self.chk_courtyard.toggled.connect(self._trigger_preview_update)
        param_layout.addWidget(self.chk_courtyard)

        tt_black = (
            "<h3>Black Monumental Buildings & Churches</h3>"
            "<p>Extracts massive churches and public monuments drawn in black ink rather than red watercolor.</p>"
        )
        self.chk_black_bld = QCheckBox("Black Monumental Buildings / Churches")
        self.chk_black_bld.setChecked(False)
        self.chk_black_bld.setToolTip(tt_black)
        self.chk_black_bld.toggled.connect(self._trigger_preview_update)
        param_layout.addWidget(self.chk_black_bld)

        # Compatibility placeholders
        self.chk_paper_norm = type("DummyCheck", (), {"isChecked": lambda self=self: True})()
        self.chk_curved_walls = type("DummyCheck", (), {"isChecked": lambda self=self: True})()
        self.combo_geom_mode = type("DummyCombo", (), {"currentIndex": lambda self=self: 0})()

        layout.addWidget(param_group)

        # Action Button with tooltip
        tt_run = (
            "<h3>Extract Buildings</h3>"
            "<p>Launches vectorization for the selected settlement polygon. The resulting vector layer is "
            "automatically added to QGIS styled in historical brick red.</p>"
        )
        self.btn_extract_buildings = QPushButton("🏛️ Extract Buildings")
        self.btn_extract_buildings.setToolTip(tt_run)
        self.btn_extract_buildings.setStyleSheet(
            "background-color: #b22222; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_extract_buildings.clicked.connect(self._run_building_extraction)
        layout.addWidget(self.btn_extract_buildings)

        layout.addStretch(1)
        self.tabs.addTab(tab, "🏛️ Buildings")

    def _create_landuse_tab(self):
        """Creates Tab 2: Land-Use Classification."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 1. Dedicated Land Use ROI
        roi_group = QGroupBox("A. Untersuchungsgebiet (Land Use Area)")
        roi_layout = QVBoxLayout(roi_group)

        btn_box = QHBoxLayout()
        self.btn_lu_roi = QPushButton("📐 Landnutzungs-Polygon einzeichnen")
        self.btn_lu_roi.setToolTip("Klicke punktweise auf die Karte, um ein beliebiges Polygon für die Landnutzung zu zeichnen (Rechtsklick/Doppelklick zum Abschließen).")
        self.btn_lu_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.btn_lu_roi.clicked.connect(self._activate_lu_roi_tool)
        btn_box.addWidget(self.btn_lu_roi)

        self.btn_clear_lu_roi = QPushButton("❌ Reset")
        self.btn_clear_lu_roi.setToolTip("Untersuchungsgebiet zurücksetzen")
        self.btn_clear_lu_roi.clicked.connect(self._clear_lu_roi)
        btn_box.addWidget(self.btn_clear_lu_roi)
        roi_layout.addLayout(btn_box)

        # Scope Radio Buttons
        scope_box = QHBoxLayout()
        self.rb_lu_roi = QRadioButton("Gezeichnetes Gebiet (ROI)", checked=True)
        self.rb_lu_sheet = QRadioButton("Gesamtblatt")
        self.rb_lu_roi.toggled.connect(self._trigger_lu_preview_update)
        self.rb_lu_sheet.toggled.connect(self._trigger_lu_preview_update)
        scope_box.addWidget(self.rb_lu_roi)
        scope_box.addWidget(self.rb_lu_sheet)
        roi_layout.addLayout(scope_box)

        self.lu_roi_info_lbl = QLabel("Untersuchungsgebiet: Noch kein Polygon gezeichnet (oben einzeichnen oder Gesamtblatt wählen)")
        self.lu_roi_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        roi_layout.addWidget(self.lu_roi_info_lbl)
        layout.addWidget(roi_group)

        # 2. Multi-Nuance Pipette Sampler
        pip_group = QGroupBox("B. Farbmuster-Sampling (Mehrere Nuancen)")
        pip_layout = QVBoxLayout(pip_group)

        row_class = QHBoxLayout()
        row_class.addWidget(QLabel("Ziel-Klasse:"))
        self.combo_lu_sample_class = QComboBox()
        self.combo_lu_sample_class.addItem("🌲 Wald (Forest)", "forest")
        self.combo_lu_sample_class.addItem("🌱 Wiese & Weiden (Meadow)", "meadow")
        self.combo_lu_sample_class.addItem("💧 Gewässer (Water)", "water")
        self.combo_lu_sample_class.addItem("🏡 Gärten & Baumgärten (Garden)", "garden")
        self.combo_lu_sample_class.addItem("🍇 Weinberge & Hänge (Vineyard)", "vineyard")
        self.combo_lu_sample_class.addItem("🏖️ Kies- & Sandbänke (Gravel)", "gravel")
        row_class.addWidget(self.combo_lu_sample_class, 1)
        pip_layout.addLayout(row_class)

        # Radius slider for sampling
        row_rad = QHBoxLayout()
        row_rad.addWidget(QLabel("Stempel-Radius (px):"))
        self.spin_stamp_radius = QSpinBox()
        self.spin_stamp_radius.setRange(4, 80)
        self.spin_stamp_radius.setValue(20)
        self.slider_stamp_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_stamp_radius.setRange(4, 80)
        self.slider_stamp_radius.setValue(20)
        self.slider_stamp_radius.valueChanged.connect(self.spin_stamp_radius.setValue)
        self.spin_stamp_radius.valueChanged.connect(self.slider_stamp_radius.setValue)
        row_rad.addWidget(self.slider_stamp_radius, 1)
        row_rad.addWidget(self.spin_stamp_radius)
        pip_layout.addLayout(row_rad)

        self.btn_lu_sample_stamp = QPushButton("🎯 Farbmuster auf Karte aufnehmen")
        self.btn_lu_sample_stamp.setToolTip("Klicke auf die Karte, um Farbmuster für die gewählte Klasse zu lernen. Mehrere Klicks erfassen mehrere Nuancen (Rechtsklick/Esc zum Beenden).")
        self.btn_lu_sample_stamp.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.btn_lu_sample_stamp.clicked.connect(self._activate_lu_stamp_tool)
        pip_layout.addWidget(self.btn_lu_sample_stamp)

        self.lbl_lu_stamps_summary = QLabel("0 Nuancen gesampelt (Standard-Farbprofile aktiv)")
        self.lbl_lu_stamps_summary.setStyleSheet("color: #888888; font-size: 10px;")
        pip_layout.addWidget(self.lbl_lu_stamps_summary)

        btn_clear_box = QHBoxLayout()
        self.btn_clear_class_stamps = QPushButton("🗑️ Nuancen d. Klasse leeren")
        self.btn_clear_class_stamps.setToolTip("Löscht die gelernten Stempel/Nuancen für die aktuell gewählte Klasse.")
        self.btn_clear_class_stamps.clicked.connect(self._clear_class_stamps)
        btn_clear_box.addWidget(self.btn_clear_class_stamps)

        self.btn_reset_all_stamps = QPushButton("🔄 Alle zurücksetzen")
        self.btn_reset_all_stamps.setToolTip("Setzt alle Farbprofile aller Klassen auf die historischen Standardwerte zurück.")
        self.btn_reset_all_stamps.clicked.connect(self._reset_all_stamps)
        btn_clear_box.addWidget(self.btn_reset_all_stamps)
        pip_layout.addLayout(btn_clear_box)
        layout.addWidget(pip_group)

        # 3. Live Sliders & Parameters
        param_group = QGroupBox("C. Farbtoleranz & Klassifikations-Regler")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)

        def add_lu_slider(layout, label, tooltip, min_v, max_v, step, def_v, scale=1.0, decimals=0):
            row = QVBoxLayout()
            h_layout = QHBoxLayout()
            lbl = QLabel(f"<b>{label}</b>")
            lbl.setToolTip(tooltip)
            h_layout.addWidget(lbl)
            h_layout.addStretch(1)

            if decimals > 0:
                sp = QDoubleSpinBox()
                sp.setDecimals(decimals)
            else:
                sp = QSpinBox()
            sp.setRange(min_v, max_v)
            sp.setSingleStep(step)
            sp.setValue(def_v)
            sp.setFixedWidth(75)
            h_layout.addWidget(sp)
            row.addLayout(h_layout)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(int(round(min_v * scale)), int(round(max_v * scale)))
            sl.setValue(int(round(def_v * scale)))
            sl.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; } "
                             "QSlider::handle:horizontal { width: 14px; margin-top: -5px; margin-bottom: -5px; "
                             "background: #2e7d32; border-radius: 7px; }")

            def on_sl_change(val):
                sp.blockSignals(True)
                sp.setValue(val / scale if decimals > 0 else int(round(val / scale)))
                sp.blockSignals(False)

            def on_sp_change(val):
                sl.blockSignals(True)
                sl.setValue(int(round(val * scale)))
                sl.blockSignals(False)

            sl.valueChanged.connect(on_sl_change)
            sp.valueChanged.connect(on_sp_change)
            sl.valueChanged.connect(self._trigger_lu_preview_update)
            sp.valueChanged.connect(self._trigger_lu_preview_update)
            row.addWidget(sl)
            layout.addLayout(row)
            return sp, sl

        tt_tol = "<h3>Farbtoleranz / Sensitivität (CIE-Lab ΔE)</h3><p>Steuert den Akzeptanzradius im Farbraum. Höhere Werte (30–60) fassen breitere Farbvariationen zusammen, niedrigere Werte (10–25) sind strenger.</p>"
        self.spin_lu_tol, self.slider_lu_tol = add_lu_slider(
            param_layout, "🎨 Farbtoleranz (CIE-Lab ΔE):", tt_tol,
            min_v=10, max_v=80, step=1, def_v=24
        )

        tt_area = "<h3>Mindestfläche (px)</h3><p>Filtert isolierte Einzelflecken und Rauschen unterhalb dieser Flächengröße heraus.</p>"
        self.spin_lu_min_area, self.slider_lu_min_area = add_lu_slider(
            param_layout, "📐 Mindestfläche (px):", tt_area,
            min_v=20, max_v=1500, step=10, def_v=150
        )

        tt_tex = "<h3>Textur-Gewichtung (Baumkronen / Schraffur)</h3><p>Gewichtet die Texturvarianz. Höher (1.0–2.0) betont eingestochene Baumkronen im Wald, niedriger (0.0–0.5) reagiert primär auf reine Farblasuren.</p>"
        self.spin_lu_tex_w, self.slider_lu_tex_w = add_lu_slider(
            param_layout, "🌿 Textur-Filter:", tt_tex,
            min_v=0.0, max_v=2.5, step=0.1, def_v=0.8, scale=10.0, decimals=1
        )

        self.chk_lu_live_preview = QCheckBox("⚡ Live Preview (sofortige Reaktion auf Schieberegler)")
        self.chk_lu_live_preview.setChecked(True)
        self.chk_lu_live_preview.toggled.connect(self._on_lu_live_preview_toggled)
        param_layout.addWidget(self.chk_lu_live_preview)
        layout.addWidget(param_group)

        # 4. Categories & Final Vectorization
        cat_group = QGroupBox("D. Kategorien & Vektorisierung")
        cat_layout = QVBoxLayout(cat_group)
        self.chk_lu_forest = QCheckBox("🌲 Wald (Forest - Olivgrün & Kronentextur)")
        self.chk_lu_forest.setChecked(True)
        self.chk_lu_forest.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_forest)

        self.chk_lu_meadow = QCheckBox("🌱 Wiese & Weiden (Meadow - Cyan-/Grünlasur)")
        self.chk_lu_meadow.setChecked(True)
        self.chk_lu_meadow.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_meadow)

        self.chk_lu_water = QCheckBox("💧 Gewässer (Water - Bäche, Flüsse, Teiche)")
        self.chk_lu_water.setChecked(True)
        self.chk_lu_water.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_water)

        self.chk_lu_garden = QCheckBox("🏡 Gärten & Nutzkulturen (Garden)")
        self.chk_lu_garden.setChecked(True)
        self.chk_lu_garden.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_garden)

        self.chk_lu_vineyard = QCheckBox("🍇 Weinberge & Hänge (Vineyard)")
        self.chk_lu_vineyard.setChecked(True)
        self.chk_lu_vineyard.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_vineyard)

        self.chk_lu_gravel = QCheckBox("🏖️ Kies- & Sandbänke (Gravel)")
        self.chk_lu_gravel.setChecked(False)
        self.chk_lu_gravel.toggled.connect(self._trigger_lu_preview_update)
        cat_layout.addWidget(self.chk_lu_gravel)
        layout.addWidget(cat_group)

        self.btn_extract_landuse = QPushButton("🌲 Landnutzung berechnen & speichern")
        self.btn_extract_landuse.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_extract_landuse.clicked.connect(self._run_landuse_extraction)
        layout.addWidget(self.btn_extract_landuse)

        layout.addStretch(1)
        self.tabs.addTab(tab, "🌲 Land Use")

    def _create_text_tab(self):
        """Creates Tab for Text and Toponyms."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        info_lbl = QLabel("Text Extraction relies on Tesseract OCR and the Historical Gazetteer.")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        btn_extract = QPushButton("🔍 Extract Text & Toponyms (ROI)")
        btn_extract.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        btn_extract.clicked.connect(self._run_text_extraction)
        layout.addWidget(btn_extract)

        layout.addStretch()
        self.tabs.addTab(tab, "📝 Text")



    def _create_roads_tab(self):
        """Creates Tab 3: Road network."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        road_group = QGroupBox("Road Network Parameters")
        road_layout = QVBoxLayout(road_group)

        r_box = QHBoxLayout()
        r_box.addWidget(QLabel("Ridge Threshold:"))
        spin_ridge = QDoubleSpinBox()
        spin_ridge.setRange(10.0, 80.0)
        spin_ridge.setValue(32.0)
        r_box.addWidget(spin_ridge)
        road_layout.addLayout(r_box)

        l_box = QHBoxLayout()
        l_box.addWidget(QLabel("Min. Road Length (px):"))
        spin_len = QSpinBox()
        spin_len.setRange(20, 200)
        spin_len.setValue(60)
        l_box.addWidget(spin_len)
        road_layout.addLayout(l_box)
        layout.addWidget(road_group)

        btn_roads = QPushButton("🛣️ Extract Road Centerlines")
        btn_roads.setStyleSheet("background-color: #8b5a2b; color: white; font-weight: bold; padding: 6px;")
        btn_roads.clicked.connect(lambda: self.status_lbl.setText("Road network extraction initializing..."))
        layout.addWidget(btn_roads)

        layout.addStretch(1)
        self.tabs.addTab(tab, "🛣️ Roads")

    def _create_system_tab(self):
        """Creates Tab 4: System info and diagnostic tools."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        diag_group = QGroupBox("Environment & Dependencies")
        diag_layout = QVBoxLayout(diag_group)

        # Check key modules
        mods = [
            ("NumPy", "numpy"),
            ("SciPy", "scipy"),
            ("Shapely", "shapely"),
            ("GeoPandas", "geopandas"),
            ("OpenCV", "cv2"),
            ("GDAL / OGR", "osgeo.gdal"),
            ("PyTorch / CUDA", "torch"),
        ]
        for name, mod in mods:
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "OK")
                lbl = QLabel(f"✓ {name}: {ver}")
                lbl.setStyleSheet("color: #4CAF50; font-size: 11px;")
            except Exception:
                lbl = QLabel(f"✗ {name}: Not installed")
                lbl.setStyleSheet("color: #F44336; font-size: 11px;")
            diag_layout.addWidget(lbl)

        layout.addWidget(diag_group)

        btn_info = QPushButton("ℹ️ About HistMap Extractor")
        btn_info.clicked.connect(self._show_about_dialog)
        layout.addWidget(btn_info)

        layout.addStretch(1)
        self.tabs.addTab(tab, "⚙️ System")

    def _init_map_tools(self):
        """Initializes custom QGIS MapTools for canvas interaction."""
        self.roi_tool = PolygonRoiMapTool(self.canvas, on_polygon_callback=self._on_polygon_roi_selected)
        self.pipette_tool = PipetteMapTool(self.canvas, on_sample_callback=self._on_pipette_sampled)
        self.gazetteer_pick_tool = GazetteerPickMapTool(self.canvas, on_pick_callback=self._on_gazetteer_point_picked)
        self.gazetteer_pick_tool.canceled.connect(self._cancel_gazetteer_picker)
        self.lu_roi_tool = PolygonRoiMapTool(self.canvas, on_polygon_callback=self._on_lu_polygon_roi_selected)
        self.lu_stamp_tool = LandUseStampMapTool(self.canvas, on_stamp_callback=self._on_lu_stamp_sampled)
        self.lu_stamp_tool.finished.connect(self._finish_lu_stamp_sampling)

    def _activate_roi_tool(self):
        """Switches QGIS map tool to irregular polygon drawing mode."""
        # Auto-commit previous village if it has features and an active polygon
        if hasattr(self, 'current_preview_features') and self.current_preview_features and getattr(self, 'current_roi_geom', None) is not None:
            self._save_current_village(silent=True)

        self.canvas.setMapTool(self.roi_tool)
        self.status_lbl.setText("Click points on map (Right-click / double-click to finish)...")
        self.btn_roi.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; padding: 6px; border-radius: 4px;")

    def _on_polygon_roi_selected(self, geom: QgsGeometry):
        """Callback when user finishes drawing an irregular settlement polygon."""
        if geom is None or geom.isEmpty():
            return
        self.current_roi_geom = geom
        self.current_roi = geom.boundingBox()
        self.roi_info_lbl.setText(
            f"Settlement Polygon: Area ≈ {geom.area():.0f} m²"
        )
        self.roi_info_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
        self.btn_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")

        # Display persistent ROI polygon on map canvas (100% transparent interior)
        if not hasattr(self, 'roi_rubber_band') or self.roi_rubber_band is None:
            self.roi_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            self.roi_rubber_band.setFillColor(QColor(0, 0, 0, 0))  # 100% transparent interior
            self.roi_rubber_band.setStrokeColor(QColor(0, 180, 216, 240))  # Crisp cyan dashed outline
            self.roi_rubber_band.setWidth(2)
            self.roi_rubber_band.setLineStyle(Qt.PenStyle.DashLine)
        else:
            self.roi_rubber_band.setFillColor(QColor(0, 0, 0, 0))
            self.roi_rubber_band.setStrokeColor(QColor(0, 180, 216, 240))
            self.roi_rubber_band.setWidth(2)
            self.roi_rubber_band.setLineStyle(Qt.PenStyle.DashLine)
        self.roi_rubber_band.setToGeometry(geom, None)
        self.roi_rubber_band.show()

        # Update Gazetteer Match
        self._update_gazetteer_matches(geom)

        # Cache raster data for fast live slider vectorization
        self._cache_roi_data()

        # Trigger live preview
        if hasattr(self, 'chk_live_preview') and self.chk_live_preview.isChecked():
            self._update_live_preview()

        # Reset map tool back to pan/navigation so user can inspect immediately
        if self.canvas.mapTool() == self.roi_tool:
            self.canvas.unsetMapTool(self.roi_tool)

    # Alias for backward compatibility
    _on_roi_selected = _on_polygon_roi_selected

    def _update_gazetteer_matches(self, geom: QgsGeometry):
        self.gazetteer_combo.blockSignals(True)
        self.gazetteer_combo.clear()
        self.gazetteer_combo.addItem("None", {"name": None, "type": None})
        
        try:
            if self.gazetteer_layer is None:
                gaz_path = os.path.join(os.path.dirname(__file__), "resources", "gazeteer_gov.geojson")
                if os.path.exists(gaz_path):
                    self.gazetteer_layer = QgsVectorLayer(gaz_path, "Gazetteer", "ogr")
                    if self.gazetteer_layer.isValid():
                        self.gazetteer_index = QgsSpatialIndex(self.gazetteer_layer.getFeatures())
                    else:
                        self.gazetteer_layer = None

            if not self.gazetteer_layer:
                self.gazetteer_info_lbl.setText("Gazetteer data not found.")
                return

            crs_canvas = self.canvas.mapSettings().destinationCrs()
            crs_4326 = QgsCoordinateReferenceSystem("EPSG:4326")
            crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
            xform_ctx = QgsProject.instance().transformContext()

            geom_4326 = QgsGeometry(geom)
            geom_4326.transform(QgsCoordinateTransform(crs_canvas, crs_4326, xform_ctx))

            matches = []
            is_fallback = False
            centroid_4326 = geom_4326.centroid()

            fids = self.gazetteer_index.intersects(geom_4326.boundingBox())
            req = QgsFeatureRequest().setFilterFids(fids)
            for f in self.gazetteer_layer.getFeatures(req):
                if f.geometry().intersects(geom_4326):
                    matches.append(f)

            if not matches:
                is_fallback = True
                geom_3857 = QgsGeometry(centroid_4326)
                geom_3857.transform(QgsCoordinateTransform(crs_4326, crs_3857, xform_ctx))
                buffer_3857 = geom_3857.buffer(3500, 8)
                buffer_4326 = QgsGeometry(buffer_3857)
                buffer_4326.transform(QgsCoordinateTransform(crs_3857, crs_4326, xform_ctx))

                fids = self.gazetteer_index.intersects(buffer_4326.boundingBox())
                req = QgsFeatureRequest().setFilterFids(fids)
                for f in self.gazetteer_layer.getFeatures(req):
                    if f.geometry().intersects(buffer_4326):
                        matches.append(f)

            if matches:
                matches.sort(key=lambda f: f.geometry().distance(centroid_4326))
                for i, f in enumerate(matches[:8]):
                    name = str(f["name"]) if f["name"] else "Unknown"
                    t_val = str(f["type_value"]) if f["type_value"] else ""
                    dist_deg = f.geometry().distance(centroid_4326)
                    dist_m = int(dist_deg * 111320.0)
                    if dist_m < 80:
                        dist_label = "in Ortslage"
                    elif dist_m < 1000:
                        dist_label = f"{dist_m}m"
                    else:
                        dist_label = f"{dist_m/1000:.1f}km"
                    
                    label = f"{name} ({t_val}, {dist_label})" if t_val else f"{name} ({dist_label})"
                    self.gazetteer_combo.addItem(label, {"name": name, "type": t_val})

                if is_fallback:
                    self.gazetteer_info_lbl.setText(f"📍 {min(8, len(matches))} Treffer in Umgebung (nächster: {matches[0]['name']})")
                else:
                    self.gazetteer_info_lbl.setText(f"📍 {min(8, len(matches))} Treffer innerhalb der Ortslage")
                
                # Auto-select best match (or keep previously selected match if present in candidates)
                selected_idx = 1
                if getattr(self, 'current_gazetteer_match', None):
                    for idx in range(1, self.gazetteer_combo.count()):
                        d = self.gazetteer_combo.itemData(idx)
                        if d and isinstance(d, dict) and d.get("name") == self.current_gazetteer_match:
                            selected_idx = idx
                            break

                self.gazetteer_combo.setCurrentIndex(selected_idx)
                best_data = self.gazetteer_combo.itemData(selected_idx)
                self.current_gazetteer_match = best_data["name"] if (best_data and isinstance(best_data, dict)) else matches[0]["name"]

                best_type = best_data.get("type", "") if (best_data and isinstance(best_data, dict)) else ""
                if best_type:
                    for idx in range(self.type_combo.count()):
                        if best_type.lower() in self.type_combo.itemText(idx).lower():
                            self.type_combo.setCurrentIndex(idx)
                            break
            else:
                self.gazetteer_info_lbl.setText("Keine Treffer in der Nähe (Name frei eintippbar).")
                self.current_gazetteer_match = None
                
        except Exception as e:
            self.gazetteer_info_lbl.setText(f"Error querying gazetteer: {e}")
        finally:
            self.gazetteer_combo.blockSignals(False)

    def _on_gazetteer_match_changed(self, text):
        data = self.gazetteer_combo.currentData()
        if data and isinstance(data, dict) and data.get("name"):
            self.current_gazetteer_match = data["name"]
        elif text == "None":
            self.current_gazetteer_match = None
        else:
            if "(" in text:
                self.current_gazetteer_match = text.split("(")[0].strip()
            else:
                self.current_gazetteer_match = text.strip()
            
        if hasattr(self, 'chk_live_preview') and self.chk_live_preview.isChecked():
            self._update_live_preview()

    def _on_type_match_changed(self, text):
        if text == "Default (Building/Courtyard)":
            self.current_type_match = None
            self.current_type_uri = None
        else:
            self.current_type_match = text
            self.current_type_uri = self.type_combo.currentData()
            
        if hasattr(self, 'chk_live_preview') and self.chk_live_preview.isChecked():
            self._update_live_preview()

    def _toggle_gazetteer_map_layer(self, checked: bool):
        """Shows or hides the historical GOV gazetteer places directly on the map canvas."""
        layer_name = "📍 Historische Orte (GOV Gazetteer)"
        existing_layers = QgsProject.instance().mapLayersByName(layer_name)

        if checked:
            if existing_layers:
                vl = existing_layers[0]
                node = QgsProject.instance().layerTreeRoot().findLayer(vl.id())
                if node:
                    node.setItemVisibilityChecked(True)
            else:
                gaz_path = os.path.join(os.path.dirname(__file__), "resources", "gazeteer_gov.geojson")
                if not os.path.exists(gaz_path):
                    self.status_lbl.setText("Gazetteer file not found.")
                    return
                vl = QgsVectorLayer(gaz_path, layer_name, "ogr")
                if not vl.isValid():
                    self.status_lbl.setText("Could not load gazetteer vector layer.")
                    return

                # Style as crisp orange pin/dots
                symbol = QgsMarkerSymbol.createSimple({
                    'name': 'circle',
                    'color': '230,81,0,255',
                    'color_border': '255,255,255,255',
                    'size': '2.6',
                    'outline_width': '0.5'
                })
                vl.setRenderer(QgsSingleSymbolRenderer(symbol))

                # Add labels
                tf = QgsTextFormat()
                tf.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                tf.setColor(QColor(33, 33, 33))
                buf = QgsTextBufferSettings()
                buf.setEnabled(True)
                buf.setSize(1.5)
                buf.setColor(QColor(255, 255, 255))
                tf.setBuffer(buf)
                lbl_settings = QgsPalLayerSettings()
                lbl_settings.fieldName = "name"
                lbl_settings.setFormat(tf)
                vl.setLabeling(QgsVectorLayerSimpleLabeling(lbl_settings))
                vl.setLabelsEnabled(True)

                QgsProject.instance().addMapLayer(vl, False)
                QgsProject.instance().layerTreeRoot().insertLayer(0, vl)

            self.canvas.refresh()
            self.status_lbl.setText("📍 Gazetteer-Orte auf Karte eingeblendet.")
        else:
            if existing_layers:
                node = QgsProject.instance().layerTreeRoot().findLayer(existing_layers[0].id())
                if node:
                    node.setItemVisibilityChecked(False)
            self.canvas.refresh()
            self.status_lbl.setText("📍 Gazetteer-Orte ausgeblendet.")

    def _activate_gazetteer_picker(self):
        """Activates map tool to select a gazetteer place directly by clicking on the map canvas."""
        if hasattr(self, 'gazetteer_pick_tool') and self.canvas.mapTool() == self.gazetteer_pick_tool:
            self.canvas.unsetMapTool(self.gazetteer_pick_tool)
            self._reset_pick_button()
            self.status_lbl.setText("Ortsauswahl auf Karte abgebrochen.")
            return

        # Ensure gazetteer places are visible on map
        if not self.chk_show_gazetteer.isChecked():
            self.chk_show_gazetteer.setChecked(True)

        if hasattr(self, 'gazetteer_pick_tool'):
            self.canvas.setMapTool(self.gazetteer_pick_tool)
            self.btn_pick_gazetteer.setStyleSheet(
                "background-color: #ff9800; color: black; font-weight: bold; padding: 5px 8px; border-radius: 4px;"
            )
            self.btn_pick_gazetteer.setText("🎯 Klicke auf Ort...")
            self.status_lbl.setText("📍 Klicke auf einen Punkt oder ein Dorf auf der Karte (Rechtsklick/Esc zum Abbrechen)...")

    def _cancel_gazetteer_picker(self):
        """Resets map tool when user cancels gazetteer picking."""
        if hasattr(self, 'gazetteer_pick_tool') and self.canvas.mapTool() == self.gazetteer_pick_tool:
            self.canvas.unsetMapTool(self.gazetteer_pick_tool)
        self._reset_pick_button()
        self.status_lbl.setText("Ortsauswahl auf Karte abgebrochen.")

    def _reset_pick_button(self):
        """Restores normal appearance of gazetteer pick button."""
        if hasattr(self, 'btn_pick_gazetteer'):
            self.btn_pick_gazetteer.setStyleSheet(
                "background-color: #e65100; color: white; font-weight: bold; padding: 5px 8px; border-radius: 4px;"
            )
            self.btn_pick_gazetteer.setText("🎯 Auf Karte wählen")

    def _on_gazetteer_point_picked(self, point: QgsPointXY):
        """Handles map click when picking a place/settlement from the Gazetteer."""
        try:
            if self.gazetteer_layer is None or self.gazetteer_index is None:
                gaz_path = os.path.join(os.path.dirname(__file__), "resources", "gazeteer_gov.geojson")
                if os.path.exists(gaz_path):
                    self.gazetteer_layer = QgsVectorLayer(gaz_path, "Gazetteer", "ogr")
                    if self.gazetteer_layer.isValid():
                        self.gazetteer_index = QgsSpatialIndex(self.gazetteer_layer.getFeatures())

            if not self.gazetteer_layer or not self.gazetteer_index:
                self.status_lbl.setText("Gazetteer-Daten konnten nicht geladen werden.")
                return

            crs_canvas = self.canvas.mapSettings().destinationCrs()
            crs_4326 = QgsCoordinateReferenceSystem("EPSG:4326")
            xform_ctx = QgsProject.instance().transformContext()
            ct_to_4326 = QgsCoordinateTransform(crs_canvas, crs_4326, xform_ctx)
            ct_to_canvas = QgsCoordinateTransform(crs_4326, crs_canvas, xform_ctx)

            pt_4326 = ct_to_4326.transform(point)

            # Query nearest candidates from spatial index
            candidate_fids = self.gazetteer_index.nearestNeighbor(pt_4326, 10)
            if not candidate_fids:
                self.status_lbl.setText("Keine Gazetteer-Punkte in der Datenbank gefunden.")
                return

            da = QgsDistanceArea()
            da.setSourceCrs(crs_4326, xform_ctx)
            da.setEllipsoid("EPSG:7030")

            candidates = []
            for fid in candidate_fids:
                f = self.gazetteer_layer.getFeature(fid)
                if f.isValid() and f.hasGeometry():
                    dist_m = da.measureLine(pt_4326, f.geometry().asPoint())
                    candidates.append((f, dist_m))

            if not candidates:
                self.status_lbl.setText("Keine gültigen Orte gefunden.")
                return

            candidates.sort(key=lambda item: item[1])
            best_feat, best_dist = candidates[0]

            best_name = str(best_feat["name"]) if best_feat["name"] else "Unknown"
            best_type = str(best_feat["type_value"]) if best_feat["type_value"] else ""

            # Populate combobox with nearby candidates
            self.gazetteer_combo.blockSignals(True)
            self.gazetteer_combo.clear()
            self.gazetteer_combo.addItem("None", {"name": None, "type": None})

            for f, d_m in candidates[:8]:
                n = str(f["name"]) if f["name"] else "Unknown"
                tv = str(f["type_value"]) if f["type_value"] else ""
                if d_m < 80:
                    dist_lbl = "am Klickpunkt"
                elif d_m < 1000:
                    dist_lbl = f"{int(d_m)}m"
                else:
                    dist_lbl = f"{d_m/1000:.1f}km"
                lbl = f"{n} ({tv}, {dist_lbl})" if tv else f"{n} ({dist_lbl})"
                self.gazetteer_combo.addItem(lbl, {"name": n, "type": tv})

            self.gazetteer_combo.setCurrentIndex(1)
            self.gazetteer_combo.blockSignals(False)

            self.current_gazetteer_match = best_name

            # Match settlement type in type_combo if available
            if best_type:
                for idx in range(self.type_combo.count()):
                    if best_type.lower() in self.type_combo.itemText(idx).lower():
                        self.type_combo.setCurrentIndex(idx)
                        break

            # Highlight selected point on map canvas with a glowing rubberband ring
            try:
                feat_pt_canvas = ct_to_canvas.transform(best_feat.geometry().asPoint())
                mupp = self.canvas.mapUnitsPerPixel()
                radius = max(mupp * 16.0, 10.0)
                circle_geom = QgsGeometry.fromPointXY(feat_pt_canvas).buffer(radius, 24)

                if not hasattr(self, 'gazetteer_pick_band') or self.gazetteer_pick_band is None:
                    self.gazetteer_pick_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
                self.gazetteer_pick_band.setFillColor(QColor(255, 152, 0, 80))
                self.gazetteer_pick_band.setStrokeColor(QColor(255, 87, 34, 255))
                self.gazetteer_pick_band.setWidth(3)
                self.gazetteer_pick_band.setToGeometry(circle_geom, None)
                self.gazetteer_pick_band.show()
                QTimer.singleShot(4000, lambda: self.gazetteer_pick_band.hide() if hasattr(self, 'gazetteer_pick_band') and self.gazetteer_pick_band else None)
            except Exception:
                pass

            dist_str = f"{int(best_dist)}m" if best_dist < 1000 else f"{best_dist/1000:.1f}km"
            info_text = f"🎯 Auf Karte gewählt: {best_name}"
            if best_type:
                info_text += f" ({best_type})"
            info_text += f" [{dist_str}]"
            self.gazetteer_info_lbl.setText(info_text)
            self.gazetteer_info_lbl.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 10px;")
            self.status_lbl.setText(f"🎯 Ort '{best_name}' direkt über Karte ausgewählt.")

            if hasattr(self, 'chk_live_preview') and self.chk_live_preview.isChecked():
                self._update_live_preview()

        except Exception as e:
            self.status_lbl.setText(f"Fehler bei Ortsauswahl: {e}")
        finally:
            if hasattr(self, 'gazetteer_pick_tool') and self.canvas.mapTool() == self.gazetteer_pick_tool:
                self.canvas.unsetMapTool(self.gazetteer_pick_tool)
            self._reset_pick_button()

    def _save_current_village(self, silent: bool = False):
        """Saves current village boundary and detected buildings permanently into project layers."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            if not silent:
                QMessageBox.warning(self, "Keine Rasterkarte", "Bitte wähle zuerst eine historische Rasterkarte aus.")
            return

        if not hasattr(self, 'current_roi_geom') or self.current_roi_geom is None or self.current_roi_geom.isEmpty():
            if not silent:
                QMessageBox.information(self, "Keine Ortslage gezeichnet", "Bitte zeichne zuerst ein Polygon um das Dorf.")
            return

        features_to_save = getattr(self, 'current_preview_features', [])
        if not features_to_save:
            self._update_live_preview()
            features_to_save = getattr(self, 'current_preview_features', [])

        village_name = self.current_gazetteer_match or self.gazetteer_combo.currentText().strip()
        if not village_name or village_name == "None":
            village_name = "Unbenannte Ortslage"
        elif "(" in village_name:
            village_name = village_name.split("(")[0].strip()

        s_type = getattr(self, 'current_type_match', None) or self.type_combo.currentText()
        if s_type == "Default (Building/Courtyard)":
            s_type = "Dorf"
        s_uri = getattr(self, 'current_type_uri', None) or self.type_combo.currentData() or ""

        crs_auth = layer.crs().authid() if (layer.crs() and layer.crs().isValid()) else "EPSG:25832"

        # 1. Commit Buildings to permanent layer
        bld_layer_name = f"🏛️ Buildings ({layer.name()})"
        bld_layers = QgsProject.instance().mapLayersByName(bld_layer_name)
        if bld_layers:
            vl_bld = bld_layers[0]
        else:
            vl_bld = QgsVectorLayer(f"Polygon?crs={crs_auth}", bld_layer_name, "memory")
            pr_bld = vl_bld.dataProvider()
            fields = [
                QgsField("id", QVariant.Int),
                QgsField("type", QVariant.String),
                QgsField("area_m2", QVariant.Double),
                QgsField("perimeter_m", QVariant.Double),
                QgsField("compactness", QVariant.Double),
                QgsField("orientation", QVariant.Double),
                QgsField("settlement", QVariant.String),
                QgsField("type_uri", QVariant.String),
            ]
            pr_bld.addAttributes(fields)
            vl_bld.updateFields()

            symbol = QgsFillSymbol.createSimple({
                'color': '205,55,55,215',
                'color_border': '110,20,20,255',
                'width_border': '0.35',
                'style': 'solid',
                'style_border': 'solid'
            })
            vl_bld.setRenderer(QgsSingleSymbolRenderer(symbol))
            QgsProject.instance().addMapLayer(vl_bld, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, vl_bld)

        pr_bld = vl_bld.dataProvider()
        new_bld_features = []
        base_id = vl_bld.featureCount()
        for idx, pf in enumerate(features_to_save):
            f = QgsFeature(vl_bld.fields())
            f.setGeometry(pf.geometry())
            f.setAttribute("id", base_id + idx + 1)
            f.setAttribute("type", pf.attribute("type") or "Building")
            f.setAttribute("area_m2", pf.attribute("area_m2"))
            f.setAttribute("perimeter_m", pf.attribute("perimeter_m"))
            f.setAttribute("compactness", pf.attribute("compactness"))
            f.setAttribute("orientation", pf.attribute("orientation"))
            f.setAttribute("settlement", village_name)
            f.setAttribute("type_uri", s_uri)
            new_bld_features.append(f)

        if new_bld_features:
            pr_bld.addFeatures(new_bld_features)
            vl_bld.updateExtents()
            vl_bld.triggerRepaint()

        # 2. Commit Settlement Boundary Polygon to permanent layer
        settlement_layer_name = f"🏘️ Ortslagen ({layer.name()})"
        settlement_layers = QgsProject.instance().mapLayersByName(settlement_layer_name)
        if settlement_layers:
            vl_settlement = settlement_layers[0]
        else:
            vl_settlement = QgsVectorLayer(f"Polygon?crs={crs_auth}", settlement_layer_name, "memory")
            pr_set = vl_settlement.dataProvider()
            fields = [
                QgsField("id", QVariant.Int),
                QgsField("name", QVariant.String),
                QgsField("type", QVariant.String),
                QgsField("type_uri", QVariant.String),
                QgsField("building_count", QVariant.Int),
                QgsField("area_m2", QVariant.Double),
                QgsField("area_ha", QVariant.Double),
                QgsField("sheet", QVariant.String),
            ]
            pr_set.addAttributes(fields)
            vl_settlement.updateFields()

            symbol = QgsFillSymbol.createSimple({
                'color': '123,31,162,25',
                'color_border': '123,31,162,240',
                'width_border': '0.6',
                'style': 'solid',
                'style_border': 'dash'
            })
            vl_settlement.setRenderer(QgsSingleSymbolRenderer(symbol))

            tf = QgsTextFormat()
            tf.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            tf.setColor(QColor(106, 27, 154))
            buf = QgsTextBufferSettings()
            buf.setEnabled(True)
            buf.setSize(1.4)
            buf.setColor(QColor(255, 255, 255))
            tf.setBuffer(buf)
            lbl_settings = QgsPalLayerSettings()
            lbl_settings.fieldName = "name"
            lbl_settings.setFormat(tf)
            vl_settlement.setLabeling(QgsVectorLayerSimpleLabeling(lbl_settings))
            vl_settlement.setLabelsEnabled(True)

            QgsProject.instance().addMapLayer(vl_settlement, False)
            root = QgsProject.instance().layerTreeRoot()
            bld_node = root.findLayer(vl_bld.id())
            if bld_node:
                idx = root.children().index(bld_node)
                root.insertLayer(idx + 1, vl_settlement)
            else:
                root.insertLayer(0, vl_settlement)

        layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer)
        pr_set = vl_settlement.dataProvider()
        sf = QgsFeature(vl_settlement.fields())
        sf.setGeometry(layer_geom)
        sf.setAttribute("id", vl_settlement.featureCount() + 1)
        sf.setAttribute("name", village_name)
        sf.setAttribute("type", s_type)
        sf.setAttribute("type_uri", s_uri)
        sf.setAttribute("building_count", len(new_bld_features))
        sf.setAttribute("area_m2", round(float(layer_geom.area()), 1))
        sf.setAttribute("area_ha", round(float(layer_geom.area()) / 10000.0, 2))
        sf.setAttribute("sheet", layer.name())
        pr_set.addFeatures([sf])
        vl_settlement.updateExtents()
        vl_settlement.triggerRepaint()

        # 3. Clean up live preview layer and rubber band
        for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
            pr_prev = l.dataProvider()
            pr_prev.deleteFeatures(l.allFeatureIds())
            l.triggerRepaint()

        if hasattr(self, 'roi_rubber_band') and self.roi_rubber_band:
            self.roi_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.roi_rubber_band.hide()

        if hasattr(self, 'gazetteer_pick_band') and self.gazetteer_pick_band:
            self.gazetteer_pick_band.reset(QgsWkbTypes.PolygonGeometry)
            self.gazetteer_pick_band.hide()

        self.current_roi = None
        self.current_roi_geom = None
        self.current_preview_features = []
        self.cached_roi_rgb = None

        msg = f"✓ Dorf '{village_name}' ({len(new_bld_features)} Gebäude) gespeichert! Du kannst das nächste Dorf einzeichnen."
        self.status_lbl.setText(msg)
        self.roi_info_lbl.setText(f"Gespeichert: '{village_name}' ({len(new_bld_features)} Gebäude). Bereit für nächstes Dorf.")
        self.canvas.refresh()
        if not silent:
            self.status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")

    def _transform_canvas_to_layer_rect(self, rect: QgsRectangle, layer: QgsRasterLayer) -> QgsRectangle:
        """Transforms rectangle from Canvas CRS to Raster Layer CRS."""
        if not rect or rect.isEmpty() or not layer or not layer.isValid():
            return rect
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        layer_crs = layer.crs()
        if not canvas_crs.isValid() or not layer_crs.isValid() or canvas_crs == layer_crs:
            return rect
        try:
            from qgis.core import QgsCoordinateTransform
            ct = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            geom = QgsGeometry.fromRect(rect)
            geom.transform(ct)
            return geom.boundingBox()
        except Exception:
            return rect

    def _transform_canvas_to_layer_geom(self, geom: QgsGeometry, layer: QgsRasterLayer) -> QgsGeometry:
        """Transforms geometry from Canvas CRS to Raster Layer CRS."""
        if not geom or geom.isEmpty() or not layer or not layer.isValid():
            return geom
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        layer_crs = layer.crs()
        if not canvas_crs.isValid() or not layer_crs.isValid() or canvas_crs == layer_crs:
            return geom
        try:
            from qgis.core import QgsCoordinateTransform
            ct = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            res_geom = QgsGeometry(geom)
            res_geom.transform(ct)
            return res_geom
        except Exception:
            return geom

    def _clear_roi(self):
        """Clears active ROI, cached data, and preview layers."""
        self.current_roi = None
        self.current_roi_geom = None
        self.cached_roi_rgb = None
        if hasattr(self, 'roi_rubber_band') and self.roi_rubber_band:
            self.roi_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.roi_rubber_band.hide()
        if hasattr(self, 'gazetteer_pick_band') and self.gazetteer_pick_band:
            self.gazetteer_pick_band.reset(QgsWkbTypes.PolygonGeometry)
            self.gazetteer_pick_band.hide()
        if self.roi_tool:
            self.roi_tool.reset()
        self.roi_info_lbl.setText("Area: No settlement polygon drawn yet (use button above)")
        self.roi_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        self.btn_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")

        # Remove preview vector layer if present
        for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
            QgsProject.instance().removeMapLayer(l.id())

        self.status_lbl.setText("Area reset. Please define an area using '📐 Draw Settlement Polygon'.")
        self.canvas.refresh()

    def _cache_roi_data(self):
        """Caches the current ROI raster array and geotransform for instant live updates."""
        # Strictly require an active ROI: do NOT process without an ROI!
        if self.current_roi is None or self.current_roi.isEmpty():
            self.cached_roi_rgb = None
            return

        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            return

        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            return

        gt = ds.GetGeoTransform()
        x_origin = gt[0]
        px_w = gt[1]
        y_origin = gt[3]
        px_h = gt[5]
        img_w = ds.RasterXSize
        img_h = ds.RasterYSize

        # Transform ROI to raster layer's CRS to ensure exact pixel coordinates
        layer_roi = self._transform_canvas_to_layer_rect(self.current_roi, layer)

        x0_px = int((layer_roi.xMinimum() - x_origin) / px_w)
        x1_px = int((layer_roi.xMaximum() - x_origin) / px_w)
        y0_px = int((layer_roi.yMaximum() - y_origin) / px_h)
        y1_px = int((layer_roi.yMinimum() - y_origin) / px_h)

        x_min = max(0, min(img_w - 1, min(x0_px, x1_px)))
        x_max = max(0, min(img_w, max(x0_px, x1_px)))
        y_min = max(0, min(img_h - 1, min(y0_px, y1_px)))
        y_max = max(0, min(img_h, max(y0_px, y1_px)))

        x_off = x_min
        y_off = y_min
        win_w = max(1, x_max - x_min)
        win_h = max(1, y_max - y_min)

        sub_x_origin = x_origin + x_off * px_w
        sub_y_origin = y_origin + y_off * px_h

        # Read RGB bands
        if ds.RasterCount >= 3:
            r = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
            g = ds.GetRasterBand(2).ReadAsArray(x_off, y_off, win_w, win_h)
            b = ds.GetRasterBand(3).ReadAsArray(x_off, y_off, win_w, win_h)
            image_rgb = np.dstack((r, g, b))
        else:
            gray = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
            image_rgb = np.dstack((gray, gray, gray))

        self.cached_roi_rgb = image_rgb
        self.cached_roi_geo = (sub_x_origin, sub_y_origin, px_w, px_h)
        self.cached_roi_crs = layer.crs().authid() if layer.crs().isValid() else "EPSG:25832"
        self.cached_roi_crs_wkt = layer.crs().toWkt()
        ds = None

    def _trigger_preview_update(self):
        """Debounced trigger for live building extraction preview."""
        if hasattr(self, 'chk_live_preview') and self.chk_live_preview.isChecked():
            if self.current_roi is not None and not self.current_roi.isEmpty():
                self.preview_timer.start(150)

    def _on_live_preview_toggled(self, checked: bool):
        """Handles user toggling the live preview checkbox."""
        if not checked:
            for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
                QgsProject.instance().removeMapLayer(l.id())
            self.status_lbl.setText("Live preview deactivated.")
        else:
            self._update_live_preview()

    def _update_live_preview(self):
        """Extracts buildings on cached ROI and updates the live preview layer on canvas."""
        if not hasattr(self, 'chk_live_preview') or not self.chk_live_preview.isChecked():
            return
        # Strictly require user to have drawn or selected an ROI: do NOT extract without ROI!
        if self.current_roi is None or self.current_roi.isEmpty():
            for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
                QgsProject.instance().removeMapLayer(l.id())
            return

        global BuildingExtractor, BuildingConfig
        if BuildingExtractor is None or BuildingConfig is None:
            try:
                from tranchot_extractor.config import BuildingConfig
                from tranchot_extractor.extractors.building_extractor import BuildingExtractor
            except Exception:
                pass

        if BuildingExtractor is None or BuildingConfig is None:
            return

        if not hasattr(self, 'cached_roi_rgb') or self.cached_roi_rgb is None:
            self._cache_roi_data()

        if not hasattr(self, 'cached_roi_rgb') or self.cached_roi_rgb is None:
            return

        config = BuildingConfig()
        min_excess = float(self.spin_min_red.value())
        max_excess = float(self.spin_max_red.value())

        config.min_red_excess = min_excess
        config.max_red_excess = max_excess
        config.min_red_ratio = 1.05
        config.max_red_ratio = 999.0
        config.min_building_thickness_px = float(self.spin_thickness.value())
        config.min_building_area_px = float(self.spin_min_area.value())
        config.regularize_orthogonal = False
        config.simplify_tolerance = 0.5
        config.filter_vineyard_terraces = False
        config.preserve_courtyards = bool(self.chk_courtyard.isChecked())
        config.enable_black_buildings = bool(self.chk_black_bld.isChecked())

        sub_x_origin, sub_y_origin, px_w, px_h = self.cached_roi_geo
        m2_per_px2 = abs(px_w * px_h)

        # Scoping polygons in pixel space
        settlement_polys = []
        if hasattr(self, 'current_roi_geom') and self.current_roi_geom is not None and not self.current_roi_geom.isEmpty():
            def geo_to_pixel(gx: float, gy: float) -> Tuple[float, float]:
                return ((gx - sub_x_origin) / px_w, (gy - sub_y_origin) / px_h)
            try:
                layer = self.layer_combo.currentLayer()
                layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer)
                if layer_geom.isMultipart():
                    poly_groups = layer_geom.asMultiPolygon()
                else:
                    poly_groups = [layer_geom.asPolygon()]
                for p_group in poly_groups:
                    if p_group and len(p_group[0]) >= 3:
                        ext_px = [geo_to_pixel(pt.x(), pt.y()) for pt in p_group[0]]
                        holes_px = [[geo_to_pixel(pt.x(), pt.y()) for pt in h_ring] for h_ring in p_group[1:]]
                        p_shapely = Polygon(ext_px, holes_px)
                        if p_shapely.is_valid and not p_shapely.is_empty:
                            settlement_polys.append(p_shapely)
            except Exception:
                pass

        try:
            cls_extractor = BuildingExtractor
            try:
                import importlib
                if "tranchot_extractor.extractors.building_extractor" in sys.modules:
                    mod = importlib.reload(sys.modules["tranchot_extractor.extractors.building_extractor"])
                    cls_extractor = getattr(mod, "BuildingExtractor", BuildingExtractor)
            except Exception:
                cls_extractor = BuildingExtractor

            extractor = cls_extractor(config=config)
            result = extractor.extract(
                image_rgb=self.cached_roi_rgb,
                settlement_boundaries=settlement_polys if settlement_polys else None
            )

            # Get or create preview vector layer
            layer_name = "🔍 Historical Buildings (Live Preview)"
            preview_layers = QgsProject.instance().mapLayersByName(layer_name)
            if preview_layers:
                vl = preview_layers[0]
            else:
                uri = f"Polygon?crs={self.cached_roi_crs}"
                vl = QgsVectorLayer(uri, layer_name, "memory")
                pr = vl.dataProvider()
                fields = [
                    QgsField("id", QVariant.Int),
                    QgsField("type", QVariant.String),
                    QgsField("area_m2", QVariant.Double),
                    QgsField("perimeter_m", QVariant.Double),
                    QgsField("compactness", QVariant.Double),
                    QgsField("orientation", QVariant.Double),
                    QgsField("settlement", QVariant.String),
                    QgsField("type_uri", QVariant.String),
                ]
                pr.addAttributes(fields)
                vl.updateFields()

                symbol = QgsFillSymbol.createSimple({
                    'color': '230,45,45,190',
                    'color_border': '120,10,10,255',
                    'width_border': '0.5',
                    'style': 'solid',
                    'style_border': 'solid'
                })
                # Register layer in project without default tree insertion, then insert at very top
                QgsProject.instance().addMapLayer(vl, False)
                QgsProject.instance().layerTreeRoot().insertLayer(0, vl)

            # Transform features to map coordinates
            qgis_features = []
            for feat in result.features:
                poly = feat.geometry
                if poly is None or poly.is_empty:
                    continue

                def transform_ring(coords):
                    return [
                        (sub_x_origin + pt[0] * px_w, sub_y_origin + pt[1] * px_h)
                        for pt in coords
                    ]

                try:
                    ext_geo = transform_ring(poly.exterior.coords)
                    holes_geo = [transform_ring(interior.coords) for interior in poly.interiors]
                    geo_poly = Polygon(ext_geo, holes_geo)
                    if not geo_poly.is_valid:
                        geo_poly = geo_poly.buffer(0)
                    if geo_poly.is_valid and not geo_poly.is_empty:
                        f = QgsFeature(vl.fields())
                        f.setGeometry(QgsGeometry.fromWkt(geo_poly.wkt))
                        f.setAttribute("id", feat.id)
                        
                        if hasattr(self, 'current_type_match') and self.current_type_match:
                            f.setAttribute("type", self.current_type_match)
                            f.setAttribute("type_uri", self.current_type_uri)
                        else:
                            f.setAttribute("type", "Courtyard Complex" if len(poly.interiors) > 0 else "Building")
                            
                        f.setAttribute("area_m2", round(float(feat.area_px) * m2_per_px2, 2))
                        f.setAttribute("perimeter_m", round(float(feat.perimeter_px) * abs(px_w), 2))
                        f.setAttribute("compactness", round(float(feat.compactness), 3))
                        f.setAttribute("orientation", round(float(feat.orientation_deg), 1))
                        if hasattr(self, 'current_gazetteer_match') and self.current_gazetteer_match:
                            f.setAttribute("settlement", self.current_gazetteer_match)
                        qgis_features.append(f)
                except Exception:
                    pass

            # Update features in place
            pr = vl.dataProvider()
            pr.deleteFeatures(vl.allFeatureIds())
            if qgis_features:
                pr.addFeatures(qgis_features)
            vl.updateExtents()
            vl.triggerRepaint()

            # Save reference for permanent village saving
            self.current_preview_features = qgis_features

            # Ensure layer is checked/visible in layer tree
            node = QgsProject.instance().layerTreeRoot().findLayer(vl.id())
            if node and not node.isVisible():
                node.setItemVisibilityChecked(True)

            self.canvas.refresh()

            self.status_lbl.setText(
                f"⚡ Live: {len(qgis_features)} buildings detected | Red Threshold={min_excess:.0f}..{max_excess:.0f} | Wall={config.min_building_thickness_px:.1f}px"
            )
        except Exception as e:
            self.status_lbl.setText(f"Preview error: {e}")

    def _set_global_view_mode(self, mode: str):
        """Switches globally between authentic aged scan and normalized de-yellowed map."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            self.status_lbl.setText("No active raster sheet selected.")
            return

        norm_layer_name = f"🎨 {layer.name()} (Normalized)"
        norm_layers = QgsProject.instance().mapLayersByName(norm_layer_name)

        if mode == "normalized":
            self.btn_view_norm.setChecked(True)
            self.btn_view_orig.setChecked(False)

            # Check if norm_layer exists AND is backed by a valid (non-zero/non-empty) file
            valid_existing = False
            if norm_layers:
                nl = norm_layers[0]
                src = nl.dataProvider().dataSourceUri() if (nl and nl.isValid()) else ""
                if src and os.path.exists(src):
                    try:
                        t_ds = gdal.Open(src, gdal.GA_ReadOnly)
                        if t_ds and t_ds.RasterXSize > 0 and t_ds.RasterYSize > 0:
                            cx = t_ds.RasterXSize // 2
                            cy = t_ds.RasterYSize // 2
                            s = t_ds.GetRasterBand(1).ReadAsArray(cx, cy, min(100, t_ds.RasterXSize - cx), min(100, t_ds.RasterYSize - cy))
                            if s is not None and int(s.max()) > 0:
                                valid_existing = True
                        t_ds = None
                    except Exception:
                        valid_existing = False

                if not valid_existing:
                    # Remove corrupt / black layer from QGIS
                    for ol in norm_layers:
                        QgsProject.instance().removeMapLayer(ol.id())
                    norm_layers = []

            if norm_layers and valid_existing:
                norm_layer = norm_layers[0]
                node_norm = QgsProject.instance().layerTreeRoot().findLayer(norm_layer.id())
                if node_norm:
                    node_norm.setItemVisibilityChecked(True)
                node_orig = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
                if node_orig:
                    node_orig.setItemVisibilityChecked(False)
                self.canvas.refresh()
                self.status_lbl.setText("View: Normalized (De-yellowed) active.")
            else:
                self._create_global_normalized_layer(layer)
        else:
            self.btn_view_orig.setChecked(True)
            self.btn_view_norm.setChecked(False)
            node_orig = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            if node_orig:
                node_orig.setItemVisibilityChecked(True)
            if norm_layers:
                node_norm = QgsProject.instance().layerTreeRoot().findLayer(norm_layers[0].id())
                if node_norm:
                    node_norm.setItemVisibilityChecked(False)
            self.canvas.refresh()
            self.status_lbl.setText("View: Original (Aged paper) active.")

    def _create_global_normalized_layer(self, layer: QgsRasterLayer):
        """Creates and displays the global normalized raster layer for the active map sheet."""
        if ColorEnhancer is None:
            self.status_lbl.setText("ColorEnhancer module not available.")
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            self.status_lbl.setText("Raster file not found.")
            return

        norm_layer_name = f"🎨 {layer.name()} (Normalized)"
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        cache_dir = os.path.join(tempfile.gettempdir(), "tranchot_norm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, f"{base_name}_norm.tif")

        def _is_valid_cache(fpath: str) -> bool:
            if not os.path.exists(fpath):
                return False
            try:
                test_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
                if test_ds is None or test_ds.RasterXSize == 0 or test_ds.RasterYSize == 0:
                    return False
                cx = test_ds.RasterXSize // 2
                cy = test_ds.RasterYSize // 2
                sample = test_ds.GetRasterBand(1).ReadAsArray(cx, cy, min(100, test_ds.RasterXSize - cx), min(100, test_ds.RasterYSize - cy))
                test_ds = None
                return sample is not None and int(sample.max()) > 0
            except Exception:
                return False

        try:
            if not _is_valid_cache(cached_file):
                # Clean up existing corrupted layer in QGIS project if loaded to release file lock
                for old_l in QgsProject.instance().mapLayersByName(norm_layer_name):
                    QgsProject.instance().removeMapLayer(old_l.id())

                self.status_lbl.setText("Computing global parchment white-balance for map sheet...")
                QgsApplication.processEvents()

                ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
                if ds is None:
                    self.status_lbl.setText("GDAL could not open raster file.")
                    return

                w = ds.RasterXSize
                h = ds.RasterYSize
                gt = ds.GetGeoTransform()
                proj = ds.GetProjection()
                has_alpha = (ds.RasterCount >= 4)
                out_bands = 4 if has_alpha else 3

                driver = gdal.GetDriverByName("GTiff")
                options = ["COMPRESS=LZW", "TILED=YES"]

                # Write to temp file first to prevent partial/corrupted cache files
                tmp_file = os.path.join(cache_dir, f"{base_name}_norm_{os.getpid()}_{int(time.time())}.tmp")
                out_ds = driver.Create(tmp_file, w, h, out_bands, gdal.GDT_Byte, options=options)
                out_ds.SetGeoTransform(gt)
                if proj:
                    out_ds.SetProjection(proj)

                out_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
                out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
                out_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
                if has_alpha:
                    out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

                # 1. Estimate uniform parchment tone from full sheet thumbnail
                thumb_w = min(1500, w)
                thumb_h = min(1500, h)
                thumb_r = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h, buf_xsize=thumb_w, buf_ysize=thumb_h)
                thumb_g = ds.GetRasterBand(2).ReadAsArray(0, 0, w, h, buf_xsize=thumb_w, buf_ysize=thumb_h)
                thumb_b = ds.GetRasterBand(3).ReadAsArray(0, 0, w, h, buf_xsize=thumb_w, buf_ysize=thumb_h)
                thumb_rgb = np.dstack((thumb_r, thumb_g, thumb_b))
                global_paper_rgb = ColorEnhancer.estimate_paper_color(thumb_rgb)

                tile_size = 2048
                total_tiles = ((w + tile_size - 1) // tile_size) * ((h + tile_size - 1) // tile_size)
                current_tile = 0

                for y in range(0, h, tile_size):
                    th = min(tile_size, h - y)
                    for x in range(0, w, tile_size):
                        tw = min(tile_size, w - x)
                        r = ds.GetRasterBand(1).ReadAsArray(x, y, tw, th)
                        g = ds.GetRasterBand(2).ReadAsArray(x, y, tw, th)
                        b = ds.GetRasterBand(3).ReadAsArray(x, y, tw, th)
                        rgb = np.dstack((r, g, b))
                        enh = ColorEnhancer.enhance_map(
                            rgb,
                            paper_rgb=global_paper_rgb,
                            deyellow_strength=0.90,
                            vibrance=1.75,
                            contrast=1.00
                        )
                        for band_idx in range(3):
                            out_ds.GetRasterBand(band_idx + 1).WriteArray(enh[:, :, band_idx], x, y)

                        if has_alpha:
                            alpha = ds.GetRasterBand(4).ReadAsArray(x, y, tw, th)
                            out_ds.GetRasterBand(4).WriteArray(alpha, x, y)

                        current_tile += 1
                        if current_tile % 4 == 0 or current_tile == total_tiles:
                            pct = int((current_tile / total_tiles) * 100)
                            self.progress_bar.setValue(pct)
                            self.status_lbl.setText(f"Normalizing map sheet: {pct}% ({current_tile}/{total_tiles} tiles)...")
                            QgsApplication.processEvents()

                out_ds.FlushCache()
                out_ds = None
                ds = None

                # Atomic replace
                replaced = False
                if os.path.exists(cached_file):
                    try:
                        os.remove(cached_file)
                        shutil.move(tmp_file, cached_file)
                        replaced = True
                    except OSError:
                        pass
                if not replaced:
                    if not os.path.exists(cached_file):
                        shutil.move(tmp_file, cached_file)
                    else:
                        cached_file = os.path.join(cache_dir, f"{base_name}_norm_{int(time.time())}.tif")
                        shutil.move(tmp_file, cached_file)

            self.progress_bar.setValue(100)
            norm_layer = QgsRasterLayer(cached_file, norm_layer_name)
            if norm_layer.isValid():
                renderer = QgsMultiBandColorRenderer(norm_layer.dataProvider(), 1, 2, 3)
                norm_layer.setRenderer(renderer)
                norm_layer.triggerRepaint()

                curr = self.layer_combo.currentLayer()
                self.layer_combo.blockSignals(True)
                try:
                    excepted = list(self.layer_combo.exceptedLayerList())
                    if norm_layer not in excepted:
                        excepted.append(norm_layer)
                    self.layer_combo.setExceptedLayerList(excepted)

                    root = QgsProject.instance().layerTreeRoot()
                    orig_node = root.findLayer(layer.id())
                    QgsProject.instance().addMapLayer(norm_layer, False)
                    if orig_node:
                        parent = orig_node.parent() or root
                        parent.insertLayer(parent.children().index(orig_node), norm_layer)
                    else:
                        root.insertLayer(0, norm_layer)

                    if curr and curr.isValid():
                        self.layer_combo.setLayer(curr)

                    node_norm = root.findLayer(norm_layer.id())
                    if node_norm:
                        node_norm.setItemVisibilityChecked(True)
                    if orig_node:
                        orig_node.setItemVisibilityChecked(False)
                finally:
                    self.layer_combo.blockSignals(False)

                self.canvas.refresh()
                self.status_lbl.setText("✓ Map sheet fully normalized (de-yellowed) and active.")
            else:
                self.status_lbl.setText("Failed to load normalized layer.")
        except Exception as e:
            self.status_lbl.setText(f"Normalization error: {e}")

    def _activate_pipette_tool(self):
        """Activates color pipette tool."""
        self.canvas.setMapTool(self.pipette_tool)
        self.status_lbl.setText("Click on any feature on the canvas to sample its color...")
        self.btn_pipette.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold;")

    def _on_pipette_sampled(self, point):
        """Callback when user clicks to sample a pixel."""
        self.btn_pipette.setStyleSheet("")
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            self.status_lbl.setText("No valid raster sheet active.")
            return

        try:
            # Transform point to raster layer CRS if needed
            canvas_crs = self.canvas.mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if canvas_crs.isValid() and layer_crs.isValid() and canvas_crs != layer_crs:
                from qgis.core import QgsCoordinateTransform
                tr = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
                sample_pt = tr.transform(point)
            else:
                sample_pt = point

            ident = layer.dataProvider().identify(sample_pt, QgsRaster.IdentifyFormatValue)
            if ident.isValid():
                res = ident.results()
                if len(res) >= 3:
                    r_val = int(res.get(1, 0))
                    g_val = int(res.get(2, 0))
                    b_val = int(res.get(3, 0))
                    lum = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
                    text_col = "#000000" if lum > 130 else "#ffffff"
                    self.pip_status_lbl.setText(f"RGB: ({r_val}, {g_val}, {b_val}) | #{r_val:02x}{g_val:02x}{b_val:02x}")
                    self.pip_status_lbl.setStyleSheet(
                        f"color: {text_col}; background-color: rgb({r_val},{g_val},{b_val}); padding: 3px 6px; border-radius: 3px; font-weight: bold; font-family: monospace;"
                    )
                else:
                    self.pip_status_lbl.setText(f"Sample: {res}")
                self.status_lbl.setText(f"Sampled at X: {sample_pt.x():.1f}, Y: {sample_pt.y():.1f}")
            else:
                self.pip_status_lbl.setText("No pixel data at this location.")
        except Exception as e:
            self.pip_status_lbl.setText(f"Sampling error: {e}")


    def _on_layer_changed(self, layer):
        """Handles user selecting another raster layer."""
        if layer and (layer.name().startswith("🎨") or layer.name().startswith("🔍") or layer.name().startswith("🏛️")):
            return
        self._clear_roi()
        if layer and isinstance(layer, QgsRasterLayer):
            self.status_lbl.setText(f"Active: {layer.name()} ({layer.width()}×{layer.height()} px)")
            norm_layer_name = f"🎨 {layer.name()} (Normalized)"
            norm_layers = QgsProject.instance().mapLayersByName(norm_layer_name)
            if norm_layers:
                node_norm = QgsProject.instance().layerTreeRoot().findLayer(norm_layers[0].id())
                is_norm_vis = node_norm.isVisible() if node_norm else False
                self.btn_view_norm.setChecked(is_norm_vis)
                self.btn_view_orig.setChecked(not is_norm_vis)
            else:
                self.btn_view_orig.setChecked(True)
                self.btn_view_norm.setChecked(False)
            self._trigger_preview_update()
        else:
            self.status_lbl.setText("Please select a historical map sheet.")

    def _run_building_extraction(self):
        """Launches the BuildingExtractionTask in background thread."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "No Raster Selected", "Please select a historical GeoTIFF first.")
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            QMessageBox.critical(self, "File Error", f"Raster file does not exist:\n{raster_path}")
            return

        # Strictly require an active ROI: do NOT extract without an ROI!
        if self.current_roi is None or self.current_roi.isEmpty():
            QMessageBox.information(
                self,
                "No Area Selected",
                "Please draw a polygon around the settlement first using '📐 Draw Settlement Polygon'."
            )
            return

        # Prepare config with adaptive sensitivity
        config = BuildingConfig() if BuildingConfig else None
        if config:
            min_excess = float(self.spin_min_red.value())
            max_excess = float(self.spin_max_red.value())

            config.min_red_excess = min_excess
            config.max_red_excess = max_excess
            config.min_red_ratio = 1.05
            config.max_red_ratio = 999.0
            config.min_building_thickness_px = float(self.spin_thickness.value())
            config.min_building_area_px = float(self.spin_min_area.value())
            config.regularize_orthogonal = False
            config.simplify_tolerance = 0.5
            config.filter_vineyard_terraces = False
            config.preserve_courtyards = bool(self.chk_courtyard.isChecked())
            config.enable_black_buildings = bool(self.chk_black_bld.isChecked())

        # Transform ROI to raster layer CRS so coordinates and pixel cutouts match 100%
        layer_roi = self._transform_canvas_to_layer_rect(self.current_roi, layer)
        layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer) if hasattr(self, 'current_roi_geom') and self.current_roi_geom is not None else None
        output_crs = layer.crs().authid() if layer.crs().isValid() else "EPSG:25832"
        layer_name = f"🏛️ Buildings ({layer.name()})"

        # Start QgsTask
        self.btn_extract_buildings.setEnabled(False)
        self.progress_bar.setValue(5)
        self.status_lbl.setText("Extracting buildings (authentic contours & courtyard subtraction)...")

        self.current_task = BuildingExtractionTask(
            raster_path=raster_path,
            config=config,
            roi_extent=layer_roi,
            roi_geometry=layer_geom,
            layer_name=layer_name,
            output_crs=output_crs,
            settlement_name=self.current_gazetteer_match if hasattr(self, 'current_gazetteer_match') else None,
            settlement_type=self.current_type_match if hasattr(self, 'current_type_match') else None,
            settlement_uri=self.current_type_uri if hasattr(self, 'current_type_uri') else None
        )
        self.current_task.task_completed.connect(self._on_task_completed)
        self.current_task.task_failed.connect(self._on_task_failed)

        QgsApplication.taskManager().addTask(self.current_task)

    def _on_task_completed(self, layer_name: str, count: int):
        self.btn_extract_buildings.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText(f"Done! {count} buildings added to layer '{layer_name}'.")

        # Remove preview vector layer now that permanent layer is committed
        for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
            QgsProject.instance().removeMapLayer(l.id())

        if count == 0:
            QMessageBox.information(
                self, "No Buildings Found",
                "No buildings were detected in this area.\nTip: Adjust the Carmine Red threshold sliders."
            )

    def _on_task_failed(self, error_msg: str):
        self.btn_extract_buildings.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Extraction Error", f"Building extraction failed:\n{error_msg}")

    # -------------------------------------------------------------------------
    # Land Use Methods: Dedicated ROI, Multi-Nuance Sampling & Live Preview
    # -------------------------------------------------------------------------
    def _activate_lu_roi_tool(self):
        """Switches QGIS map tool to Land Use polygon drawing mode."""
        self.canvas.setMapTool(self.lu_roi_tool)
        self.status_lbl.setText("Landnutzungs-Gebiet zeichnen (Rechtsklick / Doppelklick zum Abschließen)...")
        self.btn_lu_roi.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; padding: 6px; border-radius: 4px;")

    def _on_lu_polygon_roi_selected(self, geom: QgsGeometry):
        """Callback when user finishes drawing a Land Use ROI polygon."""
        if geom is None or geom.isEmpty():
            return
        self.current_lu_roi_geom = geom
        self.current_lu_roi = geom.boundingBox()
        ha = geom.area() / 10000.0
        self.lu_roi_info_lbl.setText(
            f"Landnutzungs-Gebiet: Fläche ≈ {geom.area():.0f} m² ({ha:.2f} ha)"
        )
        self.lu_roi_info_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
        self.btn_lu_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")

        # Display persistent ROI polygon on map canvas (emerald green dashed border, transparent interior)
        if not hasattr(self, 'lu_roi_rubber_band') or self.lu_roi_rubber_band is None:
            self.lu_roi_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.lu_roi_rubber_band.setFillColor(QColor(0, 0, 0, 0))
        self.lu_roi_rubber_band.setStrokeColor(QColor(46, 125, 50, 240))
        self.lu_roi_rubber_band.setWidth(2)
        self.lu_roi_rubber_band.setLineStyle(Qt.PenStyle.DashLine)
        self.lu_roi_rubber_band.setToGeometry(geom, None)
        self.lu_roi_rubber_band.show()

        # Cache raster data for fast live slider updates
        self._cache_lu_roi_data()

        # Trigger live preview
        if hasattr(self, 'chk_lu_live_preview') and self.chk_lu_live_preview.isChecked():
            self._trigger_lu_preview_update()

        # Reset map tool back to pan/navigation
        if self.canvas.mapTool() == self.lu_roi_tool:
            self.canvas.unsetMapTool(self.lu_roi_tool)

    def _clear_lu_roi(self):
        """Clears active Land Use ROI, cached data, and preview layers."""
        self.current_lu_roi = None
        self.current_lu_roi_geom = None
        self.cached_lu_roi_rgb = None
        if hasattr(self, 'lu_roi_rubber_band') and self.lu_roi_rubber_band:
            self.lu_roi_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.lu_roi_rubber_band.hide()
        if hasattr(self, 'lu_roi_tool') and self.lu_roi_tool:
            self.lu_roi_tool.reset()
        if hasattr(self, 'lu_roi_info_lbl'):
            self.lu_roi_info_lbl.setText("Untersuchungsgebiet: Noch kein Polygon gezeichnet")
            self.lu_roi_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        if hasattr(self, 'btn_lu_roi'):
            self.btn_lu_roi.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")

        # Remove preview vector layer if present
        for l in QgsProject.instance().mapLayersByName("🔍 Land Use (Live Preview)"):
            QgsProject.instance().removeMapLayer(l.id())

        self.status_lbl.setText("Landnutzungs-Gebiet zurückgesetzt.")
        self.canvas.refresh()

    def _cache_lu_roi_data(self):
        """Caches raster cutout for Land Use ROI to enable real-time slider updates."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            return

        if self.rb_lu_roi.isChecked() and self.current_lu_roi is not None and not self.current_lu_roi.isEmpty():
            extent = self.current_lu_roi
        else:
            extent = layer.extent()

        layer_roi = self._transform_canvas_to_layer_rect(extent, layer)
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            return

        gt = ds.GetGeoTransform()
        x_origin, px_w, _, y_origin, _, px_h = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
        img_w, img_h = ds.RasterXSize, ds.RasterYSize

        x0_px = int((layer_roi.xMinimum() - x_origin) / px_w)
        x1_px = int((layer_roi.xMaximum() - x_origin) / px_w)
        y0_px = int((layer_roi.yMaximum() - y_origin) / px_h)
        y1_px = int((layer_roi.yMinimum() - y_origin) / px_h)

        x_off = max(0, min(img_w - 1, min(x0_px, x1_px)))
        y_off = max(0, min(img_h - 1, min(y0_px, y1_px)))
        win_w = max(1, min(img_w - x_off, abs(x1_px - x0_px)))
        win_h = max(1, min(img_h - y_off, abs(y1_px - y0_px)))

        # Limit cache size for speed (e.g. max 2400)
        max_dim = max(win_w, win_h)
        buf_w, buf_h = win_w, win_h
        if max_dim > 2400:
            scale = 2400.0 / max_dim
            buf_w = int(win_w * scale)
            buf_h = int(win_h * scale)

        r_band = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h, buf_xsize=buf_w, buf_ysize=buf_h)
        g_band = ds.GetRasterBand(2).ReadAsArray(x_off, y_off, win_w, win_h, buf_xsize=buf_w, buf_ysize=buf_h)
        b_band = ds.GetRasterBand(3).ReadAsArray(x_off, y_off, win_w, win_h, buf_xsize=buf_w, buf_ysize=buf_h)
        ds = None

        if r_band is not None and g_band is not None and b_band is not None:
            self.cached_lu_roi_rgb = np.dstack((r_band, g_band, b_band))
            eff_px_w = px_w * (float(win_w) / float(buf_w))
            eff_px_h = px_h * (float(win_h) / float(buf_h))
            self.cached_lu_sub_origin = (x_origin + x_off * px_w, y_origin + y_off * px_h)
            self.cached_lu_px_size = (eff_px_w, eff_px_h)

    def _activate_lu_stamp_tool(self):
        """Activates multi-stamp pipette tool for land use nuances."""
        if hasattr(self, 'lu_stamp_tool') and self.canvas.mapTool() == self.lu_stamp_tool:
            self._finish_lu_stamp_sampling()
            return

        if hasattr(self, 'lu_stamp_tool'):
            self.canvas.setMapTool(self.lu_stamp_tool)
            self.btn_lu_sample_stamp.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; padding: 6px; border-radius: 4px;")
            self.btn_lu_sample_stamp.setText("🎯 Klicke auf Karte (Nuance lernen)...")
            cid = self.combo_lu_sample_class.currentData()
            self.status_lbl.setText(f"📍 Klicke auf die Karte, um Farbmuster für '{cid}' zu lernen (Rechtsklick/Esc zum Beenden)...")

    def _finish_lu_stamp_sampling(self):
        """Finishes stamp sampling mode."""
        if hasattr(self, 'lu_stamp_tool') and self.canvas.mapTool() == self.lu_stamp_tool:
            self.canvas.unsetMapTool(self.lu_stamp_tool)
        if hasattr(self, 'btn_lu_sample_stamp'):
            self.btn_lu_sample_stamp.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
            self.btn_lu_sample_stamp.setText("🎯 Farbmuster auf Karte aufnehmen")
        self.status_lbl.setText("Farbmuster-Sampling beendet.")

    def _on_lu_stamp_sampled(self, point: QgsPointXY):
        """Callback when user clicks canvas to sample a color/texture nuance for a land use class."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            self.status_lbl.setText("Keine gültige Rasterkarte aktiv.")
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            return

        # Transform point to raster coordinates
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        layer_crs = layer.crs()
        if canvas_crs.isValid() and layer_crs.isValid() and canvas_crs != layer_crs:
            ct = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            layer_pt = ct.transform(point)
        else:
            layer_pt = point

        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            return

        gt = ds.GetGeoTransform()
        x_origin, px_w, _, y_origin, _, px_h = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
        px = (layer_pt.x() - x_origin) / px_w
        py = (layer_pt.y() - y_origin) / px_h
        img_w, img_h = ds.RasterXSize, ds.RasterYSize

        radius = int(self.slider_stamp_radius.value())
        ix, iy = int(round(px)), int(round(py))
        x0 = max(0, ix - radius - 5)
        x1 = min(img_w, ix + radius + 6)
        y0 = max(0, iy - radius - 5)
        y1 = min(img_h, iy + radius + 6)

        r_crop = ds.GetRasterBand(1).ReadAsArray(x0, y0, x1 - x0, y1 - y0)
        g_crop = ds.GetRasterBand(2).ReadAsArray(x0, y0, x1 - x0, y1 - y0)
        b_crop = ds.GetRasterBand(3).ReadAsArray(x0, y0, x1 - x0, y1 - y0)
        ds = None

        if r_crop is None or g_crop is None or b_crop is None:
            return

        crop_rgb = np.dstack((r_crop, g_crop, b_crop))
        local_cx = px - x0
        local_cy = py - y0

        target_cid = self.combo_lu_sample_class.currentData()
        if self.pipette_sampler is None:
            self.pipette_sampler = PipetteSampler()

        entry = self.pipette_sampler.sample_from_stamp(
            crop_rgb, target_cid, local_cx, local_cy, radius=radius
        )

        if entry:
            try:
                mupp = self.canvas.mapUnitsPerPixel()
                map_radius = max(mupp * 8.0, radius * abs(px_w))
                circle_geom = QgsGeometry.fromPointXY(point).buffer(map_radius, 24)

                rb = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
                class_color_map = {
                    "forest": QColor(39, 174, 96, 90),
                    "meadow": QColor(0, 206, 201, 90),
                    "water": QColor(9, 132, 227, 110),
                    "vineyard": QColor(214, 48, 49, 90),
                    "garden": QColor(253, 203, 110, 90),
                    "gravel": QColor(225, 112, 85, 90),
                }
                border_color_map = {
                    "forest": QColor(27, 94, 32, 255),
                    "meadow": QColor(0, 150, 140, 255),
                    "water": QColor(13, 71, 161, 255),
                    "vineyard": QColor(150, 30, 30, 255),
                    "garden": QColor(200, 160, 50, 255),
                    "gravel": QColor(180, 80, 50, 255),
                }
                rb.setFillColor(class_color_map.get(target_cid, QColor(255, 152, 0, 90)))
                rb.setStrokeColor(border_color_map.get(target_cid, QColor(255, 87, 34, 255)))
                rb.setWidth(2)
                rb.setToGeometry(circle_geom, None)
                rb.show()
                self.lu_stamp_rubber_bands.append((target_cid, rb))
            except Exception:
                pass

            self._update_lu_stamps_summary()
            total_stamps = sum(len(s.stamps) for s in self.pipette_sampler.samples.values())
            class_stamps = len(self.pipette_sampler.get_stamps(target_cid))
            self.status_lbl.setText(
                f"✓ Nuance #{entry.stamp_id} für '{target_cid}' erfasst ({entry.hex_color})! {class_stamps} Nuancen aktiv ({total_stamps} gesamt). Klicke weiter für weitere Nuancen."
            )

            # Re-trigger live preview
            if hasattr(self, 'chk_lu_live_preview') and self.chk_lu_live_preview.isChecked():
                self._trigger_lu_preview_update()

    def _update_lu_stamps_summary(self):
        """Updates the status text showing how many stamps are active per class."""
        if not hasattr(self, 'pipette_sampler') or self.pipette_sampler is None:
            return
        parts = []
        labels = {
            "forest": "🌲 Wald",
            "meadow": "🌱 Wiese",
            "water": "💧 Gewässer",
            "garden": "🏡 Gärten",
            "vineyard": "🍇 Weinberge",
            "gravel": "🏖️ Kies",
        }
        total = 0
        for cid, lbl in labels.items():
            cnt = len(self.pipette_sampler.get_stamps(cid))
            if cnt > 0:
                parts.append(f"{lbl}: {cnt}")
                total += cnt

        if parts:
            summary = " | ".join(parts) + f" (Gesamt: {total})"
            self.lbl_lu_stamps_summary.setText(f"🎨 Aktive Nuancen: {summary}")
            self.lbl_lu_stamps_summary.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
        else:
            self.lbl_lu_stamps_summary.setText("0 Nuancen gesampelt (Standard-Farbprofile aktiv)")
            self.lbl_lu_stamps_summary.setStyleSheet("color: #888888; font-size: 10px;")

    def _clear_class_stamps(self):
        """Clears all sampled stamps for the currently selected land use class."""
        if not hasattr(self, 'pipette_sampler') or self.pipette_sampler is None:
            return
        target_cid = self.combo_lu_sample_class.currentData()
        self.pipette_sampler.clear_stamps(target_cid)

        # Remove rubber bands for this class
        remaining_bands = []
        for cid, rb in self.lu_stamp_rubber_bands:
            if cid == target_cid:
                rb.reset(QgsWkbTypes.PolygonGeometry)
                rb.hide()
            else:
                remaining_bands.append((cid, rb))
        self.lu_stamp_rubber_bands = remaining_bands

        self._update_lu_stamps_summary()
        self.status_lbl.setText(f"Nuancen für '{target_cid}' geleert. Standard-Profil wieder aktiv.")
        if hasattr(self, 'chk_lu_live_preview') and self.chk_lu_live_preview.isChecked():
            self._trigger_lu_preview_update()

    def _reset_all_stamps(self):
        """Resets all classes back to their historical defaults."""
        if not hasattr(self, 'pipette_sampler') or self.pipette_sampler is None:
            return
        self.pipette_sampler._init_defaults()
        for _, rb in self.lu_stamp_rubber_bands:
            rb.reset(QgsWkbTypes.PolygonGeometry)
            rb.hide()
        self.lu_stamp_rubber_bands.clear()
        self._update_lu_stamps_summary()
        self.status_lbl.setText("Alle Farbnuancen zurückgesetzt auf Standard-Profile.")
        if hasattr(self, 'chk_lu_live_preview') and self.chk_lu_live_preview.isChecked():
            self._trigger_lu_preview_update()

    def _on_lu_live_preview_toggled(self, checked: bool):
        if checked:
            self._trigger_lu_preview_update()
        else:
            for l in QgsProject.instance().mapLayersByName("🔍 Land Use (Live Preview)"):
                QgsProject.instance().removeMapLayer(l.id())
            self.canvas.refresh()

    def _trigger_lu_preview_update(self):
        """Debounced live preview trigger for land use."""
        if hasattr(self, 'chk_lu_live_preview') and self.chk_lu_live_preview.isChecked():
            if hasattr(self, 'lu_preview_timer'):
                self.lu_preview_timer.start(200)

    def _update_lu_live_preview(self):
        """Generates live categorized vector polygons for land use on the canvas."""
        if not hasattr(self, 'chk_lu_live_preview') or not self.chk_lu_live_preview.isChecked():
            return

        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            return

        if self.rb_lu_roi.isChecked():
            if self.current_lu_roi is None or self.current_lu_roi.isEmpty():
                return
            if self.cached_lu_roi_rgb is None:
                self._cache_lu_roi_data()
        else:
            if self.cached_lu_roi_rgb is None:
                self._cache_lu_roi_data()

        if self.cached_lu_roi_rgb is None:
            return

        image_rgb = self.cached_lu_roi_rgb
        sub_x, sub_y = self.cached_lu_sub_origin
        px_w, px_h = self.cached_lu_px_size

        enabled_cats = []
        if self.chk_lu_forest.isChecked(): enabled_cats.append("forest")
        if self.chk_lu_meadow.isChecked(): enabled_cats.append("meadow")
        if self.chk_lu_water.isChecked(): enabled_cats.append("water")
        if self.chk_lu_garden.isChecked(): enabled_cats.append("garden")
        if self.chk_lu_vineyard.isChecked(): enabled_cats.append("vineyard")
        if hasattr(self, 'chk_lu_gravel') and self.chk_lu_gravel.isChecked(): enabled_cats.append("gravel")

        if not enabled_cats or self.pipette_sampler is None:
            return

        # Update sampler parameters from sliders
        tol_val = int(self.slider_lu_tol.value())
        min_area_val = float(self.slider_lu_min_area.value())
        tex_w_val = float(self.slider_lu_tex_w.value()) / 10.0

        for s in self.pipette_sampler.samples.values():
            s.tolerance = tol_val
            s.min_area_px = min_area_val
            s.texture_weight = tex_w_val

        try:
            polys_by_class = self.pipette_sampler.extract_competitive_polygons(
                image_rgb, active_class_ids=enabled_cats
            )
        except Exception as e:
            self.status_lbl.setText(f"Vorschaufehler: {e}")
            return

        layer_name = "🔍 Land Use (Live Preview)"
        existing = QgsProject.instance().mapLayersByName(layer_name)
        if existing:
            vl_prev = existing[0]
        else:
            crs_auth = layer.crs().authid() if layer.crs().isValid() else "EPSG:25832"
            vl_prev = QgsVectorLayer(f"Polygon?crs={crs_auth}", layer_name, "memory")
            pr = vl_prev.dataProvider()
            pr.addAttributes([
                QgsField("id", QVariant.Int),
                QgsField("category", QVariant.String),
                QgsField("category_label", QVariant.String),
                QgsField("area_m2", QVariant.Double),
            ])
            vl_prev.updateFields()

            categories = [
                ("forest", "Forest (Wald)", "46,125,50,150", "27,94,32,230"),
                ("meadow", "Meadow (Wiesen/Weiden)", "0,206,201,150", "0,150,140,230"),
                ("water", "Water Body (Gewässer)", "9,132,227,180", "13,71,161,240"),
                ("garden", "Gardens (Gärten)", "253,203,110,150", "200,160,50,230"),
                ("vineyard", "Vineyard (Weinberge)", "214,48,49,150", "150,30,30,230"),
                ("gravel", "Gravel (Kies/Sand)", "225,112,85,150", "180,80,50,230"),
            ]
            cats = []
            for cat_val, cat_lbl, fill_col, border_col in categories:
                sym = QgsFillSymbol.createSimple({
                    'color': fill_col,
                    'color_border': border_col,
                    'width_border': '0.4',
                    'style': 'solid',
                    'style_border': 'solid'
                })
                cats.append(QgsRendererCategory(cat_val, sym, cat_lbl))

            renderer = QgsCategorizedSymbolRenderer("category", cats)
            vl_prev.setRenderer(renderer)
            QgsProject.instance().addMapLayer(vl_prev, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, vl_prev)

        pr_prev = vl_prev.dataProvider()
        pr_prev.deleteFeatures(vl_prev.allFeatureIds())

        new_feats = []
        feat_id = 1
        m2_per_px2 = abs(px_w * px_h)

        for cat, plist in polys_by_class.items():
            for p in plist:
                if p is None or p.is_empty:
                    continue
                try:
                    coords = [(sub_x + pt[0] * px_w, sub_y + pt[1] * px_h) for pt in p.exterior.coords]
                    holes = [[(sub_x + pt[0] * px_w, sub_y + pt[1] * px_h) for pt in ring.coords] for ring in p.interiors]
                    pts_ext = [QgsPointXY(x, y) for x, y in coords]
                    pts_holes = [[QgsPointXY(x, y) for x, y in h] for h in holes] if holes else []
                    geom_qgs = QgsGeometry.fromPolygonXY([pts_ext] + pts_holes)
                    if not geom_qgs.isGeosValid():
                        geom_qgs = geom_qgs.makeValid()

                    f = QgsFeature(vl_prev.fields())
                    f.setGeometry(geom_qgs)
                    f.setAttribute("id", feat_id)
                    f.setAttribute("category", cat)
                    f.setAttribute("category_label", cat.capitalize())
                    f.setAttribute("area_m2", round(float(p.area) * m2_per_px2, 1))
                    new_feats.append(f)
                    feat_id += 1
                except Exception:
                    pass

        if new_feats:
            pr_prev.addFeatures(new_feats)

        vl_prev.updateExtents()
        vl_prev.triggerRepaint()
        self.canvas.refresh()
        self.status_lbl.setText(f"🔍 Land Use Live-Vorschau: {len(new_feats)} Flächen erkannt.")

    def _run_landuse_extraction(self):
        """Launches the LandUseExtractionTask in background thread."""
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "No Raster Selected", "Please select a historical GeoTIFF first.")
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            QMessageBox.critical(self, "File Error", f"Raster file does not exist:\n{raster_path}")
            return

        # Determine enabled categories
        enabled_cats = []
        if self.chk_lu_forest.isChecked():
            enabled_cats.append("forest")
        if self.chk_lu_meadow.isChecked():
            enabled_cats.append("meadow")
        if self.chk_lu_water.isChecked():
            enabled_cats.append("water")
        if self.chk_lu_garden.isChecked():
            enabled_cats.append("garden")
        if self.chk_lu_vineyard.isChecked():
            enabled_cats.append("vineyard")
        if hasattr(self, 'chk_lu_gravel') and self.chk_lu_gravel.isChecked():
            enabled_cats.append("gravel")

        if not enabled_cats:
            QMessageBox.information(
                self,
                "No Categories Selected",
                "Please select at least one land-use category to extract."
            )
            return

        # Check ROI vs full sheet
        use_roi = self.rb_lu_roi.isChecked()
        if use_roi:
            if self.current_lu_roi is not None and not self.current_lu_roi.isEmpty():
                layer_roi = self._transform_canvas_to_layer_rect(self.current_lu_roi, layer)
                layer_geom = self._transform_canvas_to_layer_geom(self.current_lu_roi_geom, layer) if hasattr(self, 'current_lu_roi_geom') and self.current_lu_roi_geom is not None else None
            elif hasattr(self, 'current_roi') and self.current_roi is not None and not self.current_roi.isEmpty():
                layer_roi = self._transform_canvas_to_layer_rect(self.current_roi, layer)
                layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer) if hasattr(self, 'current_roi_geom') and self.current_roi_geom is not None else None
            else:
                QMessageBox.information(
                    self,
                    "Kein Gebiet ausgewählt",
                    "Bitte zeichne zuerst ein Polygon mit '📐 Landnutzungs-Polygon einzeichnen' oder wähle 'Gesamtblatt'."
                )
                return
        else:
            layer_roi = None
            layer_geom = None

        config = LandUseConfig() if LandUseConfig else None
        if config:
            config.enable_forest = "forest" in enabled_cats
            config.enable_meadow = "meadow" in enabled_cats
            config.enable_water = "water" in enabled_cats
            config.enable_garden = "garden" in enabled_cats
            config.enable_vineyard = "vineyard" in enabled_cats

        # Apply current slider values to pipette_sampler
        if self.pipette_sampler is not None:
            tol_val = int(self.slider_lu_tol.value())
            min_area_val = float(self.slider_lu_min_area.value())
            tex_w_val = float(self.slider_lu_tex_w.value()) / 10.0
            for s in self.pipette_sampler.samples.values():
                s.tolerance = tol_val
                s.min_area_px = min_area_val
                s.texture_weight = tex_w_val

        output_crs = layer.crs().authid() if layer.crs().isValid() else "EPSG:25832"
        layer_name = f"🌲 Land Use ({layer.name()})"

        # Start QgsTask
        self.btn_extract_landuse.setEnabled(False)
        self.progress_bar.setValue(5)
        self.status_lbl.setText("Classifying historical land-use categories...")

        self.current_task = LandUseExtractionTask(
            raster_path=raster_path,
            config=config,
            enabled_categories=enabled_cats,
            roi_extent=layer_roi,
            roi_geometry=layer_geom,
            layer_name=layer_name,
            output_crs=output_crs,
            pipette_sampler=self.pipette_sampler,
        )
        self.current_task.task_completed.connect(self._on_landuse_task_completed)
        self.current_task.task_failed.connect(self._on_landuse_task_failed)

        QgsApplication.taskManager().addTask(self.current_task)

    def _on_landuse_task_completed(self, layer_name: str, count: int):
        self.btn_extract_landuse.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText(f"Done! {count} land-use parcels classified in '{layer_name}'.")

        # Remove preview vector layer now that permanent extraction is finished
        for l in QgsProject.instance().mapLayersByName("🔍 Land Use (Live Preview)"):
            QgsProject.instance().removeMapLayer(l.id())

        if count == 0:
            QMessageBox.information(
                self, "No Land-Use Features Found",
                "No parcels of the selected categories were detected in this area."
            )

    def _on_landuse_task_failed(self, error_msg: str):
        self.btn_extract_landuse.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Classification Error", f"Land-use extraction failed:\n{error_msg}")

    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            "HistMap Extractor (Tranchot)",
            "<b>HistMap Extractor</b><br><br>"
            "Bonn Center for Digital Humanities (BCDH)<br>"
            "Rheinische Friedrich-Wilhelms-Universität Bonn<br><br>"
            "Deep Learning based extraction of historical features from the Tranchot/v. Müffling maps (1801-1828)."
        )

    def _run_text_extraction(self):
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "No Raster Selected", "Please select a historical GeoTIFF first.")
            return

        raster_path = layer.dataProvider().dataSourceUri()
        if not os.path.exists(raster_path):
            QMessageBox.critical(self, "File Error", f"Raster file does not exist:\n{raster_path}")
            return
            
        if self.current_roi is None or self.current_roi.isEmpty():
            QMessageBox.information(
                self,
                "No Area Selected",
                "Please draw a polygon using '📐 Draw Settlement Polygon' in the Buildings tab first."
            )
            return

        if TextConfig is None or TextExtractionTask is None:
            QMessageBox.critical(self, "Engine Error", "Tranchot backend not found or incompatible.")
            return

        layer_roi = self._transform_canvas_to_layer_rect(self.current_roi, layer)
        layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer) if hasattr(self, 'current_roi_geom') and self.current_roi_geom is not None else None

        config = TextConfig()
        task = TextExtractionTask(
            raster_path=raster_path,
            config=config,
            roi_extent=layer_roi,
            roi_geometry=layer_geom,
            layer_name="Tranchot_Toponyms",
            output_crs=layer.crs().authid()
        )
        task.task_completed.connect(self._on_text_extraction_completed)
        task.task_failed.connect(self._on_text_extraction_failed)

        QgsApplication.taskManager().addTask(task)
        self.status_lbl.setText("Running text & toponym extraction...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #0277bd; }")

    def _on_text_extraction_completed(self, layer_name: str, count: int):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("")
        self.status_lbl.setText(f"Text Extraction Finished! Extracted {count} labels.")
        
        if count > 0:
            layer = QgsProject.instance().mapLayersByName(layer_name)
            if layer:
                self.iface.setActiveLayer(layer[0])

    def _on_text_extraction_failed(self, error_msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("")
        self.status_lbl.setText("Text extraction failed.")
        QMessageBox.critical(self, "Extraction Error", f"Text extraction failed:\n{error_msg}")


    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            "About HistMap Extractor",
            "<h3>🗺️ HistMap Extractor QGIS Plugin</h3>"
            "<p><b>Version:</b> 1.1.0<br>"
            "<b>Developed by:</b> Bonn Center for Digital Humanities (BCDH), University of Bonn<br>"
            "<b>License:</b> MIT License</p>"
            "<p>Precision AI & computer vision vectorization of historical topographical, "
            "cadastral, and regional map sheets (18th–20th century).</p>"
        )

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()
