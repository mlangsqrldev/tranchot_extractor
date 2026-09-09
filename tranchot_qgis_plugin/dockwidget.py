"""
Dockable widget panel for the Tranchot Extractor QGIS Plugin.
Provides full parameter control, interactive ROI selection, and progress monitoring.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
import os
import sys
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
    QgsWkbTypes
)

from .map_tools import PolygonRoiMapTool, RoiExtentMapTool, PipetteMapTool
from .tasks import BuildingExtractionTask, LandUseExtractionTask


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
    from tranchot_extractor.config import BuildingConfig, RoadConfig, LandUseConfig
    from tranchot_extractor.extractors.building_extractor import BuildingExtractor
    from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
except Exception:
    BuildingConfig = None
    RoadConfig = None
    LandUseConfig = None
    BuildingExtractor = None
    ColorEnhancer = None


class TranchotDockWidget(QDockWidget):
    """
    Main dockable control panel for Tranchot historical map extraction.
    """

    closingPlugin = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__("Tranchot Extractor", parent)
        self.iface = iface
        self.canvas = self.iface.mapCanvas()
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.current_roi: Optional[QgsRectangle] = None
        self.roi_tool: Optional[RoiExtentMapTool] = None
        self.pipette_tool: Optional[PipetteMapTool] = None
        self.current_task: Optional[BuildingExtractionTask] = None
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
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs Widget
        self.tabs = QTabWidget()
        self._create_building_tab()
        self._create_landuse_tab()
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

            slider = QSlider(Qt.Horizontal)
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
        sep.setFrameShape(QFrame.HLine)
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

        # 1. Interactive Pipette Tool
        pip_group = QGroupBox("Interactive Color Pipette")
        pip_layout = QVBoxLayout(pip_group)

        self.btn_pipette = QPushButton("🎯 Activate Pipette (Click on map)")
        self.btn_pipette.setToolTip("Click on the map canvas to inspect pixel RGB values and sample colors.")
        self.btn_pipette.clicked.connect(self._activate_pipette_tool)
        pip_layout.addWidget(self.btn_pipette)

        self.pip_status_lbl = QLabel("No sample picked.")
        self.pip_status_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        pip_layout.addWidget(self.pip_status_lbl)
        layout.addWidget(pip_group)

        # 2. Scope Selection
        scope_group = QGroupBox("Analysis Scope")
        scope_layout = QVBoxLayout(scope_group)
        self.rb_lu_roi = QRadioButton("Selected Settlement / ROI Polygon", checked=True)
        self.rb_lu_roi.setToolTip("Restricts land-use classification strictly to the active settlement ROI polygon.")
        self.rb_lu_sheet = QRadioButton("Full Map Sheet Extent")
        self.rb_lu_sheet.setToolTip("Classifies the entire historical GeoTIFF map sheet.")
        scope_layout.addWidget(self.rb_lu_roi)
        scope_layout.addWidget(self.rb_lu_sheet)
        layout.addWidget(scope_group)

        # 3. Category Configuration
        lu_group = QGroupBox("Land-Use Classes to Extract")
        lu_layout = QVBoxLayout(lu_group)
        
        self.chk_lu_forest = QCheckBox("🌲 Forest (Wald - Olive-green & canopy texture)")
        self.chk_lu_forest.setChecked(True)
        self.chk_lu_forest.setToolTip("Extracts contiguous forest tracts using green chromatic excess and canopy engraving variance.")
        lu_layout.addWidget(self.chk_lu_forest)

        self.chk_lu_meadow = QCheckBox("🌱 Meadow (Wiesen/Weiden - Alluvial valley wash)")
        self.chk_lu_meadow.setChecked(True)
        self.chk_lu_meadow.setToolTip("Extracts alluvial valley meadows via cyan-pastel wash minus buffered stream network.")
        lu_layout.addWidget(self.chk_lu_meadow)

        self.chk_lu_water = QCheckBox("💧 Water Bodies (Gewässer/Bäche/Teiche/Gräben)")
        self.chk_lu_water.setChecked(True)
        self.chk_lu_water.setToolTip("Extracts linear stream channels, mill canals, town moats, and standing water bodies.")
        lu_layout.addWidget(self.chk_lu_water)

        self.chk_lu_garden = QCheckBox("🏡 Gardens & Orchards (Gärten/Baumgärten)")
        self.chk_lu_garden.setChecked(True)
        self.chk_lu_garden.setToolTip("Extracts soft green garden plots and orchards surrounding settlement cores.")
        lu_layout.addWidget(self.chk_lu_garden)

        self.chk_lu_vineyard = QCheckBox("🍇 Vineyards / Arable (Weinberge/Ackerland)")
        self.chk_lu_vineyard.setChecked(True)
        self.chk_lu_vineyard.setToolTip("Extracts warm ochre/brown slope parcels and terraced vineyards.")
        lu_layout.addWidget(self.chk_lu_vineyard)

        layout.addWidget(lu_group)

        # 4. Action Button
        self.btn_extract_landuse = QPushButton("🌲 Classify & Vectorize Land Use")
        self.btn_extract_landuse.setToolTip(
            "Launches land-use classification. The resulting categorized vector layer "
            "is automatically added to QGIS with distinct historical map colors."
        )
        self.btn_extract_landuse.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_extract_landuse.clicked.connect(self._run_landuse_extraction)
        layout.addWidget(self.btn_extract_landuse)

        layout.addStretch(1)
        self.tabs.addTab(tab, "🌲 Land Use")


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

    def _activate_roi_tool(self):
        """Switches QGIS map tool to irregular polygon drawing mode."""
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
            self.roi_rubber_band.setLineStyle(Qt.DashLine)
        else:
            self.roi_rubber_band.setFillColor(QColor(0, 0, 0, 0))
            self.roi_rubber_band.setStrokeColor(QColor(0, 180, 216, 240))
            self.roi_rubber_band.setWidth(2)
            self.roi_rubber_band.setLineStyle(Qt.DashLine)
        self.roi_rubber_band.setToGeometry(geom, None)
        self.roi_rubber_band.show()

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
                        f.setAttribute("type", "Courtyard Complex" if len(poly.interiors) > 0 else "Building")
                        f.setAttribute("area_m2", round(float(feat.area_px) * m2_per_px2, 2))
                        f.setAttribute("perimeter_m", round(float(feat.perimeter_px) * abs(px_w), 2))
                        f.setAttribute("compactness", round(float(feat.compactness), 3))
                        f.setAttribute("orientation", round(float(feat.orientation_deg), 1))
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
            if norm_layers:
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

        self.status_lbl.setText("Computing global parchment white-balance for map sheet...")
        QgsApplication.processEvents()

        norm_layer_name = f"🎨 {layer.name()} (Normalized)"
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        cache_dir = os.path.join(tempfile.gettempdir(), "tranchot_norm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, f"{base_name}_norm.tif")

        try:
            if not os.path.exists(cached_file):
                ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
                if ds is None:
                    self.status_lbl.setText("GDAL could not open raster file.")
                    return

                w = ds.RasterXSize
                h = ds.RasterYSize
                gt = ds.GetGeoTransform()
                proj = ds.GetProjection()

                driver = gdal.GetDriverByName("GTiff")
                options = ["COMPRESS=LZW", "TILED=YES"]
                out_ds = driver.Create(cached_file, w, h, 3, gdal.GDT_Byte, options=options)
                out_ds.SetGeoTransform(gt)
                if proj:
                    out_ds.SetProjection(proj)

                # 1. Estimate uniform parchment tone from full sheet thumbnail to guarantee 100% seamless tiles
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
                        current_tile += 1
                        if current_tile % 4 == 0 or current_tile == total_tiles:
                            pct = int((current_tile / total_tiles) * 100)
                            self.progress_bar.setValue(pct)
                            self.status_lbl.setText(f"Normalizing map sheet: {pct}% ({current_tile}/{total_tiles} tiles)...")
                            QgsApplication.processEvents()

                out_ds.FlushCache()
                out_ds = None
                ds = None

            self.progress_bar.setValue(100)
            norm_layer = QgsRasterLayer(cached_file, norm_layer_name)
            if norm_layer.isValid():
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
            if self.current_roi is None or self.current_roi.isEmpty():
                QMessageBox.information(
                    self,
                    "No Area Selected",
                    "Please draw a polygon using '📐 Draw Settlement Polygon' first, or select 'Full Map Sheet Extent'."
                )
                return
            layer_roi = self._transform_canvas_to_layer_rect(self.current_roi, layer)
            layer_geom = self._transform_canvas_to_layer_geom(self.current_roi_geom, layer) if hasattr(self, 'current_roi_geom') and self.current_roi_geom is not None else None
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
        )
        self.current_task.task_completed.connect(self._on_landuse_task_completed)
        self.current_task.task_failed.connect(self._on_landuse_task_failed)

        QgsApplication.taskManager().addTask(self.current_task)

    def _on_landuse_task_completed(self, layer_name: str, count: int):
        self.btn_extract_landuse.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText(f"Done! {count} land-use parcels classified in '{layer_name}'.")
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
