import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import glob
import time
import zipfile
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import cv2
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsPolygonItem, QGraphicsSimpleTextItem, QGraphicsRectItem, QGraphicsItem,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox, QFileDialog,
    QFrame, QScrollArea, QSplitter, QStatusBar, QProgressBar,
    QToolButton, QMenu, QMessageBox, QDialog, QLineEdit, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF, Signal, QObject, QThread
from PySide6.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QFont, QPainter,
    QCursor, QPolygonF, QIcon, QAction, QSurfaceFormat, QPainterPath
)
from shapely.geometry import Polygon, Point, LineString, mapping, box
from shapely.affinity import translate
import geopandas as gpd

from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
from tranchot_extractor.extractors.pipette_sampler import PipetteSampler, ColorSample, StampEntry
from tranchot_extractor.extractors.building_extractor import BuildingExtractor, BuildingConfig
from tranchot_extractor.extractors.sam_extractor import sam_extractor, SAMExtractor
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.spatial_gazetteer import SpatialGazetteer

# Style themes
LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #f1f5f9;
    color: #0f172a;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}
QFrame.card {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px;
}
QFrame.card_header {
    background-color: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    font-weight: bold;
    color: #0f172a;
    padding: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    color: #0f172a;
}
QPushButton:hover {
    background-color: #e2e8f0;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background-color: #cbd5e1;
}
QPushButton:checked {
    background-color: #0284c7;
    border-color: #0369a1;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_primary {
    background-color: #16a34a;
    border-color: #16a34a;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_primary:hover {
    background-color: #15803d;
}
QPushButton.btn_danger {
    background-color: #dc2626;
    border-color: #ef4444;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_danger:hover {
    background-color: #b91c1c;
}
QPushButton.btn_glow {
    background-color: #4f46e5;
    border-color: #4338ca;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_glow:hover {
    background-color: #4338ca;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #0284c7;
    border: 2px solid #ffffff;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QStatusBar {
    background-color: #ffffff;
    border-top: 1px solid #cbd5e1;
    color: #64748b;
}
"""

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #090d16;
    color: #f8fafc;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}
QFrame.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
}
QFrame.card_header {
    background-color: #182234;
    border-bottom: 1px solid #334155;
    font-weight: bold;
    color: #f8fafc;
    padding: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    color: #f8fafc;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_primary {
    background-color: #22c55e;
    border-color: #22c55e;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_primary:hover {
    background-color: #16a34a;
}
QPushButton.btn_danger {
    background-color: #ef4444;
    border-color: #dc2626;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_danger:hover {
    background-color: #dc2626;
}
QPushButton.btn_glow {
    background-color: #6366f1;
    border-color: #4f46e5;
    color: #ffffff;
    font-weight: bold;
}
QPushButton.btn_glow:hover {
    background-color: #4f46e5;
}
QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    color: #f8fafc;
    min-height: 24px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    border: 2px solid #ffffff;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QStatusBar {
    background-color: #0f172a;
    border-top: 1px solid #334155;
    color: #94a3b8;
}
"""

CLASS_SPECS = {
    "forest": {"label": "🌲 Wald", "color": "#16a34a", "qt_color": QColor(22, 163, 74)},
    "meadow": {"label": "🌿 Wiese / Weide", "color": "#0891b2", "qt_color": QColor(8, 145, 178)},
    "water": {"label": "💧 Gewässer / See", "color": "#2563eb", "qt_color": QColor(37, 99, 235)},
    "vineyard": {"label": "🍇 Weinberge", "color": "#9333ea", "qt_color": QColor(147, 51, 234)},
    "gravel": {"label": "🟠 Kies / Sand", "color": "#ea580c", "qt_color": QColor(234, 88, 12)},
    "garden": {"label": "🟨 Gärten", "color": "#ca8a04", "qt_color": QColor(202, 138, 4)},
    "building": {"label": "🏛️ Gebäude", "color": "#ef4444", "qt_color": QColor(239, 68, 68)},
    "settlement": {"label": "🏘️ Siedlungen (Orte)", "color": "#f59e0b", "qt_color": QColor(245, 158, 11)},
}


class SettlementDialog(QDialog):
    """
    Modal Dialog to Name a newly drawn settlement, search GeoNames/OSM,
    and link the centroid Point with Name & GeoNames ID.
    """
    def __init__(self, parent, default_name: str, default_id: str, lat: Optional[float], lon: Optional[float], title: str = "🏘️ Siedlung benennen & GeoNames Verknüpfung"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(540, 480)
        self.setMinimumSize(460, 400)
        self.lat = lat
        self.lon = lon
        self.result_name = default_name
        self.result_geonames_id = default_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # Header
        lbl_h = QLabel(f"<b><font size='4'>{title}</font></b>")
        layout.addWidget(lbl_h)

        geo_txt = f"📍 Zentroid: {lat:.5f}°N, {lon:.5f}°E (WGS84)" if (lat and lon) else "📍 Zentroid-Punkt im Zentrum des gezeichneten Rahmens"
        lbl_sub = QLabel(f"<font color='#0284c7'><b>{geo_txt}</b></font>")
        layout.addWidget(lbl_sub)

        # Name Field
        layout.addWidget(QLabel("<b>Name der Siedlung (Ort / Dorf):</b>"))
        self.edit_name = QLineEdit(default_name)
        self.edit_name.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px;")
        layout.addWidget(self.edit_name)

        # GeoNames ID Field
        layout.addWidget(QLabel("<b>GeoNames / OSM ID:</b>"))
        self.edit_id = QLineEdit(default_id)
        self.edit_id.setStyleSheet("font-size: 13px; padding: 5px;")
        layout.addWidget(self.edit_id)

        # Search Action Button
        btn_search = QPushButton("🔍 In GeoNames & OSM nach Ort suchen")
        btn_search.clicked.connect(self._do_search)
        layout.addWidget(btn_search)

        # Results List
        layout.addWidget(QLabel("<b>Vorschläge (anklicken zum Auswählen):</b>"))
        self.list_results = QListWidget()
        self.list_results.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_results)

        # Initial populate
        self._populate_results(SpatialGazetteer.search_settlement(default_name, lat=self.lat, lon=self.lon))

        # Bottom Buttons
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("💾 Speichern & Gebäude extrahieren (OK)")
        btn_ok.setProperty("class", "btn_primary")
        btn_ok.setFixedHeight(38)
        btn_ok.clicked.connect(self._on_ok)

        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.setFixedHeight(38)
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def _do_search(self):
        query = self.edit_name.text().strip()
        matches = SpatialGazetteer.search_settlement(query, lat=self.lat, lon=self.lon)
        self._populate_results(matches)

    def _populate_results(self, matches: List[Dict[str, Any]]):
        self.list_results.clear()
        if not matches:
            it = QListWidgetItem("Keine Treffer in der Datenbank. Name und ID können frei eingetragen werden.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_results.addItem(it)
            return

        for m in matches:
            name = m.get("name", "")
            gn_id = str(m.get("geonames_id", ""))
            state = m.get("state", "Rheinland")
            it = QListWidgetItem(f"📍 {name}  (GeoNames ID: {gn_id}) — {state}")
            it.setData(Qt.ItemDataRole.UserRole, (name, gn_id))
            self.list_results.addItem(it)

    def _on_item_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            name, gn_id = data
            self.edit_name.setText(name)
            self.edit_id.setText(gn_id)

    def _on_ok(self):
        self.result_name = self.edit_name.text().strip()
        self.result_geonames_id = self.edit_id.text().strip()
        self.accept()


class TranchotMapView(QGraphicsView):
    """Lightning-fast Hardware-Accelerated Map View with Stamp, Box, Polygon, Add Building, Select & Pan tools."""
    stamp_clicked = Signal(float, float)
    box_selected = Signal(float, float, float, float)
    polygon_selected = Signal(list)
    manual_building_created = Signal(float, float, float, float)
    manual_building_clicked = Signal(float, float)
    sam_prompt_requested = Signal(float, float, float, float, bool)  # min_x, min_y, max_x, max_y, is_point
    building_clicked = Signal(float, float, bool)  # x, y, is_shift
    building_box_selected = Signal(float, float, float, float, bool)  # min_x, min_y, max_x, max_y, is_shift
    building_delete_requested = Signal()
    building_context_menu_requested = Signal(float, float, QPoint)
    cursor_moved = Signal(int, int)
    radius_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QBrush(QColor("#e2e8f0")))

        self.current_tool = "stamp"  # 'stamp', 'box', 'polygon', 'pan'
        self.active_stamp_radius = 22
        self.active_class_color = QColor(22, 163, 74)
        self.is_panning = False
        self.pan_start_pos = None

        # Interactive tools
        self.cursor_ring_item = None
        self.box_start_pos = None
        self.preview_box_item = None
        self.polygon_pts: List[Tuple[float, float]] = []
        self.preview_poly_item: Optional[QGraphicsPolygonItem] = None

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _update_polygon_preview(self, current_pt: QPointF):
        if not self.scene() or not self.polygon_pts:
            return
        all_pts = [QPointF(x, y) for x, y in self.polygon_pts] + [current_pt]
        qpoly = QPolygonF(all_pts)
        if not self.preview_poly_item:
            pen = QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(245, 158, 11, 40))
            self.preview_poly_item = self.scene().addPolygon(qpoly, pen, brush)
            self.preview_poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.preview_poly_item.setZValue(50)
        else:
            self.preview_poly_item.setPolygon(qpoly)

    def _clear_polygon_preview(self):
        self.polygon_pts.clear()
        if self.preview_poly_item and self.scene():
            self.scene().removeItem(self.preview_poly_item)
            self.preview_poly_item = None

    def set_stamp_color(self, color: QColor):
        self.active_class_color = color
        if self.cursor_ring_item:
            self._update_cursor_ring_style()

    def set_stamp_radius(self, r: int):
        self.active_stamp_radius = max(2, min(120, r))
        if self.cursor_ring_item and self.cursor_ring_item.isVisible():
            rect = self.cursor_ring_item.rect()
            cx, cy = rect.center().x(), rect.center().y()
            r = self.active_stamp_radius
            self.cursor_ring_item.setRect(cx - r, cy - r, r * 2, r * 2)

    def _ensure_cursor_ring(self):
        if not self.cursor_ring_item and self.scene():
            self.cursor_ring_item = self.scene().addEllipse(0, 0, self.active_stamp_radius * 2, self.active_stamp_radius * 2)
            self.cursor_ring_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.cursor_ring_item.setZValue(100)
            self._update_cursor_ring_style()

    def _update_cursor_ring_style(self):
        if self.cursor_ring_item:
            pen = QPen(self.active_class_color, 1.5, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(self.active_class_color.red(), self.active_class_color.green(), self.active_class_color.blue(), 60))
            self.cursor_ring_item.setPen(pen)
            self.cursor_ring_item.setBrush(brush)

    def wheelEvent(self, event):
        """Ctrl/Shift+Wheel adjusts stamp radius down to 2px, regular wheel zooms."""
        mods = event.modifiers()
        if (mods & Qt.KeyboardModifier.ControlModifier) or (mods & Qt.KeyboardModifier.ShiftModifier) or self.current_tool == "stamp_resize":
            delta = 1 if abs(event.angleDelta().y()) < 120 else 2
            if event.angleDelta().y() > 0:
                self.active_stamp_radius = min(120, self.active_stamp_radius + delta)
            else:
                self.active_stamp_radius = max(2, self.active_stamp_radius - delta)
            
            self.radius_changed.emit(self.active_stamp_radius)
            scene_pt = self.mapToScene(event.position().toPoint() if hasattr(event, "position") else event.pos())
            self._ensure_cursor_ring()
            if self.cursor_ring_item:
                r = self.active_stamp_radius
                self.cursor_ring_item.setRect(scene_pt.x() - r, scene_pt.y() - r, r * 2, r * 2)
                self.cursor_ring_item.setVisible(self.current_tool == "stamp")
            event.accept()
            return

        # Regular zoom
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.80
        curr_scale = self.transform().m11()
        if (curr_scale * factor < 0.05) or (curr_scale * factor > 40.0):
            return
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        scene_pt = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == "stamp":
                self.stamp_clicked.emit(scene_pt.x(), scene_pt.y())
                event.accept()
                return
            elif self.current_tool == "polygon":
                if len(self.polygon_pts) >= 3:
                    start_pt = self.polygon_pts[0]
                    dist = ((scene_pt.x() - start_pt[0]) ** 2 + (scene_pt.y() - start_pt[1]) ** 2) ** 0.5
                    if dist <= 18:
                        pts = list(self.polygon_pts)
                        self._clear_polygon_preview()
                        self.polygon_selected.emit(pts)
                        event.accept()
                        return
                self.polygon_pts.append((scene_pt.x(), scene_pt.y()))
                self._update_polygon_preview(scene_pt)
                event.accept()
                return
            elif self.current_tool == "box":
                self.box_start_pos = scene_pt
                if self.preview_box_item and self.scene():
                    self.scene().removeItem(self.preview_box_item)
                pen = QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(245, 158, 11, 40))
                self.preview_box_item = self.scene().addRect(scene_pt.x(), scene_pt.y(), 0, 0, pen, brush)
                self.preview_box_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.preview_box_item.setZValue(50)
                event.accept()
                return
            elif self.current_tool == "add_building":
                self.box_start_pos = scene_pt
                if self.preview_box_item and self.scene():
                    self.scene().removeItem(self.preview_box_item)
                pen = QPen(QColor("#ef4444"), 2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(239, 68, 68, 60))
                self.preview_box_item = self.scene().addRect(scene_pt.x(), scene_pt.y(), 0, 0, pen, brush)
                self.preview_box_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.preview_box_item.setZValue(50)
                event.accept()
                return
            elif self.current_tool == "select_building":
                self.box_start_pos = scene_pt
                if self.preview_box_item and self.scene():
                    self.scene().removeItem(self.preview_box_item)
                pen = QPen(QColor("#0284c7"), 2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(2, 132, 199, 45))
                self.preview_box_item = self.scene().addRect(scene_pt.x(), scene_pt.y(), 0, 0, pen, brush)
                self.preview_box_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.preview_box_item.setZValue(50)
                event.accept()
                return
            elif self.current_tool == "pan":
                self.is_panning = True
                self.pan_start_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            if self.current_tool == "polygon":
                if len(self.polygon_pts) >= 3:
                    pts = list(self.polygon_pts)
                    self._clear_polygon_preview()
                    self.polygon_selected.emit(pts)
                else:
                    self._clear_polygon_preview()
                event.accept()
                return
            gpos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.building_context_menu_requested.emit(scene_pt.x(), scene_pt.y(), gpos)
            event.accept()
            return
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.current_tool == "polygon" and len(self.polygon_pts) >= 3:
            pts = list(self.polygon_pts)
            self._clear_polygon_preview()
            self.polygon_selected.emit(pts)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        scene_pt = self.mapToScene(event.pos())
        self.cursor_moved.emit(int(scene_pt.x()), int(scene_pt.y()))

        # Update hover circle
        if self.current_tool == "stamp":
            self._ensure_cursor_ring()
            if self.cursor_ring_item:
                r = self.active_stamp_radius
                self.cursor_ring_item.setRect(scene_pt.x() - r, scene_pt.y() - r, r * 2, r * 2)
                self.cursor_ring_item.setVisible(True)
        else:
            if self.cursor_ring_item:
                self.cursor_ring_item.setVisible(False)

        # Update polygon preview
        if self.current_tool == "polygon" and self.polygon_pts:
            self._update_polygon_preview(scene_pt)

        # Update box preview
        if self.current_tool in ("box", "add_building", "select_building") and self.box_start_pos and self.preview_box_item:
            x1, y1 = self.box_start_pos.x(), self.box_start_pos.y()
            x2, y2 = scene_pt.x(), scene_pt.y()
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            self.preview_box_item.setRect(rx, ry, rw, rh)

        if self.is_panning and self.pan_start_pos:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        scene_pt = self.mapToScene(event.pos())
        if self.current_tool == "add_building" and self.box_start_pos:
            x1, y1 = self.box_start_pos.x(), self.box_start_pos.y()
            x2, y2 = scene_pt.x(), scene_pt.y()
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            if self.preview_box_item and self.scene():
                self.scene().removeItem(self.preview_box_item)
                self.preview_box_item = None
            self.box_start_pos = None
            if rw <= 5 and rh <= 5:
                # Single click -> detect building at point with spectral extractor
                self.manual_building_clicked.emit(scene_pt.x(), scene_pt.y())
            else:
                # Drag box -> add building box directly
                self.manual_building_created.emit(rx, ry, rx + rw, ry + rh)
            event.accept()
            return

        if self.current_tool == "select_building" and self.box_start_pos:
            x1, y1 = self.box_start_pos.x(), self.box_start_pos.y()
            x2, y2 = scene_pt.x(), scene_pt.y()
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            if self.preview_box_item and self.scene():
                self.scene().removeItem(self.preview_box_item)
                self.preview_box_item = None
            self.box_start_pos = None
            is_shift = bool(event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
            if rw > 6 and rh > 6:
                # Drag box -> box multi-selection
                self.building_box_selected.emit(rx, ry, rx + rw, ry + rh, is_shift)
            else:
                # Click -> single or shift-toggle selection
                self.building_clicked.emit(scene_pt.x(), scene_pt.y(), is_shift)
            event.accept()
            return

        if self.current_tool == "box" and self.box_start_pos:
            x1, y1 = self.box_start_pos.x(), self.box_start_pos.y()
            x2, y2 = scene_pt.x(), scene_pt.y()
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            if self.preview_box_item and self.scene():
                self.scene().removeItem(self.preview_box_item)
                self.preview_box_item = None
            self.box_start_pos = None
            if rw > 15 and rh > 15:
                self.box_selected.emit(rx, ry, rx + rw, ry + rh)

        if self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            cursor_map = {
                "stamp": Qt.CursorShape.CrossCursor,
                "box": Qt.CursorShape.CrossCursor,
                "polygon": Qt.CursorShape.CrossCursor,
                "add_building": Qt.CursorShape.CrossCursor,
                "select_building": Qt.CursorShape.PointingHandCursor,
                "pan": Qt.CursorShape.OpenHandCursor,
            }
            self.setCursor(cursor_map.get(self.current_tool, Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._clear_polygon_preview()
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.current_tool == "polygon" and self.polygon_pts:
                self.polygon_pts.pop()
                if self.polygon_pts:
                    scene_pt = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
                    self._update_polygon_preview(scene_pt)
                else:
                    self._clear_polygon_preview()
                event.accept()
                return
            else:
                self.building_delete_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        if self.cursor_ring_item:
            self.cursor_ring_item.setVisible(False)
        super().leaveEvent(event)


class TranchotStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏛️ Tranchot Extractor — Native High-Speed Studio (OpenGL 60 FPS)")
        self.resize(1500, 920)

        # Core State
        self.raw_image_rgb: Optional[np.ndarray] = None
        self.enhanced_image_rgb: Optional[np.ndarray] = None
        self.current_file_path: Optional[str] = None
        self.geo_meta: Dict[str, Any] = {}
        self.sampler: PipetteSampler = PipetteSampler()
        self.active_class: str = "forest"
        self.stamp_radius: int = 22
        self.dark_mode: bool = False

        # Vector graphics items
        self.scene = QGraphicsScene(self)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.stamp_graphics_items: List[QGraphicsItem] = []
        self.layer_polygon_items: Dict[str, List[QGraphicsPolygonItem]] = {
            cid: [] for cid in CLASS_SPECS
        }
        self.extracted_polygons_by_class: Dict[str, List[Polygon]] = {}

        # Building place linkage & selection state
        self.building_metadata: Dict[int, Dict[str, Any]] = {}
        self.building_badges: Dict[int, QGraphicsSimpleTextItem] = {}
        self.selected_building_idx: Optional[int] = None
        self.selected_building_indices: Set[int] = set()
        self.selected_highlight_items: Dict[int, QGraphicsItem] = {}
        self.sam_extractor = sam_extractor

        self._setup_ui()
        self.apply_theme(False)
        self._load_presets()

    def _setup_ui(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Navigation Toolbar with Integrated Toolstrip
        top_bar = QFrame(self)
        top_bar.setFixedHeight(54)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 4, 10, 4)
        top_layout.setSpacing(6)

        lbl_logo = QLabel("🏛️ <b>Tranchot Studio</b> <span style='background:#0284c7;color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;'>OPENGL</span>")
        top_layout.addWidget(lbl_logo)

        top_layout.addSpacing(6)
        top_layout.addWidget(QLabel("🗺️ Blatt:"))
        self.combo_presets = QComboBox()
        self.combo_presets.setMinimumWidth(190)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        top_layout.addWidget(self.combo_presets)

        btn_open = QPushButton("📁")
        btn_open.setToolTip("Eigenes GeoTIFF öffnen")
        btn_open.clicked.connect(self._open_file_dialog)
        top_layout.addWidget(btn_open)

        # Separator to tools
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        top_layout.addWidget(sep1)

        # Werkzeuge Toolbar (horizontal)
        self.btn_tool_stamp = QPushButton("🖌️ Stempel")
        self.btn_tool_stamp.setCheckable(True)
        self.btn_tool_stamp.setChecked(True)
        self.btn_tool_stamp.setToolTip("Farbpigmente stempeln (Wald, Wiese, Wasser...)")
        self.btn_tool_stamp.clicked.connect(lambda: self._set_active_tool("stamp"))
        top_layout.addWidget(self.btn_tool_stamp)

        self.btn_tool_box = QPushButton("📦 Siedlung (Box)")
        self.btn_tool_box.setCheckable(True)
        self.btn_tool_box.setToolTip("Rechteck über Siedlung ziehen & mit GeoNames verknüpfen")
        self.btn_tool_box.clicked.connect(lambda: self._set_active_tool("box"))
        top_layout.addWidget(self.btn_tool_box)

        self.btn_tool_poly = QPushButton("📐 Siedlung (Poly)")
        self.btn_tool_poly.setCheckable(True)
        self.btn_tool_poly.setToolTip("Freies Polygon um Siedlung ziehen & mit GeoNames verknüpfen (Doppel-/Rechtsklick schließt)")
        self.btn_tool_poly.clicked.connect(lambda: self._set_active_tool("polygon"))
        top_layout.addWidget(self.btn_tool_poly)

        self.btn_tool_add_building = QPushButton("➕ Gebäude")
        self.btn_tool_add_building.setCheckable(True)
        self.btn_tool_add_building.setToolTip("Gebäude manuell zeichnen (Box ziehen) oder per Klick spektral erkennen")
        self.btn_tool_add_building.clicked.connect(lambda: self._set_active_tool("add_building"))
        top_layout.addWidget(self.btn_tool_add_building)

        self.btn_tool_select = QPushButton("👆 Auswählen")
        self.btn_tool_select.setCheckable(True)
        self.btn_tool_select.setToolTip("Polygone auswählen (Box ziehen oder Klick, Shift zum Hinzufügen, Entf zum Löschen)")
        self.btn_tool_select.clicked.connect(lambda: self._set_active_tool("select_building"))
        top_layout.addWidget(self.btn_tool_select)

        self.btn_tool_pan = QPushButton("🖐️ Pan")
        self.btn_tool_pan.setCheckable(True)
        self.btn_tool_pan.setToolTip("Karte verschieben (Pan)")
        self.btn_tool_pan.clicked.connect(lambda: self._set_active_tool("pan"))
        top_layout.addWidget(self.btn_tool_pan)

        # Separator to actions
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        top_layout.addWidget(sep2)

        btn_fit = QPushButton("🔍 Fit")
        btn_fit.setToolTip("Gesamtansicht")
        btn_fit.clicked.connect(self._fit_view)
        top_layout.addWidget(btn_fit)

        self.btn_theme = QPushButton("☀️")
        self.btn_theme.setToolTip("Design umschalten (Hell / Dunkel)")
        self.btn_theme.clicked.connect(self._toggle_theme)
        top_layout.addWidget(self.btn_theme)

        top_layout.addStretch()

        btn_norm = QPushButton("🪄 Normierung")
        btn_norm.setProperty("class", "btn_glow")
        btn_norm.clicked.connect(self._apply_100pct_normalization)
        top_layout.addWidget(btn_norm)

        btn_bldgs = QPushButton("🏛️ Gebäude")
        btn_bldgs.setProperty("class", "btn_danger")
        btn_bldgs.clicked.connect(self._run_building_extraction)
        top_layout.addWidget(btn_bldgs)

        btn_extract = QPushButton("⚡ Flächen")
        btn_extract.setProperty("class", "btn_primary")
        btn_extract.clicked.connect(self._run_landuse_extraction)
        top_layout.addWidget(btn_extract)

        # Export Button with Menu
        self.btn_export = QPushButton("💾 Export ▾")
        export_menu = QMenu(self)
        act_geojson = export_menu.addAction("🗺️ GeoJSON (.geojson)")
        act_geojson.triggered.connect(lambda: self._export_gis("geojson"))
        act_shp = export_menu.addAction("📦 ESRI Shapefile (.zip)")
        act_shp.triggered.connect(lambda: self._export_gis("shapefile_zip"))
        act_gpkg = export_menu.addAction("🗄️ GeoPackage (.gpkg)")
        act_gpkg.triggered.connect(lambda: self._export_gis("gpkg"))
        self.btn_export.setMenu(export_menu)
        top_layout.addWidget(self.btn_export)

        root_layout.addWidget(top_bar)

        # 2. Main Center Area (Splitter: View -> Right Sidebar)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Center OpenGL Map View (Left toolbar removed!)
        self.view = TranchotMapView(self)
        self.view.setScene(self.scene)
        self.view.stamp_clicked.connect(self._on_canvas_stamp_clicked)
        self.view.box_selected.connect(self._on_canvas_box_selected)
        self.view.polygon_selected.connect(self._on_canvas_polygon_selected)
        self.view.manual_building_created.connect(self._on_manual_building_created)
        self.view.manual_building_clicked.connect(self._on_manual_building_clicked)
        self.view.sam_prompt_requested.connect(self._on_sam_prompt)
        self.view.building_clicked.connect(self._on_canvas_building_clicked)
        self.view.building_box_selected.connect(self._on_canvas_box_selected_buildings)
        self.view.building_delete_requested.connect(self._on_building_delete_requested)
        self.view.building_context_menu_requested.connect(self._on_building_context_menu)
        self.view.cursor_moved.connect(self._on_cursor_moved)
        self.view.radius_changed.connect(self._on_view_radius_changed)
        splitter.addWidget(self.view)

        # Right Control Sidebar
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(340)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # Section 0: Weißabgleich & Normierung
        card_enhance = QFrame()
        card_enhance.setProperty("class", "card")
        layout_enh = QVBoxLayout(card_enhance)
        layout_enh.addWidget(QLabel("<b>✨ 0. Weißabgleich & Normierung</b>"))

        btn_paper100 = QPushButton("🪄 100% Papier-Weiß & Schraffen-Schwarz")
        btn_paper100.clicked.connect(self._apply_100pct_normalization)
        layout_enh.addWidget(btn_paper100)

        # Sliders
        self.lbl_deyellow = QLabel("Entgilbung (Weißabgleich): 85%")
        self.slider_deyellow = QSlider(Qt.Orientation.Horizontal)
        self.slider_deyellow.setRange(0, 100)
        self.slider_deyellow.setValue(85)
        self.slider_deyellow.valueChanged.connect(lambda v: self.lbl_deyellow.setText(f"Entgilbung (Weißabgleich): {v}%"))
        layout_enh.addWidget(self.lbl_deyellow)
        layout_enh.addWidget(self.slider_deyellow)

        self.lbl_vibrance = QLabel("Farbbrillanz (Vibranz): 1.65x")
        self.slider_vibrance = QSlider(Qt.Orientation.Horizontal)
        self.slider_vibrance.setRange(100, 250)
        self.slider_vibrance.setValue(165)
        self.slider_vibrance.valueChanged.connect(lambda v: self.lbl_vibrance.setText(f"Farbbrillanz (Vibranz): {v/100:.2f}x"))
        layout_enh.addWidget(self.lbl_vibrance)
        layout_enh.addWidget(self.slider_vibrance)

        self.lbl_contrast = QLabel("Kontrast: 1.00x")
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(60, 180)
        self.slider_contrast.setValue(100)
        self.slider_contrast.valueChanged.connect(lambda v: self.lbl_contrast.setText(f"Kontrast: {v/100:.2f}x"))
        layout_enh.addWidget(self.lbl_contrast)
        layout_enh.addWidget(self.slider_contrast)

        self.chk_ink = QCheckBox("✒️ Schraffen tiefschwarz färben")
        self.chk_ink.setChecked(True)
        layout_enh.addWidget(self.chk_ink)

        btn_apply_filter = QPushButton("⚡ Bildfilter anwenden")
        btn_apply_filter.clicked.connect(self._apply_custom_enhancement)
        layout_enh.addWidget(btn_apply_filter)

        sidebar_layout.addWidget(card_enhance)

        # Section 1: Stempel & Farblernen
        card_stamps = QFrame()
        card_stamps.setProperty("class", "card")
        layout_stm = QVBoxLayout(card_stamps)
        layout_stm.addWidget(QLabel("<b>🌲 1. Stempel & Farblernen</b>"))

        # Class tabs grid
        self.class_buttons: Dict[str, QPushButton] = {}
        for cid, spec in CLASS_SPECS.items():
            if cid == "building":
                continue
            btn_c = QPushButton(spec["label"])
            btn_c.setCheckable(True)
            btn_c.clicked.connect(lambda _, c=cid: self._select_active_class(c))
            self.class_buttons[cid] = btn_c

        row1 = QHBoxLayout()
        row1.addWidget(self.class_buttons["forest"])
        row1.addWidget(self.class_buttons["meadow"])
        row2 = QHBoxLayout()
        row2.addWidget(self.class_buttons["water"])
        row2.addWidget(self.class_buttons["vineyard"])
        row3 = QHBoxLayout()
        row3.addWidget(self.class_buttons["gravel"])
        row3.addWidget(self.class_buttons["garden"])

        layout_stm.addLayout(row1)
        layout_stm.addLayout(row2)
        layout_stm.addLayout(row3)

        # Radius slider (Down to 2px)
        self.lbl_radius = QLabel("Stempel-Radius: 22 px (Mausrad-Zoom: 2–120 px)")
        self.slider_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_radius.setRange(2, 120)
        self.slider_radius.setValue(22)
        self.slider_radius.valueChanged.connect(self._on_radius_changed)
        layout_stm.addWidget(self.lbl_radius)
        layout_stm.addWidget(self.slider_radius)

        # Active class status box
        self.lbl_class_info = QLabel("🌲 Wald: 0 Stempel aktiv")
        layout_stm.addWidget(self.lbl_class_info)

        self._select_active_class("forest")

        btn_clear_class = QPushButton("🗑️ Stempel dieser Klasse leeren")
        btn_clear_class.clicked.connect(self._clear_active_class_stamps)
        layout_stm.addWidget(btn_clear_class)

        btn_run_ext = QPushButton("⚡ Gelernte Flächen berechnen")
        btn_run_ext.setProperty("class", "btn_primary")
        btn_run_ext.clicked.connect(self._run_landuse_extraction)
        layout_stm.addWidget(btn_run_ext)

        sidebar_layout.addWidget(card_stamps)

        # Section 2: Baukörper & Gebäude (90° Orthogonal & Hofanlagen)
        card_bldgs = QFrame()
        card_bldgs.setProperty("class", "card")
        layout_bld = QVBoxLayout(card_bldgs)
        layout_bld.addWidget(QLabel("<b>🏛️ 2. Baukörper & Hofanlagen (90° Ortho)</b>"))

        btn_poly_extract = QPushButton("📐 Polygon: Siedlung erfassen & GeoNames")
        btn_poly_extract.setProperty("class", "btn_glow")
        btn_poly_extract.setToolTip("Freies Polygon um Dorf/Siedlung zeichnen, Gebäude extrahieren und mit GeoNames verknüpfen")
        btn_poly_extract.clicked.connect(lambda: self._set_active_tool("polygon"))
        layout_bld.addWidget(btn_poly_extract)

        btn_box_extract = QPushButton("📦 Box ziehen: Siedlung erfassen & GeoNames")
        btn_box_extract.setToolTip("Rechteck über Dorf/Siedlung ziehen, Gebäude extrahieren und mit GeoNames verknüpfen")
        btn_box_extract.clicked.connect(lambda: self._set_active_tool("box"))
        layout_bld.addWidget(btn_box_extract)

        btn_run_bldgs = QPushButton("🏛️ Alle Gebäude im gesamten Blatt extrahieren")
        btn_run_bldgs.setProperty("class", "btn_danger")
        btn_run_bldgs.clicked.connect(self._run_building_extraction)
        layout_bld.addWidget(btn_run_bldgs)

        btn_add_bldg = QPushButton("➕ Gebäude manuell / Klick hinzufügen")
        btn_add_bldg.setToolTip("Klicke auf ein Gebäude zur spektralen Erkennung oder ziehe eine Box mit der Maus auf")
        btn_add_bldg.clicked.connect(lambda: self._set_active_tool("add_building"))
        layout_bld.addWidget(btn_add_bldg)

        btn_link_bldg = QPushButton("🔗 Mit Ort verknüpfen (GeoNames)")
        btn_link_bldg.setToolTip("Ausgewählte(s) Gebäude mit Ort/GeoNames verbinden")
        btn_link_bldg.clicked.connect(lambda: self._link_building_to_place())
        layout_bld.addWidget(btn_link_bldg)

        self.btn_del_bldg = QPushButton("🗑️ Ausgewählte(s) Gebäude löschen (Entf)")
        self.btn_del_bldg.setToolTip("Löscht alle aktuell ausgewählten Gebäude (per Klick oder Auswahlbox)")
        self.btn_del_bldg.clicked.connect(self._on_building_delete_requested)
        layout_bld.addWidget(self.btn_del_bldg)

        self.lbl_bldg_count = QLabel("🏛️ Extrahierte Gebäude: 0")
        layout_bld.addWidget(self.lbl_bldg_count)

        self.chk_ortho = QCheckBox("📐 90° Manhattan Begradigung (Höfe erhalten)")
        self.chk_ortho.setChecked(True)
        layout_bld.addWidget(self.chk_ortho)

        sidebar_layout.addWidget(card_bldgs)

        # Section 3: Ebenen & Deckkraft
        card_layers = QFrame()
        card_layers.setProperty("class", "card")
        layout_lay = QVBoxLayout(card_layers)
        layout_lay.addWidget(QLabel("<b>🗺️ 3. Vektor-Ebenen & Deckkraft</b>"))

        self.layer_chks: Dict[str, QCheckBox] = {}
        self.layer_sliders: Dict[str, QSlider] = {}

        for cid, spec in CLASS_SPECS.items():
            row = QHBoxLayout()
            chk = QCheckBox(spec["label"])
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _, c=cid: self._toggle_layer_visibility(c))
            self.layer_chks[cid] = chk

            slider_op = QSlider(Qt.Orientation.Horizontal)
            slider_op.setRange(0, 100)
            slider_op.setValue(85 if cid == "building" else 70)
            slider_op.setFixedWidth(65)
            slider_op.valueChanged.connect(lambda _, c=cid: self._update_layer_opacity(c))
            self.layer_sliders[cid] = slider_op

            btn_del = QPushButton("🗑️")
            btn_del.setToolTip(f"Nur Polygone für '{spec['label']}' löschen")
            btn_del.setFixedWidth(32)
            btn_del.clicked.connect(lambda _, c=cid: self._delete_single_layer(c))

            row.addWidget(chk)
            row.addWidget(slider_op)
            row.addWidget(btn_del)
            layout_lay.addLayout(row)

        btn_clear_all = QPushButton("🗑️ Alle Vektor-Ebenen leeren")
        btn_clear_all.clicked.connect(self._clear_all_layers)
        layout_lay.addWidget(btn_clear_all)

        sidebar_layout.addWidget(card_layers)
        sidebar_layout.addStretch()

        sidebar_scroll.setWidget(sidebar_content)
        splitter.addWidget(sidebar_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        root_layout.addWidget(splitter)

        # 4. Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.lbl_status_coords = QLabel("X: 0, Y: 0 px  |  Zoom: 100%")
        self.lbl_status_poly = QLabel("0 Flächen geladen")
        self.status_bar.addPermanentWidget(self.lbl_status_coords)
        self.status_bar.addPermanentWidget(self.lbl_status_poly)
        self.status_bar.showMessage("Bereit. Wähle ein Kartenblatt oder stempfele Flächen.")

    def apply_theme(self, dark: bool):
        self.dark_mode = dark
        self.setStyleSheet(DARK_THEME if dark else LIGHT_THEME)
        self.btn_theme.setText("🌙 Dunkel" if dark else "☀️ Hell")
        bg_col = QColor("#090d16" if dark else "#e2e8f0")
        self.scene.setBackgroundBrush(QBrush(bg_col))
        self.view.setBackgroundBrush(QBrush(bg_col))

    def _toggle_theme(self):
        self.apply_theme(not self.dark_mode)

    def _set_active_tool(self, tool: str):
        self.view.current_tool = tool
        self.view._clear_polygon_preview()
        self.btn_tool_stamp.setChecked(tool == "stamp")
        self.btn_tool_poly.setChecked(tool == "polygon")
        self.btn_tool_box.setChecked(tool == "box")
        self.btn_tool_add_building.setChecked(tool == "add_building")
        self.btn_tool_select.setChecked(tool == "select_building")
        self.btn_tool_pan.setChecked(tool == "pan")
        cursor_map = {
            "stamp": Qt.CursorShape.CrossCursor,
            "box": Qt.CursorShape.CrossCursor,
            "polygon": Qt.CursorShape.CrossCursor,
            "add_building": Qt.CursorShape.CrossCursor,
            "select_building": Qt.CursorShape.PointingHandCursor,
            "pan": Qt.CursorShape.OpenHandCursor,
        }
        self.view.setCursor(cursor_map.get(tool, Qt.CursorShape.ArrowCursor))
        if tool == "polygon":
            self.status_bar.showMessage("📐 Siedlungs-Polygon: Klicke Punkte um das Dorf. Doppel-/Rechtsklick schließt das Polygon.")
        elif tool == "box":
            self.status_bar.showMessage("📦 Siedlungs-Box: Ziehe mit gedrückter Maustaste ein Rechteck über die Siedlung.")
        elif tool == "add_building":
            self.status_bar.showMessage("➕ Gebäude hinzufügen: Klicke auf ein Gebäude zur spektralen Erkennung oder ziehe eine Box auf.")
        elif tool == "select_building":
            self.status_bar.showMessage("👆 Auswahl aktiv: Klicke auf Gebäude oder ziehe einen Rahmen (Shift für Mehrfachauswahl, Entf zum Löschen, 🔗 zum Verknüpfen).")
        elif tool == "stamp":
            self.status_bar.showMessage("🖌️ Stempel-Modus aktiv: Klicke auf Farbflächen, um Pigmente zu lernen.")
        else:
            self.status_bar.showMessage("🖐️ Pan-Modus aktiv: Verschieben Sie die Karte mit gedrückter Maustaste.")

    def _select_active_class(self, cid: str):
        self.active_class = cid
        for k, btn in self.class_buttons.items():
            btn.setChecked(k == cid)
        stamps_cnt = len(self.sampler.get_stamps(cid))
        spec = CLASS_SPECS.get(cid, {})
        lbl = spec.get("label", cid)
        self.lbl_class_info.setText(f"{lbl}: {stamps_cnt} Stempel aktiv")
        self.view.set_stamp_color(spec.get("qt_color", QColor(22, 163, 74)))

    def _on_radius_changed(self, val: int):
        self.stamp_radius = val
        self.view.set_stamp_radius(val)
        self.lbl_radius.setText(f"Stempel-Radius: {val} px (Mausrad-Zoom: 2–120 px)")

    def _on_view_radius_changed(self, val: int):
        self.stamp_radius = val
        self.slider_radius.blockSignals(True)
        self.slider_radius.setValue(val)
        self.slider_radius.blockSignals(False)
        self.lbl_radius.setText(f"Stempel-Radius: {val} px (Mausrad-Zoom: 2–120 px)")

    def _load_presets(self):
        self.combo_presets.clear()
        presets = []
        sample_dir = "Sample"
        if not os.path.exists(sample_dir):
            sample_dir = os.path.join("tranchot_extractor", "Sample")

        if os.path.exists(sample_dir):
            for f in glob.glob(os.path.join(sample_dir, "*.tif")) + glob.glob(os.path.join(sample_dir, "*.tiff")):
                presets.append(f)

        for p in presets:
            self.combo_presets.addItem(f"🗺️ {os.path.basename(p)}", p)

        if presets:
            self.combo_presets.setCurrentIndex(0)
            self._load_image_file(presets[0])

    def _on_preset_changed(self, idx: int):
        path = self.combo_presets.itemData(idx)
        if path and os.path.exists(path):
            self._load_image_file(path)

    def _open_file_dialog(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Kartenblatt laden", "", "Rasterkarten (*.tif *.tiff *.png *.jpg *.jpeg)")
        if fpath:
            self._load_image_file(fpath)

    def _load_image_file(self, fpath: str):
        self.current_file_path = fpath
        self.status_bar.showMessage(f"Lade {os.path.basename(fpath)}...")
        QApplication.processEvents()

        try:
            with Image.open(fpath) as pil_img:
                self.raw_image_rgb = np.array(pil_img.convert("RGB"))
        except Exception:
            img = cv2.imread(fpath)
            if img is None:
                QMessageBox.critical(self, "Fehler", f"Bild konnte nicht geladen werden:\n{fpath}")
                return
            self.raw_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = self.raw_image_rgb.shape[:2]
        self.geo_handler = GeoReferenceHandler(fpath) if fpath.lower().endswith(('.tif', '.tiff')) else None

        # Reset sampler, polygons and extracted layers
        self.sampler = PipetteSampler()
        for items in self.layer_polygon_items.values():
            for it in items:
                self.scene.removeItem(it)
            items.clear()
        self.extracted_polygons_by_class.clear()

        for it in self.stamp_graphics_items:
            self.scene.removeItem(it)
        self.stamp_graphics_items.clear()

        self.lbl_bldg_count.setText("🏛️ Extrahierte Gebäude: 0")
        self.lbl_status_poly.setText("0 Flächen geladen")
        self._select_active_class(self.active_class)

        # Apply default enhancement
        self._apply_100pct_normalization()

        self.status_bar.showMessage(f"🗺️ Geladen: {os.path.basename(fpath)} ({w}×{h} px)")

    def _update_scene_pixmap(self, img_rgb: np.ndarray):
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self.pixmap_item.setZValue(-10)
        else:
            self.pixmap_item.setPixmap(pixmap)

        self.scene.setSceneRect(0, 0, w, h)
        self._fit_view()

    def _fit_view(self):
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_100pct_normalization(self):
        if self.raw_image_rgb is None:
            return
        self.slider_deyellow.setValue(90)
        self.slider_vibrance.setValue(175)
        self.slider_contrast.setValue(100)
        self.chk_ink.setChecked(True)
        self._apply_custom_enhancement()

    def _apply_custom_enhancement(self):
        if self.raw_image_rgb is None:
            return
        self.status_bar.showMessage("✨ Wende Weißabgleich & Farbfilter an...")
        QApplication.processEvents()

        t0 = time.time()
        self.enhanced_image_rgb = ColorEnhancer.enhance_map(
            self.raw_image_rgb,
            deyellow_strength=self.slider_deyellow.value() / 100.0,
            vibrance=self.slider_vibrance.value() / 100.0,
            contrast=self.slider_contrast.value() / 100.0,
            ink_blackening=self.chk_ink.isChecked(),
        )
        self._update_scene_pixmap(self.enhanced_image_rgb)
        self.status_bar.showMessage(f"✅ Filter angewendet in {(time.time() - t0):.2f}s")

    def _on_canvas_stamp_clicked(self, x: float, y: float):
        if self.enhanced_image_rgb is None:
            return
        cid = self.active_class
        r = self.stamp_radius

        entry = self.sampler.sample_from_stamp(self.enhanced_image_rgb, cid, cx=x, cy=y, radius=r)
        if not entry:
            lbl = CLASS_SPECS.get(cid, {}).get("label", cid)
            self.status_bar.showMessage(f"⚠️ Kein Farbpigment für '{lbl}' an Position ({int(x)}, {int(y)}) gefunden.")
            return

        # Render stamp circle on canvas
        spec = CLASS_SPECS.get(cid, {})
        color = spec.get("qt_color", QColor(22, 163, 74))

        pen = QPen(color, 2)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 90))
        ellipse = self.scene.addEllipse(x - r, y - r, r * 2, r * 2, pen, brush)
        ellipse.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        ellipse.setZValue(5)

        # Text badge
        text = self.scene.addSimpleText(f"⭘ {spec.get('label', cid)} #{entry.stamp_id}", QFont("Segoe UI", 9, QFont.Weight.Bold))
        text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        text.setBrush(QBrush(QColor("#ffffff")))
        text.setPos(x - r, y - r - 16)
        text.setZValue(6)

        self.stamp_graphics_items.extend([ellipse, text])

        # Update status
        stamps_cnt = len(self.sampler.get_stamps(cid))
        self.lbl_class_info.setText(f"{spec.get('label', cid)}: {stamps_cnt} Stempel aktiv")
        self.status_bar.showMessage(f"🖌️ Stempel #{entry.stamp_id} ({spec.get('label', cid)}) erfasst! ({entry.distilled_pixels}/{entry.raw_pixels} px Pigment)")

    def _clear_active_class_stamps(self):
        cid = self.active_class
        self.sampler.clear_stamps(cid)
        self.sampler.reset_class(cid)
        for it in self.stamp_graphics_items:
            self.scene.removeItem(it)
        self.stamp_graphics_items.clear()
        self._select_active_class(cid)
        self.status_bar.showMessage(f"🗑️ Alle Stempel für {CLASS_SPECS.get(cid, {}).get('label', cid)} gelöscht.")

    def _on_canvas_box_selected(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """Extracts all buildings inside the drawn settlement box and opens GeoNames linking dialog."""
        if self.enhanced_image_rgb is None:
            return

        h, w = self.enhanced_image_rgb.shape[:2]
        bx1, by1 = max(0, int(min_x)), max(0, int(min_y))
        bx2, by2 = min(w, int(max_x)), min(h, int(max_y))
        if bx2 <= bx1 or by2 <= by1:
            return

        # 1. Georeference & GeoNames lookup
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        lat_lon = SpatialGazetteer.get_lat_lon(self.geo_handler, cx, cy)
        lat, lon = (lat_lon[0], lat_lon[1]) if lat_lon else (None, None)

        # Find best suggestion
        suggestions = SpatialGazetteer.search_settlement("", lat=lat, lon=lon)
        def_name = suggestions[0]["name"] if suggestions else "Siedlung"
        def_id = str(suggestions[0]["geonames_id"]) if suggestions else ""

        # Open Modal Dialog
        dlg = SettlementDialog(self, def_name, def_id, lat, lon)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        settlement_name = dlg.result_name or "Siedlung"
        geonames_id = dlg.result_geonames_id or ""

        # 2. Extract buildings inside box with Manhattan 90° regularization
        self.status_bar.showMessage(f"🏛️ Extrahiere Gebäude in '{settlement_name}'...")
        QApplication.processEvents()

        t0 = time.time()
        box_poly = Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])
        cfg = BuildingConfig(
            regularize_orthogonal=self.chk_ortho.isChecked(),
        )
        extractor = BuildingExtractor(cfg)
        res = extractor.extract(self.enhanced_image_rgb, settlement_boundaries=[box_poly])

        global_polys = [f.geometry for f in res.features if f.geometry.is_valid and not f.geometry.is_empty]

        # Add to building layer
        if "building" not in self.extracted_polygons_by_class:
            self.extracted_polygons_by_class["building"] = []
        start_idx = len(self.extracted_polygons_by_class["building"])
        self.extracted_polygons_by_class["building"].extend(global_polys)

        for i in range(len(global_polys)):
            self.building_metadata[start_idx + i] = {
                "settlement": settlement_name,
                "geonames_id": geonames_id
            }

        # Render building graphics items
        color = CLASS_SPECS["building"]["qt_color"]
        op = self.layer_sliders["building"].value() / 100.0 if "building" in self.layer_sliders else 0.85
        pen = QPen(color, 1.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))

        for p in global_polys:
            item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
            if item is not None:
                self.layer_polygon_items["building"].append(item)

        # Render settlement bounding box frame with label badge
        box_pen = QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine)
        box_brush = QBrush(QColor(245, 158, 11, 20))
        box_rect = self.scene.addRect(bx1, by1, bx2 - bx1, by2 - by1, box_pen, box_brush)
        box_rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        box_rect.setZValue(2)

        lbl_text = f"🏘️ {settlement_name}" + (f" (GeoNames: {geonames_id})" if geonames_id else "")
        badge = self.scene.addSimpleText(lbl_text, QFont("Segoe UI", 10, QFont.Weight.Bold))
        badge.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        badge.setBrush(QBrush(QColor("#d97706")))
        badge.setPos(bx1 + 4, by1 - 20)
        badge.setZValue(4)

        if "settlement" not in self.layer_polygon_items:
            self.layer_polygon_items["settlement"] = []
        self.layer_polygon_items["settlement"].extend([box_rect, badge])

        elapsed = time.time() - t0
        total_bldgs = len(self.extracted_polygons_by_class["building"])
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
        self.status_bar.showMessage(f"✅ Siedlung '{settlement_name}' mit {len(global_polys)} Gebäuden extrahiert & verknüpft in {elapsed:.2f}s (GeoNames ID: {geonames_id})")

    def _on_canvas_polygon_selected(self, pts: List[Tuple[float, float]]):
        """Extracts all buildings inside the drawn settlement polygon and links with GeoNames."""
        if self.enhanced_image_rgb is None or len(pts) < 3:
            return

        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not isinstance(poly, Polygon):
            return

        # 1. Georeference & GeoNames lookup from centroid
        cx, cy = poly.centroid.x, poly.centroid.y
        lat_lon = SpatialGazetteer.get_lat_lon(self.geo_handler, cx, cy)
        lat, lon = (lat_lon[0], lat_lon[1]) if lat_lon else (None, None)

        # Find best suggestion
        suggestions = SpatialGazetteer.search_settlement("", lat=lat, lon=lon)
        def_name = suggestions[0]["name"] if suggestions else "Siedlung"
        def_id = str(suggestions[0]["geonames_id"]) if suggestions else ""

        # Open Modal Dialog
        dlg = SettlementDialog(self, def_name, def_id, lat, lon)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        settlement_name = dlg.result_name or "Siedlung"
        geonames_id = dlg.result_geonames_id or ""

        # 2. Extract buildings inside polygon with Manhattan 90° regularization
        self.status_bar.showMessage(f"🏛️ Extrahiere Gebäude in '{settlement_name}'...")
        QApplication.processEvents()

        t0 = time.time()
        cfg = BuildingConfig(
            regularize_orthogonal=self.chk_ortho.isChecked(),
        )
        extractor = BuildingExtractor(cfg)
        res = extractor.extract(self.enhanced_image_rgb, settlement_boundaries=[poly])

        global_polys = [f.geometry for f in res.features if f.geometry.is_valid and not f.geometry.is_empty]

        # Add to building layer
        if "building" not in self.extracted_polygons_by_class:
            self.extracted_polygons_by_class["building"] = []
        start_idx = len(self.extracted_polygons_by_class["building"])
        self.extracted_polygons_by_class["building"].extend(global_polys)

        for i in range(len(global_polys)):
            self.building_metadata[start_idx + i] = {
                "settlement": settlement_name,
                "geonames_id": geonames_id
            }

        # Render building graphics items
        color = CLASS_SPECS["building"]["qt_color"]
        op = self.layer_sliders["building"].value() / 100.0 if "building" in self.layer_sliders else 0.85
        pen = QPen(color, 1.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))

        for p in global_polys:
            item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
            if item is not None:
                self.layer_polygon_items["building"].append(item)

        # Render settlement polygon frame with label badge
        poly_pen = QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine)
        poly_brush = QBrush(QColor(245, 158, 11, 20))
        qpoly_settlement = QPolygonF([QPointF(pt[0], pt[1]) for pt in poly.exterior.coords])
        settlement_item = self.scene.addPolygon(qpoly_settlement, poly_pen, poly_brush)
        settlement_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        settlement_item.setZValue(2)

        lbl_text = f"🏘️ {settlement_name}" + (f" (GeoNames: {geonames_id})" if geonames_id else "")
        badge = self.scene.addSimpleText(lbl_text, QFont("Segoe UI", 10, QFont.Weight.Bold))
        badge.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        badge.setBrush(QBrush(QColor("#d97706")))
        minx, miny, _, _ = poly.bounds
        badge.setPos(minx + 4, miny - 20)
        badge.setZValue(4)

        if "settlement" not in self.layer_polygon_items:
            self.layer_polygon_items["settlement"] = []
        self.layer_polygon_items["settlement"].extend([settlement_item, badge])

        elapsed = time.time() - t0
        total_bldgs = len(self.extracted_polygons_by_class["building"])
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
        self.status_bar.showMessage(f"✅ Siedlung '{settlement_name}' mit {len(global_polys)} Gebäuden extrahiert & verknüpft in {elapsed:.2f}s (GeoNames ID: {geonames_id})")

    def _add_polygon_with_holes_to_scene(
        self,
        poly: Polygon,
        pen: QPen,
        brush: QBrush,
        z_value: int = 3
    ) -> Optional[QGraphicsItem]:
        """
        Renders a Shapely polygon in the QGraphicsScene.
        If the polygon contains interior holes (such as courtyards / Hofanlagen),
        uses QPainterPath with OddEvenFill to guarantee that courtyards remain
        completely open, unshaded, and transparent.
        """
        if poly is None or not poly.is_valid or poly.is_empty:
            return None

        if len(poly.interiors) == 0:
            qpoly = QPolygonF([QPointF(pt[0], pt[1]) for pt in poly.exterior.coords])
            item = self.scene.addPolygon(qpoly, pen, brush)
        else:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in poly.exterior.coords]))
            for interior in poly.interiors:
                path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in interior.coords]))
            item = self.scene.addPath(path, pen, brush)

        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setZValue(z_value)
        return item

    def _run_building_extraction(self):
        """Runs 90-degree orthogonal building extractor with courtyard preservation."""
        if self.enhanced_image_rgb is None:
            return
        self.status_bar.showMessage("🏛️ Extrahiere 90°-orthogonale Gebäude & Hofanlagen...")
        QApplication.processEvents()

        t0 = time.time()
        cfg = BuildingConfig(
            regularize_orthogonal=self.chk_ortho.isChecked(),
        )
        extractor = BuildingExtractor(cfg)
        res = extractor.extract(self.enhanced_image_rgb)

        # Clear previous building items and badges
        for it in self.layer_polygon_items["building"]:
            self.scene.removeItem(it)
        self.layer_polygon_items["building"].clear()
        for b in self.building_badges.values():
            if b and self.scene:
                self.scene.removeItem(b)
        self.building_badges.clear()
        self.building_metadata.clear()
        self._clear_building_selection()

        polys = [f.geometry for f in res.features]
        self.extracted_polygons_by_class["building"] = polys

        color = CLASS_SPECS["building"]["qt_color"]
        op = self.layer_sliders["building"].value() / 100.0
        pen = QPen(color, 1.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))

        for p in polys:
            item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
            if item is not None:
                self.layer_polygon_items["building"].append(item)

        elapsed = time.time() - t0
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {len(polys)}")
        self.status_bar.showMessage(f"✅ {len(polys)} Gebäude extrahiert in {elapsed:.2f}s (90° Manhattan Begradigung)")

    def _run_landuse_extraction(self):
        if self.enhanced_image_rgb is None:
            return
        self.status_bar.showMessage("⚡ Berechne gelernten Flächenbestand über alle Klassen...")
        QApplication.processEvents()

        t0 = time.time()
        active_cids = [
            cid for cid, s in self.sampler.samples.items()
            if hasattr(s, "stamps") and len(s.stamps) > 0
        ]
        if not active_cids:
            active_cids = ["forest", "meadow", "water"]

        res_polys = self.sampler.extract_competitive_polygons(self.enhanced_image_rgb, active_class_ids=active_cids)
        for cid, polys in res_polys.items():
            self.extracted_polygons_by_class[cid] = polys

        # Clear previous graphics items (except buildings & settlements)
        for cid, items in self.layer_polygon_items.items():
            if cid in ("building", "settlement"):
                continue
            for it in items:
                self.scene.removeItem(it)
            items.clear()

        total_count = 0
        for cid, polys in res_polys.items():
            spec = CLASS_SPECS.get(cid, {})
            color = spec.get("qt_color", QColor(22, 163, 74))
            op = self.layer_sliders.get(cid, QSlider()).value() / 100.0 if cid in self.layer_sliders else 0.7
            pen = QPen(color, 1.5)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))

            for p in polys:
                item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=2)
                if item is not None:
                    self.layer_polygon_items[cid].append(item)
                    total_count += 1

        elapsed = time.time() - t0
        self.lbl_status_poly.setText(f"{total_count} Flächen extrahiert")
        self.status_bar.showMessage(f"✅ Fertig in {elapsed:.2f}s: {total_count} Flächen über alle Klassen berechnet!")

    def _delete_single_layer(self, cid: str):
        """Deletes all extracted polygons of a single class without affecting other classes."""
        if cid in self.layer_polygon_items:
            for it in self.layer_polygon_items[cid]:
                self.scene.removeItem(it)
            self.layer_polygon_items[cid].clear()
        
        if cid in self.extracted_polygons_by_class:
            self.extracted_polygons_by_class[cid].clear()

        if cid == "building":
            self.lbl_bldg_count.setText("🏛️ Extrahierte Gebäude: 0")
            for b in self.building_badges.values():
                if b and self.scene:
                    self.scene.removeItem(b)
            self.building_badges.clear()
            self.building_metadata.clear()
            self._clear_building_selection()

        total_count = sum(len(p) for p in self.extracted_polygons_by_class.values())
        self.lbl_status_poly.setText(f"{total_count} Flächen geladen")
        lbl = CLASS_SPECS.get(cid, {}).get("label", cid)
        self.status_bar.showMessage(f"🗑️ Alle Polygone für Ebene '{lbl}' gelöscht.")

    def _clear_all_layers(self):
        """Clears all vector polygon layers from the canvas."""
        for cid, items in self.layer_polygon_items.items():
            for it in items:
                self.scene.removeItem(it)
            items.clear()
        self.extracted_polygons_by_class.clear()
        for b in self.building_badges.values():
            if b and self.scene:
                self.scene.removeItem(b)
        self.building_badges.clear()
        self.building_metadata.clear()
        self._clear_building_selection()
        self.lbl_bldg_count.setText("🏛️ Extrahierte Gebäude: 0")
        self.lbl_status_poly.setText("0 Flächen geladen")
        self.status_bar.showMessage("🗑️ Alle Vektor-Ebenen geleert.")

    def _toggle_layer_visibility(self, cid: str):
        visible = self.layer_chks[cid].isChecked()
        for it in self.layer_polygon_items.get(cid, []):
            it.setVisible(visible)

    def _update_layer_opacity(self, cid: str):
        op = self.layer_sliders[cid].value() / 100.0
        spec = CLASS_SPECS.get(cid, {})
        color = spec.get("qt_color", QColor(22, 163, 74))
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))
        for it in self.layer_polygon_items.get(cid, []):
            it.setBrush(brush)

    @property
    def selected_highlight_item(self) -> Optional[QGraphicsItem]:
        if self.selected_building_idx in self.selected_highlight_items:
            return self.selected_highlight_items[self.selected_building_idx]
        return next(iter(self.selected_highlight_items.values())) if self.selected_highlight_items else None

    def _highlight_building(self, idx: int):
        """Draws a crisp gold highlight overlay around the selected building (single select)."""
        self.selected_building_idx = idx
        self.selected_building_indices = {idx}
        self._highlight_selected_buildings()

    def _highlight_selected_buildings(self):
        """Renders vibrant gold/amber contour highlight overlays for all selected buildings."""
        if not self.scene:
            return
        # Remove previous highlight items
        for it in self.selected_highlight_items.values():
            if it and self.scene:
                self.scene.removeItem(it)
        self.selected_highlight_items.clear()

        bldgs = self.extracted_polygons_by_class.get("building", [])
        for idx in list(self.selected_building_indices):
            if not (0 <= idx < len(bldgs)):
                continue
            poly = bldgs[idx]
            pen = QPen(QColor("#facc15"), 2.5, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(250, 204, 21, 55))
            item = self._add_polygon_with_holes_to_scene(poly, pen, brush, z_value=25)
            if item is not None:
                self.selected_highlight_items[idx] = item

        cnt = len(self.selected_building_indices)
        self.selected_building_idx = next(iter(self.selected_building_indices)) if cnt == 1 else None
        if hasattr(self, "btn_del_bldg"):
            if cnt > 1:
                self.btn_del_bldg.setText(f"🗑️ {cnt} Gebäude löschen (Entf)")
            elif cnt == 1:
                self.btn_del_bldg.setText("🗑️ Gebäude löschen (Entf)")
            else:
                self.btn_del_bldg.setText("🗑️ Ausgewählte(s) Gebäude löschen (Entf)")

        if cnt > 0:
            self.status_bar.showMessage(f"👆 {cnt} Gebäude ausgewählt (Entf zum Löschen, 🔗 zum Verknüpfen, Shift für Mehrfachauswahl)")

    def _clear_building_selection(self):
        """Clears all building selection highlights."""
        self.selected_building_idx = None
        self.selected_building_indices.clear()
        if hasattr(self, "selected_highlight_items") and self.scene:
            for it in self.selected_highlight_items.values():
                if it and self.scene:
                    self.scene.removeItem(it)
            self.selected_highlight_items.clear()
        if hasattr(self, "btn_del_bldg"):
            self.btn_del_bldg.setText("🗑️ Ausgewählte(s) Gebäude löschen (Entf)")

    def _on_canvas_building_clicked(self, x: float, y: float, is_shift: bool = False):
        """Selects clicked building (single click replaces, Shift+click toggles)."""
        click_pt = Point(x, y)
        bldgs = self.extracted_polygons_by_class.get("building", [])
        if not bldgs:
            return

        target_idx = None
        for idx, poly in enumerate(bldgs):
            if poly.contains(click_pt):
                target_idx = idx
                break
        if target_idx is None:
            min_d = 8.0
            for idx, poly in enumerate(bldgs):
                d = poly.distance(click_pt)
                if d < min_d:
                    min_d = d
                    target_idx = idx

        if target_idx is None:
            if not is_shift:
                self._clear_building_selection()
            return

        if is_shift:
            if target_idx in self.selected_building_indices:
                self.selected_building_indices.remove(target_idx)
            else:
                self.selected_building_indices.add(target_idx)
        else:
            self.selected_building_indices = {target_idx}

        self._highlight_selected_buildings()

    def _on_canvas_box_selected_buildings(self, min_x: float, min_y: float, max_x: float, max_y: float, is_shift: bool = False):
        """Selects all buildings intersecting the selection box."""
        box_geom = box(min_x, min_y, max_x, max_y)
        bldgs = self.extracted_polygons_by_class.get("building", [])
        if not bldgs:
            return

        matched = set()
        for idx, poly in enumerate(bldgs):
            if poly.intersects(box_geom):
                matched.add(idx)

        if is_shift:
            self.selected_building_indices.update(matched)
        else:
            self.selected_building_indices = matched

        self._highlight_selected_buildings()
        self.status_bar.showMessage(f"📦 {len(self.selected_building_indices)} Gebäude über Auswahlrahmen gewählt (Entf zum Löschen, 🔗 zum Verknüpfen)")

    def _on_sam_prompt(self, min_x: float, min_y: float, max_x: float, max_y: float, is_point: bool):
        """Segments building(s) with Meta SAM using a point click or box prompt."""
        if self.enhanced_image_rgb is None and self.raw_image_rgb is None:
            self.status_bar.showMessage("ℹ️ Bitte zuerst ein Kartenblatt laden.")
            return

        img = self.enhanced_image_rgb if self.enhanced_image_rgb is not None else self.raw_image_rgb
        self.status_bar.showMessage("⏳ Meta SAM segmentiert Gebäude...")
        QApplication.processEvents()

        try:
            if is_point:
                res = self.sam_extractor.segment_with_prompts(
                    img,
                    positive_points=[(min_x, min_y)],
                    granularity="compact",
                    orthogonalize=True
                )
            else:
                res = self.sam_extractor.segment_with_prompts(
                    img,
                    bounding_box=[min_x, min_y, max_x, max_y],
                    granularity="compact",
                    orthogonalize=True
                )
        except Exception as e:
            self.status_bar.showMessage(f"⚠️ SAM-Segmentierungsfehler: {e}")
            return

        polys = res.get("polygons", [])
        if not polys:
            self.status_bar.showMessage("ℹ️ SAM hat an dieser Position keinen eindeutigen Baukörper erkannt.")
            return

        if "building" not in self.extracted_polygons_by_class:
            self.extracted_polygons_by_class["building"] = []
        if "building" not in self.layer_polygon_items:
            self.layer_polygon_items["building"] = []

        new_indices = set()
        color = CLASS_SPECS["building"]["qt_color"]
        op = self.layer_sliders["building"].value() / 100.0 if "building" in self.layer_sliders else 0.85
        pen = QPen(color, 1.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))

        for p in polys:
            if p.is_valid and not p.is_empty:
                self.extracted_polygons_by_class["building"].append(p)
                new_idx = len(self.extracted_polygons_by_class["building"]) - 1
                new_indices.add(new_idx)
                item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
                if item is not None:
                    self.layer_polygon_items["building"].append(item)

        total_bldgs = len(self.extracted_polygons_by_class["building"])
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
        self.selected_building_indices = new_indices
        self._highlight_selected_buildings()
        iou = res.get("iou_score", 1.0)
        self.status_bar.showMessage(f"➕ {len(new_indices)} Gebäude mit SAM extrahiert (IoU: {iou:.2f})! (Entf zum Löschen, 🔗 zum Verknüpfen)")

    def _link_building_to_place(self, idx: Optional[int] = None):
        """Opens SettlementDialog for the selected building(s) and saves metadata."""
        bldgs = self.extracted_polygons_by_class.get("building", [])
        if not bldgs:
            return

        if idx is None:
            if self.selected_building_indices:
                idx = next(iter(self.selected_building_indices))
            else:
                self.status_bar.showMessage("ℹ️ Bitte zuerst ein Gebäude auswählen.")
                return

        if not (0 <= idx < len(bldgs)):
            return

        poly = bldgs[idx]
        cx, cy = poly.centroid.x, poly.centroid.y
        lat_lon = SpatialGazetteer.get_lat_lon(self.geo_handler, cx, cy)
        lat, lon = (lat_lon[0], lat_lon[1]) if lat_lon else (None, None)

        prev_meta = self.building_metadata.get(idx, {})
        def_name = prev_meta.get("settlement", "")
        def_id = str(prev_meta.get("geonames_id", ""))
        if not def_name:
            suggestions = SpatialGazetteer.search_settlement("", lat=lat, lon=lon)
            def_name = suggestions[0]["name"] if suggestions else "Gebäude"
            def_id = str(suggestions[0]["geonames_id"]) if suggestions else ""

        target_count = len(self.selected_building_indices) if idx in self.selected_building_indices else 1
        dlg_title = f"🏛️ {target_count} Gebäude mit Ort verbinden" if target_count > 1 else f"🏛️ Gebäude #{idx+1} mit Ort verbinden"
        dlg = SettlementDialog(self, def_name, def_id, lat, lon, title=dlg_title)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s_name = dlg.result_name or "Ort"
            s_id = dlg.result_geonames_id or ""

            target_indices = self.selected_building_indices if (idx in self.selected_building_indices and len(self.selected_building_indices) > 1) else {idx}
            for t_idx in target_indices:
                self.building_metadata[t_idx] = {
                    "settlement": s_name,
                    "geonames_id": s_id,
                }
                t_poly = bldgs[t_idx]
                self._add_or_update_building_badge(t_idx, s_name, s_id, t_poly.centroid.x, t_poly.centroid.y)

            self.status_bar.showMessage(f"✅ {len(target_indices)} Gebäude mit '{s_name}' (GeoNames ID: {s_id}) verknüpft!")

    def _add_or_update_building_badge(self, idx: int, name: str, geonames_id: str, cx: float, cy: float):
        """Creates or updates a persistent label badge over the linked building."""
        if idx in self.building_badges and self.building_badges[idx] and self.scene:
            self.scene.removeItem(self.building_badges[idx])
            del self.building_badges[idx]

        lbl_text = f"🏠 {name}" + (f" ({geonames_id})" if geonames_id else "")
        badge = self.scene.addSimpleText(lbl_text, QFont("Segoe UI", 8, QFont.Weight.Bold))
        badge.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        badge.setBrush(QBrush(QColor("#facc15")))
        badge.setPen(QPen(QColor("#0f172a"), 0.5))
        badge.setPos(cx - 15, cy - 16)
        badge.setZValue(15)
        self.building_badges[idx] = badge

    def _on_manual_building_created(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """Adds a manually drawn rectangle as a building polygon."""
        p = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
        if "building" not in self.extracted_polygons_by_class:
            self.extracted_polygons_by_class["building"] = []
        self.extracted_polygons_by_class["building"].append(p)
        new_idx = len(self.extracted_polygons_by_class["building"]) - 1

        color = CLASS_SPECS["building"]["qt_color"]
        op = self.layer_sliders["building"].value() / 100.0 if "building" in self.layer_sliders else 0.85
        pen = QPen(color, 1.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))
        item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
        if item is not None:
            if "building" not in self.layer_polygon_items:
                self.layer_polygon_items["building"] = []
            self.layer_polygon_items["building"].append(item)

        total_bldgs = len(self.extracted_polygons_by_class["building"])
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
        self._highlight_building(new_idx)
        self.status_bar.showMessage(f"➕ Gebäude #{total_bldgs} hinzugefügt! (Entf zum Löschen, 🔗 zum Verknüpfen)")

    def _on_manual_building_clicked(self, cx: float, cy: float):
        """Extracts the building at click position using the spectral BuildingExtractor (Beer-Lambert, no SAM)."""
        img = self.enhanced_image_rgb if self.enhanced_image_rgb is not None else self.raw_image_rgb
        if img is None:
            self._on_manual_building_created(cx - 10, cy - 8, cx + 10, cy + 8)
            return

        h, w = img.shape[:2]
        pad = 45
        x1, y1 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x2, y2 = min(w, int(cx + pad)), min(h, int(cy + pad))

        crop = img[y1:y2, x1:x2]
        cfg = BuildingConfig(
            regularize_orthogonal=True,
            min_building_area_px=6.0,
            min_stroke_width_px=1.0,
        )
        ext = BuildingExtractor(cfg)
        res = ext.extract(crop)

        click_local = Point(cx - x1, cy - y1)
        best_poly = None
        min_dist = 22.0

        for feat in res.features:
            poly = feat.geometry
            if poly.contains(click_local):
                best_poly = poly
                break
            d = poly.distance(click_local)
            if d < min_dist:
                min_dist = d
                best_poly = poly

        if best_poly is not None:
            p_global = translate(best_poly, xoff=x1, yoff=y1)
            p = p_global
            if "building" not in self.extracted_polygons_by_class:
                self.extracted_polygons_by_class["building"] = []
            self.extracted_polygons_by_class["building"].append(p)
            new_idx = len(self.extracted_polygons_by_class["building"]) - 1

            color = CLASS_SPECS["building"]["qt_color"]
            op = self.layer_sliders["building"].value() / 100.0 if "building" in self.layer_sliders else 0.85
            pen = QPen(color, 1.5)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), int(op * 255)))
            item = self._add_polygon_with_holes_to_scene(p, pen, brush, z_value=3)
            if item is not None:
                if "building" not in self.layer_polygon_items:
                    self.layer_polygon_items["building"] = []
                self.layer_polygon_items["building"].append(item)

            total_bldgs = len(self.extracted_polygons_by_class["building"])
            self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
            self._highlight_building(new_idx)
            self.status_bar.showMessage(f"➕ Gebäude #{total_bldgs} an Position ({int(cx)}, {int(cy)}) spektral erkannt!")
        else:
            self._on_manual_building_created(cx - 10, cy - 8, cx + 10, cy + 8)

    def _on_building_delete_requested(self):
        """Deletes all currently selected buildings in bulk."""
        if not self.selected_building_indices:
            self.status_bar.showMessage("ℹ️ Keine Gebäude ausgewählt. Klicke oder ziehe einen Rahmen um Gebäude, um sie zu löschen.")
            return

        sorted_indices = sorted(list(self.selected_building_indices), reverse=True)
        bldgs = self.extracted_polygons_by_class.get("building", [])
        items = self.layer_polygon_items.get("building", [])

        cnt_deleted = 0
        for idx in sorted_indices:
            if 0 <= idx < len(bldgs):
                bldgs.pop(idx)
                cnt_deleted += 1
            if 0 <= idx < len(items):
                item = items.pop(idx)
                if item and self.scene:
                    self.scene.removeItem(item)
            if idx in self.building_badges:
                badge = self.building_badges.pop(idx)
                if badge and self.scene:
                    self.scene.removeItem(badge)

        # Re-index remaining badges and metadata
        new_meta = {}
        new_badges = {}
        del_set = set(sorted_indices)
        shift = 0
        for old_idx in range(len(bldgs) + cnt_deleted):
            if old_idx in del_set:
                shift += 1
            else:
                new_idx = old_idx - shift
                if old_idx in self.building_metadata:
                    new_meta[new_idx] = self.building_metadata[old_idx]
                if old_idx in self.building_badges:
                    new_badges[new_idx] = self.building_badges[old_idx]

        self.building_metadata = new_meta
        self.building_badges = new_badges

        self._clear_building_selection()
        total_bldgs = len(bldgs)
        self.lbl_bldg_count.setText(f"🏛️ Extrahierte Gebäude: {total_bldgs}")
        self.status_bar.showMessage(f"🗑️ {cnt_deleted} Gebäude gelöscht. ({total_bldgs} verbleibend)")

    def _on_building_context_menu(self, x: float, y: float, screen_pos: QPoint):
        """Shows context menu for clicked building or canvas position."""
        click_pt = Point(x, y)
        bldgs = self.extracted_polygons_by_class.get("building", [])
        target_idx = None
        for idx, poly in enumerate(bldgs):
            if poly.contains(click_pt):
                target_idx = idx
                break
        if target_idx is None:
            min_d = 8.0
            for idx, poly in enumerate(bldgs):
                d = poly.distance(click_pt)
                if d < min_d:
                    min_d = d
                    target_idx = idx

        if target_idx is not None:
            if target_idx not in self.selected_building_indices:
                self.selected_building_indices = {target_idx}
                self._highlight_selected_buildings()

            cnt = len(self.selected_building_indices)
            menu = QMenu(self)
            txt_link = f"🔗 {cnt} Gebäude mit Ort verknüpfen (GeoNames)..." if cnt > 1 else f"🔗 Gebäude #{target_idx+1} mit Ort verknüpfen (GeoNames)..."
            act_link = menu.addAction(txt_link)
            act_link.triggered.connect(lambda: self._link_building_to_place(target_idx))
            txt_del = f"🗑️ {cnt} ausgewählte Gebäude löschen (Entf)" if cnt > 1 else f"🗑️ Gebäude #{target_idx+1} löschen (Entf)"
            act_del = menu.addAction(txt_del)
            act_del.triggered.connect(self._on_building_delete_requested)
            menu.exec(screen_pos)

    def _export_gis(self, fmt: str):
        if not self.extracted_polygons_by_class:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst auf '⚡ Flächen berechnen' oder '🏛️ Gebäude' klicken.")
            return

        features = []
        for cid, polys in self.extracted_polygons_by_class.items():
            spec = CLASS_SPECS.get(cid, {})
            for idx, p in enumerate(polys):
                if p.is_valid and not p.is_empty:
                    meta = self.building_metadata.get(idx, {}) if cid == "building" else {}
                    features.append({
                        "label": spec.get("label", cid),
                        "class_id": cid,
                        "area_px": round(p.area, 1),
                        "settlement": meta.get("settlement", ""),
                        "geonames_id": str(meta.get("geonames_id", "")),
                        "geometry": p
                    })

        if not features:
            return

        gdf = gpd.GeoDataFrame(features, geometry=[f["geometry"] for f in features])
        gdf.set_crs("EPSG:25832", inplace=True, allow_override=True)

        if fmt == "geojson":
            fpath, _ = QFileDialog.getSaveFileName(self, "GeoJSON exportieren", "tranchot_vektoren.geojson", "GeoJSON (*.geojson)")
            if fpath:
                gdf.to_file(fpath, driver="GeoJSON")
                self.status_bar.showMessage(f"✅ Exportiert nach: {fpath}")
        elif fmt == "gpkg":
            fpath, _ = QFileDialog.getSaveFileName(self, "GeoPackage exportieren", "tranchot_vektoren.gpkg", "GeoPackage (*.gpkg)")
            if fpath:
                gdf.to_file(fpath, layer="tranchot", driver="GPKG")
                self.status_bar.showMessage(f"✅ Exportiert nach: {fpath}")
        elif fmt == "shapefile_zip":
            fpath, _ = QFileDialog.getSaveFileName(self, "Shapefile ZIP exportieren", "tranchot_shapefiles.zip", "ZIP-Archiv (*.zip)")
            if fpath:
                import tempfile
                tdir = tempfile.mkdtemp()
                for label, group in gdf.groupby("label"):
                    sname = "".join(c for c in label if c.isalnum() or c in ('_', '-')).strip() or "layer"
                    group.to_file(os.path.join(tdir, f"{sname}.shp"), driver="ESRI Shapefile")
                with zipfile.ZipFile(fpath, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in glob.glob(os.path.join(tdir, "*")):
                        zf.write(f, os.path.basename(f))
                self.status_bar.showMessage(f"✅ Shapefile ZIP exportiert nach: {fpath}")

    def _on_cursor_moved(self, x: int, y: int):
        scale = self.view.transform().m11()
        self.lbl_status_coords.setText(f"X: {x}, Y: {y} px  |  Zoom: {int(scale*100)}%")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = TranchotStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
