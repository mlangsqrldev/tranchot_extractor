"""
Interactive map tools for the Tranchot Extractor QGIS Plugin.
Includes ROI extent selection tool and color pipette sampling tool.
"""

from typing import Callable, Optional
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.gui import QgsMapTool, QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import QgsPointXY, QgsRectangle, QgsWkbTypes, QgsGeometry


class PolygonRoiMapTool(QgsMapTool):
    """
    Map tool allowing the user to click point-by-point to draw an arbitrary irregular polygon
    (Siedlungs-Polygon / ROI) directly on the QGIS canvas with real-time rubberband feedback.
    - Left-Click: Add vertex
    - Mouse-Move: Live polygon preview to cursor
    - Right-Click or Double-Click: Finish and commit polygon
    - Backspace: Remove last vertex
    - Escape: Cancel drawing
    """

    polygon_selected = pyqtSignal(QgsGeometry)
    canceled = pyqtSignal()

    def __init__(self, canvas, on_polygon_callback: Optional[Callable[[QgsGeometry], None]] = None):
        super().__init__(canvas)
        self.canvas = canvas
        self.on_polygon_callback = on_polygon_callback
        self.points: list = []

        # Setup completely transparent interior with crisp high-contrast border
        self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        fill_color = QColor(0, 0, 0, 0)  # 100% transparent interior (no pink/red tint!)
        border_color = QColor(0, 180, 216, 240)  # Crisp cyan dashed outline
        self.rubber_band.setFillColor(fill_color)
        self.rubber_band.setStrokeColor(border_color)
        self.rubber_band.setWidth(2)
        self.rubber_band.setLineStyle(Qt.DashLine)

        self.setCursor(Qt.CrossCursor)

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pt = self.toMapCoordinates(event.pos())
            self.points.append(pt)
            self._update_rubber_band(temp_point=pt)
        elif event.button() == Qt.RightButton:
            self._finish_polygon()

    def canvasDoubleClickEvent(self, event):
        self._finish_polygon()

    def canvasMoveEvent(self, event):
        if len(self.points) > 0:
            temp_pt = self.toMapCoordinates(event.pos())
            self._update_rubber_band(temp_point=temp_pt)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reset()
            self.canceled.emit()
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if len(self.points) > 0:
                self.points.pop()
                self._update_rubber_band()

    def _update_rubber_band(self, temp_point: Optional[QgsPointXY] = None):
        pts = list(self.points)
        if temp_point is not None:
            pts.append(temp_point)

        if len(pts) < 2:
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            return

        if len(pts) == 2:
            # Show line while only 2 points
            geom = QgsGeometry.fromPolylineXY(pts)
        else:
            geom = QgsGeometry.fromPolygonXY([pts])

        self.rubber_band.setToGeometry(geom, None)
        self.rubber_band.show()

    def _finish_polygon(self):
        if len(self.points) >= 3:
            pts = list(self.points)
            geom = QgsGeometry.fromPolygonXY([pts])
            if geom.isGeosValid():
                self.polygon_selected.emit(geom)
                if self.on_polygon_callback:
                    self.on_polygon_callback(geom)
            else:
                fixed = geom.makeValid()
                self.polygon_selected.emit(fixed)
                if self.on_polygon_callback:
                    self.on_polygon_callback(fixed)
        self.reset()

    def reset(self):
        self.points.clear()
        if self.rubber_band:
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.rubber_band.hide()

    def deactivate(self):
        self.reset()
        super().deactivate()


# Backward compatibility alias
RoiExtentMapTool = PolygonRoiMapTool


class PipetteMapTool(QgsMapToolEmitPoint):
    """
    Map tool allowing the user to click any point on the canvas to sample
    pixel values (RGB/spectral) from the underlying historical raster layer.
    """

    point_sampled = pyqtSignal(QgsPointXY)

    def __init__(self, canvas, on_sample_callback: Optional[Callable[[QgsPointXY], None]] = None):
        super().__init__(canvas)
        self.canvas = canvas
        self.on_sample_callback = on_sample_callback
        self.setCursor(Qt.CrossCursor)

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())
            self.point_sampled.emit(point)
            if self.on_sample_callback:
                self.on_sample_callback(point)
