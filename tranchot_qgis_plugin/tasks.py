"""
Background processing tasks (QgsTask) for Tranchot feature extraction.
Prevents QGIS from freezing during heavy computer vision and geometry calculations.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from osgeo import gdal
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt

from qgis.PyQt.QtCore import pyqtSignal, QVariant
from qgis.core import (
    QgsTask,
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsRectangle,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsMessageLog,
    Qgis
)

# -----------------------------------------------------------------------------
# Dynamic Backend Resolution
# Ensures the core 'tranchot_extractor' processing engine can be loaded:
# 1. Directly from the Python environment (if installed via `pip install -e .`)
# 2. Relative to the plugin repository directory (symlinked or standalone)
# -----------------------------------------------------------------------------
import sys
import os

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

from tranchot_extractor.config import BuildingConfig, LandUseConfig
from tranchot_extractor.extractors.building_extractor import BuildingExtractor, BuildingExtractionResult
from tranchot_extractor.extractors.landuse_extractor import LandUseExtractor, LandUseExtractionResult



class BuildingExtractionTask(QgsTask):
    """
    QgsTask for running building, courtyard and wall extraction in a background thread.
    """

    task_completed = pyqtSignal(str, int)  # (layer_name, feature_count)
    task_failed = pyqtSignal(str)          # (error_message)

    def __init__(
        self,
        raster_path: str,
        config: BuildingConfig,
        roi_extent: Optional[QgsRectangle] = None,
        roi_geometry: Optional[QgsGeometry] = None,
        layer_name: str = "Tranchot_Gebaeude",
        output_crs: str = "EPSG:25832",
    ):
        super().__init__(f"Tranchot: Gebäude-Extraktion ({layer_name})", QgsTask.CanCancel)
        self.raster_path = raster_path
        self.config = config
        self.roi_extent = roi_extent
        self.roi_geometry = roi_geometry
        self.layer_name = layer_name
        self.output_crs = output_crs

        # Results to pass from background thread to finished()
        self.extracted_features: List[Dict[str, Any]] = []
        self.error_msg: Optional[str] = None
        self.pixel_size_m: float = 1.0

    def run(self) -> bool:
        """Executed in background worker thread."""
        try:
            self.setProgress(5.0)
            if self.isCanceled():
                return False

            if not os.path.exists(self.raster_path):
                self.error_msg = f"Rasterdatei nicht gefunden: {self.raster_path}"
                return False

            ds = gdal.Open(self.raster_path, gdal.GA_ReadOnly)
            if ds is None:
                self.error_msg = f"GDAL konnte {self.raster_path} nicht öffnen."
                return False

            gt = ds.GetGeoTransform()
            x_origin = gt[0]
            px_w = gt[1]
            y_origin = gt[3]
            px_h = gt[5]  # typically negative

            img_w = ds.RasterXSize
            img_h = ds.RasterYSize
            self.pixel_size_m = abs(px_w)

            self.setProgress(15.0)
            if self.isCanceled():
                return False

            # Determine raster window based on ROI or full sheet
            if self.roi_geometry is not None and not self.roi_geometry.isEmpty():
                extent = self.roi_geometry.boundingBox()
            elif self.roi_extent is not None and not self.roi_extent.isEmpty():
                extent = self.roi_extent
            else:
                extent = None

            if extent is not None and not extent.isEmpty():
                x0_px = int((extent.xMinimum() - x_origin) / px_w)
                x1_px = int((extent.xMaximum() - x_origin) / px_w)
                y0_px = int((extent.yMaximum() - y_origin) / px_h)
                y1_px = int((extent.yMinimum() - y_origin) / px_h)

                # Clamp to image dimensions
                x_off = max(0, min(img_w - 1, min(x0_px, x1_px)))
                y_off = max(0, min(img_h - 1, min(y0_px, y1_px)))
                win_w = max(1, min(img_w - x_off, abs(x1_px - x0_px)))
                win_h = max(1, min(img_h - y_off, abs(y1_px - y0_px)))
            else:
                x_off, y_off = 0, 0
                win_w, win_h = img_w, img_h

            sub_x_origin = x_origin + x_off * px_w
            sub_y_origin = y_origin + y_off * px_h

            self.setProgress(25.0)
            if self.isCanceled():
                return False

            # Read RGB raster bands (1, 2, 3)
            num_bands = ds.RasterCount
            if num_bands >= 3:
                r_band = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
                g_band = ds.GetRasterBand(2).ReadAsArray(x_off, y_off, win_w, win_h)
                b_band = ds.GetRasterBand(3).ReadAsArray(x_off, y_off, win_w, win_h)
                image_rgb = np.dstack((r_band, g_band, b_band))
            else:
                gray_band = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
                image_rgb = np.dstack((gray_band, gray_band, gray_band))

            self.setProgress(40.0)
            if self.isCanceled():
                return False

            # Convert ROI geometry to pixel-space Shapely Polygons
            settlement_polys = []
            if self.roi_geometry is not None and not self.roi_geometry.isEmpty():
                def geo_to_pixel(gx: float, gy: float) -> Tuple[float, float]:
                    return ((gx - sub_x_origin) / px_w, (gy - sub_y_origin) / px_h)

                try:
                    if self.roi_geometry.isMultipart():
                        poly_groups = self.roi_geometry.asMultiPolygon()
                    else:
                        poly_groups = [self.roi_geometry.asPolygon()]

                    for p_group in poly_groups:
                        if p_group and len(p_group[0]) >= 3:
                            ext_px = [geo_to_pixel(pt.x(), pt.y()) for pt in p_group[0]]
                            holes_px = [[geo_to_pixel(pt.x(), pt.y()) for pt in h_ring] for h_ring in p_group[1:]]
                            p_shapely = Polygon(ext_px, holes_px)
                            if p_shapely.is_valid and not p_shapely.is_empty:
                                settlement_polys.append(p_shapely)
                except Exception as e:
                    QgsMessageLog.logMessage(f"Fehler bei ROI Polygon-Konvertierung: {e}", "Tranchot", Qgis.Warning)

            # Run BuildingExtractor with dynamic module reload to ensure latest code is active
            try:
                import importlib
                if "tranchot_extractor.extractors.building_extractor" in sys.modules:
                    importlib.reload(sys.modules["tranchot_extractor.extractors.building_extractor"])
                from tranchot_extractor.extractors.building_extractor import BuildingExtractor
            except Exception:
                pass

            extractor = BuildingExtractor(config=self.config)
            result: BuildingExtractionResult = extractor.extract(
                image_rgb=image_rgb,
                settlement_boundaries=settlement_polys if settlement_polys else None
            )

            self.setProgress(80.0)
            if self.isCanceled():
                return False

            # Transform extracted Shapely polygons from local pixel coordinates to Map CRS
            m2_per_px2 = abs(px_w * px_h)
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
                        self.extracted_features.append({
                            "id": feat.id,
                            "wkt": geo_poly.wkt,
                            "area_px": float(feat.area_px),
                            "area_m2": round(float(feat.area_px) * m2_per_px2, 2),
                            "perimeter_m": round(float(feat.perimeter_px) * abs(px_w), 2),
                            "compactness": round(float(feat.compactness), 3),
                            "orientation": round(float(feat.orientation_deg), 1),
                            "type": "Courtyard Complex" if len(poly.interiors) > 0 else "Building"
                        })
                except Exception as e:
                    QgsMessageLog.logMessage(f"Geometry transformation error: {e}", "HistMap", Qgis.Warning)

            self.setProgress(100.0)
            return True

        except Exception as e:
            self.error_msg = str(e)
            QgsMessageLog.logMessage(f"HistMap Task Exception: {e}", "HistMap", Qgis.Critical)
            return False

    def finished(self, result: bool):
        """Executed on the main UI thread once run() completes."""
        if not result or self.error_msg is not None:
            self.task_failed.emit(self.error_msg or "Extraction canceled or failed.")
            return

        feature_count = len(self.extracted_features)

        # Remove preview vector layer now that permanent extraction is finished
        for l in QgsProject.instance().mapLayersByName("🔍 Historical Buildings (Live Preview)"):
            QgsProject.instance().removeMapLayer(l.id())

        if feature_count == 0:
            self.task_completed.emit(self.layer_name, 0)
            return

        # Check if layer already exists in QGIS project -> update in place without duplicate layers!
        existing_layers = QgsProject.instance().mapLayersByName(self.layer_name)
        if existing_layers:
            vl = existing_layers[0]
            pr = vl.dataProvider()
            if self.roi_extent is not None and not self.roi_extent.isEmpty():
                roi_geom = QgsGeometry.fromRect(self.roi_extent)
                to_delete = [f.id() for f in vl.getFeatures() if f.geometry().intersects(roi_geom)]
                if to_delete:
                    pr.deleteFeatures(to_delete)
            else:
                pr.deleteFeatures(vl.allFeatureIds())
        else:
            uri = f"Polygon?crs={self.output_crs}"
            vl = QgsVectorLayer(uri, self.layer_name, "memory")
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
                'color': '205,55,55,210',
                'color_border': '110,20,20,255',
                'width_border': '0.35',
                'style': 'solid',
                'style_border': 'solid'
            })
            vl.setRenderer(QgsSingleSymbolRenderer(symbol))
            QgsProject.instance().addMapLayer(vl, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, vl)

        # Insert features
        qgis_features = []
        for item in self.extracted_features:
            f = QgsFeature(vl.fields())
            geom = QgsGeometry.fromWkt(item["wkt"])
            f.setGeometry(geom)
            f.setAttribute("id", item["id"])
            f.setAttribute("type", item["type"])
            f.setAttribute("area_m2", item["area_m2"])
            f.setAttribute("perimeter_m", item["perimeter_m"])
            f.setAttribute("compactness", item["compactness"])
            f.setAttribute("orientation", item["orientation"])
            qgis_features.append(f)

        pr.addFeatures(qgis_features)
        vl.updateExtents()
        vl.triggerRepaint()
        node = QgsProject.instance().layerTreeRoot().findLayer(vl.id())
        if node and not node.isVisible():
            node.setItemVisibilityChecked(True)

        self.task_completed.emit(self.layer_name, feature_count)


class LandUseExtractionTask(QgsTask):
    """
    QgsTask for running historical land-use classification (Forest, Meadow, Water, Garden, Vineyard)
    in a background thread without freezing the QGIS GUI.
    """

    task_completed = pyqtSignal(str, int)  # (layer_name, feature_count)
    task_failed = pyqtSignal(str)          # (error_message)

    def __init__(
        self,
        raster_path: str,
        config: LandUseConfig,
        enabled_categories: Optional[List[str]] = None,
        roi_extent: Optional[QgsRectangle] = None,
        roi_geometry: Optional[QgsGeometry] = None,
        layer_name: str = "🌲 Land Use",
        output_crs: str = "EPSG:25832",
    ):
        super().__init__(f"HistMap: Land Use Extraction ({layer_name})", QgsTask.CanCancel)
        self.raster_path = raster_path
        self.config = config
        self.enabled_categories = enabled_categories or ["forest", "meadow", "water", "garden", "vineyard"]
        self.roi_extent = roi_extent
        self.roi_geometry = roi_geometry
        self.layer_name = layer_name
        self.output_crs = output_crs

        self.extracted_features: List[Dict[str, Any]] = []
        self.error_msg: Optional[str] = None
        self.pixel_size_m: float = 1.0

    def run(self) -> bool:
        """Executed in background worker thread."""
        try:
            self.setProgress(5.0)
            if self.isCanceled():
                return False

            if not os.path.exists(self.raster_path):
                self.error_msg = f"Raster file not found: {self.raster_path}"
                return False

            ds = gdal.Open(self.raster_path, gdal.GA_ReadOnly)
            if ds is None:
                self.error_msg = f"GDAL could not open {self.raster_path}."
                return False

            gt = ds.GetGeoTransform()
            x_origin = gt[0]
            px_w = gt[1]
            y_origin = gt[3]
            px_h = gt[5]

            img_w = ds.RasterXSize
            img_h = ds.RasterYSize
            self.pixel_size_m = abs(px_w)

            self.setProgress(15.0)
            if self.isCanceled():
                return False

            # Determine raster window based on ROI or full sheet
            if self.roi_geometry is not None and not self.roi_geometry.isEmpty():
                extent = self.roi_geometry.boundingBox()
            elif self.roi_extent is not None and not self.roi_extent.isEmpty():
                extent = self.roi_extent
            else:
                extent = None

            if extent is not None and not extent.isEmpty():
                x0_px = int((extent.xMinimum() - x_origin) / px_w)
                x1_px = int((extent.xMaximum() - x_origin) / px_w)
                y0_px = int((extent.yMaximum() - y_origin) / px_h)
                y1_px = int((extent.yMinimum() - y_origin) / px_h)

                x_off = max(0, min(img_w - 1, min(x0_px, x1_px)))
                y_off = max(0, min(img_h - 1, min(y0_px, y1_px)))
                win_w = max(1, min(img_w - x_off, abs(x1_px - x0_px)))
                win_h = max(1, min(img_h - y_off, abs(y1_px - y0_px)))
            else:
                x_off, y_off = 0, 0
                win_w, win_h = img_w, img_h

            sub_x_origin = x_origin + x_off * px_w
            sub_y_origin = y_origin + y_off * px_h

            self.setProgress(25.0)
            if self.isCanceled():
                return False

            # Read RGB raster bands (1, 2, 3)
            num_bands = ds.RasterCount
            if num_bands >= 3:
                r_band = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
                g_band = ds.GetRasterBand(2).ReadAsArray(x_off, y_off, win_w, win_h)
                b_band = ds.GetRasterBand(3).ReadAsArray(x_off, y_off, win_w, win_h)
                image_rgb = np.dstack((r_band, g_band, b_band))
            else:
                gray_band = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_w, win_h)
                image_rgb = np.dstack((gray_band, gray_band, gray_band))

            self.setProgress(40.0)
            if self.isCanceled():
                return False

            # Convert ROI geometry to pixel-space Shapely Polygons if present
            roi_polys = []
            if self.roi_geometry is not None and not self.roi_geometry.isEmpty():
                def geo_to_pixel(gx: float, gy: float) -> Tuple[float, float]:
                    return ((gx - sub_x_origin) / px_w, (gy - sub_y_origin) / px_h)

                try:
                    if self.roi_geometry.isMultipart():
                        poly_groups = self.roi_geometry.asMultiPolygon()
                    else:
                        poly_groups = [self.roi_geometry.asPolygon()]

                    for p_group in poly_groups:
                        if p_group and len(p_group[0]) >= 3:
                            ext_px = [geo_to_pixel(pt.x(), pt.y()) for pt in p_group[0]]
                            holes_px = [[geo_to_pixel(pt.x(), pt.y()) for pt in h_ring] for h_ring in p_group[1:]]
                            p_shapely = Polygon(ext_px, holes_px)
                            if p_shapely.is_valid and not p_shapely.is_empty:
                                roi_polys.append(p_shapely)
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error in LandUse ROI conversion: {e}", "HistMap", Qgis.Warning)

            # Reload LandUseExtractor dynamically to guarantee latest code
            try:
                import importlib
                if "tranchot_extractor.extractors.landuse_extractor" in sys.modules:
                    importlib.reload(sys.modules["tranchot_extractor.extractors.landuse_extractor"])
                from tranchot_extractor.extractors.landuse_extractor import LandUseExtractor
            except Exception:
                pass

            extractor = LandUseExtractor(config=self.config)
            result: LandUseExtractionResult = extractor.extract(
                image_rgb=image_rgb,
                roi_boundaries=roi_polys if roi_polys else None,
                enabled_categories=self.enabled_categories
            )

            self.setProgress(80.0)
            if self.isCanceled():
                return False

            category_label_map = {
                "forest": "Forest (Wald)",
                "meadow": "Meadow (Wiesen/Weiden)",
                "water": "Water Body (Gewässer)",
                "garden": "Gardens & Orchards (Gärten)",
                "vineyard": "Vineyard / Arable (Weinberge)",
            }

            # Transform Shapely polygons to Map CRS
            m2_per_px2 = abs(px_w * px_h)
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
                        self.extracted_features.append({
                            "id": feat.id,
                            "wkt": geo_poly.wkt,
                            "category": feat.category,
                            "category_label": category_label_map.get(feat.category, feat.category.capitalize()),
                            "area_m2": round(float(feat.area_px) * m2_per_px2, 2),
                            "perimeter_m": round(float(geo_poly.length), 2),
                        })
                except Exception as e:
                    QgsMessageLog.logMessage(f"LandUse geometry transformation error: {e}", "HistMap", Qgis.Warning)

            self.setProgress(100.0)
            return True

        except Exception as e:
            self.error_msg = str(e)
            QgsMessageLog.logMessage(f"HistMap LandUse Task Exception: {e}", "HistMap", Qgis.Critical)
            return False

    def finished(self, result: bool):
        """Executed on the main UI thread once run() completes."""
        if not result or self.error_msg is not None:
            self.task_failed.emit(self.error_msg or "Land-use extraction canceled or failed.")
            return

        feature_count = len(self.extracted_features)
        if feature_count == 0:
            self.task_completed.emit(self.layer_name, 0)
            return

        # Check if layer already exists in QGIS project -> update in place
        existing_layers = QgsProject.instance().mapLayersByName(self.layer_name)
        if existing_layers:
            vl = existing_layers[0]
            pr = vl.dataProvider()
            if self.roi_extent is not None and not self.roi_extent.isEmpty():
                roi_geom = QgsGeometry.fromRect(self.roi_extent)
                to_delete = [f.id() for f in vl.getFeatures() if f.geometry().intersects(roi_geom)]
                if to_delete:
                    pr.deleteFeatures(to_delete)
            else:
                pr.deleteFeatures(vl.allFeatureIds())
        else:
            uri = f"Polygon?crs={self.output_crs}"
            vl = QgsVectorLayer(uri, self.layer_name, "memory")
            pr = vl.dataProvider()
            fields = [
                QgsField("id", QVariant.Int),
                QgsField("category", QVariant.String),
                QgsField("category_label", QVariant.String),
                QgsField("area_m2", QVariant.Double),
                QgsField("perimeter_m", QVariant.Double),
            ]
            pr.addAttributes(fields)
            vl.updateFields()

            # Categorized Symbology styling with historical palette
            categories = [
                ("forest", "Forest (Wald)", "46,125,50,180", "27,94,32,255"),
                ("meadow", "Meadow (Wiesen/Weiden)", "129,199,132,180", "56,142,60,255"),
                ("water", "Water Body (Gewässer)", "30,136,229,210", "13,71,161,255"),
                ("garden", "Gardens & Orchards (Gärten)", "165,214,167,180", "46,125,50,255"),
                ("vineyard", "Vineyard / Arable (Weinberge)", "212,163,115,180", "141,91,76,255"),
            ]

            cats = []
            for cat_val, cat_lbl, fill_col, border_col in categories:
                sym = QgsFillSymbol.createSimple({
                    'color': fill_col,
                    'color_border': border_col,
                    'width_border': '0.35',
                    'style': 'solid',
                    'style_border': 'solid'
                })
                cats.append(QgsRendererCategory(cat_val, sym, cat_lbl))

            renderer = QgsCategorizedSymbolRenderer("category", cats)
            vl.setRenderer(renderer)
            QgsProject.instance().addMapLayer(vl, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, vl)

        # Insert features
        qgis_features = []
        for item in self.extracted_features:
            f = QgsFeature(vl.fields())
            geom = QgsGeometry.fromWkt(item["wkt"])
            f.setGeometry(geom)
            f.setAttribute("id", item["id"])
            f.setAttribute("category", item["category"])
            f.setAttribute("category_label", item["category_label"])
            f.setAttribute("area_m2", item["area_m2"])
            f.setAttribute("perimeter_m", item["perimeter_m"])
            qgis_features.append(f)

        pr.addFeatures(qgis_features)
        vl.updateExtents()
        vl.triggerRepaint()
        node = QgsProject.instance().layerTreeRoot().findLayer(vl.id())
        if node and not node.isVisible():
            node.setItemVisibilityChecked(True)

        self.task_completed.emit(self.layer_name, feature_count)

