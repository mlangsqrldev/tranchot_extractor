"""
Tranchot Map Feature Extractor — Intuitive Desktop GIS Application.
Features:
- Adaptive Light (Default) & Dark Appearance Modes.
- Left Sidebar: Map loading, fast automatic extractors, tool selection, layer toggles, user-defined Output EPSG (default EPSG:25832), GIS export with fixed historical colors.
- Right Sidebar:
  * ✨ 0. Weißabgleich & Entgilbung (Parchment de-yellowing & watercolor radiance boost).
  * 🎨 Blatt-Kalibrierung & Pipetten mit feinem Farbabstand (2..25, default 8).
  * 🏛️ 2. Gebäude & Hofanlagen mit 90°-Orthogonal-Regularisierung und zackenfreier Trennung.
- Canvas Topbar: Live appearance mode switcher (☀️ Hell / 🌙 Dunkel), one-click view toggle.
- 100% local, instant execution (< 0.15s).
"""

import sys
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import time
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import translate
import geopandas as gpd

from tranchot_extractor.config import BuildingConfig, RoadConfig, LandUseConfig
from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
from tranchot_extractor.extractors.building_extractor import BuildingExtractor
from tranchot_extractor.extractors.road_extractor import RoadExtractor
from tranchot_extractor.extractors.text_extractor import TextExtractor
from tranchot_extractor.extractors.landuse_extractor import LandUseExtractor
from tranchot_extractor.extractors.pipette_sampler import PipetteSampler
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.spatial_gazetteer import SpatialGazetteer

# Default to Light Mode as requested
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Theme color definitions (Light, Dark)
THEME_SIDEBAR_FG = ("#F8F9FA", "#181A20")
THEME_MAIN_FG = ("#EEF2F6", "#111317")
THEME_TOPBAR_FG = ("#FFFFFF", "#1E2028")
THEME_STATUSBAR_FG = ("#F1F5F9", "#14161B")
THEME_CARD_FG = ("#FFFFFF", "#20232B")
THEME_CARD_BORDER = ("#E2E8F0", "#2D3139")
THEME_TEXT_MAIN = ("#0F172A", "#F8FAFC")
THEME_TEXT_MUTED = ("#64748B", "#94A3B8")
THEME_TEXT_SECTION = ("#1D4ED8", "#60A5FA")
THEME_SWATCH_FRAME = ("#F1F5F9", "#262A35")
THEME_INPUT_FG = ("#F8FAFC", "#1E2028")

# Fixed historical cartographic colors for layers & GIS export
LAYER_COLOR_SPECS = {
    "building": {"name": "Gebäude (Karminrot)", "fill": "#e74c3c", "stroke": "#ff4757"},
    "road": {"name": "Straßen (Ocker/Orange)", "fill": "#ff793f", "stroke": "#d35400"},
    "toponym": {"name": "Beschriftung (Dunkel)", "fill": "#2c3e50", "stroke": "#00cec9"},
    "forest": {"name": "Wald (Moosgrün)", "fill": "#27ae60", "stroke": "#2ecc71"},
    "meadow": {"name": "Wiesen (Pastell-Türkis)", "fill": "#00cec9", "stroke": "#81ecec"},
    "water": {"name": "Gewässer (Flussblau)", "fill": "#0984e3", "stroke": "#74b9ff"},
    "gravel": {"name": "Kiesbänke (Lachsrosa)", "fill": "#e17055", "stroke": "#fab1a0"},
    "vineyard": {"name": "Weinberge (Rebgelb)", "fill": "#f1c40f", "stroke": "#f39c12"},
    "garden": {"name": "Gärten (Gartengelb)", "fill": "#fdcb6e", "stroke": "#ffeaa7"},
}


class MapCanvas(tk.Canvas):
    """
    High-performance map canvas with mousewheel zoom, drag-pan, and multi-layer rendering.
    Supports dynamic Light & Dark themes.
    """
    def __init__(self, parent, app, **kwargs):
        self.app = app
        bg_col = self._get_theme_canvas_bg()
        super().__init__(parent, bg=bg_col, highlightthickness=0, **kwargs)

        self.pil_image: Optional[Image.Image] = None
        self.np_image: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_panning = False

        self.box_start_x = 0
        self.box_start_y = 0
        self.drag_current_x = 0
        self.drag_current_y = 0
        self.is_dragging_box = False

        self.polygon_pts: List[Tuple[float, float]] = []
        self.cursor_canvas_x: Optional[float] = None
        self.cursor_canvas_y: Optional[float] = None

        self.bind("<ButtonPress-1>", self._on_left_press)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_release)
        self.bind("<Double-Button-1>", self._on_double_click)

        self.bind("<ButtonPress-2>", self._on_pan_press)
        self.bind("<B2-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-2>", self._on_pan_release)

        self.bind("<ButtonPress-3>", self._on_right_press)

        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", lambda e: self._on_linux_zoom(e, 1.15))
        self.bind("<Button-5>", lambda e: self._on_linux_zoom(e, 1/1.15))

        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Configure>", lambda e: self.redraw())

        self.bind_all("<Return>", lambda e: self._finish_current_polygon())
        self.bind_all("<Escape>", lambda e: self._cancel_current_polygon())
        self.bind_all("<BackSpace>", lambda e: self._undo_polygon_point())

    def _get_theme_canvas_bg(self) -> str:
        mode = ctk.get_appearance_mode().lower()
        return "#E5E7EB" if mode == "light" else "#121317"

    def update_theme(self):
        self.configure(bg=self._get_theme_canvas_bg())
        self.redraw()

    def load_image(self, pil_img: Image.Image, np_img: np.ndarray):
        self.pil_image = pil_img
        self.np_image = np_img
        self.reset_view()

    def update_image_source(self, pil_img: Image.Image, np_img: np.ndarray):
        """Updates image pixels without resetting zoom & pan."""
        self.pil_image = pil_img
        self.np_image = np_img
        self.redraw()

    def reset_view(self):
        if not self.pil_image:
            return
        cw = max(self.winfo_width(), 600)
        ch = max(self.winfo_height(), 500)
        iw, ih = self.pil_image.size
        self.scale = min(cw / iw, ch / ih) * 0.95
        self.pan_x = (cw - iw * self.scale) / 2.0
        self.pan_y = (ch - ih * self.scale) / 2.0
        self.redraw()

    def canvas_to_image_coords(self, cx: float, cy: float) -> Tuple[float, float]:
        if self.scale <= 0:
            return (0.0, 0.0)
        ix = (cx - self.pan_x) / self.scale
        iy = (cy - self.pan_y) / self.scale
        return (ix, iy)

    def image_to_canvas_coords(self, ix: float, iy: float) -> Tuple[float, float]:
        cx = ix * self.scale + self.pan_x
        cy = iy * self.scale + self.pan_y
        return (cx, cy)

    def _on_pan_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_panning = True

    def _on_pan_drag(self, event):
        if self.is_panning:
            self.pan_x += (event.x - self.drag_start_x)
            self.pan_y += (event.y - self.drag_start_y)
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.redraw()

    def _on_pan_release(self, event):
        self.is_panning = False

    def _on_left_press(self, event):
        if self.app.active_tool == "pan":
            self._on_pan_press(event)
            return

        ix, iy = self.canvas_to_image_coords(event.x, event.y)

        if self.app.active_tool in ("pattern_poly", "settlement_poly"):
            if len(self.polygon_pts) >= 3:
                p0_cx, p0_cy = self.image_to_canvas_coords(self.polygon_pts[0][0], self.polygon_pts[0][1])
                dist_px = ((event.x - p0_cx)**2 + (event.y - p0_cy)**2)**0.5
                if dist_px < 15.0:
                    self._finish_current_polygon()
                    return
            self.polygon_pts.append((ix, iy))
            self.redraw()
            return

        if self.app.active_tool == "pipette":
            self.app.handle_pipette_sample_at(ix, iy)
            return

        if self.app.active_tool in ("toponym", "bldg_box", "landuse_box"):
            self.box_start_x = event.x
            self.box_start_y = event.y
            self.drag_current_x = event.x
            self.drag_current_y = event.y
            self.is_dragging_box = True
            return

        if self.app.active_tool == "select":
            self.app.select_feature_at(ix, iy)
        elif self.app.active_tool == "extract":
            self.app.extract_single_building_at(ix, iy)
        elif self.app.active_tool == "road_snap":
            self.app.handle_road_snap_click(ix, iy)

    def _on_double_click(self, event):
        if self.app.active_tool in ("pattern_poly", "settlement_poly") and len(self.polygon_pts) >= 3:
            self._finish_current_polygon()

    def _finish_current_polygon(self):
        if len(self.polygon_pts) < 3:
            self.polygon_pts.clear()
            self.redraw()
            return
        pts = list(self.polygon_pts)
        self.polygon_pts.clear()
        tool = self.app.active_tool
        if tool == "pattern_poly":
            self.app.handle_pattern_polygon(pts)
        elif tool == "settlement_poly":
            self.app.handle_settlement_polygon(pts)
        self.redraw()

    def _cancel_current_polygon(self):
        if self.polygon_pts:
            self.polygon_pts.clear()
            self.app.lbl_status.configure(text="Polygon-Zeichnung abgebrochen.")
            self.redraw()

    def _undo_polygon_point(self):
        if self.polygon_pts:
            self.polygon_pts.pop()
            self.redraw()

    def _on_left_drag(self, event):
        if self.app.active_tool == "pan":
            self._on_pan_drag(event)
        elif self.app.active_tool in ("toponym", "bldg_box", "landuse_box") and self.is_dragging_box:
            self.drag_current_x = event.x
            self.drag_current_y = event.y
            self.redraw()

    def _on_left_release(self, event):
        if self.app.active_tool == "pan":
            self._on_pan_release(event)
        elif self.is_dragging_box:
            self.is_dragging_box = False
            ix0, iy0 = self.canvas_to_image_coords(self.box_start_x, self.box_start_y)
            ix1, iy1 = self.canvas_to_image_coords(event.x, event.y)
            if self.app.active_tool == "toponym":
                self.app.handle_toponym_box(ix0, iy0, ix1, iy1)
            elif self.app.active_tool == "bldg_box":
                self.app.handle_building_roi_box(ix0, iy0, ix1, iy1)
            elif self.app.active_tool == "landuse_box":
                self.app.handle_landuse_roi_box(ix0, iy0, ix1, iy1)

    def _on_right_press(self, event):
        if self.polygon_pts:
            self._cancel_current_polygon()
            return
        ix, iy = self.canvas_to_image_coords(event.x, event.y)
        self.app.delete_feature_at(ix, iy)

    def _on_mouse_wheel(self, event):
        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        self._zoom(factor, event.x, event.y)

    def _on_linux_zoom(self, event, factor):
        self._zoom(factor, event.x, event.y)

    def _zoom(self, factor: float, cx: float, cy: float):
        if not self.pil_image:
            return
        new_scale = max(0.02, min(50.0, self.scale * factor))
        self.pan_x = cx - (cx - self.pan_x) * (new_scale / self.scale)
        self.pan_y = cy - (cy - self.pan_y) * (new_scale / self.scale)
        self.scale = new_scale
        self.redraw()

    def _on_mouse_move(self, event):
        self.cursor_canvas_x = event.x
        self.cursor_canvas_y = event.y
        ix, iy = self.canvas_to_image_coords(event.x, event.y)
        self.app.update_coordinates_hud(ix, iy)
        if self.app.active_tool in ("pattern_poly", "settlement_poly") and len(self.polygon_pts) > 0:
            self.redraw()

    def redraw(self):
        self.delete("all")
        if not self.pil_image:
            mode = ctk.get_appearance_mode().lower()
            txt_col = "#64748B" if mode == "light" else "#94A3B8"
            self.create_text(
                self.winfo_width() / 2, self.winfo_height() / 2,
                text="🏛️ Bitte lade eine historische Karte (GeoTIFF / PNG / JPG)...",
                fill=txt_col, font=("Segoe UI", 14, "bold")
            )
            return

        cw = self.winfo_width()
        ch = self.winfo_height()
        iw, ih = self.pil_image.size

        # Visible image region calculation
        x0_img = max(0, int(-self.pan_x / self.scale))
        y0_img = max(0, int(-self.pan_y / self.scale))
        x1_img = min(iw, int((cw - self.pan_x) / self.scale) + 1)
        y1_img = min(ih, int((ch - self.pan_y) / self.scale) + 1)

        if x1_img > x0_img and y1_img > y0_img:
            crop = self.pil_image.crop((x0_img, y0_img, x1_img, y1_img))
            dw = int((x1_img - x0_img) * self.scale)
            dh = int((y1_img - y0_img) * self.scale)
            if dw > 0 and dh > 0:
                resample_filter = Image.Resampling.NEAREST if self.scale > 2.0 else Image.Resampling.BILINEAR
                resized = crop.resize((dw, dh), resample_filter)
                self.tk_image = ImageTk.PhotoImage(resized)
                pos_x = x0_img * self.scale + self.pan_x
                pos_y = y0_img * self.scale + self.pan_y
                self.create_image(pos_x, pos_y, anchor="nw", image=self.tk_image)

        # Draw Layers
        if self.app.show_forests_var.get():
            self._draw_polygons(self.app.extracted_layers.get("forest", []), LAYER_COLOR_SPECS["forest"])
        if self.app.show_meadows_var.get():
            self._draw_polygons(self.app.extracted_layers.get("meadow", []), LAYER_COLOR_SPECS["meadow"])
        if self.app.show_water_var.get():
            self._draw_polygons(self.app.extracted_layers.get("water", []), LAYER_COLOR_SPECS["water"])
        if self.app.show_gravel_var.get():
            self._draw_polygons(self.app.extracted_layers.get("gravel", []), LAYER_COLOR_SPECS["gravel"])
        if self.app.show_vineyard_var.get():
            self._draw_polygons(self.app.extracted_layers.get("vineyard", []), LAYER_COLOR_SPECS["vineyard"])
        if self.app.show_garden_var.get():
            self._draw_polygons(self.app.extracted_layers.get("garden", []), LAYER_COLOR_SPECS["garden"])

        if self.app.show_roads_var.get():
            self._draw_roads()
        if self.app.show_bldgs_var.get():
            self._draw_buildings()
        if self.app.show_toponyms_var.get():
            self._draw_toponym_labels()

        # Draw Settlement Boundaries (Exclusion Zones)
        self._draw_settlement_boundaries()

        # Draw Few-Shot Exemplar Training Polygons
        self._draw_exemplar_polygons()

        # Draw Active Polygon In-Progress
        self._draw_active_polygon()

        # Temporary road snapping point
        if self.app.road_start_pt is not None:
            cx, cy = self.image_to_canvas_coords(*self.app.road_start_pt)
            self.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#e74c3c", outline="#ffffff", width=2)
            self.create_text(cx + 10, cy, text="Startpunkt", fill="#e74c3c", font=("Segoe UI", 10, "bold"), anchor="w")

        # Interactive Drag Box
        if self.is_dragging_box:
            bx0 = min(self.box_start_x, self.drag_current_x)
            by0 = min(self.box_start_y, self.drag_current_y)
            bx1 = max(self.box_start_x, self.drag_current_x)
            by1 = max(self.box_start_y, self.drag_current_y)
            
            if self.app.active_tool == "bldg_box":
                box_color = "#e74c3c"
                box_tag = "📐 Gebäude ROI-Box"
            elif self.app.active_tool == "landuse_box":
                box_color = "#27ae60"
                box_tag = "🌲 Flächen ROI-Box"
            else:
                box_color = "#00cec9"
                box_tag = "🏷️ Toponym ROI-Box"

            self.create_rectangle(bx0, by0, bx1, by1, outline=box_color, width=2, dash=(4, 2))
            self.create_text(bx0 + 6, by0 + 12, text=box_tag, fill=box_color, font=("Segoe UI", 10, "bold"), anchor="w")

    def _draw_settlement_boundaries(self):
        for idx, poly in enumerate(self.app.settlement_boundaries, 1):
            if not poly or poly.is_empty or not poly.exterior:
                continue
            canvas_pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
            flat_pts = [c for pt in canvas_pts for c in pt]
            if len(flat_pts) >= 6:
                self.create_polygon(*flat_pts, fill="#8e44ad", outline="#8e44ad", width=3, dash=(6, 3), stipple="gray25")
                ccx, ccy = self.image_to_canvas_coords(poly.centroid.x, poly.centroid.y)
                self.create_text(ccx, ccy, text=f"🏘️ Siedlung #{idx}", fill="#8e44ad", font=("Segoe UI", 12, "bold"))

    def _draw_exemplar_polygons(self):
        for cid, polys in self.app.exemplar_polygons.items():
            spec = LAYER_COLOR_SPECS.get(cid, {"fill": "#8e44ad", "stroke": "#f1c40f"})
            for p_idx, poly in enumerate(polys, 1):
                if not poly or poly.is_empty or not poly.exterior:
                    continue
                canvas_pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
                flat_pts = [c for pt in canvas_pts for c in pt]
                if len(flat_pts) >= 6:
                    self.create_polygon(*flat_pts, fill=spec["fill"], outline="#f1c40f", width=2.5, dash=(4, 2), stipple="gray50")
                    ccx, ccy = self.image_to_canvas_coords(poly.centroid.x, poly.centroid.y)
                    self.create_text(ccx, ccy, text=f"📐 Muster #{p_idx}", fill="#f39c12", font=("Segoe UI", 10, "bold"))

    def _draw_active_polygon(self):
        if not self.polygon_pts:
            return
        canvas_pts = [self.image_to_canvas_coords(px, py) for px, py in self.polygon_pts]
        for i, (pcx, pcy) in enumerate(canvas_pts):
            r = 5
            col = "#2ecc71" if i == 0 else "#f39c12"
            self.create_oval(pcx - r, pcy - r, pcx + r, pcy + r, fill=col, outline="#ffffff", width=1.5)
            if i == 0 and len(canvas_pts) >= 3:
                self.create_text(pcx + 8, pcy - 8, text="Start (Klick zum Schließen)", fill="#27ae60", font=("Segoe UI", 9, "bold"), anchor="w")

        if len(canvas_pts) > 1:
            flat_pts = [c for pt in canvas_pts for c in pt]
            self.create_line(*flat_pts, fill="#f39c12", width=2, dash=(4, 2))

        if self.cursor_canvas_x is not None and self.cursor_canvas_y is not None and len(canvas_pts) > 0:
            last_cx, last_cy = canvas_pts[-1]
            self.create_line(last_cx, last_cy, self.cursor_canvas_x, self.cursor_canvas_y, fill="#e74c3c", width=2)

    def _draw_polygons(self, polygons: List[Polygon], spec: Dict[str, str]):
        fill_col = spec["fill"]
        stroke_col = spec["stroke"]
        for poly in polygons:
            if not poly or poly.is_empty or not poly.exterior:
                continue
            canvas_pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
            flat_pts = [c for pt in canvas_pts for c in pt]
            if len(flat_pts) >= 6:
                self.create_polygon(*flat_pts, fill=fill_col, outline=stroke_col, width=1.5, stipple="gray25")

    def _draw_roads(self):
        for idx, line in enumerate(self.app.extracted_roads):
            if not line or line.is_empty:
                continue
            is_selected = (self.app.selected_road_idx == idx)
            stroke_col = "#f1c40f" if is_selected else LAYER_COLOR_SPECS["road"]["stroke"]
            width = 3.5 if is_selected else 2.0
            canvas_pts = [self.image_to_canvas_coords(x, y) for x, y in line.coords]
            flat_pts = [c for pt in canvas_pts for c in pt]
            if len(flat_pts) >= 4:
                self.create_line(*flat_pts, fill=stroke_col, width=width, capstyle="round", joinstyle="round")

    def _draw_buildings(self):
        mode = ctk.get_appearance_mode().lower()
        hole_fill = "#E5E7EB" if mode == "light" else "#121317"

        for idx, poly in enumerate(self.app.extracted_buildings):
            if not poly or poly.is_empty or not poly.exterior:
                continue
            is_selected = (self.app.selected_building_idx == idx)
            stroke_col = "#f1c40f" if is_selected else LAYER_COLOR_SPECS["building"]["stroke"]
            fill_col = "#e74c3c"
            width = 2.5 if is_selected else 1.5

            canvas_pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
            flat_pts = [c for pt in canvas_pts for c in pt]
            if len(flat_pts) >= 6:
                self.create_polygon(*flat_pts, fill=fill_col, outline=stroke_col, width=width, stipple="gray50")

            for interior in poly.interiors:
                hole_pts = [self.image_to_canvas_coords(x, y) for x, y in interior.coords]
                flat_hole = [c for pt in hole_pts for c in pt]
                if len(flat_hole) >= 6:
                    self.create_polygon(*flat_hole, fill=hole_fill, outline="#00cec9", width=1.0)

    def _draw_toponym_labels(self):
        for idx, top in enumerate(self.app.extracted_toponyms):
            cx, cy = self.image_to_canvas_coords(top["x"], top["y"])
            is_selected = (self.app.selected_toponym_idx == idx)
            dot_col = "#f1c40f" if is_selected else "#00cec9"
            text_col = "#d35400" if is_selected else "#0984e3"

            self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=dot_col, outline="#ffffff", width=1)
            self.create_text(cx + 8, cy, text=f"🏷️ {top['text']}", fill=text_col, font=("Segoe UI", 11, "bold"), anchor="w")


class ToponymDialog(ctk.CTkToplevel):
    """Dialog with magnifier preview and real-world spatial suggestions."""
    def __init__(
        self,
        parent,
        suggested_text: str = "",
        crop_img: Optional[Image.Image] = None,
        suggestions: Optional[List[Dict[str, str]]] = None,
        geo_coord_str: str = ""
    ):
        super().__init__(parent)
        self.title("🏷️ Beschriftung erfassen / bearbeiten")
        self.geometry("540x470")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.result_text: Optional[str] = None
        self.result_category: str = "settlement"

        lbl = ctk.CTkLabel(self, text="🏷️ Historische Beschriftung erfassen", font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(padx=20, pady=(12, 4))

        if geo_coord_str:
            ctk.CTkLabel(self, text=f"📍 Geo-Position: {geo_coord_str}", font=ctk.CTkFont(size=10), text_color="#3498db").pack(padx=20, pady=(0, 4))

        if crop_img:
            w, h = crop_img.size
            disp_w = min(480, max(200, w))
            disp_h = int(h * (disp_w / max(1, w)))
            disp_h = min(110, max(40, disp_h))
            crop_resized = crop_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
            self.tk_crop = ImageTk.PhotoImage(crop_resized)
            lbl_crop = tk.Label(self, image=self.tk_crop)
            lbl_crop.pack(padx=20, pady=4)

        ctk.CTkLabel(self, text="Textinhalt / Name:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=25, pady=(6, 2))
        
        init_val = suggested_text
        if not init_val and suggestions and len(suggestions) > 0:
            init_val = suggestions[0]["name"]

        self.entry_text = ctk.CTkEntry(self, width=480, height=36, font=ctk.CTkFont(size=15, weight="bold"))
        self.entry_text.insert(0, init_val)
        self.entry_text.pack(padx=25, pady=2)
        self.entry_text.focus_set()
        self.entry_text.select_range(0, tk.END)

        ctk.CTkLabel(self, text="Vorschläge für diese Geo-Position (GeoNames / OSM):", font=ctk.CTkFont(size=11), text_color="#00cec9").pack(anchor="w", padx=25, pady=(8, 2))
        
        self.sugg_scroll = ctk.CTkScrollableFrame(self, height=55, orientation="horizontal")
        self.sugg_scroll.pack(fill="x", padx=25, pady=2)

        suggs_to_show = suggestions or [
            {"name": "Nickenich", "category": "settlement"},
            {"name": "Affenberg", "category": "hill_mountain"},
            {"name": "Rabenkopf", "category": "hill_mountain"},
            {"name": "Kotsberg", "category": "hill_mountain"},
            {"name": "Andernach", "category": "settlement"},
            {"name": "Plaidterweg", "category": "road_label"}
        ]

        for s in suggs_to_show[:8]:
            s_name = s["name"]
            s_cat = s.get("category", "settlement")
            btn_s = ctk.CTkButton(
                self.sugg_scroll, text=s_name, height=28,
                font=ctk.CTkFont(size=11),
                command=lambda n=s_name, c=s_cat: self._select_suggestion(n, c)
            )
            btn_s.pack(side="left", padx=3)

        ctk.CTkLabel(self, text="Kategorie:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=25, pady=(8, 2))
        self.opt_category = ctk.CTkOptionMenu(
            self,
            values=["Ortsname / Siedlung", "Flurname / Berg / Wald", "Gewässer / Bach / Fluss", "Infrastruktur / Weg", "Sonstige Anmerkung"],
            width=480, height=32
        )
        self.opt_category.pack(padx=25, pady=2)

        if init_val:
            self._apply_category_for_name(init_val)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=25, pady=(15, 10))

        btn_cancel = ctk.CTkButton(btn_frame, text="Abbrechen", command=self._on_cancel, width=140)
        btn_cancel.pack(side="left")

        btn_save = ctk.CTkButton(btn_frame, text="💾 Speichern (Enter)", command=self._on_save, fg_color="#27ae60", hover_color="#2ecc71", width=200, font=ctk.CTkFont(weight="bold"))
        btn_save.pack(side="right")

        self.bind("<Return>", lambda e: self._on_save())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.grab_set()

    def _select_suggestion(self, name: str, cat: str):
        self.entry_text.delete(0, tk.END)
        self.entry_text.insert(0, name)
        self._apply_category_for_name(name, cat)

    def _apply_category_for_name(self, name: str, cat: Optional[str] = None):
        if cat == "hill_mountain" or any(w in name.lower() for w in ["berg", "kopf", "stein"]):
            self.opt_category.set("Flurname / Berg / Wald")
        elif cat == "road_label" or any(w in name.lower() for w in ["weg", "straße", "chaussee", "croix", "kreuz"]):
            self.opt_category.set("Infrastruktur / Weg")
        elif cat == "water_river" or any(w in name.lower() for w in ["bach", "rhein", "mosel", "see"]):
            self.opt_category.set("Gewässer / Bach / Fluss")
        else:
            self.opt_category.set("Ortsname / Siedlung")

    def _on_save(self):
        val = self.entry_text.get().strip()
        if val:
            self.result_text = val
            cat_map = {
                "Ortsname / Siedlung": "settlement",
                "Flurname / Berg / Wald": "hill_mountain",
                "Gewässer / Bach / Fluss": "water_river",
                "Infrastruktur / Weg": "road_label",
                "Sonstige Anmerkung": "annotation"
            }
            self.result_category = cat_map.get(self.opt_category.get(), "annotation")
        self.destroy()

    def _on_cancel(self):
        self.result_text = None
        self.destroy()


class TranchotDesktopApp(ctk.CTk):
    """
    Desktop GIS application with Left Sidebar (Tools & Automation), Right Sidebar (Pipette & De-Yellowing),
    and high-performance MapCanvas. Adaptive Light/Dark appearance mode.
    """
    def __init__(self):
        super().__init__()

        self.title("🏛️ Tranchot Extractor — Historische Kartenanalyse & GIS Extraktion (BCDH)")
        self.geometry("1620x960")
        self.minsize(1180, 740)

        # Raw & Enhanced Image State
        self.current_image_path: Optional[str] = None
        self.raw_pil: Optional[Image.Image] = None
        self.raw_np: Optional[np.ndarray] = None
        self.enhanced_pil: Optional[Image.Image] = None
        self.enhanced_np: Optional[np.ndarray] = None

        self.current_pil: Optional[Image.Image] = None
        self.current_np: Optional[np.ndarray] = None
        self.geo_handler: Optional[GeoReferenceHandler] = None

        self.sampler = PipetteSampler()
        self.active_pipette_class: str = "forest"

        self.extracted_buildings: List[Polygon] = []
        self.extracted_roads: List[LineString] = []
        self.extracted_toponyms: List[Dict[str, Any]] = []
        self.extracted_layers: Dict[str, List[Polygon]] = {
            "forest": [], "meadow": [], "water": [], "gravel": [], "vineyard": [], "garden": []
        }
        self.settlement_boundaries: List[Polygon] = []
        self.exemplar_polygons: Dict[str, List[Polygon]] = {
            "forest": [], "meadow": [], "water": [], "gravel": [], "vineyard": [], "garden": []
        }

        self.selected_building_idx: Optional[int] = None
        self.selected_road_idx: Optional[int] = None
        self.selected_toponym_idx: Optional[int] = None
        self.road_start_pt: Optional[Tuple[float, float]] = None
        self.active_tool: str = "pipette"

        self.swatch_boxes: Dict[str, Any] = {}

        # Setup GUI layout
        self._setup_layout()
        self._load_default_sample()

        # Keyboard shortcuts
        self.bind("<Delete>", lambda e: self._delete_selected())
        self.bind("<BackSpace>", lambda e: self._delete_selected())
        self.bind("<Escape>", lambda e: self._cancel_road_start())

    def _setup_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Left Sidebar
        self.grid_columnconfigure(1, weight=1)  # Canvas
        self.grid_columnconfigure(2, weight=0)  # Right Sidebar

        # =========================================================================
        # 1. Left Sidebar: Map, Auto-Extractors, Tools, Layers, Export
        # =========================================================================
        self.left_sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0, fg_color=THEME_SIDEBAR_FG)
        self.left_sidebar.grid(row=0, column=0, sticky="nsew")

        # Title
        lbl_title = ctk.CTkLabel(self.left_sidebar, text="🏛️ Tranchot Extractor", font=ctk.CTkFont(size=19, weight="bold"), text_color=THEME_TEXT_MAIN)
        lbl_title.pack(anchor="w", padx=15, pady=(12, 2))
        lbl_sub = ctk.CTkLabel(self.left_sidebar, text="Kartenaufnahme der Rheinlande 1803–1828\nBCDH Universität Bonn", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MUTED, justify="left")
        lbl_sub.pack(anchor="w", padx=15, pady=(0, 12))

        # Section: Map Loading Card
        card_load = ctk.CTkFrame(self.left_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_load.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_load, "📂 1. Karte laden")
        btn_open = ctk.CTkButton(card_load, text="📁 Eigene Karte laden...", command=self._open_file_dialog, height=34, font=ctk.CTkFont(weight="bold"))
        btn_open.pack(fill="x", padx=12, pady=3)

        self.sample_dropdown = ctk.CTkOptionMenu(
            card_load,
            values=["Beispiel: Nickenich (GeoTIFF)", "Beispiel: Rommerskirchen", "Beispiel: Kruft"],
            command=self._on_sample_selected,
            height=30
        )
        self.sample_dropdown.pack(fill="x", padx=12, pady=(3, 10))

        # Section: Automatic Extraction Card
        card_auto = ctk.CTkFrame(self.left_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_auto.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_auto, "⚡ 2. Automatische Extraktion")
        
        btn_auto_landuse = ctk.CTkButton(
            card_auto,
            text="🌲 Gesamte Landnutzung extrahieren",
            command=self._run_auto_landuse,
            fg_color="#27ae60", hover_color="#2ecc71",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_landuse.pack(fill="x", padx=12, pady=3)

        btn_auto_roads = ctk.CTkButton(
            card_auto,
            text="🛣️ Hauptachsen (≥1 km) extrahieren",
            command=self._run_auto_roads,
            fg_color="#d35400", hover_color="#e67e22",
            height=32, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_roads.pack(fill="x", padx=12, pady=3)

        btn_auto_all_sampled = ctk.CTkButton(
            card_auto,
            text="🟠 Alle gesampelten Klassen extrahieren",
            command=self._run_extract_all_sampled_classes,
            fg_color="#8e44ad", hover_color="#9b59b6",
            height=32, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_all_sampled.pack(fill="x", padx=12, pady=(3, 10))

        # Section: Tools Card
        card_tools = ctk.CTkFrame(self.left_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_tools.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_tools, "🛠️ 3. Werkzeuge")
        self.tool_var = ctk.StringVar(value="pipette")

        r_pipette = ctk.CTkRadioButton(card_tools, text="🎨 Farb-Pipette (Klick zum Samplen)", variable=self.tool_var, value="pipette", command=self._on_tool_change)
        r_pipette.pack(anchor="w", padx=12, pady=2)

        r_road_snap = ctk.CTkRadioButton(card_tools, text="🛣️ Straße nachverfolgen (A ➔ B)", variable=self.tool_var, value="road_snap", command=self._on_tool_change)
        r_road_snap.pack(anchor="w", padx=12, pady=2)

        r_toponym = ctk.CTkRadioButton(card_tools, text="🏷️ Beschriftung aufziehen (Box)", variable=self.tool_var, value="toponym", command=self._on_tool_change)
        r_toponym.pack(anchor="w", padx=12, pady=2)

        r_select = ctk.CTkRadioButton(card_tools, text="👆 Objekt auswählen & löschen", variable=self.tool_var, value="select", command=self._on_tool_change)
        r_select.pack(anchor="w", padx=12, pady=2)

        r_pan = ctk.CTkRadioButton(card_tools, text="✋ Karte verschieben (Pan)", variable=self.tool_var, value="pan", command=self._on_tool_change)
        r_pan.pack(anchor="w", padx=12, pady=2)

        btn_delete_sel = ctk.CTkButton(card_tools, text="🗑️ Ausgewähltes Element löschen", command=self._delete_selected, fg_color="#c0392b", hover_color="#e74c3c", height=28)
        btn_delete_sel.pack(fill="x", padx=12, pady=(6, 3))

        btn_clear = ctk.CTkButton(card_tools, text="🧹 Alle Layer leeren", command=self._clear_all_features, fg_color="#7f8c8d", hover_color="#95a5a6", height=28)
        btn_clear.pack(fill="x", padx=12, pady=(2, 10))

        # Section: Layer Visibility Card
        card_layers = ctk.CTkFrame(self.left_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_layers.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_layers, "👁️ 4. Ebenen anzeigen")
        self.show_bldgs_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🏛️ Gebäude / Hofanlagen", variable=self.show_bldgs_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_roads_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🛣️ Straßen-Mittellinien", variable=self.show_roads_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_toponyms_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🏷️ Beschriftungen / Namen", variable=self.show_toponyms_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_forests_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🌲 Waldflächen", variable=self.show_forests_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_meadows_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🌿 Wiesen / Feuchtgrünland", variable=self.show_meadows_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_water_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="💧 Gewässer / Seen", variable=self.show_water_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_gravel_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🟠 Kies- & Schotterbänke", variable=self.show_gravel_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_vineyard_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🍇 Weinberge (Rebhänge)", variable=self.show_vineyard_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=2)

        self.show_garden_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_layers, text="🟨 Gärten / Nutzkulturen", variable=self.show_garden_var, command=self._on_layer_toggle).pack(anchor="w", padx=12, pady=(2, 10))

        # Section: Export & Target EPSG Card
        card_export = ctk.CTkFrame(self.left_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_export.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_export, "💾 5. GIS Export & KBS")

        ctk.CTkLabel(card_export, text="Ausgabe-KBS (EPSG):", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MUTED).pack(anchor="w", padx=12, pady=(2, 0))
        self.entry_epsg = ctk.CTkEntry(card_export, height=30, font=ctk.CTkFont(size=12, weight="bold"))
        self.entry_epsg.insert(0, "EPSG:25832")
        self.entry_epsg.pack(fill="x", padx=12, pady=(0, 6))

        btn_exp_gpkg = ctk.CTkButton(card_export, text="💾 GeoPackage (.gpkg) [Farbstile]", command=lambda: self._export_gis("gpkg"), fg_color="#27ae60", hover_color="#2ecc71", height=34, font=ctk.CTkFont(weight="bold"))
        btn_exp_gpkg.pack(fill="x", padx=12, pady=3)

        btn_exp_shp = ctk.CTkButton(card_export, text="💾 ESRI Shapefile (.shp)", command=lambda: self._export_gis("shp"), fg_color="#2980b9", hover_color="#3498db", height=30)
        btn_exp_shp.pack(fill="x", padx=12, pady=3)

        btn_exp_geojson = ctk.CTkButton(card_export, text="💾 GeoJSON (.geojson)", command=lambda: self._export_gis("geojson"), fg_color="#8e44ad", hover_color="#9b59b6", height=30)
        btn_exp_geojson.pack(fill="x", padx=12, pady=(3, 10))

        # =========================================================================
        # 2. Main Canvas Area (Center)
        # =========================================================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=THEME_MAIN_FG)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top Bar on Canvas
        self.topbar = ctk.CTkFrame(self.main_frame, height=44, corner_radius=0, fg_color=THEME_TOPBAR_FG)
        self.topbar.grid(row=0, column=0, sticky="ew")

        self.lbl_count = ctk.CTkLabel(self.topbar, text="🏛️ 0 | 🛣️ 0 | 🏷️ 0 | 🌲 0 | 🌿 0 | 💧 0 | 🟠 0 | 🍇 0", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME_TEXT_MAIN)
        self.lbl_count.pack(side="left", padx=15, pady=8)

        # Appearance Mode Selector (Light/Dark Switcher)
        self.opt_theme = ctk.CTkOptionMenu(
            self.topbar,
            values=["☀️ Hell (Standard)", "🌙 Dunkel", "💻 System"],
            command=self._on_theme_changed,
            width=150, height=28,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.opt_theme.pack(side="right", padx=(6, 15), pady=6)

        # Toggle Button: Raw vs Restored View
        self.btn_toggle_view = ctk.CTkButton(
            self.topbar,
            text="✨ Restauriert (Aktiv)",
            width=160,
            command=self._toggle_enhanced_view,
            fg_color="#27ae60", hover_color="#2ecc71",
            height=28, font=ctk.CTkFont(size=11, weight="bold")
        )
        self.btn_toggle_view.pack(side="right", padx=6, pady=6)

        btn_reset_view = ctk.CTkButton(self.topbar, text="🔍 Einpassen", width=110, command=lambda: self.canvas.reset_view(), height=28)
        btn_reset_view.pack(side="right", padx=6, pady=6)

        # Canvas
        self.canvas = MapCanvas(self.main_frame, app=self)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        # Status Bar
        self.status_bar = ctk.CTkFrame(self.main_frame, height=30, corner_radius=0, fg_color=THEME_STATUSBAR_FG)
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.lbl_status = ctk.CTkLabel(self.status_bar, text="Bereit.", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN)
        self.lbl_status.pack(side="left", padx=15, pady=4)

        self.lbl_coords = ctk.CTkLabel(self.status_bar, text="X: 0 | Y: 0 | Zoom: 100%", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MUTED)
        self.lbl_coords.pack(side="right", padx=15, pady=4)

        # =========================================================================
        # 3. Right Sidebar: Dedicated Pipette & Color Calibration Panel
        # =========================================================================
        self.right_sidebar = ctk.CTkScrollableFrame(self, width=350, corner_radius=0, fg_color=THEME_SIDEBAR_FG)
        self.right_sidebar.grid(row=0, column=2, sticky="nsew")

        # Section 0 Card: White-Balance & De-Yellowing
        card_enhance = ctk.CTkFrame(self.right_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_enhance.pack(fill="x", padx=10, pady=4)

        lbl_enhance_title = ctk.CTkLabel(card_enhance, text="✨ 0. Weißabgleich & Entgilbung", font=ctk.CTkFont(size=14, weight="bold"), text_color="#27ae60")
        lbl_enhance_title.pack(anchor="w", padx=12, pady=(10, 2))

        self.enhance_active_var = ctk.BooleanVar(value=True)
        cb_enhance = ctk.CTkCheckBox(
            card_enhance,
            text="✨ Entgilbung & Farbverstärkung aktiv",
            variable=self.enhance_active_var,
            command=self._apply_enhancement_settings,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        cb_enhance.pack(anchor="w", padx=12, pady=(2, 6))

        self.lbl_deyellow_header = ctk.CTkLabel(card_enhance, text="Entgilbung / Weißabgleich (85%):", font=ctk.CTkFont(size=11))
        self.lbl_deyellow_header.pack(anchor="w", padx=12, pady=(2, 0))

        self.slider_deyellow = ctk.CTkSlider(card_enhance, from_=0.0, to=1.0, number_of_steps=20, command=self._on_deyellow_slider_moved)
        self.slider_deyellow.set(0.85)
        self.slider_deyellow.pack(fill="x", padx=12, pady=2)

        self.lbl_vibrance_header = ctk.CTkLabel(card_enhance, text="Farb-Leuchtkraft (1.85×):", font=ctk.CTkFont(size=11))
        self.lbl_vibrance_header.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_vibrance = ctk.CTkSlider(card_enhance, from_=1.0, to=3.0, number_of_steps=20, command=self._on_vibrance_slider_moved)
        self.slider_vibrance.set(1.85)
        self.slider_vibrance.pack(fill="x", padx=12, pady=(2, 10))

        # Section 1 Card: Pipette Calibration
        card_pipette = ctk.CTkFrame(self.right_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_pipette.pack(fill="x", padx=10, pady=4)

        lbl_right_title = ctk.CTkLabel(card_pipette, text="🎨 1. Blatt-Kalibrierung & Pipetten", font=ctk.CTkFont(size=14, weight="bold"), text_color="#0984e3")
        lbl_right_title.pack(anchor="w", padx=12, pady=(10, 2))
        
        lbl_right_desc = ctk.CTkLabel(
            card_pipette,
            text="Wähle eine Pipette und klicke auf die Karte, um den Farbton für dieses Blatt zu lernen.",
            font=ctk.CTkFont(size=10), text_color=THEME_TEXT_MUTED, justify="left", wraplength=310
        )
        lbl_right_desc.pack(anchor="w", padx=12, pady=(0, 6))

        self.pipette_dropdown = ctk.CTkOptionMenu(
            card_pipette,
            values=[
                "🌲 Wald (Laub/Nadel)",
                "🌿 Wiese / Aue / Grünland",
                "💧 Gewässer / Rhein / Bäche",
                "🟠 Kies- & Schotterbänke (Flussbett)",
                "🍇 Weinberge (Rebhänge)",
                "🟨 Gärten / Nutzkulturen",
                "🏛️ Gebäude (Karminrot)",
                "📜 Pergament / Hintergrund"
            ],
            command=self._on_pipette_class_selected,
            height=34,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.pipette_dropdown.pack(fill="x", padx=12, pady=4)

        self.swatch_frame = ctk.CTkFrame(card_pipette, fg_color=THEME_SWATCH_FRAME, height=46)
        self.swatch_frame.pack(fill="x", padx=12, pady=4)

        self.swatch_box = ctk.CTkFrame(
            self.swatch_frame,
            fg_color="#27ae60",
            width=28,
            height=24,
            corner_radius=4,
            border_width=1,
            border_color=("#CBD5E1", "#475569")
        )
        self.swatch_box.pack_propagate(False)
        self.swatch_box.pack(side="left", padx=8, pady=6)

        self.lbl_swatch_info = ctk.CTkLabel(self.swatch_frame, text="Farbe: Standard-Vorlage\nKlicke Karte zum Samplen", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN, justify="left")
        self.lbl_swatch_info.pack(side="left", padx=4, pady=4)

        self.lbl_tol_header = ctk.CTkLabel(card_pipette, text="Toleranz / Farbabstand (ΔE = 8):", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME_TEXT_MAIN)
        self.lbl_tol_header.pack(anchor="w", padx=12, pady=(6, 0))

        self.slider_tol = ctk.CTkSlider(card_pipette, from_=2, to=25, number_of_steps=23, command=self._on_tolerance_changed)
        self.slider_tol.set(8)
        self.slider_tol.pack(fill="x", padx=12, pady=2)

        btn_pattern_poly = ctk.CTkButton(
            card_pipette,
            text="📐 Muster-Polygon zeichnen (Few-Shot)",
            command=lambda: self._set_active_tool("pattern_poly"),
            fg_color="#8e44ad", hover_color="#9b59b6",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_pattern_poly.pack(fill="x", padx=12, pady=(6, 2))

        btn_landuse_box = ctk.CTkButton(
            card_pipette,
            text="📐 Flächen ROI-Box aufziehen (Testen)",
            command=lambda: self._set_active_tool("landuse_box"),
            fg_color="#27ae60", hover_color="#2ecc71",
            height=30, font=ctk.CTkFont(size=11)
        )
        btn_landuse_box.pack(fill="x", padx=12, pady=2)

        btn_extract_competitive = ctk.CTkButton(
            card_pipette,
            text="⚡ Gelernte Flächen berechnen (OK)",
            command=self._run_extract_all_sampled_classes,
            fg_color="#16a085", hover_color="#1abc9c",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_extract_competitive.pack(fill="x", padx=12, pady=3)

        btn_extract_active = ctk.CTkButton(
            card_pipette,
            text="🎯 Nur aktive Klasse berechnen (OK)",
            command=self._run_extract_active_sample,
            fg_color="#2980b9", hover_color="#3498db",
            height=30, font=ctk.CTkFont(size=11)
        )
        btn_extract_active.pack(fill="x", padx=12, pady=(3, 8))

        # Palette list card
        card_swatches = ctk.CTkFrame(self.right_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_swatches.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_swatches, "📋 Farbfelder aller Klassen")
        self._build_palette_class_list_inside(card_swatches)

        # Profile Save/Load Card
        card_prof = ctk.CTkFrame(self.right_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_prof.pack(fill="x", padx=10, pady=4)

        self._create_section_label(card_prof, "💾 Blatt-Profil (.palette.json)")
        btn_prof_frame = ctk.CTkFrame(card_prof, fg_color="transparent")
        btn_prof_frame.pack(fill="x", padx=12, pady=(2, 10))

        btn_save_prof = ctk.CTkButton(btn_prof_frame, text="💾 Speichern", command=self._save_palette_profile, height=30, width=140)
        btn_save_prof.pack(side="left")

        btn_load_prof = ctk.CTkButton(btn_prof_frame, text="📂 Laden", command=self._load_palette_profile, height=30, width=140)
        btn_load_prof.pack(side="right")

        # Section 2 Card: Building Footprint & Courtyard Extraction Panel
        card_bldg = ctk.CTkFrame(self.right_sidebar, fg_color=THEME_CARD_FG, border_color=THEME_CARD_BORDER, border_width=1, corner_radius=8)
        card_bldg.pack(fill="x", padx=10, pady=4)

        lbl_bldg_title = ctk.CTkLabel(card_bldg, text="🏛️ 2. Gebäude & Hofanlagen", font=ctk.CTkFont(size=14, weight="bold"), text_color="#c0392b")
        lbl_bldg_title.pack(anchor="w", padx=12, pady=(10, 2))

        lbl_bldg_desc = ctk.CTkLabel(
            card_bldg,
            text="Zeichne Siedlungsgrenzen zur Exklusion oder ziehe eine ROI-Box auf der Karte auf.",
            font=ctk.CTkFont(size=10), text_color=THEME_TEXT_MUTED, justify="left", wraplength=310
        )
        lbl_bldg_desc.pack(anchor="w", padx=12, pady=(0, 6))

        btn_settlement_poly = ctk.CTkButton(
            card_bldg,
            text="🏘️ Siedlungs-Grenze zeichnen (Sofort-Erkennung)",
            command=lambda: self._set_active_tool("settlement_poly"),
            fg_color="#8e44ad", hover_color="#9b59b6",
            height=36, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_settlement_poly.pack(fill="x", padx=12, pady=(4, 2))

        btn_clear_settlement = ctk.CTkButton(
            card_bldg,
            text="🗑️ Siedlungsgrenzen zurücksetzen",
            command=self._clear_settlement_boundaries,
            fg_color="#7f8c8d", hover_color="#95a5a6",
            height=26, font=ctk.CTkFont(size=10)
        )
        btn_clear_settlement.pack(fill="x", padx=12, pady=(1, 4))

        btn_bldg_click = ctk.CTkButton(
            card_bldg,
            text="🏛️ Einzelnes Gebäude anklicken",
            command=lambda: self._set_active_tool("extract"),
            fg_color="#34495e", hover_color="#2c3e50",
            height=30, font=ctk.CTkFont(size=11)
        )
        btn_bldg_click.pack(fill="x", padx=12, pady=3)

        self.lbl_bldg_diff_header = ctk.CTkLabel(card_bldg, text="Rot-Empfindlichkeit (RGB-Diff = 14):", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME_TEXT_MAIN)
        self.lbl_bldg_diff_header.pack(anchor="w", padx=12, pady=(6, 0))

        self.slider_bldg_diff = ctk.CTkSlider(card_bldg, from_=8, to=40, number_of_steps=32, command=self._on_bldg_diff_changed)
        self.slider_bldg_diff.set(14)
        self.slider_bldg_diff.pack(fill="x", padx=12, pady=2)

        self.lbl_bldg_min_area = ctk.CTkLabel(card_bldg, text="Mindestfläche (6 px²):", font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN)
        self.lbl_bldg_min_area.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_bldg_area = ctk.CTkSlider(card_bldg, from_=4, to=40, number_of_steps=36, command=self._on_bldg_area_changed)
        self.slider_bldg_area.set(6)
        self.slider_bldg_area.pack(fill="x", padx=12, pady=2)

        self.cb_filter_terraces_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_bldg, text="🍇 Weinberg-Terrassen & Schraffen filtern", variable=self.cb_filter_terraces_var, font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN).pack(anchor="w", padx=12, pady=2)

        self.cb_regularize_rect_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_bldg, text="📐 Einzelhäuser zu 4-Eck-Rechtecken", variable=self.cb_regularize_rect_var, font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN).pack(anchor="w", padx=12, pady=2)

        self.cb_regularize_ortho_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(card_bldg, text="🏛️ Hofanlagen 90° orthogonal ausrichten", variable=self.cb_regularize_ortho_var, font=ctk.CTkFont(size=11), text_color=THEME_TEXT_MAIN).pack(anchor="w", padx=12, pady=2)

        btn_auto_settlement_bldgs = ctk.CTkButton(
            card_bldg,
            text="⚡ Nur Gebäude in Siedlungsgrenzen",
            command=self._run_settlement_buildings,
            fg_color="#8e44ad", hover_color="#9b59b6",
            height=36, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_settlement_bldgs.pack(fill="x", padx=12, pady=(8, 2))

        btn_auto_all_bldgs = ctk.CTkButton(
            card_bldg,
            text="⚡ Alle Gebäude der Karte extrahieren",
            command=self._run_auto_buildings,
            fg_color="#c0392b", hover_color="#e74c3c",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_all_bldgs.pack(fill="x", padx=12, pady=(2, 3))

        btn_clear_bldgs = ctk.CTkButton(
            card_bldg,
            text="🧹 Nur Gebäude-Layer leeren",
            command=self._clear_buildings_layer,
            fg_color="#7f8c8d", hover_color="#95a5a6",
            height=28, font=ctk.CTkFont(size=11)
        )
        btn_clear_bldgs.pack(fill="x", padx=12, pady=(2, 12))

    def _create_section_label(self, parent, text: str):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME_TEXT_SECTION)
        lbl.pack(anchor="w", padx=12, pady=(8, 4))

    def _on_theme_changed(self, choice: str):
        if "Hell" in choice:
            ctk.set_appearance_mode("Light")
        elif "Dunkel" in choice:
            ctk.set_appearance_mode("Dark")
        else:
            ctk.set_appearance_mode("System")
        self.canvas.update_theme()
        self.lbl_status.configure(text=f"Erscheinungsbild gewechselt zu: {choice}")

    def _on_deyellow_slider_moved(self, val: float):
        self.lbl_deyellow_header.configure(text=f"Entgilbung / Weißabgleich ({int(val*100)}%):")
        self._apply_enhancement_settings()

    def _on_vibrance_slider_moved(self, val: float):
        self.lbl_vibrance_header.configure(text=f"Farb-Leuchtkraft ({val:.2f}×):")
        self._apply_enhancement_settings()

    def _apply_enhancement_settings(self):
        if self.raw_np is None:
            return

        if self.enhance_active_var.get():
            dey = float(self.slider_deyellow.get())
            vib = float(self.slider_vibrance.get())
            paper_sample = self.sampler.samples.get("paper")
            paper_rgb = paper_sample.rgb if (paper_sample and paper_sample.active) else None

            self.enhanced_np = ColorEnhancer.enhance_map(
                self.raw_np,
                paper_rgb=paper_rgb,
                deyellow_strength=dey,
                vibrance=vib,
            )
            self.enhanced_pil = Image.fromarray(self.enhanced_np)
            self.current_np = self.enhanced_np
            self.current_pil = self.enhanced_pil
            self.btn_toggle_view.configure(text="✨ Restauriert (Aktiv)", fg_color="#27ae60", hover_color="#2ecc71")
        else:
            self.current_np = self.raw_np
            self.current_pil = self.raw_pil
            self.btn_toggle_view.configure(text="📜 Original (Aktiv)", fg_color="#7f8c8d", hover_color="#95a5a6")

        self.canvas.update_image_source(self.current_pil, self.current_np)
        self._update_active_swatch_display()

    def _toggle_enhanced_view(self):
        new_state = not self.enhance_active_var.get()
        self.enhance_active_var.set(new_state)
        self._apply_enhancement_settings()
        status_txt = "Restaurierte Farben (Weißabgleich & Leuchtkraft)" if new_state else "Originale Kartenfarben (Roh)"
        self.lbl_status.configure(text=f"Ansicht gewechselt zu: {status_txt}")

    def _build_palette_class_list_inside(self, parent):
        for item in PipetteSampler.DEFAULT_CLASSES:
            cid = item["class_id"]
            lbl_text = item["label"]
            init_hex = item["hex"]

            row = ctk.CTkFrame(parent, fg_color=THEME_SWATCH_FRAME, height=32)
            row.pack(fill="x", padx=12, pady=2)

            swatch = ctk.CTkFrame(
                row,
                fg_color=init_hex,
                width=24,
                height=20,
                corner_radius=4,
                border_width=1,
                border_color=("#CBD5E1", "#475569")
            )
            swatch.pack_propagate(False)
            swatch.pack(side="left", padx=(6, 4), pady=3)
            self.swatch_boxes[cid] = swatch

            btn_select = ctk.CTkButton(
                row, text=lbl_text, width=130, height=24, fg_color="transparent",
                text_color=THEME_TEXT_MAIN, hover_color=("#E2E8F0", "#334155"),
                font=ctk.CTkFont(size=11, weight="bold"), anchor="w",
                command=lambda c=cid, l=lbl_text: self._activate_class_pipette(c, l)
            )
            btn_select.pack(side="left", padx=2)

            btn_reset = ctk.CTkButton(
                row, text="↺", width=26, height=24, fg_color="#64748B", hover_color="#EF4444",
                text_color="#FFFFFF", font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda c=cid, l=lbl_text: self._reset_specific_class_samples(c, l)
            )
            btn_reset.pack(side="right", padx=(2, 4))

            btn_ext = ctk.CTkButton(
                row, text="⚡", width=26, height=24, fg_color="#2c3e50", hover_color="#3498db",
                command=lambda c=cid: self._run_extract_specific_class(c)
            )
            btn_ext.pack(side="right", padx=2)

        btn_clear_all_samples = ctk.CTkButton(
            parent,
            text="🗑️ Alle Muster & Flächen zurücksetzen",
            command=self._reset_all_class_samples,
            fg_color="#64748B", hover_color="#EF4444",
            height=28, font=ctk.CTkFont(size=11)
        )
        btn_clear_all_samples.pack(fill="x", padx=12, pady=(4, 8))

    def _reset_specific_class_samples(self, class_id: str, label: str):
        if class_id in self.exemplar_polygons:
            self.exemplar_polygons[class_id].clear()
        if class_id in self.extracted_layers:
            self.extracted_layers[class_id].clear()
        self.sampler.reset_class(class_id)
        if class_id in self.swatch_boxes:
            sample = self.sampler.get_sample(class_id)
            if sample:
                self.swatch_boxes[class_id].configure(fg_color=sample.hex_color)
        if self.active_pipette_class == class_id:
            self._update_active_swatch_display()
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"↺ Alle Muster & Flächen für '{label}' zurückgesetzt.")

    def _reset_all_class_samples(self):
        for cid in list(self.exemplar_polygons.keys()):
            self.exemplar_polygons[cid].clear()
        for cid in list(self.extracted_layers.keys()):
            self.extracted_layers[cid].clear()
        self.sampler._init_defaults()
        for cid, swatch in self.swatch_boxes.items():
            sample = self.sampler.get_sample(cid)
            if sample:
                swatch.configure(fg_color=sample.hex_color)
        self._update_active_swatch_display()
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text="🗑️ Alle Few-Shot Muster und extrahierten Flächen zurückgesetzt.")

    def _activate_class_pipette(self, class_id: str, label: str):
        self.active_pipette_class = class_id
        self.pipette_dropdown.set(label)
        self.tool_var.set("pipette")
        self.active_tool = "pipette"
        self._update_active_swatch_display()
        self.lbl_status.configure(text=f"Pipette aktiviert für '{label}'. Klicke auf die Karte zum Samplen.")

    def _on_pipette_class_selected(self, choice: str):
        mapping = {
            "🌲 Wald (Laub/Nadel)": "forest",
            "🌿 Wiese / Aue / Grünland": "meadow",
            "💧 Gewässer / Rhein / Bäche": "water",
            "🟠 Kies- & Schotterbänke (Flussbett)": "gravel",
            "🍇 Weinberge (Rebhänge)": "vineyard",
            "🟨 Gärten / Nutzkulturen": "garden",
            "🏛️ Gebäude (Karminrot)": "building",
            "📜 Pergament / Hintergrund": "paper"
        }
        self.active_pipette_class = mapping.get(choice, "forest")
        self.tool_var.set("pipette")
        self.active_tool = "pipette"
        self._update_active_swatch_display()
        self.lbl_status.configure(text=f"Pipette aktiv für: {choice}. Klicke auf die Karte zum Samplen.")

    def _update_active_swatch_display(self):
        sample = self.sampler.samples.get(self.active_pipette_class)
        if sample:
            self.swatch_box.configure(fg_color=sample.hex_color)
            if self.active_pipette_class in self.swatch_boxes:
                self.swatch_boxes[self.active_pipette_class].configure(fg_color=sample.hex_color)
            self.slider_tol.set(sample.tolerance)
            self.lbl_tol_header.configure(text=f"Toleranz / Farbabstand (ΔE = {int(sample.tolerance)}):")
            status = "gesampelt" if sample.active else "Standard"
            self.lbl_swatch_info.configure(text=f"Farbe: RGB({sample.rgb[0]},{sample.rgb[1]},{sample.rgb[2]}) ({status})\nKlicke Karte zum Samplen")

    def _on_tolerance_changed(self, val: float):
        sample = self.sampler.samples.get(self.active_pipette_class)
        if sample:
            sample.tolerance = int(val)
            self.lbl_tol_header.configure(text=f"Toleranz / Farbabstand (ΔE = {int(val)}):")

    def handle_pipette_sample_at(self, ix: float, iy: float):
        if self.current_np is None:
            return

        sample = self.sampler.sample_from_coordinate(self.current_np, self.active_pipette_class, ix, iy, radius=4)
        self._update_active_swatch_display()
        
        if self.active_pipette_class == "paper":
            self._apply_enhancement_settings()
            self.lbl_status.configure(text=f"📜 Pergament-Referenz gesetzt auf RGB{tuple(sample.rgb)} — Weißabgleich angepasst!")
        else:
            self.lbl_status.configure(text=f"🎨 '{sample.label}' erfolgreich gesampelt bei ({int(ix)}, {int(iy)})!")

    def _run_extract_active_sample(self):
        self._run_extract_specific_class(self.active_pipette_class)

    def _run_extract_specific_class(self, class_id: str):
        if self.current_np is None:
            return

        sample = self.sampler.samples.get(class_id)
        tol = sample.tolerance if sample else int(self.slider_tol.get())

        self.lbl_status.configure(text=f"Extrahiere '{class_id}' mit Toleranz ΔE={tol}...")
        self.update_idletasks()

        threading.Thread(target=self._async_extract_sample, args=(class_id, tol), daemon=True).start()

    def _async_extract_sample(self, class_id: str, tol: int):
        try:
            polys = self.sampler.extract_class_polygons(self.current_np, class_id, tolerance_override=tol)
            self.extracted_layers[class_id] = polys
            self.after(0, lambda: self._on_sample_extracted(class_id, len(polys)))
        except Exception as e:
            self.after(0, lambda: self._on_error(f"Extraktionsfehler: {e}"))

    def _on_sample_extracted(self, class_id: str, count: int):
        self._update_counts()
        self.canvas.redraw()
        sample = self.sampler.samples.get(class_id)
        label = sample.label if sample else class_id
        self.lbl_status.configure(text=f"✅ {count} Flächen für '{label}' erfolgreich extrahiert!")

    def _run_extract_all_sampled_classes(self):
        if self.current_np is None:
            return

        self.lbl_status.configure(text="Extrahiere alle kalibrierten Klassen kompetitiv...")
        self.update_idletasks()

        threading.Thread(target=self._async_extract_all_classes, daemon=True).start()

    def handle_landuse_roi_box(self, ix0: float, iy0: float, ix1: float, iy1: float):
        if self.current_np is None:
            return

        min_x = max(0, int(min(ix0, ix1)))
        max_x = min(self.current_np.shape[1], int(max(ix0, ix1)))
        min_y = max(0, int(min(iy0, iy1)))
        max_y = min(self.current_np.shape[0], int(max(iy0, iy1)))

        if (max_x - min_x) < 20 or (max_y - min_y) < 20:
            return

        crop = self.current_np[min_y:max_y, min_x:max_x]
        all_res = self.sampler.extract_competitive_polygons(crop)

        total_found = 0
        for cid, polys in all_res.items():
            translated = [translate(p, xoff=min_x, yoff=min_y) for p in polys]
            self.extracted_layers[cid] = translated
            total_found += len(translated)

        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"🌲 Flächen ROI-Test: {total_found} Flächen (kompetitiv, 0 Überlappung) extrahiert.")

    def _async_extract_all_classes(self):
        try:
            total_added = 0
            all_res = self.sampler.extract_competitive_polygons(self.current_np)
            for cid, polys in all_res.items():
                if polys:
                    self.extracted_layers[cid] = polys
                    total_added += len(polys)
            self.after(0, lambda: self._on_all_classes_complete(total_added))
        except Exception as e:
            self.after(0, lambda: self._on_error(f"Fehler: {e}"))

    def _on_all_classes_complete(self, total_count: int):
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"Fertig: {total_count} Flächen über alle Klassen extrahiert (kompetitiv, 0 Überlappung)!")

    def _run_auto_landuse(self):
        if self.current_np is None:
            return

        self.lbl_status.configure(text="Extrahiere Landnutzung (Wald, Wiesen, Gewässer)...")
        self.update_idletasks()

        threading.Thread(target=self._async_landuse_extract, daemon=True).start()

    def _async_landuse_extract(self):
        try:
            extractor = LandUseExtractor()
            res = extractor.extract(self.current_np)
            self.extracted_layers["forest"] = res.forest_polygons
            self.extracted_layers["meadow"] = res.meadow_polygons
            self.extracted_layers["water"] = res.water_polygons
            self.after(0, lambda: self._on_landuse_complete(len(res.forest_polygons), len(res.meadow_polygons), len(res.water_polygons), res.execution_time_s))
        except Exception as e:
            self.after(0, lambda: self._on_error(f"Landnutzung Fehler: {e}"))

    def _on_landuse_complete(self, num_f: int, num_m: int, num_w: int, time_s: float):
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"Fertig in {time_s:.2f}s: {num_f} Wälder, {num_m} Wiesen, {num_w} Gewässer extrahiert!")

    def _on_bldg_diff_changed(self, val: float):
        if hasattr(self, "lbl_bldg_diff_header"):
            self.lbl_bldg_diff_header.configure(text=f"Rot-Empfindlichkeit (RGB-Diff = {int(val)}):")

    def _on_bldg_area_changed(self, val: float):
        if hasattr(self, "lbl_bldg_min_area"):
            self.lbl_bldg_min_area.configure(text=f"Mindestfläche ({int(val)} px²):")

    def _get_current_building_config(self) -> BuildingConfig:
        rgb_diff = int(self.slider_bldg_diff.get()) if hasattr(self, "slider_bldg_diff") else 14
        min_area = float(self.slider_bldg_area.get()) if hasattr(self, "slider_bldg_area") else 6.0
        filter_terraces = self.cb_filter_terraces_var.get() if hasattr(self, "cb_filter_terraces_var") else True
        regularize_rect = self.cb_regularize_rect_var.get() if hasattr(self, "cb_regularize_rect_var") else True
        regularize_ortho = self.cb_regularize_ortho_var.get() if hasattr(self, "cb_regularize_ortho_var") else True

        return BuildingConfig(
            rgb_diff_threshold=rgb_diff,
            min_building_area_px=min_area,
            filter_vineyard_terraces=filter_terraces,
            regularize_orthogonal=regularize_rect or regularize_ortho,
            min_stroke_width_px=1.0,
            lab_a_threshold=128,
        )

    def _set_active_tool(self, tool_name: str):
        self.tool_var.set(tool_name)
        self._on_tool_change()

    def handle_pattern_polygon(self, polygon_pts: List[Tuple[float, float]]):
        if self.current_np is None or len(polygon_pts) < 3:
            return

        poly = Polygon(polygon_pts)
        if poly.is_valid and not poly.is_empty:
            if self.active_pipette_class not in self.exemplar_polygons:
                self.exemplar_polygons[self.active_pipette_class] = []
            self.exemplar_polygons[self.active_pipette_class].append(poly)

        sample = self.sampler.sample_from_polygon(self.current_np, self.active_pipette_class, polygon_pts)
        if sample:
            self._update_active_swatch_display()
            label = sample.label
            count = len(self.exemplar_polygons.get(self.active_pipette_class, []))
            total_samples = sum(len(v) for v in self.exemplar_polygons.values())
            self.lbl_status.configure(
                text=f"📐 Muster #{count} für '{label}' gespeichert ({total_samples} Muster gesamt)! Zeichne weitere Muster oder klicke '⚡ Gelernte Flächen berechnen (OK)'."
            )
        self.canvas.redraw()

    def handle_settlement_polygon(self, polygon_pts: List[Tuple[float, float]]):
        if self.current_np is None or len(polygon_pts) < 3:
            return
        try:
            poly = Polygon(polygon_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty and isinstance(poly, Polygon):
                self.settlement_boundaries.append(poly)

                # Instant building extraction inside this settlement polygon
                minx, miny, maxx, maxy = [int(v) for v in poly.bounds]
                h, w = self.current_np.shape[:2]
                x0, y0 = max(0, minx), max(0, miny)
                x1, y1 = min(w, maxx), min(h, maxy)

                if (x1 - x0) >= 6 and (y1 - y0) >= 6:
                    crop_rgb = self.current_np[y0:y1, x0:x1]
                    local_poly = translate(poly, xoff=-x0, yoff=-y0)
                    cfg = self._get_current_building_config()
                    extractor = BuildingExtractor(cfg)
                    result = extractor.extract(crop_rgb, settlement_boundaries=[local_poly])

                    # Remove old buildings inside this settlement envelope and insert newly extracted
                    self.extracted_buildings = [
                        p for p in self.extracted_buildings if not poly.contains(p.centroid)
                    ]
                    added_count = 0
                    for f in result.features:
                        p_global = translate(f.geometry, xoff=x0, yoff=y0)
                        if p_global.is_valid and not p_global.is_empty:
                            self.extracted_buildings.append(p_global)
                            added_count += 1

                    self._update_counts()
                    self.lbl_status.configure(
                        text=f"🏘️ Siedlungsgrenze #{len(self.settlement_boundaries)} aktiv: {added_count} Gebäude sofort erkannt!"
                    )
                self.canvas.redraw()
        except Exception as e:
            self._on_error(f"Siedlung Extraktion Fehler: {e}")

    def _run_settlement_buildings(self):
        if self.current_np is None:
            return
        if not self.settlement_boundaries:
            self.lbl_status.configure(
                text="ℹ️ Keine Siedlungsgrenze aktiv. Klicke auf '🏘️ Siedlungs-Grenze zeichnen', um das Dorf einzugrenzen!"
            )
            return
        self._run_auto_buildings()

    def _clear_settlement_boundaries(self):
        self.settlement_boundaries.clear()
        self.lbl_status.configure(text="Siedlungsgrenzen zurückgesetzt. Gebäude-Extraktion gilt für die gesamte Karte.")
        self.canvas.redraw()

    def _clear_buildings_layer(self):
        count = len(self.extracted_buildings)
        self.extracted_buildings.clear()
        self.selected_building_idx = None
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"🧹 Gebäude-Layer geleert ({count} Gebäude entfernt).")

    def _run_auto_buildings(self):
        if self.current_np is None:
            return

        self.lbl_status.configure(text="Extrahiere alle Gebäude & Hofanlagen der Karte...")
        self.update_idletasks()

        cfg = self._get_current_building_config()
        extractor = BuildingExtractor(cfg)
        t0 = time.time()
        result = extractor.extract(self.current_np, settlement_boundaries=self.settlement_boundaries)
        elapsed = time.time() - t0

        self.extracted_buildings = [f.geometry for f in result.features]
        self.selected_building_idx = None
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"Fertig in {elapsed:.2f}s: {len(self.extracted_buildings)} echte Gebäude & Höfe extrahiert!")

    def handle_building_roi_box(self, ix0: float, iy0: float, ix1: float, iy1: float):
        if self.current_np is None:
            return

        h, w = self.current_np.shape[:2]
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)

        if dx < 6 and dy < 6:
            self.extract_single_building_at(ix0, iy0)
            return

        x0 = max(0, int(round(min(ix0, ix1))))
        x1 = min(w, int(round(max(ix0, ix1))))
        y0 = max(0, int(round(min(iy0, iy1))))
        y1 = min(h, int(round(max(iy0, iy1))))

        if (x1 - x0) < 6 or (y1 - y0) < 6:
            return

        crop_rgb = self.current_np[y0:y1, x0:x1]
        cfg = self._get_current_building_config()
        extractor = BuildingExtractor(cfg)

        t0 = time.time()
        result = extractor.extract(crop_rgb)
        elapsed = time.time() - t0

        box_polygon = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.extracted_buildings = [
            p for p in self.extracted_buildings if not box_polygon.contains(p.centroid)
        ]

        added_count = 0
        for f in result.features:
            p_global = translate(f.geometry, xoff=x0, yoff=y0)
            if p_global.is_valid and not p_global.is_empty:
                if self.settlement_boundaries and len(self.settlement_boundaries) > 0:
                    if not any(sb.intersects(p_global.centroid) or sb.intersects(p_global) for sb in self.settlement_boundaries):
                        continue
                self.extracted_buildings.append(p_global)
                added_count += 1

        self.selected_building_idx = None
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(
            text=f"🏛️ ROI-Test: {added_count} Gebäude in Box ({x1-x0}x{y1-y0} px) in {elapsed:.2f}s extrahiert!"
        )

    def _run_auto_roads(self):
        if self.current_np is None:
            return

        self.lbl_status.configure(text="Berechne kontinuierliche Hauptstraßenachsen (≥ 1 km)...")
        self.update_idletasks()

        threading.Thread(target=self._async_road_extract, daemon=True).start()

    def _async_road_extract(self):
        try:
            extractor = RoadExtractor(RoadConfig(min_path_length_px=200.0))
            result = extractor.extract(self.current_np)
            self.extracted_roads = [f.geometry for f in result.features]
            self.selected_road_idx = None
            self.after(0, lambda: self._on_roads_complete(len(self.extracted_roads)))
        except Exception as e:
            self.after(0, lambda: self._on_error(f"Straßenextraktion Fehler: {e}"))

    def _on_roads_complete(self, count: int):
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text=f"Fertig: {count} durchgehende Hauptachsen (≥ 1 km) extrahiert!")

    def _save_palette_profile(self):
        default_name = "palette.json"
        if self.current_image_path:
            base = os.path.splitext(self.current_image_path)[0]
            default_name = f"{base}.palette.json"

        fp = filedialog.asksaveasfilename(
            title="Kalibrierungs-Profil speichern",
            initialfile=os.path.basename(default_name),
            defaultextension=".palette.json",
            filetypes=[("Paletten-Profile", "*.palette.json;*.json")]
        )
        if fp:
            self.sampler.save_palette(fp)
            messagebox.showinfo("Profil gespeichert", f"Kalibrierungs-Profil gespeichert:\n{fp}")

    def _load_palette_profile(self):
        fp = filedialog.askopenfilename(
            title="Kalibrierungs-Profil laden",
            filetypes=[("Paletten-Profile", "*.palette.json;*.json")]
        )
        if fp:
            self.sampler.load_palette(fp)
            self._on_pipette_class_selected(self.pipette_dropdown.get())
            for cid in self.sampler.samples:
                if cid in self.swatch_boxes:
                    self.swatch_boxes[cid].configure(bg=self.sampler.samples[cid].hex_color)
            self._apply_enhancement_settings()
            messagebox.showinfo("Profil geladen", f"Kalibrierungs-Profil erfolgreich geladen!")

    def _on_layer_toggle(self):
        if getattr(self, "canvas", None):
            self.canvas.redraw()

    def _on_tool_change(self):
        self.active_tool = self.tool_var.get()
        self.road_start_pt = None
        names = {
            "pipette": "🎨 Farb-Pipette (Klick zum Samplen)",
            "pattern_poly": "📐 Muster-Polygon (Eckpunkte klicken, Doppelklick oder Enter zum Abschließen)",
            "settlement_poly": "🏘️ Siedlungs-Grenze (Eckpunkte um das Dorf klicken, Doppelklick zum Abschließen)",
            "bldg_box": "📐 Gebäude ROI-Box (Rechteck aufziehen zum Testen & Extrahieren)",
            "landuse_box": "🌲 Flächen ROI-Box (Rechteck aufziehen zum Testen & Extrahieren)",
            "extract": "🏛️ Gebäude-Klick (Einzelnes Gebäude anklicken)",
            "road_snap": "🛣️ Straßen-Nachverfolgung (Klick Startpunkt ➔ Endpunkt)",
            "toponym": "🏷️ Beschriftung aufziehen (Rechteck mit Maus um Text ziehen)",
            "select": "👆 Auswählen & Löschen",
            "pan": "✋ Karte verschieben (Pan)"
        }
        self.lbl_status.configure(text=f"Aktives Werkzeug: {names.get(self.active_tool, self.active_tool)}")
        self.canvas.redraw()

    def _cancel_road_start(self):
        self.road_start_pt = None
        self.canvas.redraw()
        self.lbl_status.configure(text="Aktion zurückgesetzt.")

    def _load_default_sample(self):
        sample_path = "tranchot_extractor/Sample/Nickenich.tif"
        if not os.path.exists(sample_path):
            sample_path = "Sample/Nickenich.tif"
        if os.path.exists(sample_path):
            self.load_image_file(sample_path)

    def _on_sample_selected(self, choice: str):
        paths = {
            "Beispiel: Nickenich (GeoTIFF)": ["tranchot_extractor/Sample/Nickenich.tif", "Sample/Nickenich.tif"],
            "Beispiel: Rommerskirchen": ["tranchot_extractor/sample_data/rommerskirchen_sample.png", "sample_data/rommerskirchen_sample.png"],
            "Beispiel: Kruft": ["tranchot_extractor/Sample/Kruft.tif", "Sample/Kruft.tif"],
        }
        candidates = paths.get(choice, [])
        for p in candidates:
            if os.path.exists(p):
                self.load_image_file(p)
                return
        messagebox.showwarning("Datei nicht gefunden", f"Die Beispieldatei für '{choice}' wurde nicht im Projektordner gefunden.")

    def _open_file_dialog(self):
        fp = filedialog.askopenfilename(
            title="Historische Karte / GeoTIFF öffnen",
            filetypes=[("Raster-Karten", "*.tif;*.tiff;*.png;*.jpg;*.jpeg"), ("Alle Dateien", "*.*")]
        )
        if fp:
            self.load_image_file(fp)

    def load_image_file(self, file_path: str):
        try:
            self.lbl_status.configure(text="Lade Bild & berechne Weißabgleich...")
            self.update_idletasks()

            pil_img = Image.open(file_path).convert("RGB")
            self.current_image_path = file_path
            self.raw_pil = pil_img
            self.raw_np = np.array(pil_img)

            try:
                self.geo_handler = GeoReferenceHandler.from_file(file_path)
            except Exception:
                self.geo_handler = None

            self._clear_all_features()

            base = os.path.splitext(file_path)[0]
            palette_file = f"{base}.palette.json"
            if os.path.exists(palette_file):
                self.sampler.load_palette(palette_file)

            self._apply_enhancement_settings()
            self.canvas.reset_view()

            iw, ih = pil_img.size
            crs_name = self.geo_handler.crs if self.geo_handler and self.geo_handler.crs else "Lokal (px)"
            self.lbl_status.configure(text=f"Geladen: {os.path.basename(file_path)} ({iw}×{ih} px) | Weißabgleich aktiv | CRS: {crs_name}")
            self._update_active_swatch_display()
        except Exception as e:
            messagebox.showerror("Fehler beim Laden", str(e))

    def _on_error(self, err_msg: str):
        self.lbl_status.configure(text=err_msg)
        messagebox.showerror("Fehler", err_msg)

    def handle_road_snap_click(self, ix: float, iy: float):
        if self.current_np is None:
            return

        if self.road_start_pt is None:
            self.road_start_pt = (ix, iy)
            self.canvas.redraw()
            self.lbl_status.configure(text=f"📍 Startpunkt bei ({int(ix)}, {int(iy)}) gesetzt. Klicke jetzt auf das Ende der Straße...")
        else:
            start_p = self.road_start_pt
            end_p = (ix, iy)
            self.lbl_status.configure(text="Folge dem Straßenkanal zwischen beiden parallelen Strichen...")
            self.update_idletasks()

            extractor = RoadExtractor()
            line = extractor.trace_road_between_points(self.current_np, start_p, end_p)

            if line is not None:
                self.extracted_roads.append(line)
                self.road_start_pt = end_p
                self._update_counts()
                self.canvas.redraw()
                self.lbl_status.configure(text=f"✅ Straße exakt erfasst: {line.length:.1f} px ({line.length*1.25:.0f} m)! Weiterklicken zum Fortführen.")
            else:
                self.road_start_pt = None
                self.canvas.redraw()
                self.lbl_status.configure(text="Kein Pfad zwischen den Punkten gefunden.")

    def handle_toponym_box(self, ix0: float, iy0: float, ix1: float, iy1: float):
        if self.current_np is None or self.current_pil is None:
            return

        h, w = self.current_np.shape[:2]
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)

        if dx < 15 and dy < 15:
            cx, cy = int(round(ix0)), int(round(iy0))
            x0 = max(0, cx - 350)
            x1 = min(w, cx + 350)
            y0 = max(0, cy - 75)
            y1 = min(h, cy + 75)
            centroid_x, centroid_y = float(cx), float(cy)
        else:
            x0 = max(0, int(round(min(ix0, ix1))))
            x1 = min(w, int(round(max(ix0, ix1))))
            y0 = max(0, int(round(min(iy0, iy1))))
            y1 = min(h, int(round(max(iy0, iy1))))
            centroid_x = (x0 + x1) / 2.0
            centroid_y = (y0 + y1) / 2.0

        if x1 <= x0 or y1 <= y0:
            return

        crop_np = self.current_np[y0:y1, x0:x1]
        crop_pil = Image.fromarray(crop_np)

        lat_lon = SpatialGazetteer.get_lat_lon(self.geo_handler, centroid_x, centroid_y)
        lat = lat_lon[0] if lat_lon else None
        lon = lat_lon[1] if lat_lon else None
        geo_str = f"{lat:.5f}° N, {lon:.5f}° E" if (lat and lon) else "Lokal"

        text_ext = TextExtractor()
        suggested = text_ext.recognize_crop(crop_np)
        spatial_suggestions = SpatialGazetteer.query_nearby_names(lat, lon, radius_m=3500, ocr_hint=suggested)

        dialog = ToponymDialog(
            self,
            suggested_text=suggested,
            crop_img=crop_pil,
            suggestions=spatial_suggestions,
            geo_coord_str=geo_str
        )
        self.wait_window(dialog)

        if dialog.result_text:
            toponym_id = len(self.extracted_toponyms) + 1
            record = {
                "id": toponym_id,
                "text": dialog.result_text,
                "category": dialog.result_category,
                "x": centroid_x,
                "y": centroid_y,
                "geometry": Point(centroid_x, centroid_y)
            }
            self.extracted_toponyms.append(record)
            self._update_counts()
            self.canvas.redraw()
            self.lbl_status.configure(text=f"🏷️ Beschriftung hinzugefügt: \"{dialog.result_text}\" ({dialog.result_category})")

    def extract_single_building_at(self, ix: float, iy: float):
        if self.current_np is None:
            return

        h, w = self.current_np.shape[:2]
        rad = 90
        x0 = max(0, int(ix - rad))
        x1 = min(w, int(ix + rad))
        y0 = max(0, int(iy - rad))
        y1 = min(h, int(iy + rad))

        crop_rgb = self.current_np[y0:y1, x0:x1]
        cfg = self._get_current_building_config()
        extractor = BuildingExtractor(cfg)
        res = extractor.extract(crop_rgb)

        click_pt = Point(ix, iy)
        added = 0
        for f in res.features:
            p_global = translate(f.geometry, xoff=x0, yoff=y0)
            if p_global.contains(click_pt) or p_global.distance(click_pt) < 18.0:
                if p_global not in self.extracted_buildings:
                    self.extracted_buildings.append(p_global)
                    added += 1

        if added > 0:
            self._update_counts()
            self.canvas.redraw()
            self.lbl_status.configure(text=f"🏛️ Gebäude / Hofanlage hinzugefügt (+{added})")
        else:
            self.lbl_status.configure(text="Kein rotes Gebäude an dieser Position gefunden.")

    def select_feature_at(self, ix: float, iy: float):
        click_pt = Point(ix, iy)
        self.selected_building_idx = None
        self.selected_road_idx = None
        self.selected_toponym_idx = None

        for idx, top in enumerate(self.extracted_toponyms):
            if np.hypot(top["x"] - ix, top["y"] - iy) < 18.0:
                self.selected_toponym_idx = idx
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Beschriftung ausgewählt: \"{top['text']}\". Drücke 'Entf' zum Löschen.")
                return

        for idx, poly in enumerate(self.extracted_buildings):
            if poly.contains(click_pt) or poly.distance(click_pt) < 10.0:
                self.selected_building_idx = idx
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Gebäude #{idx + 1} ausgewählt. Drücke 'Entf' zum Löschen.")
                return

        for idx, line in enumerate(self.extracted_roads):
            if line.distance(click_pt) < 12.0:
                self.selected_road_idx = idx
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Straße #{idx + 1} ausgewählt (Länge: {line.length:.1f} px / {line.length*1.25:.0f} m). Drücke 'Entf' zum Löschen.")
                return

        self.canvas.redraw()
        self.lbl_status.configure(text="Kein Objekt ausgewählt.")

    def delete_feature_at(self, ix: float, iy: float):
        click_pt = Point(ix, iy)
        for idx, top in enumerate(self.extracted_toponyms):
            if np.hypot(top["x"] - ix, top["y"] - iy) < 18.0:
                del self.extracted_toponyms[idx]
                self.selected_toponym_idx = None
                self._update_counts()
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Beschriftung gelöscht.")
                return

        for idx, poly in enumerate(self.extracted_buildings):
            if poly.contains(click_pt) or poly.distance(click_pt) < 6.0:
                del self.extracted_buildings[idx]
                self.selected_building_idx = None
                self._update_counts()
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Gebäude #{idx + 1} gelöscht.")
                return

        for idx, line in enumerate(self.extracted_roads):
            if line.distance(click_pt) < 10.0:
                del self.extracted_roads[idx]
                self.selected_road_idx = None
                self._update_counts()
                self.canvas.redraw()
                self.lbl_status.configure(text=f"Straße #{idx + 1} gelöscht.")
                return

        for k, polys in self.extracted_layers.items():
            for idx, p in enumerate(polys):
                if p.contains(click_pt):
                    del polys[idx]
                    self._update_counts()
                    self.canvas.redraw()
                    self.lbl_status.configure(text=f"Fläche ({k}) gelöscht.")
                    return

    def _delete_selected(self):
        deleted = False
        if self.selected_toponym_idx is not None and 0 <= self.selected_toponym_idx < len(self.extracted_toponyms):
            del self.extracted_toponyms[self.selected_toponym_idx]
            self.selected_toponym_idx = None
            deleted = True

        if self.selected_building_idx is not None and 0 <= self.selected_building_idx < len(self.extracted_buildings):
            del self.extracted_buildings[self.selected_building_idx]
            self.selected_building_idx = None
            deleted = True

        if self.selected_road_idx is not None and 0 <= self.selected_road_idx < len(self.extracted_roads):
            del self.extracted_roads[self.selected_road_idx]
            self.selected_road_idx = None
            deleted = True

        if deleted:
            self._update_counts()
            self.canvas.redraw()
            self.lbl_status.configure(text="Ausgewähltes Objekt gelöscht.")

    def _clear_all_features(self):
        self.extracted_buildings.clear()
        self.extracted_roads.clear()
        self.extracted_toponyms.clear()
        for k in self.extracted_layers:
            self.extracted_layers[k].clear()
        self.selected_building_idx = None
        self.selected_road_idx = None
        self.selected_toponym_idx = None
        self.road_start_pt = None
        self._update_counts()
        self.canvas.redraw()
        self.lbl_status.configure(text="Alle Ebenen geleert.")

    def _update_counts(self):
        num_f = len(self.extracted_layers.get("forest", []))
        num_m = len(self.extracted_layers.get("meadow", []))
        num_w = len(self.extracted_layers.get("water", []))
        num_g = len(self.extracted_layers.get("gravel", []))
        num_v = len(self.extracted_layers.get("vineyard", []))
        self.lbl_count.configure(
            text=f"🏛️ {len(self.extracted_buildings)} | 🛣️ {len(self.extracted_roads)} | 🏷️ {len(self.extracted_toponyms)} | 🌲 {num_f} | 🌿 {num_m} | 💧 {num_w} | 🟠 {num_g} | 🍇 {num_v}"
        )

    def update_coordinates_hud(self, ix: float, iy: float):
        if not self.current_pil:
            return
        iw, ih = self.current_pil.size
        zoom_pct = int(round(self.canvas.scale * 100))
        if self.geo_handler and getattr(self.geo_handler, "transform", None) is not None:
            gx, gy = self.geo_handler.pixel_to_geo(ix, iy)
            crs_name = self.geo_handler.crs or "EPSG"
            self.lbl_coords.configure(text=f"X: {int(round(ix))} | Y: {int(round(iy))} | Geo: {gx:.2f}, {gy:.2f} ({crs_name}) | Zoom: {zoom_pct}%")
        else:
            self.lbl_coords.configure(text=f"X: {int(round(ix))} | Y: {int(round(iy))} / {iw}×{ih} px | Zoom: {zoom_pct}%")

    def _export_gis(self, fmt: str):
        total_feats = (
            len(self.extracted_buildings) + len(self.extracted_roads) + len(self.extracted_toponyms)
            + sum(len(v) for v in self.extracted_layers.values())
        )
        if total_feats == 0:
            messagebox.showwarning("Keine Daten", "Es wurden noch keine Features extrahiert.")
            return

        target_crs_str = self.entry_epsg.get().strip() or "EPSG:25832"

        ext_map = {"gpkg": ".gpkg", "shp": ".shp", "geojson": ".geojson"}
        default_ext = ext_map.get(fmt, ".gpkg")
        fp = filedialog.asksaveasfilename(
            title=f"Vektordaten exportieren ({fmt.upper()}) [Ziel-KBS: {target_crs_str}]",
            defaultextension=default_ext,
            filetypes=[(f"{fmt.upper()} Datei", f"*{default_ext}")]
        )
        if not fp:
            return

        try:
            def make_gdf(records: List[Dict], geometries: List[Any], layer_key: str) -> gpd.GeoDataFrame:
                color_spec = LAYER_COLOR_SPECS.get(layer_key, {"name": layer_key, "fill": "#888888", "stroke": "#ffffff"})
                for rec in records:
                    rec["color_hex"] = color_spec["fill"]
                    rec["stroke_color"] = color_spec["stroke"]
                    rec["layer_name"] = color_spec["name"]
                
                transformed_geoms = []
                has_transform = (self.geo_handler and getattr(self.geo_handler, "transform", None) is not None)
                src_crs = str(self.geo_handler.crs).upper() if (self.geo_handler and self.geo_handler.crs) else "EPSG:3857"
                
                for g in geometries:
                    if has_transform:
                        if isinstance(g, Point):
                            gx, gy = self.geo_handler.pixel_to_geo(g.x, g.y)
                            transformed_geoms.append(Point(gx, gy))
                        elif isinstance(g, LineString):
                            t_pts = [self.geo_handler.pixel_to_geo(x, y) for x, y in g.coords]
                            transformed_geoms.append(LineString(t_pts))
                        elif isinstance(g, Polygon):
                            t_ext = [self.geo_handler.pixel_to_geo(x, y) for x, y in g.exterior.coords]
                            t_holes = [[self.geo_handler.pixel_to_geo(x, y) for x, y in h.coords] for h in g.interiors]
                            transformed_geoms.append(Polygon(t_ext, t_holes))
                        else:
                            transformed_geoms.append(g)
                    else:
                        transformed_geoms.append(g)

                gdf = gpd.GeoDataFrame(records, geometry=transformed_geoms)
                if has_transform:
                    gdf.set_crs(src_crs, inplace=True, allow_override=True)
                    if src_crs != target_crs_str.upper():
                        try:
                            gdf = gdf.to_crs(target_crs_str)
                        except Exception:
                            pass
                else:
                    gdf.set_crs(target_crs_str, inplace=True, allow_override=True)
                return gdf

            if fmt == "gpkg":
                mode = "w"
                if self.extracted_buildings:
                    bldg_recs = [{"building_id": i+1, "layer": "buildings", "area_m2": round(p.area * 1.56, 1)} for i, p in enumerate(self.extracted_buildings)]
                    gdf_bldg = make_gdf(bldg_recs, self.extracted_buildings, "building")
                    gdf_bldg.to_file(fp, driver="GPKG", layer="buildings", mode=mode)
                    mode = "a"

                if self.extracted_roads:
                    road_recs = [{"road_id": i+1, "layer": "roads", "length_m": round(l.length * 1.25, 1)} for i, l in enumerate(self.extracted_roads)]
                    gdf_road = make_gdf(road_recs, self.extracted_roads, "road")
                    gdf_road.to_file(fp, driver="GPKG", layer="roads", mode=mode)
                    mode = "a"

                if self.extracted_toponyms:
                    top_recs = [{"toponym_id": t["id"], "text": t["text"], "category": t["category"], "layer": "toponyms"} for t in self.extracted_toponyms]
                    gdf_top = make_gdf(top_recs, [t["geometry"] for t in self.extracted_toponyms], "toponym")
                    gdf_top.to_file(fp, driver="GPKG", layer="toponyms", mode=mode)
                    mode = "a"

                for k, polys in self.extracted_layers.items():
                    if polys:
                        l_recs = [{"feature_id": i+1, "category": k, "layer": k, "area_m2": round(p.area * 1.56, 1)} for i, p in enumerate(polys)]
                        gdf_l = make_gdf(l_recs, polys, k)
                        gdf_l.to_file(fp, driver="GPKG", layer=k, mode=mode)
                        mode = "a"
            else:
                all_records = []
                all_geoms = []
                if self.extracted_buildings:
                    for i, p in enumerate(self.extracted_buildings):
                        all_records.append({"feature_id": len(all_records)+1, "layer": "buildings", "name": "", "category": "building"})
                        all_geoms.append(p)
                if self.extracted_roads:
                    for i, l in enumerate(self.extracted_roads):
                        all_records.append({"feature_id": len(all_records)+1, "layer": "roads", "name": "", "category": "road"})
                        all_geoms.append(l)
                if self.extracted_toponyms:
                    for i, t in enumerate(self.extracted_toponyms):
                        all_records.append({"feature_id": len(all_records)+1, "layer": "toponyms", "name": t["text"], "category": t["category"]})
                        all_geoms.append(t["geometry"])
                for k, polys in self.extracted_layers.items():
                    for i, p in enumerate(polys):
                        all_records.append({"feature_id": len(all_records)+1, "layer": k, "name": k, "category": k})
                        all_geoms.append(p)

                gdf = make_gdf(all_records, all_geoms, "building")
                driver_name = "ESRI Shapefile" if fmt == "shp" else "GeoJSON"
                gdf.to_file(fp, driver=driver_name)

            messagebox.showinfo("Export erfolgreich", f"Erfolgreich mit KBS {target_crs_str} und historischen Farben exportiert nach:\n{fp}")
            self.lbl_status.configure(text=f"Exportiert ({target_crs_str}): {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Exportfehler", str(e))


def main():
    app = TranchotDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
