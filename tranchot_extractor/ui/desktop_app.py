"""
Tranchot Map Feature Extractor — Intuitive Desktop GIS Application.
Features:
- Left Sidebar: Map loading, fast automatic extractors, tool selection, layer toggles, user-defined Output EPSG (default EPSG:25832), GIS export with fixed historical colors.
- Right Sidebar:
  * ✨ 0. Weißabgleich & Entgilbung (Parchment de-yellowing & watercolor radiance boost).
  * 🎨 Blatt-Kalibrierung & Pipetten mit feinem Farbabstand (2..25, default 8).
- Canvas Topbar: One-click live toggle between Original / Restaurierte Karte.
- Multi-layer GIS export with defined CRS and historical color attributes.
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
import difflib
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
from pyproj import Transformer

from tranchot_extractor.config import BuildingConfig, RoadConfig, TextConfig, LandUseConfig
from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
from tranchot_extractor.extractors.building_extractor import BuildingExtractor
from tranchot_extractor.extractors.road_extractor import RoadExtractor
from tranchot_extractor.extractors.text_extractor import TextExtractor
from tranchot_extractor.extractors.landuse_extractor import LandUseExtractor
from tranchot_extractor.extractors.pipette_sampler import PipetteSampler, ColorSample
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.spatial_gazetteer import SpatialGazetteer

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


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
    """
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg="#121317", highlightthickness=0, **kwargs)
        self.app = app

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

        self.bind("<ButtonPress-1>", self._on_left_press)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_release)

        self.bind("<ButtonPress-2>", self._on_pan_press)
        self.bind("<B2-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-2>", self._on_pan_release)

        self.bind("<ButtonPress-3>", self._on_right_press)

        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", lambda e: self._on_linux_zoom(e, 1.15))
        self.bind("<Button-5>", lambda e: self._on_linux_zoom(e, 1/1.15))

        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Configure>", lambda e: self.redraw())

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

        if self.app.active_tool == "pipette":
            self.app.handle_pipette_sample_at(ix, iy)
            return

        if self.app.active_tool in ("toponym", "bldg_box"):
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

    def _on_left_drag(self, event):
        if self.app.active_tool == "pan":
            self._on_pan_drag(event)
        elif self.app.active_tool in ("toponym", "bldg_box") and self.is_dragging_box:
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
        ix, iy = self.canvas_to_image_coords(event.x, event.y)
        self.app.update_coordinates_hud(ix, iy)

    def redraw(self):
        self.delete("all")
        if not self.pil_image:
            self.create_text(
                self.winfo_width() / 2, self.winfo_height() / 2,
                text="🏛️ Bitte lade eine historische Karte (GeoTIFF / PNG / JPG)...",
                fill="gray50", font=("Segoe UI", 14)
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
            tw = max(1, int((x1_img - x0_img) * self.scale))
            th = max(1, int((y1_img - y0_img) * self.scale))
            resample_mode = Image.Resampling.NEAREST if self.scale > 2.5 else Image.Resampling.BILINEAR
            resized = crop.resize((tw, th), resample_mode)
            self.tk_image = ImageTk.PhotoImage(resized)

            cx_pos = x0_img * self.scale + self.pan_x
            cy_pos = y0_img * self.scale + self.pan_y
            self.create_image(cx_pos, cy_pos, anchor="nw", image=self.tk_image)

        # 1. Draw Land-Use & Custom Sampled Layers
        if self.app.show_forests_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("forest", []), fill_col="#27ae60", outline_col="#2ecc71", stipple="gray25")

        if self.app.show_meadows_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("meadow", []), fill_col="#00cec9", outline_col="#81ecec", stipple="gray25")

        if self.app.show_water_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("water", []), fill_col="#0984e3", outline_col="#74b9ff", stipple="")

        if self.app.show_gravel_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("gravel", []), fill_col="#e17055", outline_col="#fab1a0", stipple="gray25")

        if self.app.show_vineyard_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("vineyard", []), fill_col="#f1c40f", outline_col="#f39c12", stipple="gray25")

        if self.app.show_garden_var.get():
            self._draw_generic_polygons(self.app.extracted_layers.get("garden", []), fill_col="#fdcb6e", outline_col="#ffeaa7", stipple="gray25")

        # 2. Draw Road Network Layer
        if self.app.show_roads_var.get():
            self._draw_road_centerlines()

        # 3. Draw Building Footprints & Courtyards Layer
        if self.app.show_bldgs_var.get():
            self._draw_building_polygons()

        # 4. Draw Toponyms / Text Layer
        if self.app.show_toponyms_var.get():
            self._draw_toponym_labels()

        # 5. Draw active road start marker if in road snap mode
        if self.app.road_start_pt is not None:
            cx, cy = self.image_to_canvas_coords(self.app.road_start_pt[0], self.app.road_start_pt[1])
            self.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#00cec9", outline="#ffffff", width=2)

        # 6. Draw live dragging box for text selection or building ROI test
        if self.is_dragging_box:
            bx0, by0 = min(self.box_start_x, self.drag_current_x), min(self.box_start_y, self.drag_current_y)
            bx1, by1 = max(self.box_start_x, self.drag_current_x), max(self.box_start_y, self.drag_current_y)
            if self.app.active_tool == "bldg_box":
                self.create_rectangle(bx0, by0, bx1, by1, outline="#e74c3c", width=2, dash=(6, 3))
                self.create_text(bx0 + 8, by0 + 12, text="🏛️ Gebäude ROI-Test", anchor="w", fill="#e74c3c", font=("Segoe UI", 9, "bold"))
            elif self.app.active_tool == "landuse_box":
                self.create_rectangle(bx0, by0, bx1, by1, outline="#2ecc71", width=2, dash=(6, 3))
                self.create_text(bx0 + 8, by0 + 12, text="🌲 Flächen ROI-Test (Wettbewerb)", anchor="w", fill="#2ecc71", font=("Segoe UI", 9, "bold"))
            else:
                self.create_rectangle(bx0, by0, bx1, by1, outline="#00cec9", width=2, dash=(4, 2))

    def _draw_generic_polygons(self, polys: List[Polygon], fill_col: str, outline_col: str, stipple: str = ""):
        for poly in polys:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
            flat_pts = [c for pt in pts for c in pt]
            if len(flat_pts) >= 6:
                kwargs = {"fill": fill_col, "outline": outline_col, "width": 1.5}
                if stipple:
                    kwargs["stipple"] = stipple
                self.create_polygon(*flat_pts, **kwargs)

    def _draw_road_centerlines(self):
        for idx, line in enumerate(self.app.extracted_roads):
            if not isinstance(line, LineString) or line.is_empty or len(line.coords) < 2:
                continue
            pts = [self.image_to_canvas_coords(x, y) for x, y in line.coords]
            flat_pts = [c for pt in pts for c in pt]
            if len(flat_pts) >= 4:
                is_selected = (self.app.selected_road_idx == idx)
                line_col = "#f1c40f" if is_selected else "#ff793f"
                line_w = 4.0 if is_selected else 3.0
                self.create_line(*flat_pts, fill=line_col, width=line_w, capstyle="round", joinstyle="round")

    def _draw_building_polygons(self):
        for idx, poly in enumerate(self.app.extracted_buildings):
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue

            pts = [self.image_to_canvas_coords(x, y) for x, y in poly.exterior.coords]
            flat_pts = [c for pt in pts for c in pt]
            if len(flat_pts) >= 6:
                is_selected = (idx == self.app.selected_building_idx)
                outline_col = "#f1c40f" if is_selected else "#ff4757"
                fill_col = "#f39c12" if is_selected else "#e74c3c"
                line_w = 2.5 if is_selected else 1.5

                self.create_polygon(*flat_pts, fill=fill_col, outline=outline_col, width=line_w, stipple="gray25")

            for interior in poly.interiors:
                hole_pts = [self.image_to_canvas_coords(x, y) for x, y in interior.coords]
                flat_hole = [c for pt in hole_pts for c in pt]
                if len(flat_hole) >= 6:
                    self.create_polygon(*flat_hole, fill="#121317", outline="#00cec9", width=1.0)

    def _draw_toponym_labels(self):
        for idx, top in enumerate(self.app.extracted_toponyms):
            cx, cy = self.image_to_canvas_coords(top["x"], top["y"])
            is_selected = (self.app.selected_toponym_idx == idx)
            dot_col = "#f1c40f" if is_selected else "#00cec9"
            text_col = "#f1c40f" if is_selected else "#ffffff"

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
            lbl_crop = tk.Label(self, image=self.tk_crop, bg="#121317")
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
        
        self.sugg_scroll = ctk.CTkScrollableFrame(self, height=55, fg_color="#181a20", orientation="horizontal")
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
                fg_color="#2c3e50", hover_color="#34495e", font=ctk.CTkFont(size=11),
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

        btn_cancel = ctk.CTkButton(btn_frame, text="Abbrechen", command=self._on_cancel, fg_color="#444", hover_color="#666", width=140)
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
    and high-performance MapCanvas.
    """
    def __init__(self):
        super().__init__()

        self.title("🏛️ Tranchot Extractor — Interaktive Weißabgleich-Restaurierung & GIS Extraktion (BCDH)")
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

        self.selected_building_idx: Optional[int] = None
        self.selected_road_idx: Optional[int] = None
        self.selected_toponym_idx: Optional[int] = None
        self.road_start_pt: Optional[Tuple[float, float]] = None
        self.active_tool: str = "pipette"

        self.swatch_boxes: Dict[str, tk.Label] = {}

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
        self.left_sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0, fg_color="#181a20")
        self.left_sidebar.grid(row=0, column=0, sticky="nsew")

        # Title
        lbl_title = ctk.CTkLabel(self.left_sidebar, text="🏛️ Tranchot Extractor", font=ctk.CTkFont(size=19, weight="bold"))
        lbl_title.pack(anchor="w", padx=15, pady=(12, 2))
        lbl_sub = ctk.CTkLabel(self.left_sidebar, text="Kartenaufnahme der Rheinlande 1803–1828\nBCDH Universität Bonn", font=ctk.CTkFont(size=11), text_color="gray70", justify="left")
        lbl_sub.pack(anchor="w", padx=15, pady=(0, 12))

        # Section: Map Loading
        self._create_section_label(self.left_sidebar, "📂 1. Karte laden")
        btn_open = ctk.CTkButton(self.left_sidebar, text="📁 Eigene Karte / GeoTIFF laden...", command=self._open_file_dialog, height=34, font=ctk.CTkFont(weight="bold"))
        btn_open.pack(fill="x", padx=15, pady=3)

        self.sample_dropdown = ctk.CTkOptionMenu(
            self.left_sidebar,
            values=["Beispiel: Nickenich (GeoTIFF)", "Beispiel: Rommerskirchen", "Beispiel: Kruft"],
            command=self._on_sample_selected,
            height=30
        )
        self.sample_dropdown.pack(fill="x", padx=15, pady=3)

        # Section: Automatic Extraction
        self._create_section_label(self.left_sidebar, "⚡ 2. Automatische Extraktion")
        
        btn_auto_landuse = ctk.CTkButton(
            self.left_sidebar,
            text="🌲 Gesamte Landnutzung extrahieren",
            command=self._run_auto_landuse,
            fg_color="#27ae60", hover_color="#2ecc71",
            height=36, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_landuse.pack(fill="x", padx=15, pady=3)

        btn_auto_roads = ctk.CTkButton(
            self.left_sidebar,
            text="🛣️ Alle Hauptachsen (≥1 km) extrahieren",
            command=self._run_auto_roads,
            fg_color="#d35400", hover_color="#e67e22",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_roads.pack(fill="x", padx=15, pady=3)

        btn_auto_all_sampled = ctk.CTkButton(
            self.left_sidebar,
            text="🟠 Alle gesampelten Klassen extrahieren",
            command=self._run_extract_all_sampled_classes,
            fg_color="#8e44ad", hover_color="#9b59b6",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_all_sampled.pack(fill="x", padx=15, pady=3)

        # Section: Tools
        self._create_section_label(self.left_sidebar, "🛠️ 3. Werkzeuge")
        self.tool_var = ctk.StringVar(value="pipette")

        r_pipette = ctk.CTkRadioButton(self.left_sidebar, text="🎨 Farb-Pipette (Klick zum Samplen)", variable=self.tool_var, value="pipette", command=self._on_tool_change)
        r_pipette.pack(anchor="w", padx=15, pady=3)

        r_road_snap = ctk.CTkRadioButton(self.left_sidebar, text="🛣️ Straße nachverfolgen (Klick A ➔ B)", variable=self.tool_var, value="road_snap", command=self._on_tool_change)
        r_road_snap.pack(anchor="w", padx=15, pady=3)

        r_toponym = ctk.CTkRadioButton(self.left_sidebar, text="🏷️ Beschriftung aufziehen (Rechteck)", variable=self.tool_var, value="toponym", command=self._on_tool_change)
        r_toponym.pack(anchor="w", padx=15, pady=3)

        r_select = ctk.CTkRadioButton(self.left_sidebar, text="👆 Objekt auswählen & löschen", variable=self.tool_var, value="select", command=self._on_tool_change)
        r_select.pack(anchor="w", padx=15, pady=3)

        r_pan = ctk.CTkRadioButton(self.left_sidebar, text="✋ Karte verschieben (Pan)", variable=self.tool_var, value="pan", command=self._on_tool_change)
        r_pan.pack(anchor="w", padx=15, pady=3)

        btn_delete_sel = ctk.CTkButton(self.left_sidebar, text="🗑️ Ausgewähltes Element löschen", command=self._delete_selected, fg_color="#333", hover_color="#555", height=30)
        btn_delete_sel.pack(fill="x", padx=15, pady=5)

        btn_clear = ctk.CTkButton(self.left_sidebar, text="🧹 Alle Layer leeren", command=self._clear_all_features, fg_color="#444", hover_color="#666", height=30)
        btn_clear.pack(fill="x", padx=15, pady=3)

        # Section: Layer Visibility
        self._create_section_label(self.left_sidebar, "👁️ 4. Ebenen anzeigen")
        self.show_bldgs_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🏛️ Gebäude / Hofanlagen", variable=self.show_bldgs_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_roads_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🛣️ Straßen-Mittellinien", variable=self.show_roads_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_toponyms_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🏷️ Beschriftungen / Namen", variable=self.show_toponyms_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_forests_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🌲 Waldflächen", variable=self.show_forests_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_meadows_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🌿 Wiesen / Feuchtgrünland", variable=self.show_meadows_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_water_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="💧 Gewässer / Seen", variable=self.show_water_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_gravel_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🟠 Kies- & Schotterbänke", variable=self.show_gravel_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_vineyard_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🍇 Weinberge (Rebhänge)", variable=self.show_vineyard_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        self.show_garden_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.left_sidebar, text="🟨 Gärten / Nutzkulturen", variable=self.show_garden_var, command=self._on_layer_toggle).pack(anchor="w", padx=15, pady=2)

        # Section: Export & Target EPSG
        self._create_section_label(self.left_sidebar, "💾 5. GIS Export & KBS")

        ctk.CTkLabel(self.left_sidebar, text="Ausgabe-KBS (EPSG):", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=15, pady=(2, 0))
        self.entry_epsg = ctk.CTkEntry(self.left_sidebar, height=30, font=ctk.CTkFont(size=12, weight="bold"))
        self.entry_epsg.insert(0, "EPSG:25832")
        self.entry_epsg.pack(fill="x", padx=15, pady=(0, 6))

        btn_exp_gpkg = ctk.CTkButton(self.left_sidebar, text="💾 GeoPackage (.gpkg) [Farbstile inkl.]", command=lambda: self._export_gis("gpkg"), fg_color="#27ae60", hover_color="#2ecc71", height=34, font=ctk.CTkFont(weight="bold"))
        btn_exp_gpkg.pack(fill="x", padx=15, pady=3)

        btn_exp_shp = ctk.CTkButton(self.left_sidebar, text="💾 ESRI Shapefile (.shp)", command=lambda: self._export_gis("shp"), fg_color="#2980b9", hover_color="#3498db", height=32)
        btn_exp_shp.pack(fill="x", padx=15, pady=3)

        btn_exp_geojson = ctk.CTkButton(self.left_sidebar, text="💾 GeoJSON (.geojson)", command=lambda: self._export_gis("geojson"), fg_color="#8e44ad", hover_color="#9b59b6", height=32)
        btn_exp_geojson.pack(fill="x", padx=15, pady=3)

        # =========================================================================
        # 2. Main Canvas Area (Center)
        # =========================================================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#121317")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top Bar on Canvas
        self.topbar = ctk.CTkFrame(self.main_frame, height=42, corner_radius=0, fg_color="#1a1c23")
        self.topbar.grid(row=0, column=0, sticky="ew")

        self.lbl_count = ctk.CTkLabel(self.topbar, text="🏛️ 0 | 🛣️ 0 | 🏷️ 0 | 🌲 0 | 🌿 0 | 💧 0 | 🟠 0 | 🍇 0", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_count.pack(side="left", padx=15, pady=8)

        # Toggle Button: Raw vs Restored View
        self.btn_toggle_view = ctk.CTkButton(
            self.topbar,
            text="✨ Restaurierte Ansicht (Aktiv)",
            width=190,
            command=self._toggle_enhanced_view,
            fg_color="#27ae60", hover_color="#2ecc71",
            height=28, font=ctk.CTkFont(size=11, weight="bold")
        )
        self.btn_toggle_view.pack(side="right", padx=(6, 15), pady=6)

        btn_reset_view = ctk.CTkButton(self.topbar, text="🔍 Ansicht einpassen", width=130, command=lambda: self.canvas.reset_view(), height=28)
        btn_reset_view.pack(side="right", padx=6, pady=6)

        # Canvas
        self.canvas = MapCanvas(self.main_frame, app=self)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        # Status Bar
        self.status_bar = ctk.CTkFrame(self.main_frame, height=28, corner_radius=0, fg_color="#14161b")
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.lbl_status = ctk.CTkLabel(self.status_bar, text="Bereit.", font=ctk.CTkFont(size=11))
        self.lbl_status.pack(side="left", padx=15, pady=4)

        self.lbl_coords = ctk.CTkLabel(self.status_bar, text="X: 0 | Y: 0 | Zoom: 100%", font=ctk.CTkFont(size=11), text_color="gray60")
        self.lbl_coords.pack(side="right", padx=15, pady=4)

        # =========================================================================
        # 3. Right Sidebar: Dedicated Pipette & Color Calibration Panel
        # =========================================================================
        self.right_sidebar = ctk.CTkScrollableFrame(self, width=350, corner_radius=0, fg_color="#181a20")
        self.right_sidebar.grid(row=0, column=2, sticky="nsew")

        # Section 0: White-Balance & De-Yellowing
        lbl_enhance_title = ctk.CTkLabel(self.right_sidebar, text="✨ 0. Weißabgleich & Entgilbung", font=ctk.CTkFont(size=15, weight="bold"), text_color="#2ecc71")
        lbl_enhance_title.pack(anchor="w", padx=15, pady=(12, 2))

        self.enhance_active_var = ctk.BooleanVar(value=True)
        cb_enhance = ctk.CTkCheckBox(
            self.right_sidebar,
            text="✨ Entgilbung & Farbverstärkung aktiv",
            variable=self.enhance_active_var,
            command=self._apply_enhancement_settings,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        cb_enhance.pack(anchor="w", padx=15, pady=(2, 6))

        # De-yellowing slider
        self.lbl_deyellow_header = ctk.CTkLabel(self.right_sidebar, text="Entgilbung / Weißabgleich (85%):", font=ctk.CTkFont(size=11))
        self.lbl_deyellow_header.pack(anchor="w", padx=15, pady=(2, 0))

        self.slider_deyellow = ctk.CTkSlider(self.right_sidebar, from_=0.0, to=1.0, number_of_steps=20, command=self._on_deyellow_slider_moved)
        self.slider_deyellow.set(0.85)
        self.slider_deyellow.pack(fill="x", padx=15, pady=2)

        # Vibrancy slider
        self.lbl_vibrance_header = ctk.CTkLabel(self.right_sidebar, text="Farb-Leuchtkraft (1.85×):", font=ctk.CTkFont(size=11))
        self.lbl_vibrance_header.pack(anchor="w", padx=15, pady=(4, 0))

        self.slider_vibrance = ctk.CTkSlider(self.right_sidebar, from_=1.0, to=3.0, number_of_steps=20, command=self._on_vibrance_slider_moved)
        self.slider_vibrance.set(1.85)
        self.slider_vibrance.pack(fill="x", padx=15, pady=2)

        # Section 1: Pipette Calibration
        lbl_right_title = ctk.CTkLabel(self.right_sidebar, text="🎨 1. Blatt-Kalibrierung & Pipetten", font=ctk.CTkFont(size=15, weight="bold"), text_color="#00cec9")
        lbl_right_title.pack(anchor="w", padx=15, pady=(16, 2))
        
        lbl_right_desc = ctk.CTkLabel(
            self.right_sidebar,
            text="Wähle eine Pipette und klicke auf die Karte, um den Farbton für dieses Blatt zu lernen.",
            font=ctk.CTkFont(size=10), text_color="gray70", justify="left", wraplength=310
        )
        lbl_right_desc.pack(anchor="w", padx=15, pady=(0, 10))

        # Active Pipette Dropdown
        self.pipette_dropdown = ctk.CTkOptionMenu(
            self.right_sidebar,
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
        self.pipette_dropdown.pack(fill="x", padx=15, pady=4)

        # Active Swatch & Live Indicator
        self.swatch_frame = ctk.CTkFrame(self.right_sidebar, fg_color="#22252e", height=46)
        self.swatch_frame.pack(fill="x", padx=15, pady=6)

        self.swatch_box = tk.Label(self.swatch_frame, text="", bg="#27ae60", width=4, height=1)
        self.swatch_box.pack(side="left", padx=10, pady=8)

        self.lbl_swatch_info = ctk.CTkLabel(self.swatch_frame, text="Farbe: Standard-Vorlage\nKlicke Karte zum Samplen", font=ctk.CTkFont(size=11), justify="left")
        self.lbl_swatch_info.pack(side="left", padx=4, pady=4)

        # Fine-tuned Tolerance slider (Range 2..25, Default 8)
        self.lbl_tol_header = ctk.CTkLabel(self.right_sidebar, text="Toleranz / Farbabstand (ΔE = 8):", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_tol_header.pack(anchor="w", padx=15, pady=(8, 0))

        self.slider_tol = ctk.CTkSlider(self.right_sidebar, from_=2, to=25, number_of_steps=23, command=self._on_tolerance_changed)
        self.slider_tol.set(8)
        self.slider_tol.pack(fill="x", padx=15, pady=2)

        # Action: Extract current sampled class or all competitive
        btn_landuse_box = ctk.CTkButton(
            self.right_sidebar,
            text="📐 Flächen ROI-Box aufziehen (Testen)",
            command=lambda: self._set_active_tool("landuse_box"),
            fg_color="#27ae60", hover_color="#2ecc71",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_landuse_box.pack(fill="x", padx=15, pady=(6, 3))

        btn_extract_competitive = ctk.CTkButton(
            self.right_sidebar,
            text="⚡ Alle Flächen kompetitiv extrahieren (Wettbewerb)",
            command=self._run_extract_all_sampled_classes,
            fg_color="#16a085", hover_color="#1abc9c",
            height=34, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_extract_competitive.pack(fill="x", padx=15, pady=3)

        btn_extract_active = ctk.CTkButton(
            self.right_sidebar,
            text="🎯 Nur aktive Pipetten-Klasse extrahieren",
            command=self._run_extract_active_sample,
            fg_color="#2980b9", hover_color="#3498db",
            height=30, font=ctk.CTkFont(size=11)
        )
        btn_extract_active.pack(fill="x", padx=15, pady=(3, 8))

        # Separator & Quick Palette Class List
        self._create_section_label(self.right_sidebar, "📋 Farbfelder aller Klassen")
        self._build_palette_class_list()

        # Profile Save/Load
        self._create_section_label(self.right_sidebar, "💾 Blatt-Profil (.palette.json)")
        btn_prof_frame = ctk.CTkFrame(self.right_sidebar, fg_color="transparent")
        btn_prof_frame.pack(fill="x", padx=15, pady=4)

        btn_save_prof = ctk.CTkButton(btn_prof_frame, text="💾 Profil speichern", command=self._save_palette_profile, fg_color="#34495e", hover_color="#2c3e50", height=30, width=150)
        btn_save_prof.pack(side="left")

        btn_load_prof = ctk.CTkButton(btn_prof_frame, text="📂 Profil laden", command=self._load_palette_profile, fg_color="#34495e", hover_color="#2c3e50", height=30, width=150)
        btn_load_prof.pack(side="right")

        # Section 2: Building Footprint & Courtyard Extraction Panel
        lbl_bldg_title = ctk.CTkLabel(self.right_sidebar, text="🏛️ 2. Gebäude & Hofanlagen", font=ctk.CTkFont(size=15, weight="bold"), text_color="#e74c3c")
        lbl_bldg_title.pack(anchor="w", padx=15, pady=(18, 2))

        lbl_bldg_desc = ctk.CTkLabel(
            self.right_sidebar,
            text="Ziehe eine ROI-Box auf der Karte auf, um Gebäude in einem Ausschnitt zu testen und zu extrahieren.",
            font=ctk.CTkFont(size=10), text_color="gray70", justify="left", wraplength=310
        )
        lbl_bldg_desc.pack(anchor="w", padx=15, pady=(0, 6))

        # Mode Buttons
        btn_bldg_box = ctk.CTkButton(
            self.right_sidebar,
            text="📐 Gebäude ROI-Box aufziehen (Testen)",
            command=lambda: self._set_active_tool("bldg_box"),
            fg_color="#c0392b", hover_color="#e74c3c",
            height=36, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_bldg_box.pack(fill="x", padx=15, pady=3)

        btn_bldg_click = ctk.CTkButton(
            self.right_sidebar,
            text="🏛️ Einzelnes Gebäude anklicken",
            command=lambda: self._set_active_tool("extract"),
            fg_color="#34495e", hover_color="#2c3e50",
            height=30, font=ctk.CTkFont(size=11)
        )
        btn_bldg_click.pack(fill="x", padx=15, pady=3)

        # Red differential sensitivity slider (10..45)
        self.lbl_bldg_diff_header = ctk.CTkLabel(self.right_sidebar, text="Rot-Empfindlichkeit (RGB-Diff = 18):", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_bldg_diff_header.pack(anchor="w", padx=15, pady=(8, 0))

        self.slider_bldg_diff = ctk.CTkSlider(self.right_sidebar, from_=10, to=45, number_of_steps=35, command=self._on_bldg_diff_changed)
        self.slider_bldg_diff.set(18)
        self.slider_bldg_diff.pack(fill="x", padx=15, pady=2)

        # Minimum building area slider (4..40 px)
        self.lbl_bldg_min_area = ctk.CTkLabel(self.right_sidebar, text="Mindestfläche (6 px²):", font=ctk.CTkFont(size=11))
        self.lbl_bldg_min_area.pack(anchor="w", padx=15, pady=(4, 0))

        self.slider_bldg_area = ctk.CTkSlider(self.right_sidebar, from_=4, to=40, number_of_steps=36, command=self._on_bldg_area_changed)
        self.slider_bldg_area.set(6)
        self.slider_bldg_area.pack(fill="x", padx=15, pady=2)

        # Checkboxes for terrace filtering, rectangle regularization, and orthogonal snapping
        self.cb_filter_terraces_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.right_sidebar, text="🍇 Weinberg-Terrassen & Schraffen filtern", variable=self.cb_filter_terraces_var, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=15, pady=2)

        self.cb_regularize_rect_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.right_sidebar, text="📐 Einzelhäuser zu 4-Eck-Rechtecken", variable=self.cb_regularize_rect_var, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=15, pady=2)

        self.cb_regularize_ortho_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.right_sidebar, text="🏛️ Hofanlagen 90° orthogonal ausrichten", variable=self.cb_regularize_ortho_var, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=15, pady=2)

        # Extraction on whole map button & clear button
        btn_auto_all_bldgs = ctk.CTkButton(
            self.right_sidebar,
            text="⚡ Alle Gebäude der Karte extrahieren",
            command=self._run_auto_buildings,
            fg_color="#c0392b", hover_color="#e74c3c",
            height=36, font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_auto_all_bldgs.pack(fill="x", padx=15, pady=(10, 4))

        btn_clear_bldgs = ctk.CTkButton(
            self.right_sidebar,
            text="🧹 Nur Gebäude-Layer leeren",
            command=self._clear_buildings_layer,
            fg_color="#333333", hover_color="#555555",
            height=28, font=ctk.CTkFont(size=11)
        )
        btn_clear_bldgs.pack(fill="x", padx=15, pady=(2, 16))

    def _create_section_label(self, parent, text: str):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db")
        lbl.pack(anchor="w", padx=15, pady=(12, 4))

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

    def _build_palette_class_list(self):
        """Builds quick visual swatches and 1-click extract buttons for all classes in right sidebar."""
        for item in PipetteSampler.DEFAULT_CLASSES:
            cid = item["class_id"]
            lbl_text = item["label"]
            init_hex = item["hex"]

            row = ctk.CTkFrame(self.right_sidebar, fg_color="#1c1f27", height=34)
            row.pack(fill="x", padx=15, pady=2)

            swatch = tk.Label(row, text="", bg=init_hex, width=2, height=1)
            swatch.pack(side="left", padx=(6, 4), pady=4)
            self.swatch_boxes[cid] = swatch

            btn_select = ctk.CTkButton(
                row, text=lbl_text, width=150, height=24, fg_color="transparent", hover_color="#2a2e39",
                font=ctk.CTkFont(size=11), anchor="w",
                command=lambda c=cid, l=lbl_text: self._activate_class_pipette(c, l)
            )
            btn_select.pack(side="left", padx=2)

            btn_ext = ctk.CTkButton(
                row, text="⚡", width=28, height=24, fg_color="#2c3e50", hover_color="#34495e",
                command=lambda c=cid: self._run_extract_specific_class(c)
            )
            btn_ext.pack(side="right", padx=4)

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
            self.swatch_box.configure(bg=sample.hex_color)
            if self.active_pipette_class in self.swatch_boxes:
                self.swatch_boxes[self.active_pipette_class].configure(bg=sample.hex_color)
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
        """Samples color from the active restored map and updates the active pipette swatch."""
        if self.current_np is None:
            return

        sample = self.sampler.sample_from_coordinate(self.current_np, self.active_pipette_class, ix, iy, radius=4)
        self._update_active_swatch_display()
        
        # If paper was sampled, automatically re-balance the map with this exact paper reference!
        if self.active_pipette_class == "paper":
            self._apply_enhancement_settings()
            self.lbl_status.configure(text=f"📜 Pergament-Referenz gesetzt auf RGB{tuple(sample.rgb)} — Weißabgleich angepasst!")
        else:
            self.lbl_status.configure(text=f"🎨 '{sample.label}' erfolgreich gesampelt bei ({int(ix)}, {int(iy)})!")

    def _run_extract_active_sample(self):
        self._run_extract_specific_class(self.active_pipette_class)

    def _run_extract_specific_class(self, class_id: str):
        """Extracts polygons for a specific class using its sampled color and tolerance."""
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
        """Extracts all active sampled classes simultaneously."""
        if self.current_np is None:
            return

        self.lbl_status.configure(text="Extrahiere alle kalibrierten Klassen...")
        self.update_idletasks()

        threading.Thread(target=self._async_extract_all_classes, daemon=True).start()

    def handle_landuse_roi_box(self, ix0: float, iy0: float, ix1: float, iy1: float):
        """Extracts competitive land-use polygons inside drawn ROI box for instant parameter testing."""
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
            # Run competitive multi-class segmentation over all active classes
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
        """Full multi-scale automated land-use extraction."""
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
        rgb_diff = int(self.slider_bldg_diff.get()) if hasattr(self, "slider_bldg_diff") else 18
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
            lab_a_threshold=131,
        )

    def _set_active_tool(self, tool_name: str):
        self.tool_var.set(tool_name)
        self._on_tool_change()

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
        result = extractor.extract(self.current_np)
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

        # If click without drag, do single building extract
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

        # Remove existing buildings located inside this ROI box to allow clean re-testing
        box_polygon = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.extracted_buildings = [
            p for p in self.extracted_buildings if not box_polygon.contains(p.centroid)
        ]

        added_count = 0
        for f in result.features:
            p_global = translate(f.geometry, xoff=x0, yoff=y0)
            if p_global.is_valid and not p_global.is_empty:
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
            "bldg_box": "📐 Gebäude ROI-Box (Rechteck aufziehen zum Testen & Extrahieren)",
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

            # Apply initial white balance & enhancement
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
            # Helper to create styled and CRS-aware GeoDataFrame
            def make_gdf(records: List[Dict], geometries: List[Any], layer_key: str) -> gpd.GeoDataFrame:
                color_spec = LAYER_COLOR_SPECS.get(layer_key, {"name": layer_key, "fill": "#888888", "stroke": "#ffffff"})
                for rec in records:
                    rec["color_hex"] = color_spec["fill"]
                    rec["stroke_color"] = color_spec["stroke"]
                    rec["layer_name"] = color_spec["name"]
                
                # Transform pixel geoms to target CRS if georeference is present
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
