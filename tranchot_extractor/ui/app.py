"""
Interactive Gradio Web Application for Tranchot Historical Map Feature Extraction.
Includes both:
1. Automated Streaming Pipeline
2. Interactive SAM AI Studio (Point/Box Prompting & Semi-automatic Seed-Assisted Extraction)
"""

import os
import glob
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from shapely.geometry import Polygon
import geopandas as gpd

from tranchot_extractor.config import PipelineConfig, BuildingConfig, RoadConfig, TextConfig
from tranchot_extractor.core.pipeline import TranchotPipeline
from tranchot_extractor.core.tiled_processor import TiledMapProcessor
from tranchot_extractor.extractors.sam_extractor import SAMExtractor

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAMPLE_IMAGE_PATH = os.path.join(WORKSPACE_ROOT, "sample_data", "rommerskirchen_sample.png")
GEOTIFFS_DIR = os.path.join(WORKSPACE_ROOT, "output_geotiffs")


def get_available_presets():
    """Finds all available presets and GeoTIFFs."""
    presets = {}
    if os.path.exists(SAMPLE_IMAGE_PATH):
        presets["📋 Rommerskirchen Ausschnitt (Sample PNG)"] = SAMPLE_IMAGE_PATH

    tranchot_files = glob.glob(os.path.join(GEOTIFFS_DIR, "tranchot", "*.tif"))
    for f in tranchot_files:
        name = f"🗺️ Tranchot: {os.path.basename(f)}"
        presets[name] = f

    meilen_files = glob.glob(os.path.join(GEOTIFFS_DIR, "meilenblaetter", "*.tif"))
    for f in meilen_files:
        name = f"📜 Meilenblätter: {os.path.basename(f)}"
        presets[name] = f

    return presets


PRESETS_DICT = get_available_presets()


def load_preset_image(preset_name: str) -> np.ndarray:
    """Loads and downsamples image for interactive SAM studio if needed."""
    if not preset_name or preset_name not in PRESETS_DICT:
        if os.path.exists(SAMPLE_IMAGE_PATH):
            return np.array(Image.open(SAMPLE_IMAGE_PATH).convert("RGB"))
        return np.zeros((500, 500, 3), dtype=np.uint8)

    path = PRESETS_DICT[preset_name]
    if HAS_RASTERIO and path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(path) as src:
                # For interactive studio on huge 14k sheets, load a central 2048x2048 high-res crop
                cx, cy = src.width // 2, src.height // 2
                w_size = min(2048, src.width, src.height)
                win = rasterio.windows.Window(cx - w_size//2, cy - w_size//2, w_size, w_size)
                crop_data = src.read(window=win)
                crop_rgb = np.transpose(crop_data[:3, :, :], (1, 2, 0))
                return crop_rgb
        except Exception:
            pass

    pil_img = Image.open(path).convert("RGB")
    return np.array(pil_img)


def create_ui_preview(image_arr: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """Downsamples large images for responsive, crash-free browser display."""
    if image_arr is None or image_arr.size == 0:
        return None
    h, w = image_arr.shape[:2]
    if max(h, w) <= max_dim:
        return image_arr
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Global SAM instance
_sam_extractor = None

def get_sam_extractor():
    global _sam_extractor
    if _sam_extractor is None:
        _sam_extractor = SAMExtractor()
    return _sam_extractor


# ==========================================
# SAM Interactive Studio Callbacks
# ==========================================

def handle_sam_click(
    evt: gr.SelectData,
    current_img,
    click_mode: str,
    pos_points_state: list,
    neg_points_state: list,
    all_polys_state: list,
):
    """Triggered when user clicks on the image in SAM Interactive Studio."""
    if current_img is None:
        return current_img, pos_points_state, neg_points_state, "Kein Bild geladen."

    x, y = evt.index[0], evt.index[1]

    if "Positiv" in click_mode or "[+]" in click_mode:
        pos_points_state.append((x, y))
    else:
        neg_points_state.append((x, y))

    # Render points overlay on image
    vis = current_img.copy()

    # Draw existing saved polygons in green
    for p in all_polys_state:
        pts = np.array(p.exterior.coords, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 120), 2, cv2.LINE_AA)

    # Draw prompt points
    for px, py in pos_points_state:
        cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), -1)
        cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)

    for nx, ny in neg_points_state:
        cv2.circle(vis, (int(nx), int(ny)), 6, (255, 0, 0), -1)
        cv2.circle(vis, (int(nx), int(ny)), 8, (255, 255, 255), 2)

    status = f"Punkte: {len(pos_points_state)} Positiv (Grün), {len(neg_points_state)} Negativ (Rot). Klicke '✨ SAM Segmentieren'."
    return vis, pos_points_state, neg_points_state, status


def run_sam_segmentation(
    current_img,
    pos_points_state: list,
    neg_points_state: list,
    all_polys_state: list,
):
    """Executes SAM inference on prompt points."""
    if current_img is None or (not pos_points_state and not neg_points_state):
        return current_img, None, "⚠️ Bitte mindestens einen Klick-Punkt auf das Bild setzen."

    sam = get_sam_extractor()
    res = sam.segment_with_prompts(
        current_img,
        positive_points=pos_points_state,
        negative_points=neg_points_state
    )

    vis = current_img.copy()

    # Draw previously saved polygons
    for p in all_polys_state:
        pts = np.array(p.exterior.coords, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 220, 80), 2, cv2.LINE_AA)

    # Draw current SAM candidate polygon in vibrant Cyan/Yellow
    current_poly = None
    if res["polygons"]:
        current_poly = res["polygons"][0]
        pts = np.array(current_poly.exterior.coords, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 240, 255))
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)

    # Re-draw points
    for px, py in pos_points_state:
        cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), -1)
        cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)
    for nx, ny in neg_points_state:
        cv2.circle(vis, (int(nx), int(ny)), 6, (255, 0, 0), -1)
        cv2.circle(vis, (int(nx), int(ny)), 8, (255, 255, 255), 2)

    status = f"✅ SAM Segmentierung erfolgreich! IoU-Score: {res['iou_score']}. Klicke '➕ Als GIS-Objekt übernehmen'."
    return vis, current_poly, status


def save_sam_polygon(
    current_img,
    current_poly,
    all_polys_state: list,
    pos_points_state: list,
    neg_points_state: list,
):
    """Commits current SAM polygon into saved GIS layers and clears points."""
    if current_poly is None:
        return current_img, all_polys_state, [], [], None, None, "Kein aktives SAM-Polygon zum Speichern."

    all_polys_state.append(current_poly)

    # Re-render base view with all saved polygons
    vis = current_img.copy()
    for p in all_polys_state:
        pts = np.array(p.exterior.coords, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 120))
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        cv2.polylines(vis, [pts], True, (0, 230, 80), 2, cv2.LINE_AA)

    # Export to GeoDataFrame & Temp GeoJSON
    records = [{"id": i, "layer": "building_sam", "area_px": round(p.area, 2)} for i, p in enumerate(all_polys_state, 1)]
    gdf = gpd.GeoDataFrame(records, geometry=all_polys_state, crs="EPSG:25832")
    temp_geojson = os.path.join(tempfile.gettempdir(), "sam_interactive_buildings.geojson")
    temp_gpkg = os.path.join(tempfile.gettempdir(), "sam_interactive_buildings.gpkg")
    gdf.to_file(temp_geojson, driver="GeoJSON")
    gdf.to_file(temp_gpkg, driver="GPKG")

    status = f"💾 Gespeichert! Gesamt: {len(all_polys_state)} Objekte in der GIS-Ebene."
    return vis, all_polys_state, [], [], temp_geojson, temp_gpkg, status


def run_sam_auto_assistant(current_img, rgb_diff: int, min_area: int):
    """Runs semi-automatic SAM extraction on red building ink seeds."""
    if current_img is None:
        return current_img, [], None, None, "Kein Bild geladen."

    sam = get_sam_extractor()
    res = sam.auto_segment_buildings_with_sam(
        current_img,
        rgb_diff_thresh=int(rgb_diff),
        min_seed_area=int(min_area),
    )

    vis = current_img.copy()
    for p in res["polygons"]:
        pts = np.array(p.exterior.coords, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 120))
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        cv2.polylines(vis, [pts], True, (0, 230, 80), 2, cv2.LINE_AA)

    temp_geojson = os.path.join(tempfile.gettempdir(), "sam_auto_buildings.geojson")
    temp_gpkg = os.path.join(tempfile.gettempdir(), "sam_auto_buildings.gpkg")
    if res["gdf"].crs is None:
        res["gdf"].set_crs("EPSG:25832", inplace=True, allow_override=True)
    res["gdf"].to_file(temp_geojson, driver="GeoJSON")
    res["gdf"].to_file(temp_gpkg, driver="GPKG")

    status = f"🤖 SAM Auto-Assistent: {res['building_count']} saubere Gebäude aus {res['seed_count']} Seed-Punkten extrahiert (ohne Baum-/Hintergrundrauschen)!"
    return vis, res["polygons"], temp_geojson, temp_gpkg, status


def clear_sam_points(current_img, all_polys_state: list):
    """Clears point prompts and restores saved polygons."""
    vis = current_img.copy() if current_img is not None else None
    if vis is not None:
        for p in all_polys_state:
            pts = np.array(p.exterior.coords, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 220, 80), 2, cv2.LINE_AA)
    return vis, [], [], "Punkte zurückgesetzt."


# ==========================================
# Automated Pipeline Callbacks
# ==========================================

def run_auto_pipeline(
    preset_choice: str,
    uploaded_file,
    extract_buildings: bool,
    extract_roads: bool,
    extract_text: bool,
    rgb_diff_thresh: int,
    min_bldg_area: float,
    max_bldg_area: float,
    poly_simplify: float,
    ridge_thresh: float,
    min_road_len: float,
    ocr_conf_thresh: float,
    tile_size: int,
    max_tiles_val: int,
    progress=gr.Progress(),
):
    target_path = None
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            target_path = uploaded_file
        elif hasattr(uploaded_file, "name"):
            target_path = uploaded_file.name
    elif preset_choice and preset_choice in PRESETS_DICT:
        target_path = PRESETS_DICT[preset_choice]
    else:
        target_path = SAMPLE_IMAGE_PATH

    if not target_path or not os.path.exists(target_path):
        return (
            None, None, None, None,
            pd.DataFrame(columns=["ID", "Text", "Category", "Confidence", "X", "Y"]),
            "⚠️ Bitte eine Datei auswählen oder hochladen.",
            None, None, None, None, None
        )

    b_cfg = BuildingConfig(
        rgb_diff_threshold=int(rgb_diff_thresh),
        min_building_area_px=float(min_bldg_area),
        max_building_area_px=float(max_bldg_area),
        polygon_simplification_factor=float(poly_simplify),
    )
    r_cfg = RoadConfig(
        ridge_threshold=float(ridge_thresh),
        min_path_length_px=float(min_road_len),
    )
    t_cfg = TextConfig(
        confidence_threshold=float(ocr_conf_thresh),
        use_gpu=True,
    )
    p_cfg = PipelineConfig(buildings=b_cfg, roads=r_cfg, text=t_cfg)

    temp_dir = tempfile.mkdtemp(prefix="tranchot_ui_")

    is_large = False
    if HAS_RASTERIO and target_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(target_path) as src:
                if src.width > 2500 or src.height > 2500:
                    is_large = True
        except Exception:
            pass

    progress(0.1, desc="Initialisiere Extraktion...")

    if is_large:
        processor = TiledMapProcessor(p_cfg)
        max_t = int(max_tiles_val) if max_tiles_val > 0 else None

        def _prog_cb(cur, total):
            progress(cur / float(total), desc=f"Verarbeite Kachel {cur}/{total}...")

        res_dict = processor.process_geotiff(
            target_path,
            output_dir=temp_dir,
            tile_size=int(tile_size),
            max_tiles=max_t,
            extract_buildings=extract_buildings,
            extract_roads=extract_roads,
            extract_text=extract_text,
            progress_callback=_prog_cb,
        )

        toponyms_df = pd.DataFrame()
        if not res_dict["toponyms_gdf"].empty:
            t_gdf = res_dict["toponyms_gdf"]
            toponyms_df = pd.DataFrame(t_gdf.drop(columns=["geometry"], errors="ignore"))

        stats_md = f"""### ✅ Extraktionsergebnisse (Großformat-Streaming)

| Feature-Klasse | Anzahl extrahiert |
| :--- | :--- |
| 🏛️ **Gebäude & Höfe** | **{res_dict['building_count']}** Polygone |
| 🛣️ **Straßen & Wege** | **{res_dict['road_count']}** Segmente |
| 🏷️ **Beschriftungen (Toponyme)** | **{res_dict['toponym_count']}** Textstellen |

**Georeferenzierung:** `{res_dict['metadata']['crs']}`  
"""
        try:
            with rasterio.open(target_path) as src:
                overview_h = 1000
                overview_w = int(src.width * (1000.0 / src.height))
                ov_data = src.read(
                    out_shape=(src.count, overview_h, overview_w),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                ov_rgb = np.transpose(ov_data[:3, :, :], (1, 2, 0))
        except Exception:
            ov_rgb = np.zeros((600, 600, 3), dtype=np.uint8)

        exported = res_dict["exported_files"]

        return (
            ov_rgb, ov_rgb, ov_rgb, ov_rgb,
            toponyms_df, stats_md,
            exported.get("buildings_geojson"),
            exported.get("roads_geojson"),
            exported.get("toponyms_geojson"),
            exported.get("geopackage"),
            exported.get("zip_package"),
        )
    else:
        pipeline = TranchotPipeline(p_cfg)
        res = pipeline.process(
            target_path,
            output_dir=temp_dir,
            extract_buildings=extract_buildings,
            extract_roads=extract_roads,
            extract_text=extract_text,
        )

        toponyms_df = pd.DataFrame()
        if res.text.features:
            rows = []
            for t in res.text.features:
                rows.append({
                    "ID": t.id,
                    "Text": t.text,
                    "Kategorie": t.category,
                    "Konfidenz": f"{t.confidence:.2%}",
                    "Koordinate X": t.centroid_x,
                    "Koordinate Y": t.centroid_y,
                })
            toponyms_df = pd.DataFrame(rows)

        stats_md = f"""### ✅ Extraktionsergebnisse

| Feature-Klasse | Anzahl extrahiert | Details |
| :--- | :--- | :--- |
| 🏛️ **Gebäude & Höfe** | **{len(res.buildings.features)}** Polygone | Rohkonturen: {res.buildings.raw_contour_count} |
| 🛣️ **Straßen & Wege** | **{len(res.roads.features)}** Segmente | Gesamtlänge: {sum(f.length_px for f in res.roads.features):.1f} px |
| 🏷️ **Beschriftungen (Toponyme)** | **{len(res.text.features)}** Textstellen | Ø Konfidenz: {np.mean([f.confidence for f in res.text.features]):.2%} |
"""
        comp_view = create_ui_preview(res.visualizations.get("composite"))
        bldg_view = create_ui_preview(res.visualizations.get("buildings"))
        road_view = create_ui_preview(res.visualizations.get("roads"))
        text_view = create_ui_preview(res.visualizations.get("text"))

        return (
            comp_view, bldg_view, road_view, text_view,
            toponyms_df, stats_md,
            res.exported_files.get("buildings_geojson"),
            res.exported_files.get("roads_geojson"),
            res.exported_files.get("toponyms_geojson"),
            res.exported_files.get("geopackage"),
            res.exported_files.get("zip_package"),
        )


# ==========================================
# Main App Layout
# ==========================================

def create_app() -> gr.Blocks:
    """Builds and returns the Gradio application."""
    custom_css = """
    .main-header {
        text-align: center;
        margin-bottom: 15px;
        padding: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.1rem;
        margin-bottom: 4px;
        font-weight: 700;
    }
    .sam-box {
        background: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #3b82f6;
        margin-bottom: 10px;
    }
    """

    preset_names = list(PRESETS_DICT.keys())
    default_preset = preset_names[0] if preset_names else None

    with gr.Blocks(css=custom_css, title="Tranchot Map Feature Extractor & SAM Studio") as demo:
        with gr.Row():
            gr.HTML("""
            <div class="main-header">
                <h1>🏛️ Tranchot Map AI Extractor & SAM Studio</h1>
                <p>Automatische & Semiautomatische KI-Extraktion (Segment Anything Model) für historische Karten (1803–1828)</p>
            </div>
            """)

        with gr.Tabs():
            # ==========================================
            # TAB 1: Interaktives SAM KI-Studio
            # ==========================================
            with gr.TabItem("🎯 SAM KI-Studio (Interaktiv & Semiautomatisch)"):
                # States
                pos_points_state = gr.State([])
                neg_points_state = gr.State([])
                current_poly_state = gr.State(None)
                all_polys_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("### 🗺️ 1. Kartenausschnitt wählen")
                        sam_preset = gr.Dropdown(
                            choices=preset_names,
                            value=default_preset,
                            label="Kartenblatt / Preset"
                        )
                        btn_load_sam_img = gr.Button("🔄 Bild in SAM-Studio laden", variant="secondary")

                        gr.Markdown("### 🎯 2. Interaktive SAM-Steuerung")
                        click_mode = gr.Radio(
                            choices=["[+] Positiver Punkt (Gebäude / Feature)", "[-] Negativer Punkt (Hintergrund / Ausschluss)"],
                            value="[+] Positiver Punkt (Gebäude / Feature)",
                            label="Klick-Modus (Klicke direkt auf die Karte rechts!)"
                        )

                        with gr.Row():
                            btn_run_sam = gr.Button("✨ SAM Segmentieren", variant="primary", size="lg")
                            btn_save_sam = gr.Button("➕ Als GIS-Objekt übernehmen", variant="secondary", size="lg")

                        with gr.Row():
                            btn_clear_points = gr.Button("🧹 Punkte zurücksetzen")

                        gr.Markdown("---")
                        gr.Markdown("### 🤖 3. SAM Auto-Assistent (Semiautomatisch)")
                        gr.Markdown("Extrahiert alle Gebäude des Ausschnitts vollautomatisch mittels gefilterter SAM-Seed-Punkte.")
                        with gr.Row():
                            slider_sam_diff = gr.Slider(minimum=15, maximum=60, value=35, step=1, label="Rot-Toleranz")
                            slider_sam_min_area = gr.Slider(minimum=5, maximum=50, value=15, step=1, label="Min Seed-Fläche")
                        btn_sam_auto = gr.Button("🚀 SAM Auto-Extraktion starten", variant="primary")

                        gr.Markdown("### 💾 4. SAM GIS-Downloads")
                        with gr.Row():
                            dl_sam_geojson = gr.File(label="SAM Gebäude (GeoJSON)")
                            dl_sam_gpkg = gr.File(label="SAM GeoPackage (.gpkg)")

                    with gr.Column(scale=6):
                        sam_image_display = gr.Image(
                            value=load_preset_image(default_preset),
                            label="Interaktive Karte (Hier klicken, um SAM-Punkte zu setzen!)",
                            interactive=False,
                            type="numpy"
                        )
                        sam_status = gr.Markdown("### Status\nBereit. Klicke auf die Karte, um ein Gebäude oder Feature zu markieren.")

                # SAM Event Handlers
                btn_load_sam_img.click(
                    fn=load_preset_image,
                    inputs=[sam_preset],
                    outputs=[sam_image_display]
                )

                sam_image_display.select(
                    fn=handle_sam_click,
                    inputs=[sam_image_display, click_mode, pos_points_state, neg_points_state, all_polys_state],
                    outputs=[sam_image_display, pos_points_state, neg_points_state, sam_status]
                )

                btn_run_sam.click(
                    fn=run_sam_segmentation,
                    inputs=[sam_image_display, pos_points_state, neg_points_state, all_polys_state],
                    outputs=[sam_image_display, current_poly_state, sam_status]
                )

                btn_save_sam.click(
                    fn=save_sam_polygon,
                    inputs=[sam_image_display, current_poly_state, all_polys_state, pos_points_state, neg_points_state],
                    outputs=[sam_image_display, all_polys_state, pos_points_state, neg_points_state, dl_sam_geojson, dl_sam_gpkg, sam_status]
                )

                btn_clear_points.click(
                    fn=clear_sam_points,
                    inputs=[sam_image_display, all_polys_state],
                    outputs=[sam_image_display, pos_points_state, neg_points_state, sam_status]
                )

                btn_sam_auto.click(
                    fn=run_sam_auto_assistant,
                    inputs=[sam_image_display, slider_sam_diff, slider_sam_min_area],
                    outputs=[sam_image_display, all_polys_state, dl_sam_geojson, dl_sam_gpkg, sam_status]
                )

            # ==========================================
            # TAB 2: Automatisches Streaming Pipeline
            # ==========================================
            with gr.TabItem("⚡ Automatische Batch- & Streaming-Pipeline"):
                with gr.Row():
                    with gr.Column(scale=4):
                        preset_dropdown = gr.Dropdown(choices=preset_names, value=default_preset, label="Kartenbestand & GeoTIFF Presets")
                        file_input = gr.File(label="Oder eigenes Kartenblatt / GeoTIFF hochladen", file_count="single")

                        with gr.Group():
                            gr.Markdown("### ⚙️ Module")
                            with gr.Row():
                                chk_buildings = gr.Checkbox(label="🏛️ Gebäude", value=True)
                                chk_roads = gr.Checkbox(label="🛣️ Straßen", value=True)
                                chk_text = gr.Checkbox(label="🏷️ OCR Toponyme", value=True)

                        with gr.Accordion("🔧 Parameter", open=False):
                            slider_rgb_diff = gr.Slider(minimum=10, maximum=80, value=28, step=1, label="Rot-Farbdifferenz")
                            slider_min_bldg = gr.Slider(minimum=2, maximum=100, value=12, step=1, label="Min. Fläche")
                            slider_max_bldg = gr.Slider(minimum=500, maximum=100000, value=35000, step=500, label="Max. Fläche")
                            slider_simplify = gr.Slider(minimum=0.005, maximum=0.08, value=0.02, step=0.005, label="Vereinfachung")
                            slider_ridge = gr.Slider(minimum=5, maximum=80, value=24, step=1, label="Linien-Sensitivität")
                            slider_min_road = gr.Slider(minimum=5, maximum=100, value=15, step=1, label="Min. Weglänge")
                            slider_ocr_conf = gr.Slider(minimum=0.05, maximum=0.95, value=0.20, step=0.05, label="Min. OCR Konfidenz")
                            slider_tile_size = gr.Slider(minimum=1024, maximum=4096, value=2048, step=512, label="Kachelgröße (px)")
                            slider_max_tiles = gr.Slider(minimum=0, maximum=50, value=6, step=1, label="Max. Kacheln (0=alle)")

                        btn_run_auto = gr.Button("🚀 Vollautomatische Extraktion", variant="primary", size="lg")

                        with gr.Group():
                            gr.Markdown("### 💾 Downloads")
                            with gr.Row():
                                dl_buildings = gr.File(label="🏛️ Gebäude (GeoJSON)")
                                dl_roads = gr.File(label="🛣️ Straßen (GeoJSON)")
                            with gr.Row():
                                dl_toponyms = gr.File(label="🏷️ Toponyme (GeoJSON)")
                                dl_gpkg = gr.File(label="📦 GeoPackage (.gpkg)")
                            dl_zip = gr.File(label="🗜️ Komplett ZIP")

                    with gr.Column(scale=6):
                        with gr.Tabs():
                            with gr.TabItem("🌟 Gesamtübersicht"):
                                img_composite = gr.Image(label="Kombinierte Vektoren", interactive=False)
                            with gr.TabItem("🏛️ Gebäude-Layer"):
                                img_buildings = gr.Image(label="Gebäude", interactive=False)
                            with gr.TabItem("🛣️ Straßennetz-Layer"):
                                img_roads = gr.Image(label="Straßen", interactive=False)
                            with gr.TabItem("🏷️ Beschriftungs-Layer"):
                                img_text = gr.Image(label="OCR Textboxen", interactive=False)
                            with gr.TabItem("📊 Toponym-Datenbank"):
                                df_toponyms = gr.Dataframe(label="Erkannte Namen", wrap=True)

                        stats_output = gr.Markdown("### Status\nBereit.")

                btn_run_auto.click(
                    fn=run_auto_pipeline,
                    inputs=[
                        preset_dropdown, file_input,
                        chk_buildings, chk_roads, chk_text,
                        slider_rgb_diff, slider_min_bldg, slider_max_bldg, slider_simplify,
                        slider_ridge, slider_min_road, slider_ocr_conf,
                        slider_tile_size, slider_max_tiles,
                    ],
                    outputs=[
                        img_composite, img_buildings, img_roads, img_text,
                        df_toponyms, stats_output,
                        dl_buildings, dl_roads, dl_toponyms, dl_gpkg, dl_zip,
                    ]
                )

    return demo


def launch(server_port: int = 7860, share: bool = False):
    """Launches the Gradio web server."""
    app = create_app()
    app.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    launch()
