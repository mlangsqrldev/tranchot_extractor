"""
FastAPI Server for Tranchot Label Studio.
Provides high-performance REST APIs for interactive SAM segmentation, OCR, and GIS exports.
"""

import os
import glob
import json
import base64
import tempfile
import io
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from tranchot_extractor.extractors.sam_extractor import SAMExtractor
from tranchot_extractor.extractors.text_extractor import TextExtractor
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.exporter import GISExporter

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
GEOTIFFS_DIR = os.path.join(WORKSPACE_ROOT, "output_geotiffs")
SAMPLE_PATH = os.path.join(WORKSPACE_ROOT, "sample_data", "rommerskirchen_sample.png")

app = FastAPI(title="Tranchot Label Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory image store and AI instances
CURRENT_IMAGE_RGB: Optional[np.ndarray] = None
CURRENT_IMAGE_META: Dict[str, Any] = {}
SAM_ENGINE: Optional[SAMExtractor] = None
TEXT_ENGINE: Optional[TextExtractor] = None


def get_sam() -> SAMExtractor:
    global SAM_ENGINE
    if SAM_ENGINE is None:
        SAM_ENGINE = SAMExtractor()
    return SAM_ENGINE


def get_ocr() -> TextExtractor:
    global TEXT_ENGINE
    if TEXT_ENGINE is None:
        TEXT_ENGINE = TextExtractor()
    return TEXT_ENGINE


# ==========================================
# Pydantic Request Models
# ==========================================

class SAMPromptRequest(BaseModel):
    positive_points: List[List[float]] = []
    negative_points: List[List[float]] = []
    bounding_box: Optional[List[float]] = None
    granularity: str = "compact"  # 'compact', 'medium', 'large'
    simplification: float = 0.015


class SAMAutoRequest(BaseModel):
    rgb_diff_thresh: int = 35
    min_seed_area: int = 15
    max_seeds: int = 150


class ExportAnnotationItem(BaseModel):
    id: int
    label: str
    color: str
    type: str  # 'polygon', 'bbox', 'point', 'line'
    points: List[List[float]]
    text: Optional[str] = None
    properties: Dict[str, Any] = {}


class ExportRequest(BaseModel):
    annotations: List[ExportAnnotationItem]
    format: str = "geojson"  # 'geojson', 'gpkg', 'label_studio', 'coco', 'csv'


# ==========================================
# REST API Endpoints
# ==========================================

def _list_preset_entries() -> List[Dict[str, Any]]:
    """Builds the list of map sheets the server is willing to load."""
    presets = []
    if os.path.exists(SAMPLE_PATH):
        presets.append({
            "id": "sample_rommerskirchen",
            "name": "📋 Rommerskirchen Ausschnitt (Sample PNG)",
            "path": SAMPLE_PATH,
            "type": "png"
        })

    tranchot_files = glob.glob(os.path.join(GEOTIFFS_DIR, "tranchot", "*.tif"))
    for f in tranchot_files:
        presets.append({
            "id": os.path.basename(f),
            "name": f"🗺️ Tranchot: {os.path.basename(f)}",
            "path": f,
            "type": "geotiff"
        })

    meilen_files = glob.glob(os.path.join(GEOTIFFS_DIR, "meilenblaetter", "*.tif"))
    for f in meilen_files:
        presets.append({
            "id": os.path.basename(f),
            "name": f"📜 Meilenblätter: {os.path.basename(f)}",
            "path": f,
            "type": "geotiff"
        })

    return presets


@app.get("/api/presets")
def list_presets():
    """Returns list of available map sheets."""
    return {"presets": _list_preset_entries()}


@app.post("/api/load_image")
def load_image_api(preset_path: str = Form(None), file: UploadFile = File(None)):
    """Loads map image into studio canvas."""
    global CURRENT_IMAGE_RGB, CURRENT_IMAGE_META

    target_path = None
    if file is not None:
        safe_name = os.path.basename(file.filename or "upload")
        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{safe_name}")
        with open(temp_file, "wb") as f:
            f.write(file.file.read())
        target_path = temp_file
    elif preset_path and os.path.realpath(preset_path) in {os.path.realpath(p["path"]) for p in _list_preset_entries()}:
        target_path = preset_path
    elif os.path.exists(SAMPLE_PATH):
        target_path = SAMPLE_PATH
    else:
        raise HTTPException(status_code=400, detail="Kein gültiger Bildpfad angegeben.")

    geo_meta = {}
    img_rgb = None

    if HAS_RASTERIO and target_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(target_path) as src:
                geo_meta = {
                    "crs": src.crs.to_string() if src.crs else "EPSG:3857",
                    "transform": list(src.transform),
                    "bounds": list(src.bounds),
                    "full_width": src.width,
                    "full_height": src.height,
                }
                # Load central 3000x3000 crop or downsampled overview if huge
                if src.width > 4000 or src.height > 4000:
                    cx, cy = src.width // 2, src.height // 2
                    w_size = min(3000, src.width, src.height)
                    win = rasterio.windows.Window(cx - w_size//2, cy - w_size//2, w_size, w_size)
                    crop_data = src.read(window=win)
                    img_rgb = np.transpose(crop_data[:3, :, :], (1, 2, 0))
                    geo_meta["view_offset"] = [int(cx - w_size//2), int(cy - w_size//2)]
                else:
                    data = src.read()
                    img_rgb = np.transpose(data[:3, :, :], (1, 2, 0))
                    geo_meta["view_offset"] = [0, 0]
        except Exception as e:
            print(f"Rasterio error: {e}")

    if img_rgb is None:
        pil_img = Image.open(target_path).convert("RGB")
        img_rgb = np.array(pil_img)
        geo_meta = {"crs": "EPSG:25832", "view_offset": [0, 0]}

    CURRENT_IMAGE_RGB = img_rgb
    CURRENT_IMAGE_META = geo_meta

    # Encode to JPEG for fast web delivery
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    h, w = img_rgb.shape[:2]
    return {
        "width": w,
        "height": h,
        "image_data": f"data:image/jpeg;base64,{b64_str}",
        "metadata": geo_meta,
    }


@app.post("/api/sam_predict")
def sam_predict_api(req: SAMPromptRequest):
    """Executes SAM prompt inference in real time (< 30ms)."""
    global CURRENT_IMAGE_RGB
    if CURRENT_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    sam = get_sam()
    res = sam.segment_with_prompts(
        CURRENT_IMAGE_RGB,
        positive_points=req.positive_points,
        negative_points=req.negative_points,
        bounding_box=req.bounding_box,
        granularity=req.granularity,
        simplification_factor=req.simplification,
    )

    polygons_pts = []
    for poly in res["polygons"]:
        coords = [[float(p[0]), float(p[1])] for p in poly.exterior.coords]
        polygons_pts.append({
            "points": coords,
            "area": round(poly.area, 2),
            "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)]
        })

    return {
        "polygons": polygons_pts,
        "iou_score": res["iou_score"],
    }


@app.post("/api/sam_auto")
def sam_auto_api(req: SAMAutoRequest):
    """Semi-automatic SAM extraction on current canvas."""
    global CURRENT_IMAGE_RGB
    if CURRENT_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    sam = get_sam()
    res = sam.auto_segment_buildings_with_sam(
        CURRENT_IMAGE_RGB,
        rgb_diff_thresh=req.rgb_diff_thresh,
        min_seed_area=req.min_seed_area,
        max_seeds=req.max_seeds,
    )

    polygons_pts = []
    for poly in res["polygons"]:
        coords = [[float(p[0]), float(p[1])] for p in poly.exterior.coords]
        polygons_pts.append({
            "points": coords,
            "area": round(poly.area, 2),
            "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)]
        })

    return {
        "polygons": polygons_pts,
        "seed_count": res["seed_count"],
        "building_count": res["building_count"],
    }


@app.post("/api/ocr_predict")
def ocr_predict_api():
    """Runs OCR on current canvas."""
    global CURRENT_IMAGE_RGB
    if CURRENT_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    ocr = get_ocr()
    res = ocr.extract(CURRENT_IMAGE_RGB)

    toponyms = []
    for t in res.features:
        toponyms.append({
            "id": t.id,
            "text": t.text,
            "confidence": t.confidence,
            "category": t.category,
            "bbox": t.bounding_box,
            "centroid": [t.centroid_x, t.centroid_y],
        })

    return {"toponyms": toponyms}


@app.post("/api/export")
def export_annotations_api(req: ExportRequest):
    """Exports annotations to GeoJSON, GPKG, Label Studio JSON, COCO, or Shapefile."""
    temp_dir = tempfile.mkdtemp(prefix="tranchot_studio_export_")
    global CURRENT_IMAGE_META

    # Convert annotations into Shapely geometries
    features_data = []
    for item in req.annotations:
        geom = None
        if item.type == "polygon" and len(item.points) >= 3:
            try:
                poly = Polygon(item.points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                geom = poly
            except Exception:
                pass
        elif item.type == "bbox" and len(item.points) >= 4:
            try:
                geom = Polygon(item.points)
            except Exception:
                pass
        elif item.type == "line" and len(item.points) >= 2:
            try:
                geom = LineString(item.points)
            except Exception:
                pass
        elif item.type == "point" and len(item.points) >= 1:
            try:
                geom = Point(item.points[0])
            except Exception:
                pass

        if geom is not None and not geom.is_empty:
            record = {
                "id": item.id,
                "label": item.label,
                "text": item.text or "",
                "color": item.color,
                "area_px": round(geom.area, 2) if hasattr(geom, "area") else 0,
                "geometry": geom
            }
            record.update(item.properties)
            features_data.append(record)

    gdf = gpd.GeoDataFrame(features_data, geometry=[r["geometry"] for r in features_data]) if features_data else gpd.GeoDataFrame(geometry=[])

    # Tag with the CRS of the currently loaded image (falls back to the Rhineland default)
    crs = CURRENT_IMAGE_META.get("crs") or "EPSG:25832"
    gdf.set_crs(crs, inplace=True, allow_override=True)

    if req.format == "label_studio":
        # Label Studio JSON format
        ls_tasks = [{
            "id": 1,
            "data": {"image": "uploaded_map"},
            "annotations": [{
                "result": [
                    {
                        "id": f"result_{item.id}",
                        "type": "polygonlabels" if item.type == "polygon" else "brushlabels",
                        "value": {
                            "points": item.points,
                            "polygonlabels": [item.label],
                        }
                    }
                    for item in req.annotations
                ]
            }]
        }]
        out_file = os.path.join(temp_dir, "label_studio_export.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ls_tasks, f, indent=2, ensure_ascii=False)
        return FileResponse(out_file, filename="label_studio_export.json", media_type="application/json")

    elif req.format == "coco":
        # COCO Segmentation format
        coco_data = {
            "images": [{"id": 1, "width": 2048, "height": 2048, "file_name": "map_sheet.jpg"}],
            "categories": [
                {"id": 1, "name": "Gebäude"},
                {"id": 2, "name": "Hofanlage"},
                {"id": 3, "name": "Straße"},
                {"id": 4, "name": "Toponym"},
            ],
            "annotations": [
                {
                    "id": item.id,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[coord for pt in item.points for coord in pt]],
                    "area": 500,
                    "iscrowd": 0
                }
                for item in req.annotations if item.type == "polygon"
            ]
        }
        out_file = os.path.join(temp_dir, "coco_annotations.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)
        return FileResponse(out_file, filename="coco_annotations.json", media_type="application/json")

    elif req.format == "gpkg":
        out_file = os.path.join(temp_dir, "tranchot_studio_annotations.gpkg")
        if not gdf.empty:
            gdf.to_file(out_file, layer="annotations", driver="GPKG")
        return FileResponse(out_file, filename="tranchot_studio_annotations.gpkg")

    else:
        # Default: GeoJSON
        out_file = os.path.join(temp_dir, "tranchot_studio_annotations.geojson")
        if not gdf.empty:
            gdf.to_file(out_file, driver="GeoJSON")
        else:
            with open(out_file, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
        return FileResponse(out_file, filename="tranchot_studio_annotations.geojson", media_type="application/geo+json")


# Mount static frontend directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the Label Studio SPA interface."""
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Tranchot Label Studio Frontend loading...</h1>"
