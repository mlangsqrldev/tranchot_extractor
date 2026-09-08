"""
FastAPI Server for Tranchot Label Studio & WebGIS Suite.
Provides high-performance REST APIs for interactive Color Enhancement, Pipette/Stamp Land-Use Learning,
SAM Segmentation, 90-degree Building Extraction, Road Digitization, OCR Toponym Recognition, and GIS Exports.
"""

import os
import glob
import json
import base64
import tempfile
import io
import uuid
import time
import zipfile
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
from shapely.geometry import Polygon, Point, LineString, mapping
from shapely.affinity import translate
import geopandas as gpd

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from tranchot_extractor.preprocessing.color_enhancer import ColorEnhancer
from tranchot_extractor.extractors.pipette_sampler import PipetteSampler, ColorSample, StampEntry
from tranchot_extractor.extractors.building_extractor import BuildingExtractor, BuildingConfig
from tranchot_extractor.extractors.road_extractor import RoadExtractor, RoadConfig
from tranchot_extractor.extractors.sam_extractor import SAMExtractor
from tranchot_extractor.extractors.text_extractor import TextExtractor
from tranchot_extractor.geo.spatial_gazetteer import SpatialGazetteer
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.exporter import GISExporter

# Directory Paths
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(STUDIO_DIR)
WORKSPACE_ROOT = os.path.dirname(PACKAGE_DIR)
STATIC_DIR = os.path.join(STUDIO_DIR, "static")
SAMPLE_DIR = os.path.join(WORKSPACE_ROOT, "Sample")
SAMPLE_DATA_DIR = os.path.join(WORKSPACE_ROOT, "sample_data")
GEOTIFFS_DIR = os.path.join(WORKSPACE_ROOT, "output_geotiffs")

app = FastAPI(title="Tranchot WebGIS Label Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
CURRENT_RAW_IMAGE_RGB: Optional[np.ndarray] = None
CURRENT_ENHANCED_IMAGE_RGB: Optional[np.ndarray] = None
CURRENT_IMAGE_PATH: Optional[str] = None
CURRENT_IMAGE_META: Dict[str, Any] = {}
SAMPLER: PipetteSampler = PipetteSampler()
SAM_ENGINE: Optional[SAMExtractor] = None
TEXT_ENGINE: Optional[TextExtractor] = None
GEO_HANDLER: Optional[GeoReferenceHandler] = None

# Enhancement parameters
ENHANCE_PARAMS = {
    "deyellow_strength": 0.85,
    "vibrance": 1.65,
    "contrast": 1.0,
    "ink_blackening": True,
}


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

class EnhanceRequest(BaseModel):
    deyellow_strength: float = 0.85
    vibrance: float = 1.65
    contrast: float = 1.0
    ink_blackening: bool = True


class StampRequest(BaseModel):
    class_id: str
    cx: float
    cy: float
    radius: int = 22


class RemoveStampRequest(BaseModel):
    class_id: str
    stamp_index: int


class ClearStampsRequest(BaseModel):
    class_id: Optional[str] = None


class SAMPromptRequest(BaseModel):
    positive_points: List[List[float]] = []
    negative_points: List[List[float]] = []
    bounding_box: Optional[List[float]] = None
    granularity: str = "compact"  # 'compact', 'medium', 'large'
    simplification: float = 0.015


class ExtractLandUseRequest(BaseModel):
    active_class_ids: Optional[List[str]] = None


class ExtractBuildingsRequest(BaseModel):
    min_area_px: float = 40.0
    max_area_px: float = 8000.0
    orthogonal_regularization: bool = True


class ExtractRoadsRequest(BaseModel):
    min_length_px: float = 50.0


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
    format: str = "geojson"  # 'geojson', 'shapefile_zip', 'gpkg', 'label_studio', 'coco'


# ==========================================
# REST API Endpoints
# ==========================================

def _list_preset_entries() -> List[Dict[str, Any]]:
    """Builds the list of map sheets available for loading."""
    presets = []
    
    # 1. Sample directory
    if os.path.exists(SAMPLE_DIR):
        for f in glob.glob(os.path.join(SAMPLE_DIR, "*.tif")) + glob.glob(os.path.join(SAMPLE_DIR, "*.tiff")):
            presets.append({
                "id": os.path.basename(f),
                "name": f"🗺️ Tranchot: {os.path.basename(f)}",
                "path": f,
                "type": "geotiff"
            })
            
    # 2. sample_data directory
    if os.path.exists(SAMPLE_DATA_DIR):
        for f in glob.glob(os.path.join(SAMPLE_DATA_DIR, "*.*")):
            presets.append({
                "id": os.path.basename(f),
                "name": f"📋 Probeblatt: {os.path.basename(f)}",
                "path": f,
                "type": "png" if f.endswith(".png") else "geotiff"
            })

    # 3. output_geotiffs directory
    if os.path.exists(GEOTIFFS_DIR):
        for f in glob.glob(os.path.join(GEOTIFFS_DIR, "**", "*.tif"), recursive=True):
            presets.append({
                "id": os.path.basename(f),
                "name": f"🌍 GeoTIFF: {os.path.basename(f)}",
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
    """Loads map image into studio canvas and computes initial enhancement."""
    global CURRENT_RAW_IMAGE_RGB, CURRENT_ENHANCED_IMAGE_RGB, CURRENT_IMAGE_PATH, CURRENT_IMAGE_META, GEO_HANDLER
    global SAMPLER

    target_path = None
    if file is not None:
        safe_name = os.path.basename(file.filename or "upload")
        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{safe_name}")
        with open(temp_file, "wb") as f:
            f.write(file.file.read())
        target_path = temp_file
    elif preset_path:
        target_path = preset_path
    else:
        # Default to Nickenich.tif
        nickenich = os.path.join(SAMPLE_DIR, "Nickenich.tif")
        if os.path.exists(nickenich):
            target_path = nickenich
        else:
            presets = _list_preset_entries()
            if presets:
                target_path = presets[0]["path"]

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Kein gültiger Bildpfad gefunden.")

    CURRENT_IMAGE_PATH = target_path
    geo_meta = {}
    img_rgb = None

    if HAS_RASTERIO and target_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(target_path) as src:
                geo_meta = {
                    "crs": src.crs.to_string() if src.crs else "EPSG:25832",
                    "transform": list(src.transform),
                    "bounds": list(src.bounds),
                    "full_width": src.width,
                    "full_height": src.height,
                }
                data = src.read()
                if data.shape[0] >= 3:
                    img_rgb = np.transpose(data[:3, :, :], (1, 2, 0))
                else:
                    gray = data[0, :, :]
                    img_rgb = np.stack([gray, gray, gray], axis=-1)
        except Exception as e:
            print(f"[Studio] Rasterio read fallback: {e}")

    if img_rgb is None:
        pil_img = Image.open(target_path).convert("RGB")
        img_rgb = np.array(pil_img)
        geo_meta = {"crs": "EPSG:25832", "bounds": [0, 0, img_rgb.shape[1], img_rgb.shape[0]]}

    CURRENT_RAW_IMAGE_RGB = img_rgb
    CURRENT_IMAGE_META = geo_meta
    GEO_HANDLER = GeoReferenceHandler(target_path) if target_path.endswith(('.tif', '.tiff')) else None

    # Compute default enhanced image
    CURRENT_ENHANCED_IMAGE_RGB = ColorEnhancer.enhance_map(
        CURRENT_RAW_IMAGE_RGB,
        deyellow_strength=ENHANCE_PARAMS["deyellow_strength"],
        vibrance=ENHANCE_PARAMS["vibrance"],
        contrast=ENHANCE_PARAMS["contrast"],
        ink_blackening=ENHANCE_PARAMS["ink_blackening"],
    )

    # Re-initialize sampler for new image
    SAMPLER = PipetteSampler()

    # Encode display image to JPEG for 60 FPS web rendering (downsample for browser DOM if > 2560px)
    h, w = img_rgb.shape[:2]
    max_dim = 2560
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp_img = cv2.resize(CURRENT_ENHANCED_IMAGE_RGB, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_img = CURRENT_ENHANCED_IMAGE_RGB

    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "filename": os.path.basename(target_path),
        "width": w,
        "height": h,
        "image_data": f"data:image/jpeg;base64,{b64_str}",
        "metadata": geo_meta,
        "enhance_params": ENHANCE_PARAMS,
    }


@app.post("/api/enhance_map")
def enhance_map_api(req: EnhanceRequest):
    """Dynamically applies white-balance, de-yellowing, vibrancy, and ink-blackening."""
    global CURRENT_RAW_IMAGE_RGB, CURRENT_ENHANCED_IMAGE_RGB, ENHANCE_PARAMS
    if CURRENT_RAW_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    ENHANCE_PARAMS = req.model_dump()

    CURRENT_ENHANCED_IMAGE_RGB = ColorEnhancer.enhance_map(
        CURRENT_RAW_IMAGE_RGB,
        deyellow_strength=req.deyellow_strength,
        vibrance=req.vibrance,
        contrast=req.contrast,
        ink_blackening=req.ink_blackening,
    )

    h, w = CURRENT_ENHANCED_IMAGE_RGB.shape[:2]
    max_dim = 2560
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp_img = cv2.resize(CURRENT_ENHANCED_IMAGE_RGB, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_img = CURRENT_ENHANCED_IMAGE_RGB

    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "image_data": f"data:image/jpeg;base64,{b64_str}",
        "params": ENHANCE_PARAMS
    }


@app.get("/api/stamps")
def get_stamps_api():
    """Returns all active stamps grouped by class."""
    global SAMPLER
    result = {}
    for cid, sample in SAMPLER.samples.items():
        stamps = []
        for s in getattr(sample, "stamps", []):
            stamps.append({
                "stamp_id": s.stamp_id,
                "cx": s.cx,
                "cy": s.cy,
                "radius": s.radius,
                "rgb": s.rgb,
                "hex_color": s.hex_color,
                "raw_pixels": s.raw_pixels,
                "distilled_pixels": s.distilled_pixels,
            })
        result[cid] = {
            "class_id": cid,
            "label": sample.label,
            "hex_color": sample.hex_color,
            "stamps": stamps,
            "stamps_count": len(stamps),
        }
    return {"classes": result}


@app.post("/api/sample_stamp")
def sample_stamp_api(req: StampRequest):
    """Extracts pure pigment from a circular stamp area on the enhanced map."""
    global CURRENT_ENHANCED_IMAGE_RGB, SAMPLER
    if CURRENT_ENHANCED_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    entry = SAMPLER.sample_from_stamp(
        CURRENT_ENHANCED_IMAGE_RGB,
        req.class_id,
        cx=req.cx,
        cy=req.cy,
        radius=req.radius,
    )

    if not entry:
        raise HTTPException(
            status_code=422,
            detail=f"Kein Farbpigment für Klasse '{req.class_id}' an dieser Position gefunden (z.B. auf unbemaltes Papier oder Straßenschwärzung geklickt)."
        )

    sample = SAMPLER.get_sample(req.class_id)
    return {
        "status": "ok",
        "stamp": {
            "stamp_id": entry.stamp_id,
            "cx": entry.cx,
            "cy": entry.cy,
            "radius": entry.radius,
            "rgb": entry.rgb,
            "hex_color": entry.hex_color,
            "raw_pixels": entry.raw_pixels,
            "distilled_pixels": entry.distilled_pixels,
        },
        "class_prototype": {
            "class_id": req.class_id,
            "label": sample.label if sample else req.class_id,
            "hex_color": sample.hex_color if sample else entry.hex_color,
            "stamps_count": len(sample.stamps) if sample else 1,
        }
    }


@app.post("/api/remove_stamp")
def remove_stamp_api(req: RemoveStampRequest):
    """Removes a specific stamp."""
    global SAMPLER
    SAMPLER.remove_stamp(req.class_id, req.stamp_index)
    return {"status": "ok"}


@app.post("/api/clear_stamps")
def clear_stamps_api(req: ClearStampsRequest):
    """Clears all stamps for a specific class or all classes."""
    global SAMPLER
    if req.class_id:
        SAMPLER.clear_stamps(req.class_id)
        SAMPLER.reset_class(req.class_id)
    else:
        SAMPLER._init_defaults()
    return {"status": "ok"}


@app.post("/api/extract_landuse")
def extract_landuse_api(req: ExtractLandUseRequest):
    """Executes multi-class competitive land-use extraction with strict parcel geometry."""
    global CURRENT_ENHANCED_IMAGE_RGB, SAMPLER
    if CURRENT_ENHANCED_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    t0 = time.time()
    sampled_cids = req.active_class_ids
    if not sampled_cids:
        sampled_cids = [
            cid for cid, s in SAMPLER.samples.items()
            if hasattr(s, "stamps") and len(s.stamps) > 0
        ]

    if not sampled_cids:
        # Fallback to forest, meadow, water if no stamps set
        sampled_cids = ["forest", "meadow", "water"]

    res_polys = SAMPLER.extract_competitive_polygons(
        CURRENT_ENHANCED_IMAGE_RGB,
        active_class_ids=sampled_cids
    )

    features = []
    layer_stats = {}

    for cid, polys in res_polys.items():
        sample = SAMPLER.get_sample(cid)
        hex_color = sample.hex_color if sample else "#22c55e"
        layer_stats[cid] = len(polys)

        for idx, p in enumerate(polys):
            if not p.is_valid or p.is_empty:
                continue
            geom_json = mapping(p)
            features.append({
                "type": "Feature",
                "id": f"{cid}_{idx+1}",
                "geometry": geom_json,
                "properties": {
                    "class_id": cid,
                    "label": sample.label if sample else cid,
                    "color": hex_color,
                    "area_px": round(p.area, 1),
                    "bounds": [round(b, 1) for b in p.bounds],
                }
            })

    elapsed = time.time() - t0
    return {
        "type": "FeatureCollection",
        "features": features,
        "layer_stats": layer_stats,
        "elapsed_seconds": round(elapsed, 2),
        "total_polygons": len(features),
    }


@app.post("/api/extract_buildings")
def extract_buildings_api(req: ExtractBuildingsRequest):
    """Extracts 90-degree orthogonal buildings and farmsteads."""
    global CURRENT_ENHANCED_IMAGE_RGB
    if CURRENT_ENHANCED_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    cfg = BuildingConfig(
        min_building_area_px=req.min_area_px,
        max_building_area_px=req.max_area_px,
        regularize_orthogonal=req.orthogonal_regularization,
    )
    extractor = BuildingExtractor(cfg)
    res = extractor.extract(CURRENT_ENHANCED_IMAGE_RGB)

    features = []
    for idx, b in enumerate(res.features):
        features.append({
            "type": "Feature",
            "id": f"building_{idx+1}",
            "geometry": mapping(b.geometry),
            "properties": {
                "class_id": "building",
                "label": "Gebäude",
                "color": "#ef4444",
                "area_px": round(b.area_px, 1),
                "is_courtyard": getattr(b, "is_courtyard", False),
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "building_count": len(features),
    }


@app.post("/api/extract_roads")
def extract_roads_api(req: ExtractRoadsRequest):
    """Extracts historical road and path network."""
    global CURRENT_ENHANCED_IMAGE_RGB
    if CURRENT_ENHANCED_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    cfg = RoadConfig(min_road_length_px=req.min_length_px)
    extractor = RoadExtractor(cfg)
    res = extractor.extract(CURRENT_ENHANCED_IMAGE_RGB)

    features = []
    for idx, r in enumerate(res.features):
        features.append({
            "type": "Feature",
            "id": f"road_{idx+1}",
            "geometry": mapping(r.geometry),
            "properties": {
                "class_id": "road",
                "label": "Straße / Weg",
                "color": "#3b82f6",
                "length_px": round(r.length_px, 1),
                "road_type": getattr(r, "road_type", "unclassified"),
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "road_count": len(features),
    }


@app.post("/api/sam_predict")
def sam_predict_api(req: SAMPromptRequest):
    """Executes SAM prompt inference in real time (< 30ms)."""
    global CURRENT_ENHANCED_IMAGE_RGB
    if CURRENT_ENHANCED_IMAGE_RGB is None:
        raise HTTPException(status_code=400, detail="Kein Bild geladen.")

    sam = get_sam()
    res = sam.segment_with_prompts(
        CURRENT_ENHANCED_IMAGE_RGB,
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


@app.post("/api/export")
def export_annotations_api(req: ExportRequest):
    """Exports annotations to GeoJSON, Shapefile ZIP, GPKG, or Label Studio."""
    temp_dir = tempfile.mkdtemp(prefix="tranchot_studio_export_")
    global CURRENT_IMAGE_META

    # Convert annotations into Shapely geometries
    features_data = []
    for item in req.annotations:
        geom = None
        if item.type == "polygon" and len(item.points) >= 1:
            try:
                # Check if points is a list of rings [shell, hole1, hole2, ...]
                if len(item.points) > 0 and isinstance(item.points[0], (list, tuple)) and len(item.points[0]) > 0 and isinstance(item.points[0][0], (list, tuple)):
                    poly = Polygon(shell=item.points[0], holes=item.points[1:])
                else:
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
    crs = CURRENT_IMAGE_META.get("crs") or "EPSG:25832"
    gdf.set_crs(crs, inplace=True, allow_override=True)

    if req.format == "shapefile_zip":
        shp_dir = os.path.join(temp_dir, "shapefiles")
        os.makedirs(shp_dir, exist_ok=True)
        
        # Group by label
        for label, group in gdf.groupby("label"):
            safe_label = "".join(c for c in label if c.isalnum() or c in ('_', '-')).strip() or "layer"
            layer_path = os.path.join(shp_dir, f"{safe_label}.shp")
            group.to_file(layer_path, driver="ESRI Shapefile")
            
        zip_path = os.path.join(temp_dir, "tranchot_shapefiles.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(shp_dir):
                for f in files:
                    zipf.write(os.path.join(root, f), f)
        return FileResponse(zip_path, filename="tranchot_shapefiles.zip", media_type="application/zip")

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
    """Serves the Label Studio WebGIS SPA interface."""
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Tranchot Label Studio Frontend loading...</h1>"
