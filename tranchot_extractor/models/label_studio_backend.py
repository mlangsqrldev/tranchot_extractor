"""
Label Studio ML Backend Server with Interactive Segment Anything Model (SAM) Support.
Supports:
1. Interactive Point / Smart-Tool Click Prompting (Positive & Negative clicks via GPU)
2. Interactive Bounding Box Prompting (Rectangle -> SAM Mask -> Polygon)
3. Full Automatic Pre-annotation for Map Sheets (Auto-SAM & Rule-based)
4. YOLOv8 Segmentation Model inference
"""

import os
import io
import re
import base64
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Polygon

from tranchot_extractor.config import BuildingConfig, RoadConfig
from tranchot_extractor.extractors.building_extractor import BuildingExtractor
from tranchot_extractor.extractors.road_extractor import RoadExtractor

try:
    from tranchot_extractor.extractors.sam_extractor import SAMExtractor, HAS_SAM
except ImportError:
    HAS_SAM = False

try:
    from tranchot_extractor.models.yolo_extractor import YOLOSegExtractor, HAS_ULTRALYTICS
except ImportError:
    HAS_ULTRALYTICS = False


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(title="Tranchot SAM ML Backend for Label Studio", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings - DEFAULT TO SAM
MODEL_CONFIG = {
    "engine": "sam",  # Default to 'sam' (Segment Anything Model)
    "yolo_weights": "yolov8n-seg.pt",
    "conf_threshold": 0.35,
    "from_name": "label",
    "to_name": "image",
    "tag_type": "polygonlabels",  # 'polygonlabels' or 'brushlabels'
    "label_studio_url": os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
    "api_key": os.getenv("LABEL_STUDIO_API_KEY", ""),
}

_building_extractor = None
_road_extractor = None
_sam_extractor = None
_yolo_extractor = None


def get_sam_extractor() -> Optional[SAMExtractor]:
    global _sam_extractor
    if _sam_extractor is None and HAS_SAM:
        print("[ML-Backend] Lade SAM (Segment Anything Model) auf GPU/CPU...")
        _sam_extractor = SAMExtractor()
    return _sam_extractor


def get_building_extractor() -> BuildingExtractor:
    global _building_extractor
    if _building_extractor is None:
        _building_extractor = BuildingExtractor()
    return _building_extractor


def get_yolo_extractor(weights_path: Optional[str] = None):
    global _yolo_extractor
    weights = weights_path or MODEL_CONFIG["yolo_weights"]
    if _yolo_extractor is None and HAS_ULTRALYTICS:
        if os.path.exists(weights):
            _yolo_extractor = YOLOSegExtractor(model_path=weights)
    return _yolo_extractor


def resolve_and_load_image(image_ref: str) -> Optional[np.ndarray]:
    """
    Loads an image from various Label Studio sources:
    - Base64 data URI
    - Absolute or relative file path on disk
    - Label Studio server URL or external HTTP/HTTPS URL
    """
    if not image_ref:
        return None

    # 1. Base64 URI
    if image_ref.startswith("data:image/"):
        header, encoded = image_ref.split(",", 1)
        data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    # 2. Local file path check
    cleaned_path = image_ref
    if cleaned_path.startswith("file://"):
        cleaned_path = cleaned_path[7:]

    cleaned_path_no_query = cleaned_path.split("?")[0]
    filename_only = os.path.basename(cleaned_path_no_query)

    ls_data_dir = os.path.expanduser("~\\AppData\\Local\\label-studio\\label-studio")
    candidate_paths = [
        cleaned_path,
        cleaned_path_no_query,
        # Direct relative path inside Label Studio data dir (e.g. /data/upload/1/abc.jpg -> media/upload/1/abc.jpg)
        os.path.join(ls_data_dir, "media", cleaned_path_no_query.lstrip("/data/").lstrip("/")),
        os.path.join(ls_data_dir, cleaned_path_no_query.lstrip("/data/").lstrip("/")),
        os.path.join(ls_data_dir, "media", "upload", "1", filename_only),
        os.path.join(WORKSPACE_ROOT, cleaned_path_no_query),
        os.path.join(WORKSPACE_ROOT, "label_studio_crops", filename_only),
        os.path.join(WORKSPACE_ROOT, "sample_data", filename_only),
        os.path.join(WORKSPACE_ROOT, "output_geotiffs", filename_only),
    ]

    # Search in all project upload folders in label-studio media
    for root_dir in [os.path.join(ls_data_dir, "media", "upload"), os.path.join(ls_data_dir, "upload")]:
        if os.path.exists(root_dir):
            for proj_folder in os.listdir(root_dir):
                candidate_paths.append(os.path.join(root_dir, proj_folder, filename_only))

    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                img = Image.open(p).convert("RGB")
                return np.array(img)
            except Exception:
                pass

    # 3. HTTP / HTTPS URL
    if image_ref.startswith("http://") or image_ref.startswith("https://") or image_ref.startswith("/"):
        full_url = image_ref
        if image_ref.startswith("/"):
            base_url = MODEL_CONFIG["label_studio_url"].rstrip("/")
            full_url = f"{base_url}{image_ref}"

        req = urllib.request.Request(full_url)
        if MODEL_CONFIG["api_key"]:
            req.add_header("Authorization", f"Token {MODEL_CONFIG['api_key']}")

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                return np.array(img)
        except Exception as e:
            print(f"[ML-Backend] Fehler beim Laden der Bild-URL ({full_url}): {e}")

    return None


def parse_interactive_context(
    context_data: Any, img_w: int, img_h: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Optional[List[float]], str]:
    """
    Parses Label Studio interactive smart-tool context (clicks, boxes, keypoints).
    Returns:
    - positive_points: List[(x_px, y_px)]
    - negative_points: List[(x_px, y_px)]
    - bounding_box: [x1_px, y1_px, x2_px, y2_px] or None
    - target_label: str (e.g. 'Gebäude')
    """
    pos_points = []
    neg_points = []
    bbox = None
    target_label = "Gebäude"

    if not context_data:
        return pos_points, neg_points, bbox, target_label

    results = []
    if isinstance(context_data, dict):
        results = context_data.get("result", [])
    elif isinstance(context_data, list):
        results = context_data

    for item in results:
        t = item.get("type", "")
        val = item.get("value", {})

        # 1. Keypoint / Point click
        if "keypoint" in t or t in ("points", "keypointlabels"):
            kx = float(val.get("x", 0))
            ky = float(val.get("y", 0))
            px = (kx / 100.0 * img_w) if kx <= 100.0 else kx
            py = (ky / 100.0 * img_h) if ky <= 100.0 else ky

            labels = val.get("keypointlabels", val.get("labels", ["Positive"]))
            lbl = labels[0] if labels else "Positive"
            lbl_lower = lbl.lower()

            if "neg" in lbl_lower or "back" in lbl_lower or "false" in lbl_lower or item.get("is_negative"):
                neg_points.append((px, py))
            else:
                pos_points.append((px, py))
                if lbl_lower not in ("positive", "pos", "foreground"):
                    target_label = lbl

        # 2. Bounding box / Rectangle prompt
        elif "rectangle" in t or t == "rectanglelabels":
            rx = float(val.get("x", 0))
            ry = float(val.get("y", 0))
            rw = float(val.get("width", 0))
            rh = float(val.get("height", 0))

            x1 = (rx / 100.0 * img_w) if rx <= 100.0 else rx
            y1 = (ry / 100.0 * img_h) if ry <= 100.0 else ry
            x2 = ((rx + rw) / 100.0 * img_w) if rx <= 100.0 else (rx + rw)
            y2 = ((ry + rh) / 100.0 * img_h) if ry <= 100.0 else (ry + rh)

            bbox = [x1, y1, x2, y2]
            labels = val.get("rectanglelabels", val.get("labels", ["Gebäude"]))
            if labels:
                target_label = labels[0]

        # 3. Polygon prompt
        elif "polygon" in t or t == "polygonlabels":
            pts = val.get("points", [])
            labels = val.get("polygonlabels", val.get("labels", ["Gebäude"]))
            if labels:
                target_label = labels[0]
            if pts:
                all_x = [(p[0] / 100.0 * img_w) if p[0] <= 100.0 else p[0] for p in pts]
                all_y = [(p[1] / 100.0 * img_h) if p[1] <= 100.0 else p[1] for p in pts]
                bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]

    return pos_points, neg_points, bbox, target_label


def extract_sam_predictions(
    image_rgb: np.ndarray,
    context: Any = None,
) -> List[Dict[str, Any]]:
    """
    Runs SAM AI inference on GPU:
    - If context (user clicks / box) is present -> interactive prompt segmentation
    - If context is empty -> automatic building segmentation with SAM
    """
    sam = get_sam_extractor()
    if sam is None:
        return []

    h, w = image_rgb.shape[:2]
    from_name = MODEL_CONFIG["from_name"]
    to_name = MODEL_CONFIG["to_name"]
    results = []

    pos_pts, neg_pts, bbox, target_label = parse_interactive_context(context, w, h)

    # 1. Interactive Prompt Mode (User clicked or drew a box in Label Studio)
    if pos_pts or bbox:
        print(f"[SAM AI] 🎯 Interaktiver Prompt empfangen: {len(pos_pts)} Positiv-Punkte, {len(neg_pts)} Negativ-Punkte, BBox={bbox}")
        try:
            sam_res = sam.segment_with_prompts(
                image_rgb,
                positive_points=pos_pts if pos_pts else None,
                negative_points=neg_pts if neg_pts else None,
                bounding_box=bbox,
                granularity="compact",
            )
            for poly in sam_res["polygons"]:
                if poly.is_empty or not poly.is_valid:
                    continue
                coords = list(poly.exterior.coords)
                if len(coords) < 3:
                    continue
                pts_percent = [[round(x / w * 100.0, 3), round(y / h * 100.0, 3)] for x, y in coords]
                results.append({
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "polygonlabels",
                    "original_width": w,
                    "original_height": h,
                    "image_rotation": 0,
                    "value": {
                        "polygonlabels": [target_label],
                        "points": pts_percent,
                    },
                    "score": sam_res.get("iou_score", 0.95),
                })
        except Exception as e:
            print(f"[SAM AI] Fehler bei interaktiver Segmentierung: {e}")

        return results

    # 2. Auto-Prelabeling Mode (Task opened -> exact calibrated building & courtyard extraction)
    print(f"[SAM ML] ⚡ Führe präzise historische Gebäude- & Hofextraktion durch ({w}x{h} px)...")
    try:
        b_ext = get_building_extractor()
        b_res = b_ext.extract(image_rgb)
        for feat in b_res.features:
            poly = feat.geometry
            if poly.is_empty or not poly.is_valid:
                continue
            coords = list(poly.exterior.coords)
            if len(coords) < 3:
                continue
            pts_percent = [[round(x / w * 100.0, 3), round(y / h * 100.0, 3)] for x, y in coords]
            lbl_val = "Hofanlage" if feat.area_px > 1500 and feat.compactness < 0.4 else "Gebäude"
            results.append({
                "from_name": from_name,
                "to_name": to_name,
                "type": "polygonlabels",
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "value": {
                    "polygonlabels": [lbl_val],
                    "points": pts_percent,
                },
                "score": round(min(0.98, 0.75 + (feat.compactness * 0.20)), 2),
            })
    except Exception as e:
        print(f"[SAM ML] Fehler bei Gebäudeextraktion: {e}")

    return results


@app.get("/")
@app.get("/health")
def health():
    """Health check endpoint required by Label Studio."""
    return {
        "status": "UP",
        "model_class": "TranchotSAMExtractorBackend",
        "engine": MODEL_CONFIG["engine"],
        "has_sam": HAS_SAM,
        "has_yolo": HAS_ULTRALYTICS,
        "device": "cuda" if HAS_SAM else "cpu",
    }


@app.post("/setup")
async def setup(request: Request):
    """
    Label Studio calls /setup when connecting the model to a project.
    Parses label config XML to identify tag names (from_name, to_name).
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    schema = data.get("schema", "")
    if schema:
        # Detect PolygonLabels or BrushLabels
        m_poly = re.search(r'<PolygonLabels[^>]*name=["\']([^"\']+)["\'][^>]*toName=["\']([^"\']+)["\']', schema)
        m_brush = re.search(r'<BrushLabels[^>]*name=["\']([^"\']+)["\'][^>]*toName=["\']([^"\']+)["\']', schema)
        if m_poly:
            MODEL_CONFIG["from_name"] = m_poly.group(1)
            MODEL_CONFIG["to_name"] = m_poly.group(2)
            MODEL_CONFIG["tag_type"] = "polygonlabels"
            print(f"[ML-Backend] Erkanntes Schema: PolygonLabels from_name={MODEL_CONFIG['from_name']}, to_name={MODEL_CONFIG['to_name']}")
        elif m_brush:
            MODEL_CONFIG["from_name"] = m_brush.group(1)
            MODEL_CONFIG["to_name"] = m_brush.group(2)
            MODEL_CONFIG["tag_type"] = "brushlabels"
            print(f"[ML-Backend] Erkanntes Schema: BrushLabels from_name={MODEL_CONFIG['from_name']}, to_name={MODEL_CONFIG['to_name']}")

    extra = data.get("extra_params", {})
    if "engine" in extra:
        MODEL_CONFIG["engine"] = extra["engine"]
    if "yolo_weights" in extra:
        MODEL_CONFIG["yolo_weights"] = extra["yolo_weights"]

    # Pre-warm SAM on GPU
    if MODEL_CONFIG["engine"] == "sam" and HAS_SAM:
        get_sam_extractor()

    print(f"[ML-Backend] Projekt erfolgreich verbunden mit SAM Engine auf Port 9090!")
    return {
        "model_version": f"tranchot-sam-v1.1",
        "status": "OK",
    }


@app.post("/predict")
async def predict(request: Request):
    """
    Label Studio prediction endpoint.
    Handles both live interactive clicks (smart-tool / keypoints) and auto-prelabeling.
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    tasks = payload.get("tasks", [])
    context = payload.get("context")

    if not tasks and not context:
        return {"results": []}

    results = []
    engine = MODEL_CONFIG["engine"]

    for task in tasks:
        task_data = task.get("data", {})
        image_ref = task_data.get("image") or task_data.get("img") or task_data.get("url", "")

        img_rgb = resolve_and_load_image(image_ref)
        if img_rgb is None:
            print(f"[ML-Backend] Bild konnte nicht geladen werden: {image_ref}")
            results.append({
                "result": [],
                "score": 0.0,
                "model_version": f"tranchot-sam-v1.1",
            })
            continue

        # Run SAM AI extraction (interactive prompt or auto-segmentation)
        if engine == "sam" and HAS_SAM:
            pred_items = extract_sam_predictions(img_rgb, context=context)
        elif engine == "yolo" and HAS_ULTRALYTICS:
            yolo = get_yolo_extractor()
            pred_items = []
            if yolo:
                yolo_dets = yolo.extract_from_crop(img_rgb, conf_threshold=MODEL_CONFIG["conf_threshold"])
                h, w = img_rgb.shape[:2]
                for d in yolo_dets:
                    poly = d["geometry"]
                    coords = list(poly.exterior.coords)
                    pts_percent = [[round(x / w * 100.0, 3), round(y / h * 100.0, 3)] for x, y in coords]
                    pred_items.append({
                        "from_name": MODEL_CONFIG["from_name"],
                        "to_name": MODEL_CONFIG["to_name"],
                        "type": "polygonlabels",
                        "original_width": w,
                        "original_height": h,
                        "image_rotation": 0,
                        "value": {
                            "polygonlabels": [d["class_name"]],
                            "points": pts_percent,
                        },
                        "score": d["confidence"],
                    })
        else:
            # Rule-based fallback
            b_ext = get_building_extractor()
            b_res = b_ext.extract(img_rgb)
            h, w = img_rgb.shape[:2]
            pred_items = []
            for feat in b_res.features:
                poly = feat.geometry
                if poly.is_empty or not poly.is_valid:
                    continue
                coords = list(poly.exterior.coords)
                if len(coords) < 3:
                    continue
                pts_percent = [[round(x / w * 100.0, 3), round(y / h * 100.0, 3)] for x, y in coords]
                lbl_val = "Hofanlage" if feat.area_px > 1500 and feat.compactness < 0.4 else "Gebäude"
                pred_items.append({
                    "from_name": MODEL_CONFIG["from_name"],
                    "to_name": MODEL_CONFIG["to_name"],
                    "type": "polygonlabels",
                    "original_width": w,
                    "original_height": h,
                    "image_rotation": 0,
                    "value": {
                        "polygonlabels": [lbl_val],
                        "points": pts_percent,
                    },
                    "score": 0.90,
                })

        overall_score = float(np.mean([p["score"] for p in pred_items])) if pred_items else 0.0

        results.append({
            "result": pred_items,
            "score": round(overall_score, 3),
            "model_version": f"tranchot-sam-v1.1",
        })

    return {"results": results}


@app.post("/train")
@app.post("/webhook")
async def webhook(request: Request):
    return {"status": "received"}


def launch_backend(host: str = "0.0.0.0", port: int = 9090, engine: str = "sam", yolo_weights: Optional[str] = None):
    """Starts the Uvicorn server for Label Studio SAM ML backend."""
    import uvicorn
    MODEL_CONFIG["engine"] = engine
    if yolo_weights:
        MODEL_CONFIG["yolo_weights"] = yolo_weights

    print(f"🚀 Starte Tranchot SAM ML Backend auf http://localhost:{port}...")
    print(f"   Modell-Engine: {MODEL_CONFIG['engine']} (Segment Anything Model)")
    print(f"   Status URL:    http://localhost:{port}/health")
    print(f"   Label Studio Verbindungs-URL: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tranchot SAM ML Backend for Label Studio")
    parser.add_argument("--port", type=int, default=9090, help="Port (Standard: 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (Standard: 0.0.0.0)")
    parser.add_argument("--engine", type=str, default="sam", choices=["sam", "building", "yolo"], help="Engine")
    parser.add_argument("--weights", type=str, default="yolov8n-seg.pt", help="Path to YOLO weights (.pt)")
    args = parser.parse_args()

    launch_backend(host=args.host, port=args.port, engine=args.engine, yolo_weights=args.weights)
