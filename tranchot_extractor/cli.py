"""
Command-Line Interface (CLI) for Tranchot Feature Extractor.
Supports single-file processing, batch directory scanning, and UI launch.
"""

import sys
import os
import argparse
import time
import glob
from pathlib import Path
from typing import Optional, Set

# Ensure UTF-8 for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
from tranchot_extractor.config import PipelineConfig, BuildingConfig, RoadConfig, TextConfig, GeoConfig
from tranchot_extractor.core.pipeline import TranchotPipeline
from tranchot_extractor.ui.app import launch
from tranchot_extractor.core.tiled_processor import TiledMapProcessor

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="🏛️ Tranchot Map AI Feature Extractor - Gebäude, Straßen & Beschriftungen"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Pfad zum Eingabe-Kartenblatt (PNG, JPG, TIFF) oder Verzeichnis für Batch-Verarbeitung"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="extracted_output",
        help="Ausgabeverzeichnis für GIS-Layer (GeoJSON, GeoPackage, Shapefile, Masken)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Zu extrahierende Layer: 'all' oder kommagetrennt z. B. 'buildings,roads,text'"
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:25832",
        help="Ziel-Koordinatenreferenzsystem (Standard: EPSG:25832 ETRS89 / UTM 32N)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Kachelgröße für hochauflösende GeoTIFFs (Standard: 2048)"
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximale Anzahl an Kacheln pro Blatt (optional, z. B. für Schnelltests)"
    )
    parser.add_argument(
        "--gui", "--desktop",
        action="store_true",
        help="Startet die native CustomTkinter Desktop-Benutzeroberfläche"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Startet die interaktive Benutzeroberfläche"
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Startet das WebGIS Tranchot Label Studio (Leaflet + SAM AI auf Port 8000)"
    )
    parser.add_argument(
        "--labelme",
        action="store_true",
        help="Öffnet das native LabelMe Desktop-Tool auf dem Bild"
    )
    parser.add_argument(
        "--train-yolo",
        type=str,
        help="Trainiert ein YOLOv8-Segmentierungsmodell anhand einer Label Studio JSON Datei oder eines Dataset-Ordners"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Anzahl der Trainings-Epochen für YOLOv8 (Standard: 50)"
    )
    parser.add_argument(
        "--yolo",
        type=str,
        help="Führt die Extraktion mit einem trainierten YOLOv8-Seg Modell (.pt) auf dem Eingabebild/GeoTIFF durch"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Konfidenzschwelle für YOLO-Extraktion (Standard: 0.20)"
    )
    parser.add_argument(
        "--build-hybrid-dataset",
        type=str,
        nargs="?",
        const="output_geotiffs",
        help="Erstellt automatisch einen MapReader-Hybrid-Trainingsdatensatz aus den GeoTIFFs (Farbtrennung + Watershed)"
    )
    parser.add_argument(
        "--train-hybrid",
        action="store_true",
        help="Trainiert ein domänenspezifisches YOLOv8-Seg Modell auf dem generierten Hybrid-Datensatz"
    )
    parser.add_argument(
        "--ml-backend",
        action="store_true",
        help="Startet das Label Studio ML Backend für automatische KI-Vorannotationen (Port 9090)"
    )
    parser.add_argument(
        "--ml-engine",
        type=str,
        default="building",
        choices=["building", "sam", "yolo", "combined"],
        help="Engine für das ML Backend: 'building' (Farbe/Heuristik), 'sam' (Segment Anything), 'yolo' (YOLOv8) oder 'combined'"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port für die Web-Benutzeroberfläche (Standard: 8000 für Studio, 7860 für Gradio, 9090 für ML Backend)"
    )
    return parser.parse_args()


def process_single(
    pipeline: TranchotPipeline,
    img_path: str,
    out_dir: str,
    layers_set: set,
    tile_size: int = 2048,
    max_tiles: Optional[int] = None
):
    print(f"\n========================================================")
    print(f"🔄 Verarbeite: {img_path}")
    start_t = time.time()

    ext_bldg = "all" in layers_set or "buildings" in layers_set or "gebaeude" in layers_set
    ext_bound = "all" in layers_set or "boundaries" in layers_set or "grenzen" in layers_set
    ext_roads = "all" in layers_set or "roads" in layers_set or "strassen" in layers_set
    ext_text = "all" in layers_set or "text" in layers_set or "ocr" in layers_set

    # Check if large GeoTIFF
    is_large_geotiff = False
    if HAS_RASTERIO and img_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(img_path) as src:
                if src.width > 3000 or src.height > 3000:
                    is_large_geotiff = True
        except Exception:
            pass

    if is_large_geotiff:
        print(f"📦 Großformatige GeoTIFF erkannt -> Kachelung aktiv (Tile Size: {tile_size}px)...")
        processor = TiledMapProcessor(pipeline.config)
        res = processor.process_geotiff(
            img_path,
            output_dir=out_dir,
            tile_size=tile_size,
            max_tiles=max_tiles,
            extract_buildings=ext_bldg,
            extract_roads=ext_roads,
            extract_text=ext_text,
        )
        elapsed = time.time() - start_t
        print(f"✨ Fertig in {elapsed:.2f}s:")
        print(f"   🏛️ Gebäude:       {res['building_count']} Polygone")
        print(f"   🚩 Grenzlinien:   {res.get('boundary_count', 0)} Segmente")
        print(f"   🛣️ Straßen/Wege:  {res['road_count']} Segmente")
        print(f"   🏷️ Beschriftung:  {res['toponym_count']} Toponyme")
        print(f"   💾 Gespeichert in: {out_dir}")
    else:
        res = pipeline.process(
            img_path,
            output_dir=out_dir,
            extract_buildings=ext_bldg,
            extract_boundaries=ext_bound,
            extract_roads=ext_roads,
            extract_text=ext_text,
        )
        elapsed = time.time() - start_t
        print(f"✨ Fertig in {elapsed:.2f}s:")
        print(f"   🏛️ Gebäude:       {len(res.buildings.features)} Polygone")
        print(f"   🚩 Grenzlinien:   {len(res.boundaries.features)} Segmente")
        print(f"   🛣️ Straßen/Wege:  {len(res.roads.features)} Segmente")
        print(f"   🏷️ Beschriftung:  {len(res.text.features)} Toponyme")
        print(f"   💾 Gespeichert in: {out_dir}")


def main():
    args = parse_args()

    if args.ml_backend:
        port = args.port or 9090
        from tranchot_extractor.models.label_studio_backend import launch_backend
        launch_backend(port=port, engine=args.ml_engine)
        return

    if args.studio:
        port = args.port or 8000
        print(f"🚀 Starte Tranchot Label Studio auf http://localhost:{port}...")
        from tranchot_extractor.studio import launch_studio
        launch_studio(port=port)
        return

    if args.gui or args.desktop:
        from tranchot_extractor.ui.desktop_app import main as launch_desktop
        print("🏛️ Starte Tranchot Extractor CustomTkinter Desktop-Applikation...")
        launch_desktop()
        return

    if args.ui:
        from tranchot_extractor.ui.desktop_app import main as launch_desktop
        print("🏛️ Starte Tranchot Extractor Desktop-Applikation...")
        launch_desktop()
        return

    if args.labelme:
        from tranchot_extractor.geo.labelme_bridge import launch_labelme_on_crop
        img_target = args.input or "sample_data/rommerskirchen_sample.png"
        launch_labelme_on_crop(img_target)
        return

    if args.build_hybrid_dataset:
        from tranchot_extractor.models.mapreader_grid import MapReaderGridDatasetBuilder
        target_dir = args.build_hybrid_dataset
        builder = MapReaderGridDatasetBuilder()
        yaml_path = builder.process_all_geotiffs(target_dir, patches_per_sheet=12)
        print(f"✅ Hybrid-Datensatz fertig generiert: {yaml_path}")
        return

    if args.train_hybrid:
        from tranchot_extractor.models.yolo_trainer import YOLOSegTrainer
        yaml_path = "datasets/tranchot_hybrid_yolo/tranchot_hybrid.yaml"
        if not os.path.exists(yaml_path):
            print("⚠️ Datensatz nicht gefunden. Bitte zuerst '--build-hybrid-dataset' ausführen.")
            return
        trainer = YOLOSegTrainer(base_model="yolov8s-seg.pt")
        res = trainer.train(
            dataset_yaml_path=yaml_path,
            epochs=args.epochs,
            project_name="tranchot_hybrid_runs",
            run_name="tranchot_deep_seg",
        )
        print(f"🎉 Hybrid YOLOv8-Seg Training erfolgreich abgeschlossen!")
        print(f"   Beste Modellgewichte: {res['best_weights']}")
        return

    if args.train_yolo:
        from tranchot_extractor.models import YOLODatasetBuilder, YOLOSegTrainer
        target = args.train_yolo
        yaml_path = target
        if target.endswith(".json"):
            print(f"📦 Konvertiere Label Studio Export ({target}) in YOLOv8-Format...")
            builder = YOLODatasetBuilder()
            builder.convert_label_studio_export(target, images_dir="sample_data")
            yaml_path = builder.generate_yaml()

        trainer = YOLOSegTrainer()
        res = trainer.train(yaml_path, epochs=args.epochs)
        print(f"✅ Training abgeschlossen! Beste Gewichte: {res['best_weights']}")
        return

    if args.yolo:
        from tranchot_extractor.models import YOLOSegExtractor
        if not args.input:
            print("⚠️ Bitte ein Eingabebild oder GeoTIFF mit '--input <Pfad>' angeben.")
            return

        extractor = YOLOSegExtractor(model_path=args.yolo)
        if args.input.lower().endswith(('.tif', '.tiff')):
            res = extractor.process_large_geotiff(args.input, output_dir=args.output, tile_size=args.tile_size, conf_threshold=args.conf)
            print(f"✨ YOLO GeoTIFF Extraktion fertig: {res['feature_count']} Features extrahiert.")
        else:
            img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
            dets = extractor.extract_from_crop(img, conf_threshold=args.conf)
            print(f"✨ YOLO Extraktion fertig: {len(dets)} Features erkannt.")
        return

    if not args.input:
        print("⚠️ Kein Eingabepfad angegeben. Nutze '--input <Pfad>' oder '--ui' für die Weboberfläche.")
        print("Beispiel: python -m tranchot_extractor.cli --input sample_data/rommerskirchen_sample.png -o output/")
        return

    input_path = Path(args.input)
    layers_set = set(layer.strip().lower() for layer in args.layers.split(","))

    cfg = PipelineConfig()
    cfg.geo.default_crs = args.crs
    pipeline = TranchotPipeline(cfg)

    if input_path.is_file():
        process_single(
            pipeline, str(input_path), args.output, layers_set,
            tile_size=args.tile_size, max_tiles=args.max_tiles
        )
    elif input_path.is_dir():
        image_exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        files = []
        for ext in image_exts:
            files.extend(glob.glob(str(input_path / ext)))
            files.extend(glob.glob(str(input_path / ext.upper())))

        print(f"📂 Batch-Modus: {len(files)} Bilddateien gefunden in {input_path}")
        for idx, f in enumerate(files, 1):
            stem = Path(f).stem
            sub_out = os.path.join(args.output, stem)
            print(f"\n[{idx}/{len(files)}] ----------------------------------------")
            process_single(
                pipeline, f, sub_out, layers_set,
                tile_size=args.tile_size, max_tiles=args.max_tiles
            )
    else:
        print(f"❌ Fehler: Pfad existiert nicht: {args.input}")


if __name__ == "__main__":
    main()
