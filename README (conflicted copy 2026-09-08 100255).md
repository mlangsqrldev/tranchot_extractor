# 🗺️ Tranchot Extractor: AI-Powered Historical Cartography Vectorization Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Geospatial](https://img.shields.io/badge/GIS-GeoPandas%20%7C%20Rasterio%20%7C%20Shapely-orange.svg)](https://geopandas.org/)
[![AI-Backend](https://img.shields.io/badge/AI-Meta%20SAM%20%7C%20EasyOCR-purple.svg)](https://github.com/facebookresearch/segment-anything)

A high-performance, GIS-native feature extraction and vectorization system designed specifically for the **Tranchot and von Müffling Historical Map of the Rhineland (1803–1820)** (*Kartenaufnahme der Rheinlande durch Tranchot und von Müffling, 1:20.000 / 1:25.000*).

Developed by the **Bonn Center for Digital Humanities (BCDH)** at the University of Bonn.

---

## 📸 Key Capabilities

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              TRANCHOT MAP INPUT (GeoTIFF)               │
                  └────────────────────────────┬────────────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
     🏛️ BUILDINGS & HOFS            🌲 LAND-USE & WATER             📜 TOPONYMS & ROADS
  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
  │ • Orthogonal 90° Footpr.│     │ • Competitive Argmin    │     │ • Bilingual OCR (FR/DE) │
  │ • 0-Spike Regularization│     │ • Texture-Aware Var.    │     │ • TGN / GeoNames / GND  │
  │ • Vierkanthof Hole Subt.│     │ • Zero Polygon Overlap  │     │ • Road Skeletons (skan) │
  │ • Black Stone & Churches│     │ • Pipette Multi-Point   │     │ • Graph Cleanup (NX)    │
  └────────────┬────────────┘     └────────────┬────────────┘     └────────────┬────────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               │
                                               ▼
                              🗺️ GIS EXPORT (EPSG:25832)
                       GeoPackage (.gpkg) │ GeoJSON │ Shapefile
```

---

## ✨ Features at a Glance

### 1. 🏛️ Precision Building & Courtyard Extraction (`BuildingExtractor`)
- **Zero-Spike Orthogonal Regularization**: Solves the classic raster contour problem (hairpin spikes and $45^\circ$ bevels) by fitting exact $90^\circ$ architectural Manhattan footprints.
- **Historical Farmsteads & Courtyards**: Preserves L-shapes (*Hakenhöfe*), U-shapes (*Dreiseithöfe*), and enclosed *Vierkanthöfe* with oriented courtyard yard subtraction.
- **Thin Walls & Enclosures**: Captures thin historical garden walls, courtyard enclosures, and narrow barn wings down to $1.0\text{ px}$ stroke width.
- **Black Stone Buildings & Churches**: Dedicated segmentation for public massiv buildings, castles, and churches drawn in black ink.
- **Street Row Separation**: Distance-transform core detection cleanly separates adjacent attached houses along village streets.

### 2. 🌲 Competitive Texture-Aware Land-Use Engine (`LandUseExtractor` / `PipetteSampler`)
- **Competitive Multi-Class Classifier**: All active land-use categories (*Wald, Wiese, Gewässer, Weinberge, Gärten, Kies, Gebäude*) compete per pixel via weighted Mahalanobis/$\Delta E$ distance. **Zero polygon overlap and zero conflicting boundaries.**
- **Texture-Aware Discrimination**: Integrates local variance and gradient filters:
  - **Forest (*Wald*)**: Olive-green wash **+** high texture variance (engraved tree crown signatures).
  - **Meadow (*Wiese*)**: Cyan-pastel wash **+** smooth, homogeneous watercolor.
  - **Vineyard (*Weinberg*)**: Yellow/ochre tint **+** periodic horizontal hillside hatching.
  - **Water (*Gewässer*)**: Rhine River, mill streams, and ponds with wave smoothness.
- **Interactive Multi-Point Pipette**: Sample multiple points (light, dark, shadow) per class; save/load sheet profiles via `.palette.json`.
- **ROI Test Box**: Drag a green test box on the canvas to evaluate land-use classification in $<0.1\text{s}$.

### 3. 📜 Historical Toponym OCR & Linked Open Data Gazetteers (`ToponymExtractor`)
- **Bilingual Antique OCR**: Optimized for French and German historical geographic toponyms (Fraktur, Antiqua, cursive script).
- **LOD Gazetteer Alignment**: Spatial and phonetic matching against **Getty Thesaurus of Geographic Names (TGN)**, **GeoNames**, and **Integrated Authority File (GND)**.

### 4. 🛣️ Road Network Vectorization (`RoadExtractor`)
- **Dual-Casing Centerline Extraction**: Detects double-cased historical highways (*Chaussées*), country roads (*Landstraßen*), and sunken paths (*Hohlwege*).
- **Topology Analysis**: Medial-axis thinning via `skan` and graph cleanup via `networkx` to generate connected linear GIS networks.

### 5. 🤖 Meta SAM GPU Zero-Shot Segmentation (`SAMExtractor`)
- **Meta Segment Anything (ViT-B)** with full CUDA acceleration.
- Multi-point positive and negative prompt support for complex historical landmarks.
- Integrated LabelStudio ML Backend (`label_studio_ml_backend.py`) for collaborative annotation workflows.

---

## 📦 Installation & Setup

### Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python**: `3.10` or `3.11` recommended
- **GPU (Optional)**: NVIDIA GPU with CUDA for SAM AI acceleration (CPU fallback included)

### 1. Clone the Repository
```bash
git clone https://github.com/mlangsqrldev/tranchot_extractor.git
cd tranchot_extractor
```

### 2. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Usage

### 🖥️ Desktop GUI (Recommended)
Start the modern CustomTkinter dark-mode desktop application:
```bash
python gui.py
```
*Or via installed entry point:*
```bash
tranchot-gui
```

#### GUI Workflow:
1. **Load Map**: Drag & drop or open a historical GeoTIFF (e.g. `Sample/Nickenich.tif`).
2. **Color Restoration**: Adjust the *Entgilbung (Deyellow)* and *Leuchtkraft (Vibrance)* sliders in the left sidebar for instant white balance restoration.
3. **Building ROI Test**: Click `📐 Gebäude ROI-Box aufziehen` and drag a red dashed box over a village to test building extraction in $<0.05\text{s}$.
4. **Land-Use ROI Test**: Click `📐 Flächen ROI-Box aufziehen` and drag a green dashed box over fields to test competitive land-use classification.
5. **Pipette Calibration**: Click on any map feature to calibrate custom color classes and save the profile to `.palette.json`.
6. **Export**: Click `💾 Export Layer (GPKG / Shapefile / GeoJSON)` to generate GIS vector datasets.

---

### 💻 Command Line Interface (CLI)

Run full-sheet batch processing from terminal:

```bash
# Extract all buildings and courtyards to GeoPackage
tranchot-extract buildings --input Sample/Nickenich.tif --output output/buildings.gpkg --rgb-diff 18

# Extract full land-use (Forest, Meadow, Water)
tranchot-extract landuse --input Sample/Nickenich.tif --output output/landuse.gpkg

# Extract road network centerlines
tranchot-extract roads --input Sample/Nickenich.tif --output output/roads.geojson

# Extract historical toponyms with TGN/GeoNames gazetteer lookup
tranchot-extract toponyms --input Sample/Nickenich.tif --output output/toponyms.gpkg --align-gazetteer
```

---

## 📁 Repository Structure

```
tranchot_extractor/
├── gui.py                              # Desktop Application Entry Point
├── setup.py                            # Package Setup & Dependencies
├── requirements.txt                    # Pip Dependencies
├── .gitignore                          # Clean Git Ignore Rules
├── README.md                           # Main Project Documentation
├── WIKI.md                             # Comprehensive Developer & Architecture Wiki
│
├── tests/                              # Automated Unit & Regression Test Suite
│   ├── test_building_extractor.py      # Building extraction, despiking & thin wall tests
│   ├── test_landuse_extractor.py       # Competitive land-use segmentation tests
│   ├── test_road_extractor.py          # Road skeleton & topology tests
│   ├── test_text_extractor.py          # Toponym OCR tests
│   ├── test_color_enhancer.py          # White balancing & color restoration tests
│   └── test_desktop_app.py             # GUI headless integration tests
│
└── tranchot_extractor/                 # Main Python Package
    ├── __init__.py
    ├── config.py                       # Dataclass configs for all extractors
    ├── cli.py                          # Command Line Interface (Argparse)
    │
    ├── core/                           # Core Image Processing & Geometry
    │   ├── color_segmenter.py          # Multi-space color thresholding
    │   ├── skeletonizer.py             # Morphological thinning & medial axis
    │   └── vectorizer.py               # Polygon & Linestring vectorization
    │
    ├── extractors/                     # Domain-Specific Feature Extractors
    │   ├── building_extractor.py       # Orthogonal building & courtyard extractor
    │   ├── landuse_extractor.py        # Competitive texture-aware land-use extractor
    │   ├── pipette_sampler.py          # Interactive multi-point palette engine
    │   ├── road_extractor.py           # Road centerline & graph extractor
    │   ├── text_extractor.py           # Historical toponym OCR extractor
    │   └── sam_extractor.py            # Meta SAM AI GPU zero-shot extractor
    │
    ├── geo/                            # Geospatial & GIS Data Management
    │   ├── exporter.py                 # GeoPackage, GeoJSON, Shapefile & GeoTIFF export
    │   ├── georeferencer.py            # CRS transformations & pixel-to-world mapping
    │   └── spatial_gazetteer.py        # LOD gazetteer connector (TGN, GeoNames, GND)
    │
    ├── preprocessing/                  # Historical Map Preprocessing
    │   └── color_enhancer.py           # Parchment normalization, de-yellowing & vibrance
    │
    ├── studio/                         # LabelStudio Annotation Backend
    │   └── label_studio_ml_backend.py  # SAM ML-Backend for LabelStudio port 9090
    │
    └── ui/                             # Desktop User Interface
        └── desktop_app.py              # CustomTkinter Dark Mode GUI
```

---

## 🧪 Testing

Run the automated regression test suite:
```bash
python -m unittest discover -s tests
```
All 25 unit tests run in $<10\text{s}$ and cover all modules.

---

## 🤝 Handover & Next Steps for Collaborators

For technical architecture details, algorithm walkthroughs, and developer guidelines, see [WIKI.md](WIKI.md).

### Immediate Roadmap:
1. **Multi-Sheet Batch Processing**: Extend CLI to process full Tranchot map series (180+ map sheets).
2. **Fine-Tuned Historical OCR**: Integrate a custom Tranchot Fraktur/Antiqua model trained on curated crop datasets.
3. **Graph-Cut Superpixel Snapping**: Further refine parcel boundaries along natural hedgerows and sunken lanes.

---

## 📄 Citation & License

This project is licensed under the [MIT License](LICENSE).

If you use this software in your research, please cite:
```bibtex
@software{tranchot_extractor_2026,
  author = {Bonn Center for Digital Humanities (BCDH)},
  title = {Tranchot Extractor: AI-Powered Historical Cartography Vectorization Engine},
  year = {2026},
  url = {https://github.com/mlangsqrldev/tranchot_extractor}
}
```
