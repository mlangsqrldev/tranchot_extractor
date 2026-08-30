# 📚 Tranchot Extractor Developer Wiki & Technical Architecture Manual

Welcome to the internal technical documentation of the **Tranchot Extractor** project. This manual is written for developers, digital humanities researchers, and GIS engineers continuing work on this codebase.

---

## 📑 Table of Contents
1. [Historical Map Context: The Tranchot & von Müffling Series](#1-historical-map-context)
2. [Software Architecture & Core Pipeline](#2-software-architecture--core-pipeline)
3. [Building & Courtyard Extraction Algorithm (`BuildingExtractor`)](#3-building--courtyard-extraction-algorithm)
4. [Competitive Texture-Aware Land-Use Engine (`LandUseExtractor` & `PipetteSampler`)](#4-competitive-texture-aware-land-use-engine)
5. [Road Network & Linear Topology Vectorization (`RoadExtractor`)](#5-road-network--linear-topology-vectorization)
6. [Toponym OCR & LOD Gazetteer Alignment (`ToponymExtractor`)](#6-toponym-ocr--lod-gazetteer-alignment)
7. [Meta SAM GPU AI Backend & LabelStudio Integration](#7-meta-sam-gpu-ai-backend--labelstudio-integration)
8. [Configuration & Parameter Reference (`config.py`)](#8-configuration--parameter-reference)
9. [Developer Guide: How to Extend & Calibrate](#9-developer-guide-how-to-extend--calibrate)

---

## 1. Historical Map Context

The **Kartenaufnahme der Rheinlande durch Tranchot und von Müffling (1803–1820)** was surveyed under French military leadership by Colonel Jean Joseph Tranchot and continued by the Prussian Major General Karl von Müffling.
- **Original Scale**: 1:20.000 (French survey) / 1:25.000 (Prussian continuation).
- **Modern Projection**: ETRS89 / UTM Zone 32N (`EPSG:25832`).
- **Cartographic Layer Conventions**:
  - **Carmine Red (*Karminrot*)**: Standard residential houses, agricultural barn wings, thin courtyard garden walls.
  - **Black Tusche (*Schwarz*)**: Public stone buildings, churches (*Kirchen*), chapels, monasteries, castles, and town gates.
  - **Olive Green (*Olivgrün*)**: Deciduous and coniferous forests with engraved tree crown signatures.
  - **Cyan/Pastel Green (*Pastellgrün*)**: River meadows (*Auen*), wet pastures, and cleared gardens.
  - **Ochre/Yellow with Hatching (*Ocker*)**: Hillside vineyards (*Rebhänge / Weinberge*) with parallel terrace hatching.
  - **Blue/Cyan (*Preußischblau*)**: Rhine River, tributaries, mill ponds, and drainage ditches.

---

## 2. Software Architecture & Core Pipeline

The codebase follows a modular, GIS-native separation of concerns:

```
tranchot_extractor/
├── tranchot_extractor/
│   ├── config.py              <-- Central dataclasses for all module parameters
│   ├── cli.py                 <-- CLI commands and batch runner
│   │
│   ├── core/                  <-- Reusable mathematical primitives
│   │   ├── color_segmenter.py <-- Multi-space color masks (HSV, LAB, RGB-diff)
│   │   ├── skeletonizer.py    <-- Morphological medial axis & graph thinning
│   │   └── vectorizer.py      <-- Shapely Polygon & Linestring builders
│   │
│   ├── extractors/            <-- Feature-specific extraction engines
│   │   ├── building_extractor.py <-- 90° Orthogonal building/courtyard engine
│   │   ├── landuse_extractor.py  <-- Competitive texture-aware land-use engine
│   │   ├── pipette_sampler.py    <-- Interactive multi-point palette classifier
│   │   ├── road_extractor.py     <-- Dual-line casing & road network extractor
│   │   ├── text_extractor.py     <-- Toponym OCR & LOD gazetteer aligner
│   │   └── sam_extractor.py      <-- Meta SAM ViT-B GPU segmentation engine
│   │
│   ├── geo/                   <-- GIS export & spatial transforms
│   │   ├── exporter.py        <-- GeoPackage, GeoJSON, Shapefile & GeoTIFF writer
│   │   ├── georeferencer.py   <-- Affine transform & CRS reprojection
│   │   └── spatial_gazetteer.py <-- TGN, GeoNames & GND SPARQL/REST connectors
│   │
│   ├── preprocessing/         <-- Color restoration & radiometric balance
│   │   └── color_enhancer.py  <-- Parchment normalization & de-yellowing
│   │
│   ├── studio/                <-- LabelStudio ML Backend
│   │   └── label_studio_ml_backend.py <-- SAM REST service on port 9090
│   │
│   └── ui/                    <-- Modern GUI
│       └── desktop_app.py     <-- CustomTkinter dark-mode map workbench
```

---

## 3. Building & Courtyard Extraction Algorithm

### A. The Challenge of Historical Building Extraction
1. **Raster Staircase & Spikes**: Douglas-Peucker polygonization on 1–2 px strokes creates acute triangular spikes ($2^\circ–15^\circ$) at line ends and corners.
2. **Village Street Merging**: Attached houses touching at single pixels form giant merged blobs.
3. **Garden & Orchard Interference**: Diffuse pink watercolor washes over village gardens create false giant bounding boxes if global bounding boxes are naively fitted.

### B. The 4-Stage Solution in `BuildingExtractor`:
1. **Multi-Space Color Segmentation**:
   $$\text{Carmine} = (H \in [0, 18] \cup [162, 180]) \cap (S \ge 26) \cap (a^* \ge 131) \cap (R - G \ge 18) \cap (R - B \ge 18)$$
   $$\text{Black Stone} = (R < 75) \cap (G < 75) \cap (B < 75) \cap (\text{Gray} < 75)$$
2. **Distance-Transform Peak Separation**:
   For each connected component, $D(x,y) = \text{distanceTransform}(\text{mask})$ is calculated. Local maxima $D(x,y) \ge 0.5 \cdot \max(D)$ identify distinct building cores, cleanly splitting attached houses along village streets.
3. **Solidity-Based Village Wash Filter**:
   - If $\text{Area} > 350\text{ px}^2$ and $\text{Solidity} < 0.55$ without courtyard holes: Classified as a diffuse village wash/garden. The global bounding box is **discarded**, and only dense building cores ($\le 450\text{ px}^2$) inside are extracted.
4. **Orthogonal $90^\circ$ Regularization (Manhattan Footprint)**:
   - Freestanding houses, barn wings, and thin walls are fitted with oriented bounding boxes (`minAreaRect`), guaranteeing **exakt 4 vertices, 4 right angles ($90^\circ$), and 0 spikes**.
   - Large courtyards with interior yard holes subtract angle-aligned hole bounding boxes (`outer_box.difference(h)`).
   - Iterative despiking collapses any remaining acute vertices with inner angle $< 45^\circ$.

---

## 4. Competitive Texture-Aware Land-Use Engine

### A. Competitive $argmin$ Multi-Class Segmentation
Unlike traditional independent thresholding (which creates polygon overlaps and gaps), the `PipetteSampler` and `LandUseExtractor` implement a **competitive distance classification**:

For every pixel $(x, y)$ and each active class $c \in \{\text{Wald}, \text{Wiese}, \text{Gewässer}, \text{Weinberg}, \text{Garten}, \text{Pergament}\}$:

$$d_{\text{color}}(x, y, c) = \sqrt{0.45 \cdot (L - L_c)^2 + 1.8 \cdot (a^* - a^*_c)^2 + 1.8 \cdot (b^* - b^*_c)^2}$$

$$d_{\text{total}}(x, y, c) = d_{\text{color}}(x, y, c) - w_{\text{tex}}(c) \cdot \text{TextureScore}(x, y)$$

$$\text{Class}(x, y) = \arg\min_{c} d_{\text{total}}(x, y, c)$$

### B. Texture Feature Computation:
- $\text{Var}_{\text{local}}(x, y) = \text{GaussianBlur}\left((I - \text{GaussianBlur}(I, \sigma))^2, \sigma\right)$
- **Forest**: $w_{\text{tex}} = +1.2$ (prefers high texture / tree stamps).
- **Meadow**: $w_{\text{tex}} = -0.8$ (prefers smooth pastel watercolor).
- **Water**: $w_{\text{tex}} = -1.0$ (prefers smooth blue).

### C. Multi-Point Pipette Sampling:
When the user samples multiple points for a class, the sample vector $\mathbf{x}_i \in \mathbb{R}^3$ updates the running mean $\boldsymbol{\mu}_c$ and covariance matrix $\boldsymbol{\Sigma}_c$, enabling calibration across light, shadow, and parchment variations.

---

## 5. Road Network & Linear Topology Vectorization

Implemented in `RoadExtractor`:
1. **Dual-Casing Detection**: Detects parallel orange/brown road boundaries.
2. **Medial-Axis Thinning**: Uses `skan.csr.skeleton_to_csgraph` to generate single-pixel topological skeletons.
3. **Graph Cleaning (`networkx`)**:
   - Collapses spurious dead-end spurs ($< 15\text{ px}$).
   - Merges degree-2 collinear vertices.
   - Extracts continuous GIS `LineString` geometries with attributes (length, road class).

---

## 6. Toponym OCR & LOD Gazetteer Alignment

Implemented in `ToponymExtractor` and `SpatialGazetteer`:
1. **Text Detection**: EasyOCR CRAFT text detector with French & German language models.
2. **Parchment Bounding Box Snapping**: Snaps text boxes to remove background color and isolate antique typography.
3. **Gazetteer Harmonization**:
   - Queries **Getty TGN**, **GeoNames**, and **GND** via SPARQL and REST APIs.
   - Computes Levenshtein edit distance and spatial proximity to assign historical place IDs (URI, coordinates, historical variants).

---

## 7. Meta SAM GPU AI Backend & LabelStudio Integration

Implemented in `SAMExtractor` and `label_studio_ml_backend.py`:
- Loads `facebook/sam-vit-base` on CUDA GPU with float16 inference.
- Generates high-resolution masks from positive/negative prompt coordinates.
- Exposes LabelStudio ML Backend API on port `9090`.

To launch the SAM LabelStudio Backend:
```bash
python -m tranchot_extractor.studio.label_studio_ml_backend
```

---

## 8. Configuration & Parameter Reference

All extractors are configurable via strongly typed dataclasses in [`config.py`](file:///c:/Users/langm/sciebo/BCDH_Projektbox/1_BCDH%20Intern/Scripts/Tranchot_Extractor/tranchot_extractor/tranchot_extractor/config.py):

### `BuildingConfig`
| Parameter | Default | Description |
|---|---|---|
| `rgb_diff_threshold` | `18` | Minimum $(R - G)$ and $(R - B)$ difference for carmine ink |
| `min_building_area_px` | `6.0` | Minimum building area in square pixels |
| `max_building_area_px` | `25000.0` | Maximum complex area |
| `min_stroke_width_px` | `1.0` | Minimum wall / wing thickness |
| `max_aspect_ratio` | `50.0` | Maximum length/width ratio for long thin walls |
| `despike_angle_deg` | `45.0` | Threshold below which acute corners are eliminated |
| `filter_vineyard_terraces` | `True` | Rejects non-solid wavy hillside hatching |

### `LandUseConfig`
| Parameter | Default | Description |
|---|---|---|
| `min_forest_area_px` | `600.0` | Minimum forest polygon size |
| `min_meadow_area_px` | `300.0` | Minimum meadow polygon size |
| `min_water_area_px` | `150.0` | Minimum water body size |
| `pyramid_max_dim` | `1800` | Pyramidal downsampling dimension for $100\times$ speedup |

---

## 9. Developer Guide: How to Extend & Calibrate

### Adding a New Extractor Module:
1. Create `tranchot_extractor/extractors/my_new_extractor.py`.
2. Define a dataclass result (e.g. `MyExtractionResult`) with Shapely geometries and GeoDataFrame.
3. Add a configuration dataclass in `config.py`.
4. Register CLI command in `cli.py` and add UI control in `desktop_app.py`.
5. Add unit tests in `tests/test_my_new_extractor.py`.

### Running Tests:
```bash
python -m unittest discover -s tests
```

---

## 🏛️ Project Contact
**Bonn Center for Digital Humanities (BCDH)**  
University of Bonn  
Repository: [https://github.com/mlangsqrldev/tranchot_extractor](https://github.com/mlangsqrldev/tranchot_extractor)
