"""
Master orchestration pipeline for Tranchot Map AI Feature Extraction.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import geopandas as gpd

from tranchot_extractor.config import PipelineConfig
from tranchot_extractor.core.preprocessor import MapPreprocessor
from tranchot_extractor.extractors.building_extractor import BuildingExtractor, BuildingExtractionResult
from tranchot_extractor.extractors.boundary_extractor import BoundaryExtractor, BoundaryExtractionResult
from tranchot_extractor.extractors.road_extractor import RoadExtractor, RoadExtractionResult
from tranchot_extractor.extractors.text_extractor import TextExtractor, TextExtractionResult
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.exporter import GISExporter


@dataclass
class PipelineResult:
    """Master result containing all extracted features, layers, masks, and metadata."""
    original_image: np.ndarray
    buildings: BuildingExtractionResult
    boundaries: BoundaryExtractionResult
    roads: RoadExtractionResult
    text: TextExtractionResult
    geo_metadata: Optional[Dict[str, Any]]
    visualizations: Dict[str, np.ndarray]
    exported_files: Dict[str, str]


class TranchotPipeline:
    """
    End-to-end pipeline for extracting Buildings, Boundaries, Roads, and Toponyms
    from Tranchot and historical Rhineland map sheets.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.preprocessor = MapPreprocessor()
        self.building_extractor = BuildingExtractor(self.config.buildings)
        self.boundary_extractor = BoundaryExtractor(self.config.boundaries, self.config.buildings)
        self.road_extractor = RoadExtractor(self.config.roads)
        self.text_extractor = TextExtractor(self.config.text)

    def process(
        self,
        image_input: Any,
        output_dir: Optional[str] = None,
        extract_buildings: bool = True,
        extract_boundaries: bool = True,
        extract_roads: bool = True,
        extract_text: bool = True,
    ) -> PipelineResult:
        """
        Executes full extraction pipeline on the provided map image.
        """
        # 1. Load image and metadata
        image_rgb, geo_meta = self.preprocessor.load_image(image_input)
        h, w = image_rgb.shape[:2]

        # 2. Text Extraction (first, so text mask can be used for road exclusion)
        text_result = None
        if extract_text:
            text_result = self.text_extractor.extract(image_rgb)
        else:
            text_result = TextExtractionResult(
                features=[],
                mask=np.zeros((h, w), dtype=np.uint8),
                dilated_mask=np.zeros((h, w), dtype=np.uint8),
                gdf=gpd.GeoDataFrame(geometry=[])
            )

        # 3. Building Extraction (solid Baukörper)
        building_result = None
        if extract_buildings:
            building_result = self.building_extractor.extract(image_rgb)
        else:
            building_result = BuildingExtractionResult(
                features=[],
                mask=np.zeros((h, w), dtype=np.uint8),
                gdf=gpd.GeoDataFrame(geometry=[]),
                raw_contour_count=0
            )

        # 4. Boundary Extraction (thin red Gemarkungs- & Flurgrenzen lines)
        boundary_result = None
        if extract_boundaries:
            boundary_result = self.boundary_extractor.extract(
                image_rgb,
                building_mask=building_result.mask
            )
        else:
            boundary_result = BoundaryExtractionResult(
                features=[],
                mask=np.zeros((h, w), dtype=np.uint8),
                gdf=gpd.GeoDataFrame(geometry=[])
            )

        # 5. Road Extraction (with building, boundary, and text exclusion)
        road_result = None
        if extract_roads:
            road_result = self.road_extractor.extract(
                image_rgb,
                building_mask=building_result.mask,
                text_mask=text_result.dilated_mask
            )
        else:
            road_result = RoadExtractionResult(
                features=[],
                ridge_map=np.zeros((h, w), dtype=np.uint8),
                skeleton_mask=np.zeros((h, w), dtype=np.uint8),
                gdf=gpd.GeoDataFrame(geometry=[])
            )

        # 6. Generate Rich Visualizations
        visualizations = self._generate_visualizations(
            image_rgb, building_result, boundary_result, road_result, text_result
        )

        # 7. Export to GIS Formats if output directory specified
        exported_files = {}
        if output_dir:
            crs_value = (geo_meta or {}).get("crs") or self.config.geo.default_crs
            georef = GeoReferenceHandler.from_metadata({**(geo_meta or {}), "crs": crs_value})
            exporter = GISExporter(output_dir, georef=georef)
            masks_to_save = {
                "buildings_mask": building_result.mask,
                "boundaries_mask": boundary_result.mask,
                "roads_skeleton": road_result.skeleton_mask,
                "roads_ridge": road_result.ridge_map,
                "text_mask": text_result.mask,
                "composite_overlay": visualizations.get("composite"),
            }
            exported_files = exporter.export_all(
                buildings_gdf=building_result.gdf,
                boundaries_gdf=boundary_result.gdf,
                roads_gdf=road_result.gdf,
                toponyms_gdf=text_result.gdf,
                masks=masks_to_save,
                create_zip=True,
            )

        return PipelineResult(
            original_image=image_rgb,
            buildings=building_result,
            boundaries=boundary_result,
            roads=road_result,
            text=text_result,
            geo_metadata=geo_meta,
            visualizations=visualizations,
            exported_files=exported_files,
        )

    def _generate_visualizations(
        self,
        image_rgb: np.ndarray,
        b_res: BuildingExtractionResult,
        bound_res: BoundaryExtractionResult,
        r_res: RoadExtractionResult,
        t_res: TextExtractionResult,
    ) -> Dict[str, np.ndarray]:
        """Creates distinct color-coded layer overlays for preview and UI."""
        h, w = image_rgb.shape[:2]

        # 1. Buildings Only Overlay
        from shapely.geometry import Polygon
        bldg_vis = image_rgb.copy()
        for f in b_res.features:
            geoms = [f.geometry] if isinstance(f.geometry, Polygon) else list(getattr(f.geometry, 'geoms', []))
            for g in geoms:
                if hasattr(g, 'exterior'):
                    pts = np.array(g.exterior.coords, dtype=np.int32)
                    overlay = bldg_vis.copy()
                    cv2.fillPoly(overlay, [pts], (0, 230, 120))
                    cv2.addWeighted(overlay, 0.45, bldg_vis, 0.55, 0, bldg_vis)
                    cv2.polylines(bldg_vis, [pts], True, (0, 255, 100), 2, cv2.LINE_AA)
                    for interior in getattr(g, 'interiors', []):
                        hole_pts = np.array(interior.coords, dtype=np.int32)
                        cv2.polylines(bldg_vis, [hole_pts], True, (0, 200, 255), 2, cv2.LINE_AA)

        # 2. Boundaries Only Overlay (Orange / Gold Lines)
        bound_vis = image_rgb.copy()
        for f in bound_res.features:
            coords = np.array(f.geometry.coords, dtype=np.int32)
            cv2.polylines(bound_vis, [coords], False, (0, 165, 255), 2, cv2.LINE_AA)

        # 3. Roads Only Overlay (Electric Blue / Cyan)
        road_vis = image_rgb.copy()
        for f in r_res.features:
            coords = np.array(f.geometry.coords, dtype=np.int32)
            color = (30, 144, 255) if f.category == "main_road" else (0, 220, 255)
            thickness = 3 if f.category == "main_road" else 2
            cv2.polylines(road_vis, [coords], False, color, thickness, cv2.LINE_AA)

        # 4. Text Only Overlay (Amber / Gold Boxes & Labels)
        text_vis = image_rgb.copy()
        for f in t_res.features:
            pts = np.array(f.bounding_box, dtype=np.int32)
            cv2.polylines(text_vis, [pts], True, (255, 191, 0), 2, cv2.LINE_AA)
            label = f"{f.text} ({f.confidence:.2f})"
            cv2.putText(
                text_vis,
                label,
                (int(f.centroid_x) - 10, max(15, int(f.centroid_y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                text_vis,
                label,
                (int(f.centroid_x) - 10, max(15, int(f.centroid_y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 220, 50),
                1,
                cv2.LINE_AA
            )

        # 5. Master Composite All-in-One Overlay
        composite = image_rgb.copy()

        # Add boundaries
        for f in bound_res.features:
            coords = np.array(f.geometry.coords, dtype=np.int32)
            cv2.polylines(composite, [coords], False, (0, 165, 255), 2, cv2.LINE_AA)

        # Add roads
        for f in r_res.features:
            coords = np.array(f.geometry.coords, dtype=np.int32)
            color = (0, 180, 255)
            cv2.polylines(composite, [coords], False, color, 2, cv2.LINE_AA)

        # Add buildings
        for f in b_res.features:
            geoms = [f.geometry] if isinstance(f.geometry, Polygon) else list(getattr(f.geometry, 'geoms', []))
            for g in geoms:
                if hasattr(g, 'exterior'):
                    pts = np.array(g.exterior.coords, dtype=np.int32)
                    overlay = composite.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 120))
                    cv2.addWeighted(overlay, 0.4, composite, 0.6, 0, composite)
                    cv2.polylines(composite, [pts], True, (0, 230, 80), 2, cv2.LINE_AA)
                    for interior in getattr(g, 'interiors', []):
                        hole_pts = np.array(interior.coords, dtype=np.int32)
                        cv2.polylines(composite, [hole_pts], True, (0, 200, 255), 2, cv2.LINE_AA)

        # Add text
        for f in t_res.features:
            pts = np.array(f.bounding_box, dtype=np.int32)
            cv2.polylines(composite, [pts], True, (255, 215, 0), 2, cv2.LINE_AA)
            label = f.text
            cv2.putText(
                composite,
                label,
                (int(f.centroid_x), int(f.centroid_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 235, 80),
                1,
                cv2.LINE_AA
            )

        return {
            "buildings": bldg_vis,
            "roads": road_vis,
            "text": text_vis,
            "composite": composite,
        }
