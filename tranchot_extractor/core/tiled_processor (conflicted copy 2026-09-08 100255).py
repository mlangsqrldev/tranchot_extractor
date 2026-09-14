"""
Tiled streaming processor for large historical map sheets and GeoTIFFs (10k+ pixels).
Processes maps in windowed chunks to avoid memory bottlenecks and combines features into global GIS layers.
"""

import os
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import translate
import geopandas as gpd
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from tranchot_extractor.config import PipelineConfig
from tranchot_extractor.extractors.building_extractor import BuildingExtractor, BuildingFeature, BuildingExtractionResult
from tranchot_extractor.extractors.boundary_extractor import BoundaryExtractor, BoundaryFeature, BoundaryExtractionResult
from tranchot_extractor.extractors.road_extractor import RoadExtractor, RoadSegmentFeature, RoadExtractionResult
from tranchot_extractor.extractors.text_extractor import TextExtractor, ToponymFeature, TextExtractionResult
from tranchot_extractor.geo.georeference import GeoReferenceHandler
from tranchot_extractor.geo.exporter import GISExporter


class TiledMapProcessor:
    """Streams and processes large historical map GeoTIFFs in tile windows."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.building_extractor = BuildingExtractor(self.config.buildings)
        self.boundary_extractor = BoundaryExtractor(self.config.boundaries, self.config.buildings)
        self.road_extractor = RoadExtractor(self.config.roads)
        self.text_extractor = TextExtractor(self.config.text)

    def process_geotiff(
        self,
        geotiff_path: str,
        output_dir: str,
        tile_size: int = 2048,
        overlap: int = 128,
        max_tiles: Optional[int] = None,
        extract_buildings: bool = True,
        extract_roads: bool = True,
        extract_text: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Streamingly processes a high-resolution GeoTIFF in windowed tiles.
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for GeoTIFF processing.")

        if not os.path.exists(geotiff_path):
            raise FileNotFoundError(f"GeoTIFF file not found: {geotiff_path}")

        os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(geotiff_path) as src:
            width = src.width
            height = src.height
            crs = src.crs.to_string() if src.crs else self.config.geo.default_crs
            transform = src.transform
            bounds = src.bounds

            geo_meta = {
                "crs": crs,
                "transform": transform,
                "bounds": bounds,
                "width": width,
                "height": height,
            }

            # Generate windows
            windows: List[Tuple[Window, int, int]] = []
            step = tile_size - overlap
            for row in range(0, height, step):
                for col in range(0, width, step):
                    w_w = min(tile_size, width - col)
                    w_h = min(tile_size, height - row)
                    win = Window(col, row, w_w, w_h)
                    windows.append((win, col, row))

            if max_tiles and max_tiles > 0:
                windows = windows[:max_tiles]

            print(f"🗺️ Verarbeite GeoTIFF: {os.path.basename(geotiff_path)} ({width}x{height} px, {len(windows)} Kacheln)")

            all_buildings: List[Dict[str, Any]] = []
            all_boundaries: List[Dict[str, Any]] = []
            all_roads: List[Dict[str, Any]] = []
            all_toponyms: List[Dict[str, Any]] = []

            bldg_id_counter = 1
            bound_id_counter = 1
            road_id_counter = 1
            toponym_id_counter = 1

            for idx, (win, col_off, row_off) in enumerate(tqdm(windows, desc="Verarbeite Kacheln")):
                if progress_callback:
                    progress_callback(idx + 1, len(windows))

                # Read tile
                tile_data = src.read(window=win)
                if tile_data.shape[0] >= 3:
                    tile_rgb = np.transpose(tile_data[:3, :, :], (1, 2, 0))
                elif tile_data.shape[0] == 1:
                    tile_rgb = cv2.cvtColor(tile_data[0], cv2.COLOR_GRAY2RGB)
                else:
                    tile_rgb = np.transpose(tile_data, (1, 2, 0))

                # Skip completely blank / black / transparent tiles
                if np.mean(tile_rgb) < 10 or np.mean(tile_rgb) > 250 and np.std(tile_rgb) < 5:
                    continue

                # 1. Text extraction
                tile_text_res = None
                if extract_text:
                    tile_text_res = self.text_extractor.extract(tile_rgb)
                    for t in tile_text_res.features:
                        # Translate geometry by tile offset
                        shifted_pt = Point(t.centroid_x + col_off, t.centroid_y + row_off)
                        shifted_poly = translate(t.geometry_polygon, xoff=col_off, yoff=row_off)
                        all_toponyms.append({
                            "toponym_id": toponym_id_counter,
                            "text": t.text,
                            "category": t.category,
                            "confidence": t.confidence,
                            "centroid_x": t.centroid_x + col_off,
                            "centroid_y": t.centroid_y + row_off,
                            "geometry": shifted_pt,
                            "layer": "toponym",
                        })
                        toponym_id_counter += 1

                # 2. Building extraction (solid Baukörper)
                tile_bldg_res = None
                if extract_buildings:
                    tile_bldg_res = self.building_extractor.extract(tile_rgb)
                    for b in tile_bldg_res.features:
                        shifted_geom = translate(b.geometry, xoff=col_off, yoff=row_off)
                        all_buildings.append({
                            "building_id": bldg_id_counter,
                            "area_px": b.area_px,
                            "perimeter_px": b.perimeter_px,
                            "compactness": round(b.compactness, 4),
                            "orientation_deg": round(b.orientation_deg, 2),
                            "centroid_x": b.centroid_x + col_off,
                            "centroid_y": b.centroid_y + row_off,
                            "geometry": shifted_geom,
                            "layer": "building",
                        })
                        bldg_id_counter += 1

                # 3. Boundary extraction (thin red border and parcel lines)
                if getattr(self, 'boundary_extractor', None) is not None:
                    b_mask = tile_bldg_res.mask if tile_bldg_res else None
                    tile_bound_res = self.boundary_extractor.extract(tile_rgb, building_mask=b_mask)
                    for b_feat in tile_bound_res.features:
                        shifted_line = translate(b_feat.geometry, xoff=col_off, yoff=row_off)
                        all_boundaries.append({
                            "boundary_id": bound_id_counter,
                            "length_px": b_feat.length_px,
                            "geometry": shifted_line,
                            "layer": "boundary",
                        })
                        bound_id_counter += 1

                # 4. Road extraction
                if extract_roads:
                    b_mask = tile_bldg_res.mask if tile_bldg_res else None
                    t_mask = tile_text_res.dilated_mask if tile_text_res else None
                    tile_road_res = self.road_extractor.extract(
                        tile_rgb, building_mask=b_mask, text_mask=t_mask
                    )
                    for r in tile_road_res.features:
                        shifted_line = translate(r.geometry, xoff=col_off, yoff=row_off)
                        all_roads.append({
                            "road_id": road_id_counter,
                            "category": r.category,
                            "length_px": r.length_px,
                            "geometry": shifted_line,
                            "layer": "road",
                        })
                        road_id_counter += 1

            # Global topological line merging across all tile boundaries
            from shapely.ops import linemerge, unary_union
            if all_roads:
                try:
                    all_lines = [r["geometry"] for r in all_roads]
                    merged_roads = linemerge(unary_union(all_lines))
                    if isinstance(merged_roads, LineString):
                        merged_road_list = [merged_roads]
                    elif hasattr(merged_roads, 'geoms'):
                        merged_road_list = list(merged_roads.geoms)
                    else:
                        merged_road_list = all_lines
                except Exception:
                    merged_road_list = [r["geometry"] for r in all_roads]
            else:
                merged_road_list = []

            min_road_len = getattr(self.config.roads, 'min_path_length_px', 85.0)
            valid_roads = []
            for idx, l in enumerate(merged_road_list, 1):
                if isinstance(l, LineString) and not l.is_empty and l.length >= min_road_len:
                    valid_roads.append({
                        "road_id": idx,
                        "category": "main_road" if l.length > 120 else "path",
                        "length_px": round(l.length, 2),
                        "geometry": l,
                        "layer": "road",
                    })

            # Global topological line merging for boundaries
            if all_boundaries:
                try:
                    all_bound_lines = [b["geometry"] for b in all_boundaries]
                    merged_bounds = linemerge(unary_union(all_bound_lines))
                    if isinstance(merged_bounds, LineString):
                        merged_bound_list = [merged_bounds]
                    elif hasattr(merged_bounds, 'geoms'):
                        merged_bound_list = list(merged_bounds.geoms)
                    else:
                        merged_bound_list = all_bound_lines
                except Exception:
                    merged_bound_list = [b["geometry"] for b in all_boundaries]
            else:
                merged_bound_list = []

            min_bound_len = getattr(self.config.boundaries, 'min_boundary_length_px', 20.0) if hasattr(self.config, 'boundaries') else 20.0
            valid_boundaries = []
            for idx, l in enumerate(merged_bound_list, 1):
                if isinstance(l, LineString) and not l.is_empty and l.length >= min_bound_len:
                    valid_boundaries.append({
                        "boundary_id": idx,
                        "length_px": round(l.length, 2),
                        "geometry": l,
                        "layer": "boundary",
                    })

            # Deduplicate buildings that were independently detected in overlapping tiles
            if all_buildings:
                all_buildings = self._deduplicate_buildings(all_buildings)

            bldg_gdf = gpd.GeoDataFrame(all_buildings, geometry=[r["geometry"] for r in all_buildings]) if all_buildings else gpd.GeoDataFrame(geometry=[])
            boundaries_gdf = gpd.GeoDataFrame(valid_boundaries, geometry=[r["geometry"] for r in valid_boundaries]) if valid_boundaries else gpd.GeoDataFrame(geometry=[])
            roads_gdf = gpd.GeoDataFrame(valid_roads, geometry=[r["geometry"] for r in valid_roads]) if valid_roads else gpd.GeoDataFrame(geometry=[])
            toponyms_gdf = gpd.GeoDataFrame(all_toponyms, geometry=[r["geometry"] for r in all_toponyms]) if all_toponyms else gpd.GeoDataFrame(geometry=[])

            # Deduplicate text labels that appear in tile overlaps
            if not toponyms_gdf.empty:
                toponyms_gdf = self._deduplicate_toponyms(toponyms_gdf)

            # Export
            georef = GeoReferenceHandler.from_metadata(geo_meta)
            exporter = GISExporter(output_dir, georef=georef)
            exported = exporter.export_all(
                buildings_gdf=bldg_gdf,
                boundaries_gdf=boundaries_gdf,
                roads_gdf=roads_gdf,
                toponyms_gdf=toponyms_gdf,
                masks=None,
                create_zip=True,
            )

            return {
                "metadata": geo_meta,
                "building_count": len(bldg_gdf),
                "boundary_count": len(boundaries_gdf),
                "road_count": len(roads_gdf),
                "toponym_count": len(toponyms_gdf),
                "exported_files": exported,
                "buildings_gdf": bldg_gdf,
                "boundaries_gdf": boundaries_gdf,
                "roads_gdf": roads_gdf,
                "toponyms_gdf": toponyms_gdf,
            }

    @staticmethod
    def _deduplicate_buildings(buildings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes buildings independently re-detected in adjacent overlapping tiles."""
        clean: List[Dict[str, Any]] = []
        for record in buildings:
            poly = record["geometry"]
            if not poly.is_valid or poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            is_dup = False
            for existing in clean:
                e_minx, e_miny, e_maxx, e_maxy = existing["geometry"].bounds
                if maxx < e_minx or minx > e_maxx or maxy < e_miny or miny > e_maxy:
                    continue
                try:
                    inter_area = poly.intersection(existing["geometry"]).area
                    if inter_area > 0.70 * min(poly.area, existing["geometry"].area):
                        is_dup = True
                        break
                except Exception:
                    pass
            if not is_dup:
                clean.append(record)
        return clean

    @staticmethod
    def _deduplicate_toponyms(gdf: gpd.GeoDataFrame, distance_threshold: float = 30.0) -> gpd.GeoDataFrame:
        """Removes duplicate text detections in overlap zones."""
        if len(gdf) <= 1:
            return gdf

        keep_indices = []
        visited = set()

        coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        texts = list(gdf["text"])

        for i in range(len(gdf)):
            if i in visited:
                continue
            keep_indices.append(i)
            visited.add(i)

            # Check neighbors
            for j in range(i + 1, len(gdf)):
                if j in visited:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < distance_threshold and (texts[i].lower() == texts[j].lower()):
                    visited.add(j)

        return gdf.iloc[keep_indices].reset_index(drop=True)
