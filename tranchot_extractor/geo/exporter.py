"""
GIS exporter module for saving extracted Tranchot features to GeoJSON, Shapefile, GeoPackage, and CSV.
"""

import os
import zipfile
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import pandas as pd
import geopandas as gpd

from tranchot_extractor.geo.georeference import GeoReferenceHandler


class GISExporter:
    """Exports vectorized map features and raster masks to standard GIS formats."""

    def __init__(self, output_dir: str, georef: Optional[GeoReferenceHandler] = None):
        self.output_dir = output_dir
        self.georef = georef or GeoReferenceHandler()
        os.makedirs(self.output_dir, exist_ok=True)

    def export_all(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        roads_gdf: gpd.GeoDataFrame,
        toponyms_gdf: gpd.GeoDataFrame,
        boundaries_gdf: Optional[gpd.GeoDataFrame] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        create_zip: bool = True,
    ) -> Dict[str, str]:
        """
        Exports all layers to GeoJSON, GeoPackage, Shapefiles, and CSV.
        Returns dictionary mapping export format/layer to file paths.
        """
        saved_files = {}

        # Transform to geographic coordinates if georeference is available
        default_crs = self.georef.crs if (self.georef and self.georef.crs) else "EPSG:25832"
        b_gdf = self.georef.apply_to_geodataframe(buildings_gdf)
        r_gdf = self.georef.apply_to_geodataframe(roads_gdf)
        t_gdf = self.georef.apply_to_geodataframe(toponyms_gdf)
        bound_gdf = self.georef.apply_to_geodataframe(boundaries_gdf) if boundaries_gdf is not None else gpd.GeoDataFrame(geometry=[])

        for g in [b_gdf, r_gdf, t_gdf, bound_gdf]:
            if not g.empty and g.crs is None:
                g.set_crs(default_crs, inplace=True, allow_override=True)

        def safe_write_geojson(gdf: gpd.GeoDataFrame, path: str):
            try:
                gdf.to_file(path, driver="GeoJSON")
                return path
            except PermissionError:
                alt_path = path.replace(".geojson", "_updated.geojson")
                gdf.to_file(alt_path, driver="GeoJSON")
                return alt_path
            except Exception as e:
                print(f"Warning: Error writing {path}: {e}")
                return None

        # 1. GeoJSON Exports
        if not b_gdf.empty:
            b_path = os.path.join(self.output_dir, "buildings.geojson")
            res_p = safe_write_geojson(b_gdf, b_path)
            if res_p:
                saved_files["buildings_geojson"] = res_p

        if not bound_gdf.empty:
            bound_path = os.path.join(self.output_dir, "boundaries.geojson")
            res_p = safe_write_geojson(bound_gdf, bound_path)
            if res_p:
                saved_files["boundaries_geojson"] = res_p

        if not r_gdf.empty:
            r_path = os.path.join(self.output_dir, "roads.geojson")
            res_p = safe_write_geojson(r_gdf, r_path)
            if res_p:
                saved_files["roads_geojson"] = res_p

        if not t_gdf.empty:
            t_path = os.path.join(self.output_dir, "toponyms.geojson")
            res_p = safe_write_geojson(t_gdf, t_path)
            if res_p:
                saved_files["toponyms_geojson"] = res_p

        # 2. GeoPackage Export (Single file with multiple layers)
        gpkg_path = os.path.join(self.output_dir, "tranchot_features.gpkg")
        try:
            if not b_gdf.empty:
                b_gdf.to_file(gpkg_path, layer="buildings", driver="GPKG")
            if not bound_gdf.empty:
                bound_gdf.to_file(gpkg_path, layer="boundaries", driver="GPKG")
            if not r_gdf.empty:
                r_gdf.to_file(gpkg_path, layer="roads", driver="GPKG")
            if not t_gdf.empty:
                t_gdf.to_file(gpkg_path, layer="toponyms", driver="GPKG")
            saved_files["geopackage"] = gpkg_path
        except PermissionError:
            alt_gpkg = os.path.join(self.output_dir, "tranchot_features_updated.gpkg")
            try:
                if not b_gdf.empty:
                    b_gdf.to_file(alt_gpkg, layer="buildings", driver="GPKG")
                if not bound_gdf.empty:
                    bound_gdf.to_file(alt_gpkg, layer="boundaries", driver="GPKG")
                if not r_gdf.empty:
                    r_gdf.to_file(alt_gpkg, layer="roads", driver="GPKG")
                if not t_gdf.empty:
                    t_gdf.to_file(alt_gpkg, layer="toponyms", driver="GPKG")
                saved_files["geopackage"] = alt_gpkg
            except Exception as e:
                print(f"Warning: GPKG fallback error ({e})")
        except Exception as e:
            print(f"Warning: GPKG export error ({e})")

        # 3. CSV Toponym Export
        if not t_gdf.empty:
            csv_path = os.path.join(self.output_dir, "toponyms_catalog.csv")
            t_df = pd.DataFrame(t_gdf.drop(columns=["geometry"], errors="ignore"))
            t_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            saved_files["toponyms_csv"] = csv_path

        # 4. Raster Mask Exports
        if masks:
            mask_dir = os.path.join(self.output_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            for name, mask_arr in masks.items():
                if mask_arr is not None and mask_arr.size > 0:
                    m_path = os.path.join(mask_dir, f"{name}.png")
                    # If RGB, convert from RGB to BGR for OpenCV
                    if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
                        cv2.imwrite(m_path, cv2.cvtColor(mask_arr, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(m_path, mask_arr)
                    saved_files[f"mask_{name}"] = m_path

        # 5. Create ZIP package
        if create_zip:
            zip_path = os.path.join(self.output_dir, "tranchot_extracted_layers.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in saved_files.values():
                    if os.path.exists(fpath):
                        arcname = os.path.relpath(fpath, self.output_dir)
                        zf.write(fpath, arcname)
            saved_files["zip_package"] = zip_path

        return saved_files
