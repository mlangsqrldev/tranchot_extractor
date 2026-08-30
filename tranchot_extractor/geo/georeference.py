"""
Georeferencing and spatial coordinate transformation for Tranchot map features.
"""

import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform
import geopandas as gpd

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class GeoReferenceHandler:
    """Handles pixel-to-geographic coordinate transformations."""

    def __init__(
        self,
        crs: Optional[str] = "EPSG:25832",
        affine_transform: Optional[Any] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ):
        # Handle case where a file path string was passed directly to constructor
        if isinstance(crs, str) and (crs.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.jp2')) or os.path.exists(crs)):
            file_handler = self.from_file(crs)
            self.crs = file_handler.crs
            self.affine_transform = file_handler.affine_transform
            self.bounds = file_handler.bounds
            self.image_shape = file_handler.image_shape
            return

        self.crs = crs or "EPSG:25832"
        self.affine_transform = affine_transform
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.image_shape = image_shape  # (height, width)

        if self.affine_transform is None and self.bounds is not None and self.image_shape is not None:
            h, w = self.image_shape
            min_x, min_y, max_x, max_y = self.bounds
            res_x = (max_x - min_x) / max(1, w)
            res_y = (max_y - min_y) / max(1, h)
            if HAS_RASTERIO:
                self.affine_transform = Affine(res_x, 0.0, min_x, 0.0, -res_y, max_y)

    @property
    def transform(self) -> Optional[Any]:
        """Alias for affine_transform."""
        return self.affine_transform

    @classmethod
    def from_file(cls, file_path: str) -> "GeoReferenceHandler":
        """Factory method to load georeferencing directly from a GeoTIFF or image file."""
        if not os.path.exists(file_path):
            return cls(crs="EPSG:25832", affine_transform=None)

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    crs_str = src.crs.to_string() if src.crs else "EPSG:25832"
                    bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    return cls(
                        crs=crs_str,
                        affine_transform=src.transform,
                        bounds=bounds,
                        image_shape=(src.height, src.width)
                    )
            except Exception:
                pass

        return cls(crs="EPSG:25832", affine_transform=None)

    @classmethod
    def from_metadata(cls, geo_meta: Optional[Dict[str, Any]]) -> "GeoReferenceHandler":
        """Factory from rasterio metadata dictionary."""
        if not geo_meta:
            return cls(crs="EPSG:25832", affine_transform=None)
        return cls(
            crs=geo_meta.get("crs", "EPSG:25832"),
            affine_transform=geo_meta.get("transform"),
            bounds=geo_meta.get("bounds"),
            image_shape=(geo_meta.get("height", 0), geo_meta.get("width", 0))
        )

    def pixel_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """Transforms a single pixel (col, row) coordinate to geographic (X, Y)."""
        if self.affine_transform is not None:
            geo_x, geo_y = self.affine_transform * (x, y)
            return float(geo_x), float(geo_y)
        return float(x), float(y)

    def transform_geometry(self, geom: Any) -> Any:
        """Transforms a Shapely geometry from pixel coords to geographic coords."""
        if self.affine_transform is None:
            return geom

        def _coord_trans(x, y, z=None):
            if z is not None:
                gx, gy = self.affine_transform * (x, y)
                return gx, gy, z
            gx, gy = self.affine_transform * (x, y)
            return gx, gy

        return transform(_coord_trans, geom)

    def apply_to_geodataframe(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Applies spatial transform and CRS to a GeoDataFrame."""
        if gdf.empty:
            return gdf

        geo_gdf = gdf.copy()
        if self.affine_transform is not None:
            geo_gdf["geometry"] = geo_gdf["geometry"].apply(self.transform_geometry)
            geo_gdf.set_crs(self.crs or "EPSG:25832", inplace=True, allow_override=True)
        else:
            # Default to Rhineland standard ETRS89 / UTM 32N
            geo_gdf.set_crs(self.crs or "EPSG:25832", inplace=True, allow_override=True)

        return geo_gdf
