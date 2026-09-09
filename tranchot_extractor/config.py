"""
Configuration settings and default parameters for Tranchot feature extraction.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BuildingConfig:
    """Configuration for building footprint, courtyard, and thin wall extraction."""
    # Pure Carmine / Vermilion red HSV ranges
    hsv_lower_red1: Tuple[int, int, int] = (0, 18, 25)
    hsv_upper_red1: Tuple[int, int, int] = (18, 255, 255)
    hsv_lower_red2: Tuple[int, int, int] = (160, 18, 25)
    hsv_upper_red2: Tuple[int, int, int] = (180, 255, 255)
    
    # Differential RGB threshold (R - max(G, B))
    rgb_diff_threshold: int = 15
    min_red_intensity: int = 60
    
    # LAB a* channel threshold (green-red axis: isolates deep carmine from brown ink and yellow parchment)
    lab_a_threshold: int = 133
    
    # Normalized Red Excess and Ratio thresholds (Bandpass Filter)
    min_red_excess: float = 18.0
    max_red_excess: float = 255.0
    min_red_ratio: float = 1.05
    max_red_ratio: float = 999.0
    min_building_thickness_px: float = 0.5
    
    # Morphological filtering & geometric constraints
    morph_kernel_size: int = 1
    min_building_area_px: float = 5.0
    max_building_area_px: float = 35000.0
    min_stroke_width_px: float = 0.2
    max_aspect_ratio: float = 120.0  # Allows long thin courtyard walls, attached street rows, and rectangular wings
    min_compactness: float = 0.01  # Sensitive enough to capture long thin walls
    filter_vineyard_terraces: bool = False
    preserve_curved_walls: bool = True
    preserve_courtyards: bool = True
    enable_black_buildings: bool = False
    enable_paper_normalization: bool = True
    polygon_simplification_factor: float = 0.015
    simplify_tolerance: float = 0.6
    min_edge_length_px: float = 2.0
    despike_angle_deg: float = 30.0
    regularize_orthogonal: bool = False
    orthogonal_snap_deg: float = 15.0




@dataclass
class RoadConfig:
    """Configuration for road and linear path network extraction."""
    # Multiscale Frangi / Ridge filter sigmas
    frangi_sigmas: Tuple[float, ...] = (1.2, 2.2, 3.5)
    ridge_threshold: float = 32.0  # Higher threshold suppresses fine mountain hachures (Schraffen)
    
    # Mask dilation around buildings and text to prevent false positive road segments
    building_exclusion_dilation: int = 5
    text_exclusion_dilation: int = 6
    
    # Minimum length of through-road segments (filters out mountain hachures and small specks)
    min_path_length_px: float = 60.0
    min_path_length_m: float = 70.0
    simplify_tolerance: float = 2.0



@dataclass
class TextConfig:
    """Configuration for text and toponym detection and OCR."""
    languages: List[str] = field(default_factory=lambda: ["de", "fr", "en"])
    use_gpu: bool = True
    confidence_threshold: float = 0.20
    text_mask_dilation: int = 4


@dataclass
class BoundaryConfig:
    """Configuration for historical boundary lines (Gemarkungs- & Flurgrenzen)."""
    min_boundary_length_px: float = 20.0
    bridge_kernel_size: int = 7
    max_stroke_thickness: float = 2.0
    simplification_tolerance: float = 2.0


@dataclass
class LandUseConfig:
    """Configuration for historical land-use (Forest, Meadow, Water, Vineyards, Gardens) extraction."""
    min_forest_area_px: float = 400.0
    min_meadow_area_px: float = 250.0
    min_water_area_px: float = 50.0
    min_vineyard_area_px: float = 300.0
    min_garden_area_px: float = 150.0
    max_processing_dim: int = 2048
    enable_forest: bool = True
    enable_meadow: bool = True
    enable_water: bool = True
    enable_vineyard: bool = True
    enable_garden: bool = True


@dataclass
class GeoConfig:
    """Configuration for georeferencing and GIS exports."""
    default_crs: str = "EPSG:25832"  # ETRS89 / UTM zone 32N (Standard Rhineland)
    export_formats: List[str] = field(default_factory=lambda: ["geojson", "gpkg", "shapefile", "csv"])


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all module settings."""
    buildings: BuildingConfig = field(default_factory=BuildingConfig)
    roads: RoadConfig = field(default_factory=RoadConfig)
    landuse: LandUseConfig = field(default_factory=LandUseConfig)
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    text: TextConfig = field(default_factory=TextConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)

    tile_size: int = 2048
    tile_overlap: int = 256
