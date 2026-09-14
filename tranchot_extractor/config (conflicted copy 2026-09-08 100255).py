"""
Configuration settings and default parameters for Tranchot feature extraction.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BuildingConfig:
    """Configuration for building footprint, courtyard, and thin wall extraction."""
    # Carmine, Terracotta & Brownish-red building HSV ranges
    hsv_lower_red1: Tuple[int, int, int] = (0, 26, 35)
    hsv_upper_red1: Tuple[int, int, int] = (18, 255, 255)
    hsv_lower_red2: Tuple[int, int, int] = (160, 26, 35)
    hsv_upper_red2: Tuple[int, int, int] = (180, 255, 255)
    
    # Differential RGB threshold (R - max(G, B))
    rgb_diff_threshold: int = 18
    min_red_intensity: int = 145
    min_norm_red_intensity: int = 135
    
    # LAB a* channel threshold (green-red axis: distinguishes red/terracotta ink from parchment, lettering and vineyard hachures)
    lab_a_threshold: int = 133

    # Paper normalization & Beer-Lambert Optical Density deconvolution
    enable_paper_normalization: bool = True
    carmine_od_threshold: float = 0.035
    norm_rgb_diff_threshold: int = 14
    min_building_thickness_px: float = 1.6
    min_courtyard_fill_ratio: float = 0.18
    max_unified_hof_area_px: float = 8500.0  # Max area for a genuine single farmstead courtyard
    max_hof_span_px: float = 115.0  # Above 115 px span, always separate clusters with watershed
    
    # Morphological filtering & geometric constraints
    morph_kernel_size: int = 1
    min_building_area_px: float = 12.0
    max_building_area_px: float = 25000.0
    min_stroke_width_px: float = 1.0  # Sensitive enough to capture fine 1.0-2.0 px historical courtyard walls and barn wings
    max_aspect_ratio: float = 16.0  # Max architectural aspect ratio for elongated barn wings and farmstead enclosures
    min_compactness: float = 0.02  # Sensitive enough to capture long thin walls
    filter_vineyard_terraces: bool = True
    
    # Black stone building extraction (default False to strictly prevent text lettering and hachure false positives)
    enable_black_buildings: bool = False
    min_black_building_area_px: float = 45.0
    min_black_building_thickness_px: float = 4.5
    max_black_building_aspect_ratio: float = 3.8
    min_black_building_solidity: float = 0.72
    
    polygon_simplification_factor: float = 0.035
    simplify_tolerance: float = 2.5
    min_edge_length_px: float = 3.5
    despike_angle_deg: float = 45.0
    regularize_orthogonal: bool = True
    orthogonal_snap_deg: float = 18.0




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
    """Configuration for historical land-use (Forest, Meadow, Water) extraction."""
    min_forest_area_px: float = 600.0
    min_meadow_area_px: float = 250.0
    min_water_area_px: float = 40.0
    max_processing_dim: int = 2048

    # Multi-cue differentiation: Watercourses (Flüsse/Bäche) vs. Meadows (Wiesen/Weiden)
    enable_hierarchical_valley_subtraction: bool = True
    stream_ridge_sigmas: Tuple[float, ...] = (1.0, 1.8, 2.8, 4.0)
    stream_max_width_px: float = 24.0
    stream_min_length_px: float = 35.0
    stream_bank_gradient_threshold: float = 600.0
    meadow_min_breadth_px: float = 28.0
    stream_dilation_buffer_px: int = 4


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
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    text: TextConfig = field(default_factory=TextConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    tile_size: int = 2048
    tile_overlap: int = 256
