"""
Tranchot Map Feature Extractor
Automated & Semiautomatic AI/CV pipeline for extracting Buildings, Roads/Paths,
and Historical Toponyms from Tranchot & von Müffling maps (1803-1828).
"""

__version__ = "0.2.0"
__author__ = "Bonn Center for Digital Humanities (BCDH)"

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from tranchot_extractor.core.pipeline import TranchotPipeline
from tranchot_extractor.core.tiled_processor import TiledMapProcessor
from tranchot_extractor.extractors.building_extractor import BuildingExtractor
from tranchot_extractor.extractors.road_extractor import RoadExtractor
from tranchot_extractor.extractors.text_extractor import TextExtractor
from tranchot_extractor.extractors.sam_extractor import SAMExtractor

__all__ = [
    "TranchotPipeline",
    "TiledMapProcessor",
    "BuildingExtractor",
    "RoadExtractor",
    "TextExtractor",
    "SAMExtractor",
]
