import sys
import os
sys.path.insert(0, r'c:\Users\Edouard Grigowski\Desktop\tranchot_extractor')
from qgis.core import QgsApplication
app = QgsApplication([], False)
app.initQgis()

from tranchot_extractor.config import TextConfig
from tranchot_qgis_plugin.tasks import TextExtractionTask

task = TextExtractionTask(
    raster_path=r'c:\Users\Edouard Grigowski\Desktop\tranchot_extractor\test.tif',  # we just want to see if it starts
    config=TextConfig()
)
task.run()
print(f"Task error: {task.error_msg}")
print(f"Task features: {len(task.extracted_features)}")
