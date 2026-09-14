import sys
import os
sys.path.insert(0, r'c:\Users\Edouard Grigowski\Desktop\tranchot_extractor')
from qgis.core import QgsApplication
app = QgsApplication([], False)
app.initQgis()

class MockCanvas:
    pass

class MockIface:
    def mapCanvas(self):
        return MockCanvas()

from tranchot_qgis_plugin.dockwidget import TranchotDockWidget
try:
    w = TranchotDockWidget(MockIface(), None)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
