"""
Tranchot Extractor - QGIS Plugin
Automated feature extraction and vectorization for historical maps (Tranchot & von Müffling 1803-1820).

Developed by the Bonn Center for Digital Humanities (BCDH), University of Bonn.
"""


import sys
import io

# QGIS on Windows runs as a GUI application without an attached console;
# sys.stderr and sys.stdout can be None, causing libraries emitting warnings to crash.
class _SafeStream(io.StringIO):
    def write(self, s):
        pass
    def flush(self):
        pass

if getattr(sys, "stderr", None) is None:
    sys.stderr = _SafeStream()
if getattr(sys, "stdout", None) is None:
    sys.stdout = _SafeStream()


def classFactory(iface):
    """Load TranchotPlugin class from file plugin.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .plugin import TranchotPlugin
    return TranchotPlugin(iface)

