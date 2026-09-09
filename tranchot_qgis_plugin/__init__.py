"""
Tranchot Extractor - QGIS Plugin
Automated feature extraction and vectorization for historical maps (Tranchot & von Müffling 1803-1820).

Developed by the Bonn Center for Digital Humanities (BCDH), University of Bonn.
"""


def classFactory(iface):
    """Load TranchotPlugin class from file plugin.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .plugin import TranchotPlugin
    return TranchotPlugin(iface)
