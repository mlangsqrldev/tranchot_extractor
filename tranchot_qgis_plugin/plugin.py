"""
Main QGIS Plugin class for Tranchot Extractor.
Registers menus, toolbars, and controls the lifecycle of the dockwidget panel.
"""

import os
from qgis.PyQt.QtCore import Qt, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .dockwidget import TranchotDockWidget


class TranchotPlugin:
    """
    QGIS Plugin implementation for Tranchot Extractor.
    """

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.dockwidget: TranchotDockWidget = None
        self.action: QAction = None
        self.toolbar = None

    def tr(self, message: str) -> str:
        return QCoreApplication.translate("TranchotPlugin", message)

    def initGui(self):
        """Create the menu entries and toolbar icons inside QGIS."""
        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        self.action = QAction(icon, self.tr("HistMap Extractor"), self.iface.mainWindow())
        self.action.setToolTip(self.tr("Historical Map Vectorization (BCDH)"))
        self.action.triggered.connect(self.run)

        # Add toolbar icon
        self.toolbar = self.iface.addToolBar("HistMap Extractor")
        self.toolbar.setObjectName("HistMapExtractorToolBar")
        self.toolbar.addAction(self.action)

        # Add to QGIS Raster menu and main Plugins menu
        self.iface.addPluginToRasterMenu(self.tr("&HistMap Extractor"), self.action)
        self.iface.addPluginToMenu(self.tr("&HistMap Extractor"), self.action)

    def unload(self):
        """Removes the plugin menu items, toolbar and dock widget."""
        if self.action:
            self.iface.removePluginRasterMenu(self.tr("&HistMap Extractor"), self.action)
            self.iface.removePluginMenu(self.tr("&HistMap Extractor"), self.action)

        if self.toolbar:
            del self.toolbar

        if self.dockwidget:
            self.iface.removeDockWidget(self.dockwidget)
            self.dockwidget.deleteLater()
            self.dockwidget = None

    def run(self):
        """Toggles or displays the Tranchot dock widget panel with hot-reloading."""
        import sys
        import importlib

        # Dynamically reload all related modules so on-disk code and translation updates take effect immediately
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("tranchot_qgis_plugin") or mod_name.startswith("tranchot_extractor"):
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

        # Re-import fresh TranchotDockWidget class
        from .dockwidget import TranchotDockWidget

        if self.dockwidget is not None:
            try:
                self.iface.removeDockWidget(self.dockwidget)
                self.dockwidget.deleteLater()
            except Exception:
                pass
            self.dockwidget = None

        self.dockwidget = TranchotDockWidget(self.iface, self.iface.mainWindow())
        self.dockwidget.closingPlugin.connect(self._on_dockwidget_closed)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dockwidget)

        self.dockwidget.show()
        self.dockwidget.raise_()

    def _on_dockwidget_closed(self):
        """Handle cleanup when user closes dockwidget."""
        if self.dockwidget is not None:
            try:
                self.iface.removeDockWidget(self.dockwidget)
                self.dockwidget.deleteLater()
            except Exception:
                pass
            self.dockwidget = None
