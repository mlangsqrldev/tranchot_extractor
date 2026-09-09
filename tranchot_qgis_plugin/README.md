# 🗺️ HistMap Extractor – QGIS 3 Plugin

> **AI- & Computer-Vision-basierte Vektorisierung historischer Kartenblätter (Tranchot & von Müffling 1803–1820, Preußische Uraufnahme, etc.) direkt in QGIS.**  
> Entwickelt am **Bonn Center for Digital Humanities (BCDH)**, Universität Bonn.

---

## 🏗️ Architektur & Modulübersicht

Das Plugin ist modular aufgebaut und trennt Benutzeroberfläche (UI), interaktive Karten-Werkzeuge (Map Tools), Hintergrund-Berechnungen (QgsTask Threading) und Rendering/Symbolik sauber voneinander:

```
tranchot_qgis_plugin/
├── __init__.py              # QGIS Plugin Entry Point (classFactory)
├── metadata.txt             # Plugin-Metadaten (Name, Version, QGIS-Mindestversion, Tags)
├── plugin.py                # Plugin-Lifecycle, Toolbar-Actions, Menü-Einträge
├── dockwidget.py            # Hauptbedienpanel (QDockWidget): Parameter, Tabs, Live-Feedback
├── map_tools.py             # Interaktive Werkzeuge auf dem QGIS-Kartenfenster (ROI-Polygon, Pipette)
├── tasks.py                 # Asynchrone QgsTask-Worker (Multithreading ohne UI-Blockade)
├── symbology-style.db       # SQLite QGIS-Stildatenbank für automatische Layer-Gestaltung
├── resources/
│   ├── icon.png             # Plugin-Icon (Toolbar / Menü)
│   └── siedlungstypen_lvr_vokabular.csv  # LVR-Siedlungstyp-Vokabular mit SKOS/URI
└── README.md                # Entwickler-Dokumentation (diese Datei)
```

---

## 📋 Funktionsweise der Komponenten

### 1. `plugin.py` (Plugin-Lifecycle & Integration)
- Implementiert `initGui()` und `unload()` gemäß QGIS-Standard.
- Fügt das Icon zur Toolbar und zu den Menüs (*Raster* und *Erweiterungen*) hinzu.
- Initialisiert das `TranchotDockWidget` als andockbares Seitenpanel (rechts) im Hauptfenster.

### 2. `dockwidget.py` (UI & Workflow-Steuerung)
- Verwaltet die GUI mit 4 spezialisierten Reitern:
  1. **🎯 1. ROI & Eingabe**: Auswahl des Raster-Layers, interaktive Definition des Siedlungsgebiets (Polygon/Rechteck), Farbpipette für Schwellenwerte.
  2. **🏛️ 2. Gebäude & Höfe**: Parameter für orthogonale Gebäudeextraktion ($90^\circ$ Manhattan-Geometrien, Innenhof-Subtraktion für Vierkanthöfe/Dreiseithöfe, Mauern & Scheunenflügel).
  3. **🌲 3. Landnutzung**: Mehrklassen-Klassifikation (*Wald, Wiese, Gewässer, Weinberge, Gärten, Acker*) mittels spektraler Distanz ($\Delta E$, Mahalanobis) und Texturvarianz.
  4. **🎨 4. Symbolik & Export**: Automatische Zuweisung von BCDH-Stilen aus der Stildatenbank und Export nach GeoPackage/GeoJSON.
- Prüft Eingaben und startet die Extraktions-Tasks thread-sicher.

### 3. `map_tools.py` (Karteninteraktion auf dem Canvas)
- **`PolygonRoiMapTool`**:
  - Ermöglicht das präzise Einzeichnen unregelmäßiger Siedlungsumringe per Mausklick.
  - Live-Vorschau mittels transparenter `QgsRubberBand` mit hochkontrastierender Umrandung.
  - *Steuerung*: Linksklick = Punkt setzen, Mausbewegung = Vorschau, Rechtsklick/Doppelklick = Abschließen, Rücktaste = Punkt löschen, Esc = Abbrechen.
- **`PipetteMapTool`**:
  - Klick auf das historische Kartenblatt tastet Farbwerte (RGB/LAB) pixelgenau ab und befüllt Schwellenwert-Felder.

### 4. `tasks.py` (Asynchrones Multithreading via `QgsTask`)
- **`BuildingExtractionTask`**:
  - Liest das Rasterfenster des ROI via GDAL aus.
  - Führt die Bildverarbeitungs-Pipeline im Hintergrundthread aus (kein Einfrieren der QGIS-Oberfläche).
  - Wandelt segmentierte Masken in Shapely-Geometrien um und regularisiert sie.
  - Erzeugt im Hauptthread einen `QgsVectorLayer` (EPSG:25832) und fügt ihn mit Attributen (`building_id`, `area_sqm`, `confidence`, `type`) direkt zum QGIS-Projekt hinzu.
- **`LandUseExtractionTask`**:
  - Berechnet kompetitive Pixel-Klassifikation und erzeugt überlappungsfreie Landnutzungs-Multipolygone.

### 5. `symbology-style.db`
- SQLite-basierte QGIS Stildatenbank mit vorgefertigten Signaturen und Farbschemata im Stil historischer Atlanten (Gebäude in Ziegelrot/Grafit, Innenhöfe gestrichelt, Gewässer in Cyan/Aquarellblau, Wald in Olivegrün).

---

## 🛠️ Lokale Entwicklung & Installation

### Live-Link via Directory Junction / Symlink
Damit Quellcode-Änderungen sofort in QGIS wirksam werden (ohne ständiges manuelles Kopieren), nutze das Skript `install_to_qgis.py`:

```bash
# Aus dem Repository-Hauptverzeichnis:
python install_to_qgis.py
```

Das Skript:
1. Erkennt das aktive QGIS-Profilverzeichnis unter Windows (`%APPDATA%/QGIS/QGIS3/profiles/default/python/plugins`), Linux oder macOS.
2. Erstellt eine symbolische Verknüpfung (`mklink /J` bzw. `os.symlink`) auf den Ordner `tranchot_qgis_plugin`.
3. Prüft, ob benötigte Python-Pakete (`opencv-python`, `shapely`, `numpy`) in der QGIS-Python-Umgebung installiert sind.

### Schneller Entwicklungszyklus (QGIS Plugin Reloader)
1. Installiere in QGIS die Erweiterung **Plugin Reloader** (über *Erweiterungen* -> *Erweiterungen verwalten und installieren*).
2. Konfiguriere `HistMap Extractor` im Reloader.
3. Nach jeder Code-Änderung in Python einfach die Taste **F5** (bzw. Reloader-Button) drücken, um das Plugin in laufendem QGIS neu zu laden.

---

## 🔌 Abhängigkeiten & Backend-Anbindung

Das Plugin greift auf die Verarbeitungsmodule des Hauptpakets `tranchot_extractor` zu:
- `tranchot_extractor.config`: Konfigurationsobjekte (`BuildingConfig`, `LandUseConfig`)
- `tranchot_extractor.extractors.building_extractor`: Orthogonale Vektorisierung & Hofsubtraktion
- `tranchot_extractor.extractors.landuse_extractor`: Spektrale Textur-Segmentierung
- `tranchot_extractor.preprocessing.color_enhancer`: LAB-Farbnormalisierung

Die Modulauflösung erfolgt in `tasks.py` und `dockwidget.py` dynamisch relativ zum Repository-Pfad oder über die installierte `pip`-Entwicklungsumgebung (`pip install -e .`).
