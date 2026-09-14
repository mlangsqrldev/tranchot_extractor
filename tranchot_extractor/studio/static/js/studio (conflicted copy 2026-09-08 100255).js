/**
 * Tranchot WebGIS Studio — Modern High-Performance Leaflet GPU Engine
 * Dual Bright/Dark Theme, Rock-Solid 60 FPS Zoom/Pan, Interactive Stamp Land-Use Sampling,
 * 90-Degree Orthogonal Building Extraction (Courtyards Preserved), and GIS Export.
 */

class TranchotWebGISStudio {
  constructor() {
    this.map = null;
    this.imageOverlay = null;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.imageLoaded = false;
    this.imageMeta = {};

    // Tool & Class State
    this.currentTool = "stamp"; // 'stamp', 'pan'
    this.activeClassId = "forest";
    this.activeClassLabel = "🌲 Wald";
    this.activeClassColor = "#16a34a";
    this.stampRadius = 22;

    // Theme (Bright by default)
    this.theme = localStorage.getItem("tranchot_theme") || "light";

    // Vector Layers
    this.layers = {
      forest: L.featureGroup(),
      meadow: L.featureGroup(),
      water: L.featureGroup(),
      vineyard: L.featureGroup(),
      gravel: L.featureGroup(),
      garden: L.featureGroup(),
      building: L.featureGroup(),
      stampMarkers: L.featureGroup(),
    };

    // Stamped Circles for Canvas rendering
    this.stampsByClass = {
      forest: [],
      meadow: [],
      water: [],
      vineyard: [],
      gravel: [],
      garden: [],
    };

    this.allExtractedFeatures = [];

    this.applyTheme(this.theme);
    this.initMap();
    this.setupUI();
    this.setupKeyboardShortcuts();
    this.loadPresets();
  }

  // ==========================================
  // Coordinate Conversions (Pixel <-> LatLng)
  // ==========================================
  pixelToLatLng(x, y) {
    return [this.imageHeight - y, x];
  }

  latLngToPixel(lat, lng) {
    return {
      x: Math.round(lng),
      y: Math.round(this.imageHeight - lat),
    };
  }

  // ==========================================
  // Leaflet Map Initialization (60 FPS GPU)
  // ==========================================
  initMap() {
    this.map = L.map("map", {
      crs: L.CRS.Simple,
      minZoom: -5,
      maxZoom: 5,
      zoomSnap: 0.1,
      zoomDelta: 0.25,
      wheelPxPerZoomLevel: 60,
      zoomControl: true,
      attributionControl: false,
      preferCanvas: true,
    });

    // Add all layer groups to map
    for (const group of Object.values(this.layers)) {
      group.addTo(this.map);
    }

    // Live cursor stamp ring
    this.cursorRing = L.circle([0, 0], {
      radius: this.stampRadius,
      color: this.activeClassColor,
      weight: 1.5,
      dashArray: "4, 4",
      fillColor: this.activeClassColor,
      fillOpacity: 0.25,
      interactive: false,
    });

    // Cursor tracking
    this.map.on("mousemove", (e) => {
      if (!this.imageLoaded) return;
      const px = this.latLngToPixel(e.latlng.lat, e.latlng.lng);
      document.getElementById("statusCoords").innerText = `X: ${px.x}, Y: ${px.y} px`;

      if (this.currentTool === "stamp" && px.x >= 0 && px.y >= 0 && px.x <= this.imageWidth && px.y <= this.imageHeight) {
        this.cursorRing.setLatLng(e.latlng);
        this.cursorRing.setRadius(this.stampRadius);
        this.cursorRing.setStyle({ color: this.activeClassColor, fillColor: this.activeClassColor });
        if (!this.map.hasLayer(this.cursorRing)) {
          this.cursorRing.addTo(this.map);
        }
      } else {
        if (this.map.hasLayer(this.cursorRing)) {
          this.map.removeLayer(this.cursorRing);
        }
      }
    });

    this.map.on("mouseout", () => {
      if (this.map.hasLayer(this.cursorRing)) {
        this.map.removeLayer(this.cursorRing);
      }
    });

    // Wheel radius adjustment (Ctrl/Shift+Wheel)
    const mapContainer = document.getElementById("map");
    if (mapContainer) {
      mapContainer.addEventListener("wheel", (e) => {
        if (e.ctrlKey || e.shiftKey) {
          e.preventDefault();
          e.stopPropagation();
          const delta = e.deltaY < 0 ? 1 : -1;
          this.stampRadius = Math.max(2, Math.min(120, this.stampRadius + delta * 2));
          const slider = document.getElementById("sliderStampRadius");
          if (slider) slider.value = this.stampRadius;
          const lbl = document.getElementById("lblStampRadiusVal");
          if (lbl) lbl.innerText = `${this.stampRadius} px (Mausrad: 2–120 px)`;
          if (this.map.hasLayer(this.cursorRing)) {
            this.cursorRing.setRadius(this.stampRadius);
          }
        }
      }, { passive: false });
    }

    this.map.on("zoomend", () => {
      const zoom = Math.round(Math.pow(2, this.map.getZoom()) * 100);
      document.getElementById("statusZoom").innerText = `Zoom: ${zoom}%`;
    });

    // Canvas click dispatcher
    this.map.on("click", (e) => this.handleMapClick(e));
  }

  // ==========================================
  // Theme Toggle (Bright / Dark)
  // ==========================================
  applyTheme(theme) {
    this.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("tranchot_theme", theme);
    
    const btn = document.getElementById("btnThemeToggle");
    if (btn) {
      btn.innerText = theme === "light" ? "☀️ Hell" : "🌙 Dunkel";
    }
  }

  toggleTheme() {
    const nextTheme = this.theme === "light" ? "dark" : "light";
    this.applyTheme(nextTheme);
  }

  // ==========================================
  // Map Clicks & Tool Handlers
  // ==========================================
  handleMapClick(e) {
    if (!this.imageLoaded) return;
    const px = this.latLngToPixel(e.latlng.lat, e.latlng.lng);
    if (px.x < 0 || px.y < 0 || px.x > this.imageWidth || px.y > this.imageHeight) return;

    if (this.currentTool === "stamp") {
      this.executeStampSample(px.x, px.y);
    }
  }

  // ==========================================
  // Stempel (Color & Texture Sampling)
  // ==========================================
  async executeStampSample(cx, cy) {
    this.showStatus(`🖌️ Sampele Nuance für ${this.activeClassLabel} bei (${cx}, ${cy})...`);

    try {
      const res = await fetch("/api/sample_stamp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_id: this.activeClassId,
          cx: cx,
          cy: cy,
          radius: this.stampRadius,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        this.showStatus(`⚠️ ${err.detail || "Stempelfehler"}`, true);
        return;
      }

      const data = await res.json();
      const stamp = data.stamp;

      this.stampsByClass[this.activeClassId].push(stamp);
      this.renderStampOnMap(stamp, this.activeClassId, this.activeClassLabel);
      this.refreshStampChipsUI();

      this.showStatus(
        `✅ Stempel #${stamp.stamp_id} für '${this.activeClassLabel}' erfasst! (${stamp.distilled_pixels}/${stamp.raw_pixels} px reines Pigment)`
      );
    } catch (e) {
      this.showStatus(`Netzwerkfehler: ${e.message}`, true);
    }
  }

  renderStampOnMap(stamp, classId, label) {
    const center = this.pixelToLatLng(stamp.cx, stamp.cy);

    const circle = L.circle(center, {
      radius: stamp.radius,
      color: stamp.hex_color,
      weight: 2,
      fillColor: stamp.hex_color,
      fillOpacity: 0.4,
    });

    circle.bindTooltip(`⭘ ${label} #${stamp.stamp_id}`, {
      permanent: true,
      direction: "top",
      className: "leaflet-stamp-badge",
      offset: [0, -stamp.radius],
    });

    circle.addTo(this.layers.stampMarkers);
  }

  renderAllStamps() {
    this.layers.stampMarkers.clearLayers();
    const classLabels = {
      forest: "🌲 Wald",
      meadow: "🌿 Wiese",
      water: "💧 Gewässer",
      vineyard: "🍇 Weinberge",
      gravel: "🟠 Kies",
      garden: "🟨 Gärten",
    };

    for (const [cid, stamps] of Object.entries(this.stampsByClass)) {
      const lbl = classLabels[cid] || cid;
      for (const s of stamps) {
        this.renderStampOnMap(s, cid, lbl);
      }
    }
  }

  refreshStampChipsUI() {
    const list = document.getElementById("stampChipsList");
    list.innerHTML = "";

    const stamps = this.stampsByClass[this.activeClassId] || [];
    document.getElementById("activeClassStatus").innerText = `${stamps.length} Stempel aktiv`;

    if (stamps.length === 0) {
      list.innerHTML = `<span class="empty-chips-msg">Noch keine Stempel. Klicke 3–10 Stellen auf der Karte an.</span>`;
      return;
    }

    stamps.forEach((s, idx) => {
      const chip = document.createElement("div");
      chip.className = "stamp-chip";
      chip.innerHTML = `
        <span class="stamp-chip-dot" style="background: ${s.hex_color};"></span>
        <span>#${s.stamp_id}</span>
        <span class="stamp-chip-del" title="Diesen Stempel löschen">✕</span>
      `;
      chip.querySelector(".stamp-chip-del").onclick = (e) => {
        e.stopPropagation();
        this.removeStamp(this.activeClassId, idx);
      };
      list.appendChild(chip);
    });
  }

  async removeStamp(classId, index) {
    try {
      await fetch("/api/remove_stamp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: classId, stamp_index: index }),
      });
      this.stampsByClass[classId].splice(index, 1);
      this.renderAllStamps();
      this.refreshStampChipsUI();
      this.showStatus(`🗑️ Stempel entfernt.`);
    } catch (e) {
      this.showStatus(`Fehler beim Löschen: ${e.message}`, true);
    }
  }

  async clearClassStamps(classId) {
    try {
      await fetch("/api/clear_stamps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: classId }),
      });
      this.stampsByClass[classId] = [];
      this.renderAllStamps();
      this.refreshStampChipsUI();
      this.showStatus(`🗑️ Alle Stempel für '${this.activeClassLabel}' geleert.`);
    } catch (e) {
      this.showStatus(`Fehler: ${e.message}`, true);
    }
  }

  // ==========================================
  // Building Extraction (`🏛️ Gebäude extrahieren`)
  // ==========================================
  async runBuildingExtraction() {
    this.showStatus("🏛️ Extrahiere 90°-orthogonale Gebäude & Hofanlagen (Courtyards erhalten)...");

    const minArea = parseInt(document.getElementById("sliderMinBldgArea")?.value || 35);
    const ortho = document.getElementById("chkOrthoRegularization") ? document.getElementById("chkOrthoRegularization").checked : true;

    try {
      const res = await fetch("/api/extract_buildings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          min_area_px: minArea,
          max_area_px: 10000.0,
          orthogonal_regularization: ortho,
        }),
      });

      if (!res.ok) throw new Error("Fehler bei Gebäudeextraktion");

      const geojson = await res.json();
      this.layers.building.clearLayers();

      const bldgFeatures = geojson.features || [];
      for (const feat of bldgFeatures) {
        const leafletLayer = L.geoJSON(feat, {
          coordsToLatLng: (coords) => this.pixelToLatLng(coords[0], coords[1]),
          renderer: L.svg({ padding: 0.5 }),
          style: {
            color: "#ef4444",
            weight: 1.8,
            opacity: 0.95,
            fillColor: "#ef4444",
            fillOpacity: 0.85,
            fillRule: "evenodd",
          },
        });

        leafletLayer.bindTooltip(
          `<strong>🏛️ Gebäude</strong><br>Fläche: ${feat.properties.area_px} px²`,
          { sticky: true }
        );

        leafletLayer.addTo(this.layers.building);
      }

      // Add to exported list
      this.allExtractedFeatures = this.allExtractedFeatures.filter(f => f.properties.class_id !== "building").concat(bldgFeatures);

      if (document.getElementById("cntBuildings")) {
        document.getElementById("cntBuildings").innerText = bldgFeatures.length;
      }

      this.showStatus(`✅ ${bldgFeatures.length} 90°-orthogonale Gebäude extrahiert!`);
    } catch (e) {
      this.showStatus(`Gebäudefehler: ${e.message}`, true);
    }
  }

  // ==========================================
  // Land-Use Extraction (`⚡ Flächen berechnen`)
  // ==========================================
  async runLandUseExtraction() {
    this.showStatus("⚡ Berechne gelernten Flächenbestand über alle Klassen (kompetitiv, 0 Überlappung)...");

    try {
      const res = await fetch("/api/extract_landuse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!res.ok) throw new Error(`Serverfehler (${res.status})`);

      const geojson = await res.json();

      // Clear previous land-use vector layers (keep buildings)
      this.layers.forest.clearLayers();
      this.layers.meadow.clearLayers();
      this.layers.water.clearLayers();
      this.layers.vineyard.clearLayers();
      this.layers.gravel.clearLayers();
      this.layers.garden.clearLayers();

      const landuseFeatures = geojson.features || [];
      this.allExtractedFeatures = this.allExtractedFeatures.filter(f => f.properties.class_id === "building").concat(landuseFeatures);

      let counts = { forest: 0, meadow: 0, water: 0, vineyard: 0, gravel: 0, garden: 0 };

      for (const feat of landuseFeatures) {
        const cid = feat.properties.class_id || "forest";
        counts[cid] = (counts[cid] || 0) + 1;

        const targetGroup = this.layers[cid] || this.layers.forest;
        const color = feat.properties.color || "#16a34a";

        const leafletLayer = L.geoJSON(feat, {
          coordsToLatLng: (coords) => this.pixelToLatLng(coords[0], coords[1]),
          style: {
            color: color,
            weight: 2,
            opacity: 0.9,
            fillColor: color,
            fillOpacity: 0.65,
          },
        });

        leafletLayer.bindTooltip(
          `<strong>${feat.properties.label}</strong><br>Fläche: ${feat.properties.area_px} px²`,
          { sticky: true }
        );

        leafletLayer.addTo(targetGroup);
      }

      if (document.getElementById("cntForest")) document.getElementById("cntForest").innerText = counts.forest;
      if (document.getElementById("cntMeadow")) document.getElementById("cntMeadow").innerText = counts.meadow;
      if (document.getElementById("cntWater")) document.getElementById("cntWater").innerText = counts.water;
      if (document.getElementById("cntVineyard")) document.getElementById("cntVineyard").innerText = counts.vineyard;
      if (document.getElementById("cntGravel")) document.getElementById("cntGravel").innerText = counts.gravel;
      if (document.getElementById("cntGarden")) document.getElementById("cntGarden").innerText = counts.garden;

      document.getElementById("statusPolyCount").innerText = `${this.allExtractedFeatures.length} Flächen geladen`;
      this.showStatus(
        `✅ Fertig in ${geojson.elapsed_seconds}s: ${geojson.total_polygons} Flächen über alle Klassen berechnet!`
      );
    } catch (e) {
      this.showStatus(`Extraktionsfehler: ${e.message}`, true);
    }
  }

  // ==========================================
  // Image Enhancement & Normalization
  // ==========================================
  async applyEnhancement(params) {
    this.showStatus("✨ Wende Weißabgleich & Bildfilter an...");

    try {
      const res = await fetch("/api/enhance_map", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      if (!res.ok) throw new Error("Fehler beim Bildfilter");

      const data = await res.json();
      this.updateImageOverlay(data.image_data);
      this.showStatus("✅ Bildfilter erfolgreich angewendet!");
    } catch (e) {
      this.showStatus(`Fehler: ${e.message}`, true);
    }
  }

  updateImageOverlay(dataUrl) {
    const bounds = [
      [0, 0],
      [this.imageHeight, this.imageWidth],
    ];

    if (this.imageOverlay) {
      this.map.removeLayer(this.imageOverlay);
      this.imageOverlay = null;
    }

    this.imageOverlay = L.imageOverlay(dataUrl, bounds, {
      opacity: 1.0,
      interactive: false,
    }).addTo(this.map);

    this.map.fitBounds(bounds, { padding: [10, 10] });
    this.map.setMaxBounds([
      [-this.imageHeight * 0.2, -this.imageWidth * 0.2],
      [this.imageHeight * 1.2, this.imageWidth * 1.2],
    ]);
  }

  // ==========================================
  // Preset & File Loading
  // ==========================================
  async loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      const select = document.getElementById("presetSelect");
      select.innerHTML = "";

      let defaultPath = null;
      for (const p of data.presets) {
        const opt = document.createElement("option");
        opt.value = p.path;
        opt.innerText = p.name;
        select.appendChild(opt);
        if (!defaultPath || p.name.includes("Nickenich")) {
          defaultPath = p.path;
        }
      }

      if (defaultPath) {
        select.value = defaultPath;
        this.loadImage(defaultPath);
      }
    } catch (e) {
      console.error("Presets loading error:", e);
    }
  }

  async loadImage(path) {
    this.showStatus(`Lade Kartenblatt: ${path}...`);
    try {
      const fd = new FormData();
      fd.append("preset_path", path);

      const res = await fetch("/api/load_image", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error("Fehler beim Laden des Bildes");

      const data = await res.json();
      this.imageWidth = data.width;
      this.imageHeight = data.height;
      this.imageLoaded = true;
      this.imageMeta = data.metadata || {};

      // Reset all vector layers, stamps, and counters
      for (const group of Object.values(this.layers)) {
        group.clearLayers();
      }
      for (const k of Object.keys(this.stampsByClass)) {
        this.stampsByClass[k] = [];
      }
      this.allExtractedFeatures = [];
      this.refreshStampChipsUI();

      ["cntForest", "cntMeadow", "cntWater", "cntBuildings", "cntVineyard", "cntGravel", "cntGarden"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = "0";
      });
      const polyCnt = document.getElementById("statusPolyCount");
      if (polyCnt) polyCnt.innerText = "0 Flächen geladen";

      this.updateImageOverlay(data.image_data);

      const bounds = [
        [0, 0],
        [this.imageHeight, this.imageWidth],
      ];
      this.map.fitBounds(bounds, { padding: [10, 10] });

      this.showStatus(`🗺️ Geladen: ${data.filename} (${this.imageWidth}×${this.imageHeight} px).`);
    } catch (e) {
      this.showStatus(`Fehler beim Laden: ${e.message}`, true);
    }
  }

  // ==========================================
  // Exports
  // ==========================================
  async exportData(format) {
    this.showStatus(`💾 Bereite ${format.toUpperCase()}-Export vor...`);

    const annotations = [];
    let idCounter = 1;

    for (const feat of this.allExtractedFeatures) {
      const coords = feat.geometry.coordinates;
      annotations.push({
        id: idCounter++,
        label: feat.properties.label || "Landnutzung",
        color: feat.properties.color || "#16a34a",
        type: "polygon",
        points: (coords && coords.length === 1) ? coords[0] : coords,
        properties: feat.properties,
      });
    }

    try {
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations: annotations, format: format }),
      });

      if (!res.ok) throw new Error("Export fehlgeschlagen");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = format === "shapefile_zip" ? "tranchot_shapefiles.zip" : `tranchot_layers.${format === "gpkg" ? "gpkg" : "geojson"}`;
      document.body.appendChild(a);
      a.click();
      a.remove();

      this.showStatus(`✅ Export erfolgreich heruntergeladen!`);
    } catch (e) {
      this.showStatus(`Exportfehler: ${e.message}`, true);
    }
  }

  // ==========================================
  // UI & Event Bindings
  // ==========================================
  setupUI() {
    // Theme Toggle
    document.getElementById("btnThemeToggle").onclick = () => this.toggleTheme();

    // Tool buttons
    const toolBtns = document.querySelectorAll(".tool-btn");
    toolBtns.forEach((btn) => {
      btn.onclick = () => {
        toolBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.currentTool = btn.dataset.tool;

        const bannerText = {
          stamp: "🖌️ Stempel-Modus aktiv: Klicke auf Wald- oder Wiesenflächen, um Farbtöne aufzunehmen.",
          pan: "🖐️ Pan-Modus: Klicke und ziehe, um die Karte frei zu bewegen.",
        }[this.currentTool] || "";

        document.getElementById("bannerText").innerText = bannerText;
      };
    });

    // Class selection tabs
    const classTabs = document.querySelectorAll(".class-tab");
    classTabs.forEach((tab) => {
      tab.onclick = () => {
        classTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        this.activeClassId = tab.dataset.class;
        this.activeClassLabel = tab.querySelector("span:last-child").innerText;
        this.activeClassColor = tab.style.getPropertyValue("--c");

        document.getElementById("activeClassName").innerText = this.activeClassLabel;
        document.getElementById("activeClassSwatch").style.background = this.activeClassColor;
        if (this.cursorRing) {
          this.cursorRing.setStyle({ color: this.activeClassColor, fillColor: this.activeClassColor });
        }
        this.refreshStampChipsUI();
      };
    });

    // Stamp radius slider
    const sliderRadius = document.getElementById("sliderStampRadius");
    sliderRadius.oninput = (e) => {
      this.stampRadius = parseInt(e.target.value);
      document.getElementById("lblStampRadiusVal").innerText = `${this.stampRadius} px (Mausrad: 2–120 px)`;
      if (this.cursorRing) {
        this.cursorRing.setRadius(this.stampRadius);
      }
    };

    // Calculate Land Use button
    document.getElementById("btnRunLandUseExtraction").onclick = () => this.runLandUseExtraction();
    document.getElementById("btnExtractLandUse").onclick = () => this.runLandUseExtraction();

    // Building Extraction button
    if (document.getElementById("btnExtractBuildings")) {
      document.getElementById("btnExtractBuildings").onclick = () => this.runBuildingExtraction();
    }
    if (document.getElementById("btnExtractBuildingsTop")) {
      document.getElementById("btnExtractBuildingsTop").onclick = () => this.runBuildingExtraction();
    }

    // Min Building Area Slider
    const sliderMinBldg = document.getElementById("sliderMinBldgArea");
    if (sliderMinBldg) {
      sliderMinBldg.oninput = (e) => {
        document.getElementById("lblMinBldgArea").innerText = `${e.target.value} px²`;
      };
    }

    // Clear stamps button
    document.getElementById("btnClearCurrentStamps").onclick = () => this.clearClassStamps(this.activeClassId);

    // Normalization & Enhancement buttons
    document.getElementById("btnPaper100").onclick = () => {
      document.getElementById("sliderDeyellow").value = 90;
      document.getElementById("lblDeyellowVal").innerText = "90%";
      document.getElementById("sliderVibrance").value = 175;
      document.getElementById("lblVibranceVal").innerText = "1.75x";
      document.getElementById("chkInkBlack").checked = true;
      this.applyEnhancement({
        deyellow_strength: 0.9,
        vibrance: 1.75,
        contrast: 1.0,
        ink_blackening: true,
      });
    };

    document.getElementById("btnQuickNormalize").onclick = () => {
      document.getElementById("btnPaper100").click();
    };

    document.getElementById("btnApplyEnhance").onclick = () => {
      this.applyEnhancement({
        deyellow_strength: parseInt(document.getElementById("sliderDeyellow").value) / 100.0,
        vibrance: parseInt(document.getElementById("sliderVibrance").value) / 100.0,
        contrast: parseInt(document.getElementById("sliderContrast").value) / 100.0,
        ink_blackening: document.getElementById("chkInkBlack").checked,
      });
    };

    // Sliders event listeners
    document.getElementById("sliderDeyellow").oninput = (e) => {
      document.getElementById("lblDeyellowVal").innerText = `${e.target.value}%`;
    };
    document.getElementById("sliderVibrance").oninput = (e) => {
      document.getElementById("lblVibranceVal").innerText = `${(parseInt(e.target.value) / 100).toFixed(2)}x`;
    };
    document.getElementById("sliderContrast").oninput = (e) => {
      document.getElementById("lblContrastVal").innerText = `${(parseInt(e.target.value) / 100).toFixed(2)}x`;
    };

    // Layer Opacity & Visibility Listeners
    const bindLayerControls = (cid, chkId, sliderId) => {
      const chk = document.getElementById(chkId);
      const slider = document.getElementById(sliderId);
      if (chk) {
        chk.onchange = () => {
          if (this.layers[cid]) {
            if (chk.checked) {
              this.layers[cid].addTo(this.map);
            } else {
              this.map.removeLayer(this.layers[cid]);
            }
          }
        };
      }
      if (slider) {
        slider.oninput = (e) => {
          const val = parseInt(e.target.value) / 100.0;
          if (this.layers[cid]) {
            this.layers[cid].eachLayer((l) => {
              if (l.setStyle) l.setStyle({ fillOpacity: val });
            });
          }
        };
      }
    };

    bindLayerControls("forest", "chkLayerForest", "opacityForest");
    bindLayerControls("meadow", "chkLayerMeadow", "opacityMeadow");
    bindLayerControls("water", "chkLayerWater", "opacityWater");
    bindLayerControls("building", "chkLayerBuildings", "opacityBuildings");
    bindLayerControls("vineyard", "chkLayerVineyard", "opacityVineyard");
    bindLayerControls("gravel", "chkLayerGravel", "opacityGravel");
    bindLayerControls("garden", "chkLayerGarden", "opacityGarden");

    // Delete single layer buttons
    document.querySelectorAll(".btn-del-layer").forEach((btn) => {
      btn.onclick = (e) => {
        e.stopPropagation();
        const layerId = btn.dataset.layer;
        if (this.layers[layerId]) {
          this.layers[layerId].clearLayers();
        }
        this.allExtractedFeatures = this.allExtractedFeatures.filter((f) => f.properties.class_id !== layerId);
        const cntMap = {
          forest: "cntForest",
          meadow: "cntMeadow",
          water: "cntWater",
          building: "cntBuildings",
          vineyard: "cntVineyard",
          gravel: "cntGravel",
          garden: "cntGarden",
        };
        if (cntMap[layerId] && document.getElementById(cntMap[layerId])) {
          document.getElementById(cntMap[layerId]).innerText = "0";
        }
        const polyCnt = document.getElementById("statusPolyCount");
        if (polyCnt) polyCnt.innerText = `${this.allExtractedFeatures.length} Flächen geladen`;
        this.showStatus(`🗑️ Ebene '${layerId}' erfolgreich gelöscht.`);
      };
    });

    // Clear all layers
    const btnClearAll = document.getElementById("btnClearAllLayers");
    if (btnClearAll) {
      btnClearAll.onclick = () => {
        for (const [k, group] of Object.entries(this.layers)) {
          if (k !== "stampMarkers") group.clearLayers();
        }
        this.allExtractedFeatures = [];
        ["cntForest", "cntMeadow", "cntWater", "cntBuildings", "cntVineyard", "cntGravel", "cntGarden"].forEach((id) => {
          const el = document.getElementById(id);
          if (el) el.innerText = "0";
        });
        const polyCnt = document.getElementById("statusPolyCount");
        if (polyCnt) polyCnt.innerText = "0 Flächen geladen";
        this.showStatus("🗑️ Alle Vektor-Ebenen geleert.");
      };
    }

    // Export Menu
    const btnExportMenu = document.getElementById("btnExportMenu");
    const exportDropdown = document.getElementById("exportDropdown");
    btnExportMenu.onclick = (e) => {
      e.stopPropagation();
      exportDropdown.classList.toggle("show");
    };
    window.onclick = () => exportDropdown.classList.remove("show");

    document.getElementById("exportGeoJSON").onclick = (e) => {
      e.preventDefault();
      this.exportData("geojson");
    };
    document.getElementById("exportShapefile").onclick = (e) => {
      e.preventDefault();
      this.exportData("shapefile_zip");
    };
    document.getElementById("exportGPKG").onclick = (e) => {
      e.preventDefault();
      this.exportData("gpkg");
    };

    // Preset selection change
    document.getElementById("presetSelect").onchange = (e) => {
      if (e.target.value) this.loadImage(e.target.value);
    };

    // File Upload
    document.getElementById("btnUploadMap").onclick = () => document.getElementById("fileInputMap").click();
    document.getElementById("fileInputMap").onchange = (e) => {
      if (e.target.files && e.target.files[0]) {
        const fd = new FormData();
        fd.append("file", e.target.files[0]);
        this.showStatus(`Lade Datei ${e.target.files[0].name}...`);
        fetch("/api/load_image", { method: "POST", body: fd })
          .then((r) => r.json())
          .then((data) => {
            this.imageWidth = data.width;
            this.imageHeight = data.height;
            this.imageLoaded = true;
            this.updateImageOverlay(data.image_data);
            this.showStatus(`🗺️ Geladen: ${data.filename}`);
          });
      }
    };

    // Fit view button
    document.getElementById("btnFitView").onclick = () => {
      if (this.imageLoaded) {
        this.map.fitBounds([
          [0, 0],
          [this.imageHeight, this.imageWidth],
        ], { padding: [10, 10] });
      }
    };

    // Accordion card toggles
    document.querySelectorAll(".card-header").forEach((header) => {
      header.onclick = () => {
        header.parentElement.classList.toggle("active");
      };
    });
  }

  setupKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "s" || e.key === "S") document.getElementById("toolStamp").click();
      if (e.key === "h" || e.key === "H") document.getElementById("toolPan").click();
    });
  }

  showStatus(msg, isError = false) {
    const el = document.getElementById("statusMessage");
    if (el) {
      el.innerText = msg;
      el.style.color = isError ? "#ef4444" : "var(--text-muted)";
    }
  }
}

// Instantiate on load
document.addEventListener("DOMContentLoaded", () => {
  window.app = new TranchotWebGISStudio();
});
