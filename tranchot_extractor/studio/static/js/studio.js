/**
 * Tranchot Label Studio - Leaflet WebGIS & SAM AI Interactive Engine
 * Precise Pixel-to-LatLng Coordinates with Instant GPU SAM Preview
 */

class LeafletLabelStudio {
  constructor() {
    this.map = null;
    this.imageOverlay = null;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.imageLoaded = false;
    this.imageMeta = {};

    // Tool & Mode
    this.currentTool = "sam"; // 'sam', 'draw_poly', 'edit', 'pan'
    this.granularity = "compact"; // 'compact', 'medium', 'large'
    this.activeClass = {
      id: "building",
      name: "Gebäude",
      color: "#ef4444",
      type: "polygon"
    };

    // Layer Groups
    this.annotationsLayer = null;
    this.samCandidateLayer = null;
    this.samMarkersLayer = null;

    // Prompt & State
    this.positivePoints = []; // [[x, y], ...] in image pixel space
    this.negativePoints = []; // [[x, y], ...] in image pixel space
    this.currentCandidate = null;
    this.annotations = [];
    this.selectedAnnotationId = null;

    this.initMap();
    this.setupUI();
    this.setupKeyboardShortcuts();
    this.loadPresets();
  }

  // ==========================================
  // Coordinate Transformations (Pixel <-> LatLng)
  // ==========================================

  pixelToLatLng(x, y) {
    return [this.imageHeight - y, x];
  }

  latLngToPixel(lat, lng) {
    return {
      x: Math.round(lng),
      y: Math.round(this.imageHeight - lat)
    };
  }

  // ==========================================
  // Leaflet Map Initialization
  // ==========================================

  initMap() {
    this.map = L.map("map", {
      crs: L.CRS.Simple,
      minZoom: -4,
      maxZoom: 6,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      wheelPxPerZoomLevel: 60,
      zoomControl: true,
      attributionControl: false,
    });

    this.annotationsLayer = L.featureGroup().addTo(this.map);
    this.samCandidateLayer = L.featureGroup().addTo(this.map);
    this.samMarkersLayer = L.featureGroup().addTo(this.map);

    // Track Cursor
    this.map.on("mousemove", (e) => {
      if (!this.imageLoaded) return;
      const px = this.latLngToPixel(e.latlng.lat, e.latlng.lng);
      document.getElementById("statusCoords").innerText = `X: ${px.x}, Y: ${px.y} px`;
    });

    this.map.on("zoomend", () => {
      const zoom = Math.round(Math.pow(2, this.map.getZoom()) * 100);
      document.getElementById("statusZoom").innerText = `Zoom: ${zoom}%`;
    });

    // Left Click: Positive Point / Tool action
    this.map.on("click", (e) => {
      if (this.currentTool !== "sam" || !this.imageLoaded) return;

      const px = this.latLngToPixel(e.latlng.lat, e.latlng.lng);
      if (px.x < 0 || px.y < 0 || px.x > this.imageWidth || px.y > this.imageHeight) return;

      if (e.originalEvent.shiftKey || e.originalEvent.button === 2) {
        // Negative Point
        this.negativePoints.push([px.x, px.y]);
        this.addMarker(px.x, px.y, "#ef4444", "-");
      } else {
        // Positive Point
        this.positivePoints.push([px.x, px.y]);
        this.addMarker(px.x, px.y, "#22c55e", "+");
      }

      this.requestSAMPrediction();
    });

    // Right Click: Negative Point
    this.map.on("contextmenu", (e) => {
      if (this.currentTool !== "sam" || !this.imageLoaded) return;
      const px = this.latLngToPixel(e.latlng.lat, e.latlng.lng);
      if (px.x >= 0 && px.y >= 0 && px.x <= this.imageWidth && px.y <= this.imageHeight) {
        this.negativePoints.push([px.x, px.y]);
        this.addMarker(px.x, px.y, "#ef4444", "-");
        this.requestSAMPrediction();
      }
    });

    // Geoman setup
    if (this.map.pm) {
      this.map.pm.setLang("de");
      this.map.on("pm:create", (e) => {
        const shape = e.layer;
        const latlngs = shape.getLatLngs()[0];
        const points = latlngs.map((ll) => {
          const px = this.latLngToPixel(ll.lat, ll.lng);
          return [px.x, px.y];
        });

        if (this.currentTool === "sam_box") {
          const x1 = Math.min(...points.map((p) => p[0]));
          const y1 = Math.min(...points.map((p) => p[1]));
          const x2 = Math.max(...points.map((p) => p[0]));
          const y2 = Math.max(...points.map((p) => p[1]));
          this.map.removeLayer(shape);
          this.requestSAMBoxPrediction([x1, y1, x2, y2]);
          return;
        }

        const newAnn = {
          id: Date.now(),
          label: this.activeClass.name,
          color: this.activeClass.color,
          type: "polygon",
          points: points,
          leafletLayer: shape,
          area: this.calcPolygonArea(points),
          properties: { manual_drawn: true }
        };

        shape.setStyle({
          color: this.activeClass.color,
          fillColor: this.activeClass.color,
          fillOpacity: 0.45,
          weight: 2,
        });

        shape.on("click", () => this.selectAnnotation(newAnn.id));
        this.annotations.push(newAnn);
        this.updateObjectList();
      });
    }
  }

  addMarker(px_x, px_y, color, symbol) {
    const latlng = this.pixelToLatLng(px_x, px_y);
    L.circleMarker(latlng, {
      radius: 7,
      fillColor: color,
      color: "#ffffff",
      weight: 2.5,
      opacity: 1,
      fillOpacity: 0.95,
    }).addTo(this.samMarkersLayer);
  }

  // ==========================================
  // SAM Inferenz Request & Rendering
  // ==========================================

  async requestSAMPrediction() {
    if (this.positivePoints.length === 0 && this.negativePoints.length === 0) {
      this.clearPrompts();
      return;
    }

    try {
      document.getElementById("statusMsg").innerText = "✨ SAM berechnet Segmentierung auf GPU...";
      const resp = await fetch("/api/sam_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          positive_points: this.positivePoints,
          negative_points: this.negativePoints,
          granularity: this.granularity,
          simplification: 0.015
        })
      });

      const data = await resp.json();
      this.samCandidateLayer.clearLayers();

      if (data.polygons && data.polygons.length > 0) {
        this.currentCandidate = data;

        data.polygons.forEach((poly, idx) => {
          // Convert all [x, y] polygon vertices to correct Leaflet [lat, lng]
          const latlngs = poly.points.map((pt) => this.pixelToLatLng(pt[0], pt[1]));

          const candidatePoly = L.polygon(latlngs, {
            color: "#38bdf8",
            fillColor: "#0284c7",
            fillOpacity: 0.55,
            weight: 3.5,
            dashArray: idx === 0 ? "6, 6" : "2, 4",
          }).addTo(this.samCandidateLayer);

          if (idx === 0) {
            this.updateBanner(true, data.iou_score, poly.area);
          }
        });

        document.getElementById("statusMsg").innerText = `✅ SAM Segmentierung (IoU: ${(data.iou_score * 100).toFixed(0)}%). Drücke Enter zum Speichern.`;
      } else {
        document.getElementById("statusMsg").innerText = "⚠️ SAM hat an diesem Punkt kein klares Objekt gefunden. Klicke erneut.";
      }
    } catch (err) {
      console.error("SAM Prediction Error:", err);
      document.getElementById("statusMsg").innerText = "SAM Fehler beim Verbinden zum Server.";
    }
  }

  async requestSAMBoxPrediction(box) {
    try {
      document.getElementById("statusMsg").innerText = "✨ SAM berechnet Bounding-Box...";
      const resp = await fetch("/api/sam_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bounding_box: box,
          granularity: this.granularity,
          simplification: 0.015
        })
      });

      const data = await resp.json();
      this.samCandidateLayer.clearLayers();

      if (data.polygons && data.polygons.length > 0) {
        this.currentCandidate = data;
        const poly = data.polygons[0];
        const latlngs = poly.points.map((pt) => this.pixelToLatLng(pt[0], pt[1]));

        L.polygon(latlngs, {
          color: "#38bdf8",
          fillColor: "#0284c7",
          fillOpacity: 0.55,
          weight: 3.5,
          dashArray: "6, 6"
        }).addTo(this.samCandidateLayer);

        this.updateBanner(true, data.iou_score, poly.area);
        document.getElementById("statusMsg").innerText = `✅ Gebäude in Box erkannt (${poly.area} px²). Drücke Enter zum Speichern.`;
      }

      // Keep draw rectangle active for next box
      if (this.currentTool === "sam_box" && this.map.pm) {
        this.map.pm.enableDraw("Rectangle", {
          snappingOption: false,
          templineStyle: { color: "#38bdf8", dashArray: [4, 4] },
          hintlineStyle: { color: "#38bdf8", dashArray: [4, 4] },
        });
      }
    } catch (err) {
      console.error("SAM Box Error:", err);
    }
  }

  commitSAMCandidate() {
    if (!this.currentCandidate || !this.currentCandidate.polygons || this.currentCandidate.polygons.length === 0) {
      return;
    }

    const poly = this.currentCandidate.polygons[0];
    const latlngs = poly.points.map((pt) => this.pixelToLatLng(pt[0], pt[1]));

    const layer = L.polygon(latlngs, {
      color: this.activeClass.color,
      fillColor: this.activeClass.color,
      fillOpacity: 0.45,
      weight: 2.5,
    }).addTo(this.annotationsLayer);

    const newAnn = {
      id: Date.now(),
      label: this.activeClass.name,
      color: this.activeClass.color,
      type: "polygon",
      points: poly.points,
      area: poly.area,
      leafletLayer: layer,
      properties: {
        iou_score: this.currentCandidate.iou_score,
        class_id: this.activeClass.id,
      }
    };

    layer.on("click", () => this.selectAnnotation(newAnn.id));

    this.annotations.push(newAnn);
    this.clearPrompts();
    this.updateObjectList();
    document.getElementById("statusMsg").innerText = `💾 Objekt gespeichert: #${this.annotations.length} ${newAnn.label}`;
  }

  clearPrompts() {
    this.positivePoints = [];
    this.negativePoints = [];
    this.currentCandidate = null;
    this.samCandidateLayer.clearLayers();
    this.samMarkersLayer.clearLayers();
    this.updateBanner(false);
  }

  updateBanner(active, iou = 0, area = 0) {
    const banner = document.getElementById("samActionBanner");
    if (active) {
      banner.classList.add("active");
      document.getElementById("bannerIou").innerText = `IoU: ${(iou * 100).toFixed(0)}%`;
      document.getElementById("bannerArea").innerText = `Fläche: ${Math.round(area)} px²`;
    } else {
      banner.classList.remove("active");
    }
  }

  // ==========================================
  // Object List & Layer Management
  // ==========================================

  selectAnnotation(id) {
    this.selectedAnnotationId = this.selectedAnnotationId === id ? null : id;
    this.annotations.forEach((ann) => {
      if (ann.leafletLayer) {
        if (ann.id === this.selectedAnnotationId) {
          ann.leafletLayer.setStyle({ color: "#ffffff", weight: 4, fillOpacity: 0.7 });
          if (ann.leafletLayer.getBounds) {
            this.map.panTo(ann.leafletLayer.getBounds().getCenter());
          }
        } else {
          ann.leafletLayer.setStyle({ color: ann.color, weight: 2.5, fillOpacity: 0.45 });
        }
      }
    });
    this.updateObjectList();
  }

  updateObjectList() {
    const listEl = document.getElementById("objectList");
    const countEl = document.getElementById("objCount");
    listEl.innerHTML = "";
    countEl.innerText = this.annotations.length;

    this.annotations.forEach((ann, idx) => {
      const item = document.createElement("div");
      item.className = `object-item ${this.selectedAnnotationId === ann.id ? "selected" : ""}`;
      item.innerHTML = `
        <div class="obj-left">
          <div class="obj-color-box" style="background: ${ann.color};"></div>
          <div>
            <div class="obj-name">#${idx + 1} ${ann.label}</div>
            <div class="obj-meta">${ann.area ? Math.round(ann.area) + " px²" : ann.text || "Objekt"}</div>
          </div>
        </div>
        <div class="obj-actions">
          <button class="icon-btn delete-btn" title="Löschen">🗑️</button>
        </div>
      `;

      item.addEventListener("click", (e) => {
        if (!e.target.classList.contains("delete-btn")) {
          this.selectAnnotation(ann.id);
        }
      });

      item.querySelector(".delete-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        if (ann.leafletLayer) {
          this.annotationsLayer.removeLayer(ann.leafletLayer);
        }
        this.annotations = this.annotations.filter((a) => a.id !== ann.id);
        this.updateObjectList();
      });

      listEl.appendChild(item);
    });
  }

  // ==========================================
  // Image Loading & Presets
  // ==========================================

  async loadPresets() {
    try {
      const resp = await fetch("/api/presets");
      const data = await resp.json();
      const select = document.getElementById("presetSelect");
      select.innerHTML = "";
      data.presets.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.path;
        opt.innerText = p.name;
        select.appendChild(opt);
      });

      if (data.presets.length > 0) {
        this.loadImage(data.presets[0].path);
      }
    } catch (err) {
      console.error("Failed to load presets:", err);
    }
  }

  async loadImage(path) {
    try {
      document.getElementById("statusMsg").innerText = "Lade Kartenblatt...";
      const formData = new FormData();
      formData.append("preset_path", path);

      const resp = await fetch("/api/load_image", {
        method: "POST",
        body: formData
      });

      const data = await resp.json();

      this.imageWidth = data.width;
      this.imageHeight = data.height;
      this.imageMeta = data.metadata;
      this.imageLoaded = true;

      // Clear existing layers
      if (this.imageOverlay) {
        this.map.removeLayer(this.imageOverlay);
      }
      this.annotationsLayer.clearLayers();
      this.samCandidateLayer.clearLayers();
      this.samMarkersLayer.clearLayers();
      this.annotations = [];
      this.clearPrompts();

      // In Leaflet CRS.Simple bounds: [[0, 0], [height, width]]
      const bounds = [[0, 0], [this.imageHeight, this.imageWidth]];
      this.imageOverlay = L.imageOverlay(data.image_data, bounds).addTo(this.map);

      this.map.setMaxBounds([[-1000, -1000], [this.imageHeight + 1000, this.imageWidth + 1000]]);
      this.map.fitBounds(bounds);

      document.getElementById("statusMsg").innerText = `Geladen: ${this.imageWidth} × ${this.imageHeight} px (${data.metadata.crs || "EPSG:3857"})`;
      this.updateObjectList();
    } catch (err) {
      document.getElementById("statusMsg").innerText = "Fehler beim Laden des Bildes.";
    }
  }

  // ==========================================
  // AI Assist / Auto-ML Functions
  // ==========================================

  async runAutoSAM() {
    document.getElementById("statusMsg").innerText = "🤖 SAM Auto-Assistent analysiert Baukörper...";
    try {
      const resp = await fetch("/api/sam_auto", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rgb_diff_thresh: 35, min_seed_area: 15 })
      });
      const data = await resp.json();
      if (data.polygons) {
        data.polygons.forEach((poly) => {
          const latlngs = poly.points.map((pt) => this.pixelToLatLng(pt[0], pt[1]));
          const layer = L.polygon(latlngs, {
            color: "#ef4444",
            fillColor: "#ef4444",
            fillOpacity: 0.45,
            weight: 2,
          }).addTo(this.annotationsLayer);

          const ann = {
            id: Date.now() + Math.random(),
            label: "Gebäude",
            color: "#ef4444",
            type: "polygon",
            points: poly.points,
            area: poly.area,
            leafletLayer: layer,
            properties: { auto_detected: true }
          };

          layer.on("click", () => this.selectAnnotation(ann.id));
          this.annotations.push(ann);
        });

        this.updateObjectList();
        document.getElementById("statusMsg").innerText = `✅ ${data.building_count} Gebäude automatisch mit SAM extrahiert!`;
      }
    } catch (err) {
      document.getElementById("statusMsg").innerText = "Fehler bei SAM Auto-Extraktion.";
    }
  }

  async runOCR() {
    document.getElementById("statusMsg").innerText = "🔍 OCR liest historische Toponyme...";
    try {
      const resp = await fetch("/api/ocr_predict", { method: "POST" });
      const data = await resp.json();
      if (data.toponyms) {
        data.toponyms.forEach((t) => {
          const latlngs = t.bbox.map((pt) => this.pixelToLatLng(pt[0], pt[1]));
          const layer = L.polygon(latlngs, {
            color: "#f59e0b",
            fillColor: "#f59e0b",
            fillOpacity: 0.35,
            weight: 2,
          }).addTo(this.annotationsLayer);

          layer.bindTooltip(t.text, { permanent: true, direction: "top", className: "ocr-tooltip" });

          const ann = {
            id: Date.now() + Math.random(),
            label: "Toponym",
            text: t.text,
            color: "#f59e0b",
            type: "bbox",
            points: t.bbox,
            leafletLayer: layer,
            properties: { confidence: t.confidence, category: t.category }
          };

          layer.on("click", () => this.selectAnnotation(ann.id));
          this.annotations.push(ann);
        });

        this.updateObjectList();
        document.getElementById("statusMsg").innerText = `✅ ${data.toponyms.length} Toponyme erkannt!`;
      }
    } catch (err) {
      document.getElementById("statusMsg").innerText = "Fehler bei OCR.";
    }
  }

  calcPolygonArea(points) {
    let area = 0;
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += points[i][0] * points[j][1];
      area -= points[j][0] * points[i][1];
    }
    return Math.abs(area) / 2.0;
  }

  // ==========================================
  // UI & Tool Handlers
  // ==========================================

  setupUI() {
    document.getElementById("presetSelect").addEventListener("change", (e) => {
      this.loadImage(e.target.value);
    });

    document.getElementById("btnFitView").addEventListener("click", () => {
      if (this.imageLoaded) {
        this.map.fitBounds([[0, 0], [this.imageHeight, this.imageWidth]]);
      }
    });

    const toolSAM = document.getElementById("toolSAM");
    const toolSAMBox = document.getElementById("toolSAMBox");
    const toolPoly = document.getElementById("toolPoly");
    const toolEdit = document.getElementById("toolEdit");
    const toolPan = document.getElementById("toolPan");

    const clearActiveTools = () => {
      [toolSAM, toolSAMBox, toolPoly, toolEdit, toolPan].forEach((b) => {
        if (b) b.classList.remove("active");
      });
      if (this.map.pm) {
        this.map.pm.disableDraw();
        this.map.pm.disableGlobalEditMode();
      }
    };

    toolSAM.addEventListener("click", () => {
      clearActiveTools();
      toolSAM.classList.add("active");
      this.currentTool = "sam";
      this.map.dragging.enable();
    });

    if (toolSAMBox) {
      toolSAMBox.addEventListener("click", () => {
        clearActiveTools();
        toolSAMBox.classList.add("active");
        this.currentTool = "sam_box";
        if (this.map.pm) {
          this.map.pm.enableDraw("Rectangle", {
            snappingOption: false,
            templineStyle: { color: "#38bdf8", dashArray: [4, 4] },
            hintlineStyle: { color: "#38bdf8", dashArray: [4, 4] },
          });
        }
      });
    }

    toolPoly.addEventListener("click", () => {
      clearActiveTools();
      toolPoly.classList.add("active");
      this.currentTool = "draw_poly";
      if (this.map.pm) {
        this.map.pm.enableDraw("Polygon", {
          snappingOption: true,
          templineStyle: { color: this.activeClass.color },
          hintlineStyle: { color: this.activeClass.color, dashArray: [5, 5] },
        });
      }
    });

    toolEdit.addEventListener("click", () => {
      clearActiveTools();
      toolEdit.classList.add("active");
      this.currentTool = "edit";
      if (this.map.pm) {
        this.map.pm.enableGlobalEditMode();
      }
    });

    toolPan.addEventListener("click", () => {
      clearActiveTools();
      toolPan.classList.add("active");
      this.currentTool = "pan";
      this.map.dragging.enable();
    });

    document.querySelectorAll(".class-chip").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        document.querySelectorAll(".class-chip").forEach((c) => c.classList.remove("active"));
        const target = e.currentTarget;
        target.classList.add("active");
        this.activeClass = {
          id: target.dataset.id,
          name: target.dataset.name,
          color: target.dataset.color,
          type: "polygon"
        };
      });
    });

    // Granularity Toggle buttons
    document.querySelectorAll(".gran-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".gran-btn").forEach((b) => b.classList.remove("active"));
        const target = e.currentTarget;
        target.classList.add("active");
        this.granularity = target.dataset.gran;
        if (this.positivePoints.length > 0) {
          this.requestSAMPrediction();
        }
      });
    });

    document.getElementById("btnCommitSAM").addEventListener("click", () => this.commitSAMCandidate());
    document.getElementById("btnCancelSAM").addEventListener("click", () => this.clearPrompts());
    document.getElementById("btnAutoSAM").addEventListener("click", () => this.runAutoSAM());
    document.getElementById("btnRunOCR").addEventListener("click", () => this.runOCR());

    document.getElementById("btnExport").addEventListener("click", () => {
      document.getElementById("exportModal").classList.add("active");
    });
    document.getElementById("btnCloseExportModal").addEventListener("click", () => {
      document.getElementById("exportModal").classList.remove("active");
    });

    document.querySelectorAll(".export-format-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const format = e.currentTarget.dataset.format;
        await this.downloadExport(format);
        document.getElementById("exportModal").classList.remove("active");
      });
    });
  }

  async downloadExport(format) {
    try {
      const resp = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotations: this.annotations.map((a) => ({
            id: a.id,
            label: a.label,
            color: a.color,
            type: a.type,
            points: a.points,
            text: a.text || "",
            properties: a.properties || {}
          })),
          format: format
        })
      });
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `tranchot_studio_export.${format === "label_studio" || format === "coco" ? "json" : format === "gpkg" ? "gpkg" : "geojson"}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert("Export fehlgeschlagen.");
    }
  }

  setupKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        this.commitSAMCandidate();
      } else if (e.key === "Escape") {
        this.clearPrompts();
      } else if (e.key === "w" || e.key === "W") {
        document.getElementById("toolSAM").click();
      } else if (e.key === "b" || e.key === "B" || e.key === "r" || e.key === "R") {
        const b = document.getElementById("toolSAMBox");
        if (b) b.click();
      } else if (e.key === "p" || e.key === "P") {
        document.getElementById("toolPoly").click();
      } else if (e.key === "e" || e.key === "E") {
        document.getElementById("toolEdit").click();
      } else if (e.key === "h" || e.key === "H") {
        document.getElementById("toolPan").click();
      } else if (e.key >= "1" && e.key <= "7") {
        const chips = document.querySelectorAll(".class-chip");
        const idx = parseInt(e.key) - 1;
        if (chips[idx]) chips[idx].click();
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new LeafletLabelStudio();
});
