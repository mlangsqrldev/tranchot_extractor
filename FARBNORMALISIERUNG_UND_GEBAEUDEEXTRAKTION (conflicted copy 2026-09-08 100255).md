# 🎨 Farbraum-Verschiebung, Optische Dichte & Morphologische Trennung für historische Karten (Tranchot 1803–1820)

Dieses Dokument erklärt die physikalischen, mathematischen und bildverarbeitungstechnischen Grundlagen, warum Farben auf über 200 Jahre alten Karten extrem dicht beieinander liegen und wie dieses Problem durch **Pergament-Referenzierung**, **spektrale Entmischung (Optical Density)** und **morphologische Wandstärkenfilterung** gelöst wird.

---

## 1. Das Kernproblem: Warum liegen die Farben so extrem dicht beieinander?

Auf den Tranchot-Karten (1803–1820) treffen drei physikalische und historische Gegebenheiten aufeinander:

### A. Subtraktive Aquarell-Lasuren statt deckender Farben
Die roten Gebäude wurden von den französischen Ingenieurgeographen (Jean Joseph Tranchot) nicht mit deckender Farbe gemalt, sondern mit **halbtransparenten Wasserfarben-Lasuren** (meist Karminrot aus Cochenille-Schildläusen oder Krapplack auf pflanzlicher/mineralischer Basis). 
Das Licht durchdringt die Farbschicht, wird vom darunterliegenden Papier reflektiert und durchquert die Farbschicht ein zweites Mal, bevor es das Auge bzw. den Kamerasensor erreicht.

### B. 200 Jahre Pergament- und Papiervergilbung
Das handgeschöpfte Hadernpapier (Leinen-/Baumwollfasern) ist durch Ligninabbau, Lichteinwirkung, Verschmutzung und Feuchtigkeit stark gealtert und nachgedunkelt.
Eine Messung auf dem Kartenblatt `Sample_Kruft.tif` zeigt die tatsächlichen Farbwerte des unbeschrifteten Papiers:

| Kanal | Rohwert im leeren Papier | Bedeutung |
|---|---|---|
| **Rot ($R$)** | $\approx 221$ | Extrem hoher Grundwert |
| **Grün ($G$)** | $\approx 209$ | Leicht abgedämpft |
| **Blau ($B$)** | $\approx 183$ | Stark absorbiert durch Vergilbung |
| **Differenz $R - G$** | **$+12$** | Das leere Papier ist bereits rötlich-gelb! |
| **Differenz $R - B$** | **$+38$** | Massiver Ockerton des Trägerpapiers |

### C. Das Schwellenwert-Dilemma im Standard-RGB/HSV-Farbraum
Weil das weiße Trägerpapier von Natur aus bereits einen Rotüberschuss von $R - G \approx 12$ aufweist, stauchen sich alle Karteninhalte in einem winzigen Kegel im RGB-Würfel zusammen:

* **Leeres Pergament**: $R - G \approx 10..13$
* **Braune Wegkante / Ackerfurche / Erdboden**: $R - G \approx 13..16$
* **Blasses / verwittertes rotes Gebäude**: $R - G \approx 14..18$
* **Kräftiges rotes Gebäude**: $R - G \approx 25..55$

Der euklidische Farbabstand zwischen einem blassen Ziegeldach und dem gelben Papier beträgt oft **weniger als 3 bis 5 Helligkeitsstufen** ($\Delta E < 5$).
* Ein starrer Schwellenwert von $R - G \ge 18$ **verliert alle blassen Gebäude** (z. B. im Osten von Kruft).
* Senkt man den Schwellenwert blind auf $\ge 12$, werden sofort **Straßenränder, Zäune und Pergamentflecken** fälschlicherweise als Gebäude erkannt.

---

## 2. Die Lösung: Gezielte Farbraum-Verschiebung (Chromatic Adaptation)

Die Lösung besteht darin, **alle Farben vor der Erkennung relativ zum lokalen Trägerpapier zu verschieben**.

```
[Rohbild RGB] 
      │
      ▼
[1. Lokale Pergament-Erkennung] ──► Ermittelt Weißpunkt P = [P_R, P_G, P_B]
      │
      ▼
[2. Von-Kries-Normalisierung]   ──► Papier wird [255, 255, 255] (R'-G' = 0)
      │
      ▼
[3. Beer-Lambert Optische Dichte] ──► OD_G - OD_R trennt Karmin von Tinte & Schmutz
      │
      ▼
[4. Morphologische Wandstärkenprüfung] ──► Dünne Gartenmauern (1-2 px) verwerfen,
                                          echte Gebäudekerne (>= 3.5 px) behalten
```

---

## 3. Mathematische Grundlagen

### Stufe 1: Von-Kries Chromatische Adaption (Lokale Papier-Referenzierung)

Wenn der Benutzer ein Dorf mit einem Polygon umfährt (Settlement-ROI), wird innerhalb dieser Begrenzung der **lokale Pergament-Weißpunkt** $P = [P_R, P_G, P_B]$ bestimmt (z. B. das 85. Perzentil der hellsten, schwach gesättigten Pixel):

$$I'_{\text{norm}}(x, y) = \mathrm{clip}\left(\frac{I(x, y)}{P} \cdot 255,\; 0,\; 255\right)$$

#### Was bewirkt diese Verschiebung?
1. Das gelb-braune Papier wird rechnerisch exakt zu **reinem Neutralweiß** $[255, 255, 255]$.
2. Auf dem neutralen Papier gilt:
   $$R' - G' = 0 \quad \text{und} \quad R' - B' = 0$$
3. Selbst die blasseste rote Wasserfarbe hat nach der Normalisierung einen klaren positiven Abstand $R' - G' > 12$.
4. Lokale Helligkeitsverläufe (Vignettierung, Scan-Schatten, biologische Flecken) werden vollständig kompensiert.

---

### Stufe 2: Beer-Lambert-Gesetz & Optische Dichte (Optical Density Deconvolution)

In der Physik mischen sich Farbstoffe auf Papier **subtraktiv durch Lichtabsorption**. Die Transmission $T$ folgt dem Lambert-Beerschen Gesetz:

$$T(\lambda) = \frac{I(\lambda)}{I_{\text{paper}}(\lambda)} = 10^{-A(\lambda)}$$

Die Optische Dichte (Optical Density / Absorbance $A$) ist der negative Logarithmus der Transmission:

$$OD_R = -\log_{10}\left(\frac{R}{P_R}\right), \quad OD_G = -\log_{10}\left(\frac{G}{P_G}\right), \quad OD_B = -\log_{10}\left(\frac{B}{P_B}\right)$$

#### Warum ist der OD-Raum dem RGB-Raum überlegen?
Im RGB-Farbraum mischen sich Pigmente nichtlinear. Im $OD$-Raum verhalten sich Absorptionswerte **linear und additiv**:
* **Schwarze Tusche / Carbon Ink**: Ruß absorbiert alle sichtbaren Wellenlängen gleichmäßig:
  $$OD_R \approx OD_G \approx OD_B$$
* **Karminrotes Pigment**: Reflektiert Rot, absorbiert aber Grün und Blau sehr stark:
  $$OD_G \gg OD_R$$

Daraus ergibt sich der **reine Karmin-Signal-Kanal**:

$$\Delta OD_{\text{carmine}} = OD_G - OD_R$$

#### Reale Messwerte auf `Sample_Kruft.tif`:
* **Leeres Pergament**: $\Delta OD = -0.0007 \pm 0.013$
* **Schwarze Tinte / Schraffen / Text**: $\Delta OD = +0.0040 \pm 0.024$
* **Karminrote Gebäude**: $\mathbf{\Delta OD = +0.1323 \pm 0.0832}$

> [!NOTE]
> Der Abstand zwischen Karminrot und schwarzer Tinte bzw. Pergament beträgt im $OD$-Raum **mehr als 10 Standardabweichungen**! 
> Dadurch können selbst extrem verblasste Gebäude sicher segmentiert werden, ohne dass Schraffen oder Linien anschlagen.

---

## 4. Morphologische Trennung: Gebäude vs. Gartenumfriedungen & Wege

Auf historischen Karten wurden Gärten, Klostergüter und Weinberge oft mit dünnen Mauern oder Hecken eingefasst, die mitunter ebenfalls mit einem braun-rötlichen Strich gezeichnet wurden.

### Warum entstanden links im Bild fehlerhafte Riesenpolygone?
1. Ein umfriedeter Gemüsegarten hat eine Fläche von $3.000$ bis $6.000\,\text{px}$.
2. Die Umfriedung besteht aus einer **dünnen Linie von 1 bis 2 Pixeln Breite**.
3. Wenn ein Algorithmus nur nach geschlossenen Ringen sucht, hält er das riesige leere Gartenareal für den "Innenhof" einer Hofanlage und erzeugt ein gigantisches $5.000\,\text{px}$-Gebäudepolygon.

### Die 3 morphologischen Kriterien zur Unterscheidung:

| Eigenschaft | Echtes Gebäude / Hofanlage | Gartenumfriedung / Wegschleife |
|---|---|---|
| **Wandstärke (Distance Transform)** | $\ge 3.5\,\text{px}$ bis $15\,\text{px}$ | $1.0$ bis $2.2\,\text{px}$ (dünne 1D-Linie) |
| **Füllungsgrad (Fill-Ratio = Area / BoundingBox)** | $\ge 0.25$ bis $0.85$ | $< 0.12$ (hohle Schleife) |
| **Verhältnis Bausubstanz zu Innenhof** | $\ge 25\%$ Baukörper | $< 8\%$ (fast nur freie Wiese/Garten) |
| **Maximale Baukörper-Dimension** | Typisch $\le 1.200\,\text{px}$ pro Trakt | $> 2.500\,\text{px}$ ohne Baukörper |

#### Technische Umsetzung:
1. **Wandstärken-Filter via Euklidische Distanztransformation (`cv2.distanceTransform`)**:
   Flächen, deren maximale Kerndicke unter $2.8\,\text{px}$ liegt, werden als Zäune/Mauern eingestuft und nicht als Gebäudehäuser verarbeitet.
2. **Hof-Plausibilitätsprüfung**:
   Ein Innenhof wird nur dann als historischer Bauernhof-Innenhof akzeptiert, wenn die umgebenden Trakte eine echte Gebäudestärke aufweisen und die Hoffläche im plausiblen Rahmen historischer Gehöfte liegt ($50$ bis $800\,\text{px}$).
3. **Watershed-Zerlegung**:
   Große zusammenhängende Hofkomplexe werden an ihren architektonischen Einschnürungen in die tatsächlichen Einzelgebäude zerlegt.

---

## 5. Parameter in `BuildingConfig`

In [config.py](file:///c:/Users/langm/sciebo/BCDH_Projektbox/1_BCDH%20Intern/Scripts/Tranchot_Extractor/tranchot_extractor/tranchot_extractor/config.py) werden diese Verfahren über folgende Parameter gesteuert:

```python
@dataclass
class BuildingConfig:
    # 1. Papier-Referenzierung & Beer-Lambert Optische Dichte
    enable_paper_normalization: bool = True
    carmine_od_threshold: float = 0.038   # Schwellenwert für OD_G - OD_R
    
    # 2. Farb-Differenz im normalisierten Raum
    norm_rgb_diff_threshold: int = 12     # (R' - G') und (R' - B')
    min_norm_red_intensity: int = 105
    
    # 3. Morphologische Wandstärken- & Geometriebeschränkungen
    min_building_thickness_px: float = 3.2 # Unterhalb 3.2 px: Zaun/Linie, kein Gebäude
    min_building_area_px: float = 24.0
    max_single_wing_area_px: float = 2500.0 # Zerlegt Riesenstrukturen per Watershed
    min_courtyard_fill_ratio: float = 0.20 # Verhindert hohle Gartenschleifen
    
    # 4. Orthogonale 90°-Regularisierung
    regularize_orthogonal: bool = True
    orthogonal_snap_deg: float = 18.0
```

---

## 6. Zusammenfassung

| Problem | Ursache | Lösung |
|---|---|---|
| **Blasse Gebäude nicht erkannt** | 200 Jahre Vergilbung hob das Pergament auf $R-G=12$; Schwellenwert $16$ war zu hoch. | **Lokale Papier-Normalisierung & Beer-Lambert OD ($OD_G - OD_R$)**: Vergilbung wird neutralisiert, Karmin-Signal hebt sich um $>10\sigma$ ab. |
| **Schraffen / Straßen fälschlich erkannt** | Straßenränder haben denselben RGB-Farbton wie blasse Ziegel. | **Tusche-Entkopplung**: Ruß-Tinte hat $OD_G - OD_R \approx 0$ und wird im OD-Kanal vollständig ausgeblendet. |
| **Riesige Fehlflächen auf Gärten/Feldern** | Dünne 1–2 px Gartenmauern wurden als "Innenhöfe" interpretiert. | **Distance-Transform Wandstärkenfilter**: Verwirft Strukturen unter $2.8\,\text{px}$ Wandstärke und hohle Schleifen mit $<20\%$ Füllgrad. |
| **Treppenstufen & Zacken an Mauern** | Rasterartefakte bei $45^\circ$/schrägen Häusern. | **Manhattan 90°-Regularisierung**: Snapped Kanten an die dominante Bauausrichtung des Gehöfts. |
