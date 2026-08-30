"""
Spatial Gazetteer & GeoNames / OSM Name Resolver for Historical Maps.
Queries real-world geographic place names, peaks, water bodies, and historical landmarks
based on the exact georeferenced coordinates of the click or selection box.
"""

from typing import List, Dict, Any, Optional, Tuple
import urllib.request
import urllib.parse
import json
import difflib
from pyproj import Transformer


# Comprehensive offline fallback gazetteer for the historical Rhineland (Tranchot & v. Müffling)
OFFLINE_RHINELAND_GAZETTEER = [
    # Siedlungen & Orte
    "Nickenich", "Andernach", "Kruft", "Plaidt", "Saffig", "Miesenheim", "Ochtendung",
    "Kretz", "Eich", "Kell", "Wassenach", "Maria Laach", "Bell", "Mendig", "Niedermendig",
    "Obermendig", "Thür", "Kottenheim", "Mayen", "Neuwied", "Weißenthurm", "Bassenheim",
    "Kärlich", "Mülheim", "Urmitz", "Kaltenengers", "St. Sebastian", "Koblenz", "Metternich",
    "Güls", "Moselweiß", "Lützel", "Bubenheim", "Rübenach", "Wolken",
    "Polch", "Rüber", "Trimbs", "Welling", "Fraukirch", "Hausen", "Betzing", "Monreal",
    "Ettringen", "St. Johann", "Kürrenberg", "Alzheim", "Kehrig", "Gering", "Kollig",
    "Mertloch", "Einig", "Gierschnach", "Münstermaifeld", "Kalt", "Lonnig",
    "Brohl", "Brohl-Lützing", "Niederlützingen", "Oberlützingen", "Burgbrohl", "Weiler",
    "Glees", "Wehr", "Rieden", "Volkesfeld", "Hausten", "Morswiesen",
    "Sinzig", "Bad Bodendorf", "Westum", "Löhndorf", "Remagen", "Kripp", "Rolandseck",
    "Oberwinter", "Unkelbach", "Bandorf", "Oedingen", "Ahrweiler", "Bad Neuenahr", "Walporzheim",
    "Marienthal", "Dernau", "Rech", "Mayschoß", "Altenahr", "Kreuzberg", "Altenburg",
    
    # Berge, Kuppen & Vulkankegel
    "Affenberg", "Rabenkopf", "Kotsberg", "Hartenberg", "Korretsberg", "Thelenberg",
    "Wingertsberg", "Krahnenberg", "Hummerich", "Veitskopf", "Laacher Kopf", "Nastberg",
    "Gänsehals", "Hochstein", "Sulzbusch", "Bocksberg", "Ettringer Bellerberg", "Mayener Bellberg",
    "Gollasch", "Sattelberg", "Heidekopf", "Steinberg", "Michelsberg", "Karmelenberg",
    
    # Flurnamen, Wege & Landmarken
    "Plaidterweg", "Andernach-Chaussee", "Mayener Chaussee", "Koblenzer Chaussee",
    "Croix", "Kreuz", "Kapelle", "Mühle", "Ölmühle", "Papiermühle", "Walkmühle", "Klostermühle",
    "Hof", "Gutshof", "Schloss", "Burg", "Kloster", "Kirche", "Kirchhof", "Hospital",
    "Zollhaus", "Forsthaus", "Jagdschloss", "Ziegelei", "Steinbruch", "Basaltbruch", "Tuffsteinbruch",
    
    # Gewässer
    "Rhein", "Mosel", "Nette", "Brohlbach", "Ahr", "Elzbach", "Laacher See", "Krufter Bach",
    "Segbach", "Mühlenbach", "Krummbach", "Wannbach", "Nitzbach"
]


class SpatialGazetteer:
    """
    Resolves geographic names based on spatial proximity to map coordinates.
    """

    @staticmethod
    def get_lat_lon(geo_handler, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Converts pixel coordinates to (latitude, longitude) WGS84.
        """
        if not geo_handler or getattr(geo_handler, "transform", None) is None:
            return None

        gx, gy = geo_handler.pixel_to_geo(px, py)
        crs = str(geo_handler.crs).upper() if geo_handler.crs else "EPSG:3857"

        try:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(gx, gy)
            return (float(lat), float(lon))
        except Exception:
            return None

    @classmethod
    def query_nearby_names(
        cls,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_m: int = 3000,
        ocr_hint: str = ""
    ) -> List[Dict[str, str]]:
        """
        Queries real-world names near the coordinate and ranks them against OCR hint.
        Returns list of dicts: [{'name': 'Nickenich', 'category': 'settlement'}, ...]
        """
        results: List[Dict[str, str]] = []
        seen = set()

        def add_item(name: str, cat: str):
            clean = name.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                results.append({"name": clean, "category": cat})

        # 1. Query OSM / Overpass spatial names if coordinates are available
        if lat is not None and lon is not None:
            try:
                query = f"""
                [out:json][timeout:3];
                (
                  node(around:{radius_m}, {lat}, {lon})["place"];
                  node(around:{radius_m}, {lat}, {lon})["natural"="peak"];
                  node(around:{radius_m}, {lat}, {lon})["natural"="hill"];
                  node(around:{radius_m}, {lat}, {lon})["historic"];
                  way(around:{radius_m}, {lat}, {lon})["place"];
                  way(around:{radius_m}, {lat}, {lon})["waterway"];
                  way(around:{radius_m}, {lat}, {lon})["highway"];
                );
                out tags 20;
                """
                url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query)
                req = urllib.request.Request(url, headers={"User-Agent": "TranchotExtractor/1.0 (BCDH Uni Bonn)"})
                with urllib.request.urlopen(req, timeout=3.5) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    for el in data.get("elements", []):
                        tags = el.get("tags", {})
                        n = tags.get("name")
                        if n:
                            # Infer category
                            if "place" in tags:
                                cat = "settlement"
                            elif "natural" in tags:
                                cat = "hill_mountain"
                            elif "waterway" in tags:
                                cat = "water_river"
                            elif "highway" in tags:
                                cat = "road_label"
                            else:
                                cat = "annotation"
                            add_item(n, cat)
            except Exception:
                pass

        # 2. Add matching historical Rhineland gazetteer names
        if ocr_hint:
            matches = difflib.get_close_matches(ocr_hint.replace(" ", ""), OFFLINE_RHINELAND_GAZETTEER, n=6, cutoff=0.20)
            for m in matches:
                cat = cls._infer_category(m)
                add_item(m, cat)

        # 3. Always include prominent regional key names as baseline choices
        for fallback in ["Nickenich", "Affenberg", "Rabenkopf", "Kotsberg", "Hartenberg", "Andernach", "Plaidterweg", "Croix"]:
            cat = cls._infer_category(fallback)
            add_item(fallback, cat)

        return results[:10]

    @classmethod
    def _infer_category(cls, name: str) -> str:
        name_lower = name.lower()
        if any(w in name_lower for w in ["berg", "kopf", "stein", "busch", "höhe", "kuppe"]):
            return "hill_mountain"
        elif any(w in name_lower for w in ["bach", "rhein", "mosel", "see", "fluss", "riv"]):
            return "water_river"
        elif any(w in name_lower for w in ["weg", "straße", "chaussee", "croix", "kreuz"]):
            return "road_label"
        elif any(w in name_lower for w in ["hof", "mühle", "burg", "schloss", "kloster", "kapelle"]):
            return "farm_landmark"
        else:
            return "settlement"
