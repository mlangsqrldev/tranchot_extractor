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

    # Extended dictionary of Rhineland settlements with verified GeoNames IDs and coordinates
    RHINELAND_GEONAMES_DB = {
        "leutesdorf": {"name": "Leutesdorf", "geonames_id": "2878413", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.4522, "lon": 7.2894},
        "andernach": {"name": "Andernach", "geonames_id": "2956277", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.4397, "lon": 7.4011},
        "nickenich": {"name": "Nickenich", "geonames_id": "2862828", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.4136, "lon": 7.3242},
        "kruft": {"name": "Kruft", "geonames_id": "2886290", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.3886, "lon": 7.3364},
        "plaidt": {"name": "Plaidt", "geonames_id": "2854378", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.3908, "lon": 7.3872},
        "saffig": {"name": "Saffig", "geonames_id": "2842426", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.3800, "lon": 7.4167},
        "miesenheim": {"name": "Miesenheim", "geonames_id": "2870425", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.4167, "lon": 7.4000},
        "ochtendung": {"name": "Ochtendung", "geonames_id": "2860431", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.3500, "lon": 7.3833},
        "neuwied": {"name": "Neuwied", "geonames_id": "2863777", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.4289, "lon": 7.4614},
        "weißenthurm": {"name": "Weißenthurm", "geonames_id": "2814421", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.4167, "lon": 7.4667},
        "mayen": {"name": "Mayen", "geonames_id": "2873133", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.3275, "lon": 7.2208},
        "mendig": {"name": "Mendig", "geonames_id": "2872323", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.3739, "lon": 7.2794},
        "koblenz": {"name": "Koblenz", "geonames_id": "2886946", "type": "PPLA2", "state": "Rheinland-Pfalz", "lat": 50.3569, "lon": 7.5890},
        "rommerskirchen": {"name": "Rommerskirchen", "geonames_id": "2845620", "type": "PPL", "state": "Nordrhein-Westfalen", "lat": 51.0333, "lon": 6.6833},
        "sinzig": {"name": "Sinzig", "geonames_id": "2830841", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.5447, "lon": 7.2514},
        "remagen": {"name": "Remagen", "geonames_id": "2849887", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.5786, "lon": 7.2264},
        "bad neuenahr": {"name": "Bad Neuenahr-Ahrweiler", "geonames_id": "2953416", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.5447, "lon": 7.1136},
        "bad hönningen": {"name": "Bad Hönningen", "geonames_id": "2953497", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.5147, "lon": 7.3075},
        "linz am rhein": {"name": "Linz am Rhein", "geonames_id": "2877395", "type": "PPLA3", "state": "Rheinland-Pfalz", "lat": 50.5667, "lon": 7.2833},
        "rheinbrohl": {"name": "Rheinbrohl", "geonames_id": "2847247", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.4967, "lon": 7.3347},
        "hammerstein": {"name": "Hammerstein", "geonames_id": "2910793", "type": "PPL", "state": "Rheinland-Pfalz", "lat": 50.4789, "lon": 7.3094},
    }

    @classmethod
    def search_settlement(
        cls,
        query: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        max_results: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Searches for settlements by name or proximity, querying online OSM Nominatim / GeoNames
        and falling back to local regional database.
        """
        results: List[Dict[str, Any]] = []
        seen_names = set()

        q_clean = query.strip().lower()

        # 1. Check local offline database for exact or fuzzy matches
        if q_clean:
            for k, v in cls.RHINELAND_GEONAMES_DB.items():
                if q_clean in k or k in q_clean:
                    results.append(v)
                    seen_names.add(v["name"].lower())

        # If spatial coordinates are provided, find closest towns
        if lat is not None and lon is not None:
            sorted_by_dist = []
            for k, v in cls.RHINELAND_GEONAMES_DB.items():
                d = (v["lat"] - lat) ** 2 + (v["lon"] - lon) ** 2
                sorted_by_dist.append((d, v))
            sorted_by_dist.sort(key=lambda x: x[0])
            for _, v in sorted_by_dist[:3]:
                if v["name"].lower() not in seen_names:
                    results.append(v)
                    seen_names.add(v["name"].lower())

        # 2. Query Online Nominatim / GeoNames if available
        if query and len(query.strip()) >= 2:
            try:
                url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query.strip())}+Germany&format=json&limit=5"
                req = urllib.request.Request(url, headers={"User-Agent": "TranchotExtractor/1.0 (BCDH Uni Bonn)"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    data = json.loads(resp.read().decode())
                    for item in data:
                        display = item.get("display_name", "").split(",")[0].strip()
                        if display.lower() not in seen_names:
                            osm_id = str(item.get("osm_id", ""))
                            results.append({
                                "name": display,
                                "geonames_id": osm_id,
                                "type": item.get("type", "settlement"),
                                "state": item.get("display_name", ""),
                                "lat": float(item.get("lat", 0.0)),
                                "lon": float(item.get("lon", 0.0)),
                            })
                            seen_names.add(display.lower())
            except Exception:
                pass

        # Return top matches
        return results[:max_results]
