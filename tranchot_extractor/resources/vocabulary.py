"""
Controlled Vocabulary Handler for Historical Settlements & Topographical Objects.
Integrated with BCDH Skosmos / LVR Fundansprachen Vocabulary (https://vocabs.bcdh.uni-bonn.de/LVR_Fundansprachen/de/).
"""

import os
import csv
from typing import Dict, Optional, List, Any


class SettlementVocabulary:
    """Provides structured access to historical settlement and object URIs."""

    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "siedlungstypen_lvr_vokabular.csv")
        self.csv_path = csv_path
        self._entries: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                term = row.get("Siedlungstyp", "").strip()
                if term:
                    self._entries[term.lower()] = {
                        "term": term,
                        "uri": row.get("URI", "").strip(),
                        "prefLabel_de": row.get("PrefLabel_DE", "").strip(),
                        "category": row.get("Kategorie", "").strip(),
                        "description_en": row.get("Description_EN", "").strip()
                    }

    def get_uri(self, term: str) -> Optional[str]:
        """Returns the BCDH / LVR URI for a given settlement or object term."""
        entry = self.get_entry(term)
        return entry.get("uri") if entry else None

    def get_entry(self, term: str) -> Optional[Dict[str, str]]:
        """Returns the full metadata dictionary for a term."""
        term_clean = term.strip().lower()
        if term_clean in self._entries:
            return self._entries[term_clean]
        
        # Fuzzy prefix search
        for k, v in self._entries.items():
            if k in term_clean or term_clean in k:
                return v
        return None

    def all_terms(self) -> List[str]:
        """Returns list of all available settlement and object terms."""
        return [v["term"] for v in self._entries.values()]

    def all_entries(self) -> List[Dict[str, str]]:
        """Returns all entries."""
        return list(self._entries.values())
