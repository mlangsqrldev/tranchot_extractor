"""
Tranchot Extractor - Native CustomTkinter Desktop Application
Start script for launching the Tranchot historical map AI extraction suite.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tranchot_extractor.ui.desktop_app import main

if __name__ == "__main__":
    main()
