from setuptools import setup, find_packages

setup(
    name="tranchot-extractor",
    version="0.1.0",
    description="AI-based Feature Extractor for Tranchot historical maps (Buildings, Roads, Toponyms)",
    author="Bonn Center for Digital Humanities (BCDH)",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "shapely>=2.0.0",
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "easyocr>=1.7.0",
        "skan>=0.11.0",
        "networkx>=3.0",
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "customtkinter>=5.2.0",
    ],
    entry_points={
        "console_scripts": [
            "tranchot-extract=tranchot_extractor.cli:main",
            "tranchot-gui=tranchot_extractor.ui.desktop_app:main",
        ],
    },
    python_requires=">=3.9",
)
