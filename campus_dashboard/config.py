"""Central configuration: paths, limits, the land-use class schema and map defaults.

Edit this file (not the page code) when the survey team finalises its class list,
the campus reference location, or file-size limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
USER_DIR = DATA_DIR / "user"

# Sub-folders per dataset kind. Both demo and user trees share this layout.
KIND_DIRS = {"vector": "vector", "raster": "raster", "3d": "3d"}

# --------------------------------------------------------------------------- campus
# Approximate map centre for SSUET, University Road, Karachi. It is used ONLY to
# centre maps and to warn when uploaded data lies far away (likely a CRS mistake).
# It is not a surveyed control point - replace it with the project's own value.
CAMPUS_NAME = "Sir Syed University of Engineering & Technology (SSUET)"
CAMPUS_CENTER_LATLON = (24.9157, 67.0920)
FAR_FROM_CAMPUS_WARN_KM = 5.0

# --------------------------------------------------------------------------- CRS
DISPLAY_CRS = "EPSG:4326"  # web maps and stored GeoJSON
# Karachi lies in UTM zone 42N. Used as a fallback when the CRS cannot be estimated.
FALLBACK_METRIC_CRS = "EPSG:32642"

# --------------------------------------------------------------------------- limits
MAX_UPLOAD_MB = {"vector": 200, "raster": 2000, "3d": 300}
# Largest side (pixels) when reading an orthomosaic for display / analysis. The full
# resolution file stays on disk; only a decimated view is loaded into memory.
RASTER_PREVIEW_MAX_PX = 2048
RASTER_ANALYSIS_MAX_PX = 1500
# 3D models are embedded in the page as base64; beyond this the browser struggles.
MODEL_VIEWER_MAX_MB = 80
POINT_CLOUD_MAX_POINTS = 60_000

# --------------------------------------------------------------------------- formats
VECTOR_EXTENSIONS = (".geojson", ".json", ".gpkg", ".zip", ".kml", ".csv")
RASTER_EXTENSIONS = (".tif", ".tiff")
MODEL_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")
POINT_CLOUD_EXTENSIONS = (".las", ".laz")
ASSET_3D_EXTENSIONS = MODEL_EXTENSIONS + POINT_CLOUD_EXTENSIONS

# Dataset categories decide which page uses a layer.
CATEGORIES = {
    "boundary": "Campus / study-area boundary",
    "landuse": "Land-use / land-cover polygons",
    "green_cover": "Green-cover polygons",
    "buildings": "Building footprints",
    "paths": "Roads & footpaths",
    "trees": "Tree / vegetation points",
    "orthomosaic": "Orthomosaic (RGB raster)",
    "index_raster": "Index raster (e.g. NDVI, single band)",
    "dsm": "Elevation model (DSM / DTM)",
    "model_3d": "3D model / mesh",
    "point_cloud": "Point cloud",
    "other": "Other spatial analysis result",
}
VECTOR_CATEGORIES = ("boundary", "landuse", "green_cover", "buildings", "paths", "trees", "other")
RASTER_CATEGORIES = ("orthomosaic", "index_raster", "dsm", "other")
ASSET_3D_CATEGORIES = ("model_3d", "point_cloud")

# --------------------------------------------------------------------------- land use
# Attribute names accepted as the land-use class field (first match wins).
LANDUSE_CLASS_FIELDS = ("class", "landuse", "lu_class", "land_use", "category")
BUILDING_HEIGHT_FIELDS = ("height_m", "height", "bldg_h", "elev_m")
GREEN_LABEL_FIELDS = ("is_green", "green", "label")


@dataclass(frozen=True)
class LandUseClass:
    label: str
    color: str
    is_green: bool


# Colours follow land-use conventions (vegetation green, paved grey, water blue).
# The order is the adjacent-pair order validated for colour-vision deficiency; the
# grey paved classes are deliberately low-chroma, so the map always shows a legend
# and tooltips, and charts label classes on the axis - colour is never the only cue.
LANDUSE_CLASSES: dict[str, LandUseClass] = {
    "building": LandUseClass("Building / roof", "#8c5a3c", False),
    "road": LandUseClass("Road", "#3d3c3a", False),
    "parking": LandUseClass("Parking", "#a3a19c", False),
    "paved_open": LandUseClass("Paved open space", "#e6c35c", False),
    "bare_soil": LandUseClass("Bare soil", "#b5651d", False),
    "lawn_grass": LandUseClass("Lawn / grass", "#5fb83f", True),
    "trees_shrubs": LandUseClass("Trees / shrubs", "#1f6e35", True),
    "sports_field": LandUseClass("Sports field", "#1baf7a", False),
    "water": LandUseClass("Water", "#2a78d6", False),
    "other": LandUseClass("Other / unclassified", "#c25d9a", False),
}
UNKNOWN_CLASS_COLOR = "#7a7974"

# Generic layer styling (non land-use vector layers).
LAYER_COLORS = {
    "boundary": "#d03b3b",
    "green_cover": "#008300",
    "buildings": "#8c5a3c",
    "paths": "#3d3c3a",
    "trees": "#1f6e35",
    "other": "#6250d6",
}

# --------------------------------------------------------------------------- basemaps
BASEMAPS = {
    "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": None},
    "Esri World Imagery (reference only)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    },
    "None (blank)": {"tiles": None, "attr": None},
}

DEMO_LABEL = "DEMO"
DEMO_DISCLAIMER = (
    "Synthetic demonstration data generated by scripts/generate_demo_data.py. "
    "It is NOT a drone survey result and does not depict real SSUET buildings, "
    "land use or vegetation."
)
DEFAULT_ACCURACY_STATEMENT = "Not validated - no accuracy assessment has been performed."
