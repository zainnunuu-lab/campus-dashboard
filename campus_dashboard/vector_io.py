"""Reading vector GIS files into GeoDataFrames with friendly error messages.

Supported: GeoJSON (.geojson/.json), GeoPackage (.gpkg), zipped ESRI Shapefile
(.zip containing .shp/.shx/.dbf/.prj), KML (.kml) and CSV with either
latitude/longitude columns or a WKT geometry column.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

from . import config

LAT_NAMES = ("lat", "latitude", "y", "lat_dd")
LON_NAMES = ("lon", "lng", "long", "longitude", "x", "lon_dd")
WKT_NAMES = ("wkt", "geometry", "geom")


class VectorReadError(Exception):
    """Raised with a message that can be shown directly to the user."""


def find_column(columns, candidates) -> str | None:
    """Case-insensitive lookup of the first candidate column name."""
    lookup = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _read_csv(path: Path, crs: str | None) -> gpd.GeoDataFrame:
    df = None
    # Excel on Windows saves "CSV UTF-8" with a BOM, or plain CSV as cp1252.
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # pandas raises several parser error types
            raise VectorReadError(f"Could not parse CSV: {exc}") from exc
    if df is None:
        raise VectorReadError("Could not decode the CSV file; save it as 'CSV UTF-8'.")
    if df.empty:
        raise VectorReadError("The CSV file has no rows.")

    wkt_col = find_column(df.columns, WKT_NAMES)
    lat_col = find_column(df.columns, LAT_NAMES)
    lon_col = find_column(df.columns, LON_NAMES)
    if wkt_col is not None:
        try:
            geom = df[wkt_col].astype(str).map(shapely_wkt.loads)
        except Exception as exc:
            raise VectorReadError(f"Column '{wkt_col}' does not contain valid WKT geometry: {exc}") from exc
        df = df.drop(columns=[wkt_col])
    elif lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        if lat.isna().all() or lon.isna().all():
            raise VectorReadError(f"Columns '{lat_col}'/'{lon_col}' contain no numeric coordinates.")
        geom = gpd.points_from_xy(lon, lat)
    else:
        raise VectorReadError(
            "CSV needs either latitude/longitude columns "
            f"(one of {LAT_NAMES} and one of {LON_NAMES}) or a WKT column ({WKT_NAMES})."
        )
    # Plain CSV carries no CRS; WGS84 is assumed unless the user specifies otherwise.
    return gpd.GeoDataFrame(df, geometry=geom, crs=crs or "EPSG:4326")


def _zip_shapefile_target(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        raise VectorReadError("The .zip file is corrupt or not a zip archive.") from exc
    shp = [n for n in names if n.lower().endswith(".shp")]
    gpkg = [n for n in names if n.lower().endswith((".gpkg", ".geojson"))]
    if not shp and not gpkg:
        raise VectorReadError("The .zip archive contains no .shp, .gpkg or .geojson file.")
    if shp:
        stem = shp[0][:-4]
        lower = {n.lower() for n in names}
        missing = [ext for ext in (".shx", ".dbf") if (stem + ext).lower() not in lower]
        if missing:
            raise VectorReadError(f"Shapefile '{shp[0]}' is incomplete - missing {', '.join(missing)}.")
        inner = shp[0]
    else:
        inner = gpkg[0]
    return f"zip://{path.as_posix()}!{inner}"


def read_vector(path: str | Path, assumed_crs: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector file. ``assumed_crs`` is applied only when the file has no CRS."""
    path = Path(path)
    if not path.exists():
        raise VectorReadError(f"File not found: {path.name}")
    ext = path.suffix.lower()
    if ext not in config.VECTOR_EXTENSIONS:
        raise VectorReadError(
            f"Unsupported vector format '{ext}'. Supported: {', '.join(config.VECTOR_EXTENSIONS)}"
        )
    if ext == ".csv":
        return _read_csv(path, assumed_crs)

    target = _zip_shapefile_target(path) if ext == ".zip" else str(path)
    try:
        gdf = gpd.read_file(target, engine="pyogrio")
    except Exception as exc:  # GDAL errors surface as several exception classes
        raise VectorReadError(f"GDAL could not read '{path.name}': {exc}") from exc
    if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf:
        raise VectorReadError(f"'{path.name}' has no geometry column.")
    if gdf.crs is None and assumed_crs:
        gdf = gdf.set_crs(assumed_crs)
    return gdf


def to_display_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise VectorReadError("Layer has no CRS; it cannot be placed on the map.")
    return gdf if gdf.crs.to_string() == config.DISPLAY_CRS else gdf.to_crs(config.DISPLAY_CRS)


def metric_crs_for(gdf: gpd.GeoDataFrame):
    """A local projected CRS (UTM) for area and length calculations."""
    try:
        crs = gdf.estimate_utm_crs()
        if crs is not None:
            return crs
    except Exception:
        pass
    return config.FALLBACK_METRIC_CRS


def to_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise VectorReadError("Layer has no CRS; areas and lengths cannot be computed.")
    if gdf.crs.is_projected and gdf.crs.axis_info and gdf.crs.axis_info[0].unit_name in ("metre", "meter"):
        return gdf
    return gdf.to_crs(metric_crs_for(gdf))


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Store a validated layer as RFC 7946 GeoJSON (always WGS84)."""
    to_display_crs(gdf).to_file(path, driver="GeoJSON", engine="pyogrio")
    return path
