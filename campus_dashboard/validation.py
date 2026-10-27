"""Validation of uploaded and catalogued datasets.

Every check returns messages in three levels:

* **errors**   - the dataset cannot be used (upload is blocked);
* **warnings** - usable, but the user must know (e.g. geometries were repaired);
* **info**     - neutral facts (feature count, CRS, extent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

from . import config
from .raster_io import RasterReadError, summarize_raster
from .vector_io import find_column, to_display_crs, to_metric_crs


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationResult") -> "ValidationResult":
        self.errors += other.errors
        self.warnings += other.warnings
        self.info += other.info
        return self


# --------------------------------------------------------------------------- files
def validate_upload_file(filename: str, size_bytes: int, kind: str) -> ValidationResult:
    res = ValidationResult()
    allowed = {
        "vector": config.VECTOR_EXTENSIONS,
        "raster": config.RASTER_EXTENSIONS,
        "3d": config.ASSET_3D_EXTENSIONS,
    }[kind]
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        res.errors.append(f"'{ext or 'no extension'}' is not a supported {kind} format. Use: {', '.join(allowed)}.")
    if size_bytes == 0:
        res.errors.append("The file is empty (0 bytes).")
    limit = config.MAX_UPLOAD_MB[kind]
    if size_bytes > limit * 1024 * 1024:
        res.errors.append(f"File is {size_bytes / 1e6:.0f} MB; the limit for {kind} data is {limit} MB.")
    return res


# --------------------------------------------------------------------------- geometry
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def check_extent(bounds_wgs84, res: ValidationResult) -> None:
    """Plausibility of a (west, south, east, north) extent in degrees."""
    west, south, east, north = bounds_wgs84
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
        res.errors.append(
            "Coordinates fall outside valid latitude/longitude ranges after reprojection - "
            "the declared CRS is probably wrong (e.g. projected metres labelled as WGS84)."
        )
        return
    lat, lon = (south + north) / 2, (west + east) / 2
    dist = haversine_km(lat, lon, *config.CAMPUS_CENTER_LATLON)
    res.info.append(f"Extent centre: {lat:.5f} N, {lon:.5f} E ({dist:.2f} km from configured campus centre).")
    if dist > config.FAR_FROM_CAMPUS_WARN_KM:
        res.warnings.append(
            f"Data centre is {dist:,.1f} km from the configured campus location. "
            "Check the CRS / axis order, or update CAMPUS_CENTER_LATLON in config.py."
        )


def _polygonal_overlap_ratio(gdf_metric: gpd.GeoDataFrame) -> float:
    polys = gdf_metric[gdf_metric.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(polys) < 2:
        return 0.0
    total = float(polys.area.sum())
    union = float(polys.union_all().area)
    return 0.0 if total == 0 else max(0.0, (total - union) / total)


def validate_vector(gdf: gpd.GeoDataFrame, category: str = "other") -> tuple[gpd.GeoDataFrame, ValidationResult]:
    """Validate and lightly clean a layer. Returns the cleaned copy and the messages."""
    res = ValidationResult()
    if gdf is None or len(gdf) == 0:
        res.errors.append("The layer contains no features.")
        return gdf, res
    gdf = gdf.copy()

    if gdf.crs is None:
        res.errors.append(
            "The layer has no coordinate reference system (CRS). Re-export it with a .prj / CRS, "
            "or choose the CRS it was created in."
        )
        return gdf, res
    res.info.append(f"Source CRS: {gdf.crs.to_string()}")

    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty.all():
        res.errors.append("All features have empty or missing geometry.")
        return gdf, res
    if empty.any():
        res.warnings.append(f"{int(empty.sum())} feature(s) with empty/missing geometry were removed.")
        gdf = gdf[~empty].copy()

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].map(make_valid)
        res.warnings.append(
            f"{int(invalid.sum())} invalid geometr{'y was' if invalid.sum() == 1 else 'ies were'} "
            "repaired with make_valid (e.g. self-intersecting polygons). Review them in your GIS."
        )

    types = sorted(set(gdf.geom_type))
    res.info.append(f"{len(gdf):,} feature(s); geometry type(s): {', '.join(types)}")
    if len({t.replace("Multi", "") for t in types} - {"GeometryCollection"}) > 1:
        res.warnings.append("Layer mixes geometry types; some charts only use the polygon features.")

    try:
        wgs = to_display_crs(gdf)
        check_extent(tuple(wgs.total_bounds), res)
    except Exception as exc:
        res.errors.append(f"Could not reproject the layer to WGS84: {exc}")
        return gdf, res

    _validate_category(gdf, category, res)
    return gdf, res


def _validate_category(gdf: gpd.GeoDataFrame, category: str, res: ValidationResult) -> None:
    polygonal = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    if category in ("landuse", "green_cover", "buildings", "boundary") and not polygonal.any():
        res.errors.append(f"A '{config.CATEGORIES[category]}' layer must contain polygons.")
        return
    if category == "trees" and not gdf.geom_type.isin(["Point", "MultiPoint"]).any():
        res.warnings.append("Tree layers are expected to contain points; other geometries are counted as-is.")

    if category == "landuse":
        field_name = find_column(gdf.columns, config.LANDUSE_CLASS_FIELDS)
        if field_name is None:
            res.errors.append(
                "No land-use class attribute found. Add a column named one of: "
                f"{', '.join(config.LANDUSE_CLASS_FIELDS)}."
            )
        else:
            values = gdf[field_name].astype(str).str.strip().str.lower()
            unknown = sorted(set(values) - set(config.LANDUSE_CLASSES))
            if unknown:
                res.warnings.append(
                    f"Unrecognised class value(s) in '{field_name}': {', '.join(unknown[:10])}. "
                    "They are shown in grey as 'unknown'. Allowed: " + ", ".join(config.LANDUSE_CLASSES)
                )
        if polygonal.sum() >= 2:
            ratio = _polygonal_overlap_ratio(to_metric_crs(gdf))
            if ratio > 0.01:
                res.warnings.append(
                    f"Polygons overlap ({ratio:.1%} of total area is counted more than once). "
                    "Class areas and percentages will be overstated until overlaps are removed."
                )

    if category == "buildings":
        hfield = find_column(gdf.columns, config.BUILDING_HEIGHT_FIELDS)
        if hfield is None:
            res.warnings.append(
                "No building height attribute found; 3D extrusion will be unavailable. "
                f"Expected one of: {', '.join(config.BUILDING_HEIGHT_FIELDS)}."
            )
        else:
            h = pd.to_numeric(gdf[hfield], errors="coerce")
            if h.isna().any():
                res.warnings.append(f"{int(h.isna().sum())} building(s) have a missing/non-numeric '{hfield}'.")
            if (h < 0).any():
                res.errors.append(f"'{hfield}' contains negative heights.")
            if (h > 300).any():
                res.warnings.append(f"'{hfield}' contains values above 300 m - check the unit (m vs cm/ft).")


# --------------------------------------------------------------------------- raster
def validate_raster(path: str | Path, category: str = "orthomosaic") -> ValidationResult:
    res = ValidationResult()
    try:
        s = summarize_raster(path)
    except RasterReadError as exc:
        res.errors.append(str(exc))
        return res
    res.info.append(s.size_label)
    if s.crs is None or s.pixel_size is None:
        res.errors.append(
            "The raster is not georeferenced (no CRS or geotransform). Export a GeoTIFF from the "
            "photogrammetry software (WebODM / Agisoft Metashape / Pix4D) rather than a plain image."
        )
        return res
    res.info.append(f"CRS: {s.crs}")
    if s.pixel_unit in ("metre", "meter"):
        res.info.append(f"Pixel size (from file): {s.pixel_size[0] * 100:.1f} x {s.pixel_size[1] * 100:.1f} cm")
    else:
        res.info.append(f"Pixel size (from file): {s.pixel_size[0]:.3g} x {s.pixel_size[1]:.3g} {s.pixel_unit}")
    if s.bounds_wgs84:
        check_extent(s.bounds_wgs84, res)
    if category == "orthomosaic" and s.band_count < 3:
        res.warnings.append(
            f"Orthomosaics are expected to have >= 3 bands (RGB); this file has {s.band_count}. "
            "Use category 'Index raster' or 'Elevation model' for single-band data."
        )
    if category in ("index_raster", "dsm") and s.band_count != 1:
        res.warnings.append(f"Expected a single band for this category; band 1 of {s.band_count} will be used.")
    if s.nodata is None and not s.has_alpha:
        res.warnings.append("No nodata value or alpha band - areas outside the survey may appear as data.")
    return res


# --------------------------------------------------------------------------- labels
def validate_label_points(gdf: gpd.GeoDataFrame) -> tuple[str | None, ValidationResult]:
    """Ground-truth points for green-cover accuracy: need a 0/1 'is_green' style field."""
    res = ValidationResult()
    if gdf is None or len(gdf) == 0:
        res.errors.append("Reference point file has no features.")
        return None, res
    if not gdf.geom_type.isin(["Point"]).all():
        res.errors.append("Reference data must contain only Point geometries.")
    col = find_column(gdf.columns, config.GREEN_LABEL_FIELDS)
    if col is None:
        res.errors.append(f"Reference points need a 0/1 label column named one of {config.GREEN_LABEL_FIELDS}.")
        return None, res
    vals = pd.to_numeric(gdf[col], errors="coerce")
    if vals.isna().any() or not set(np.unique(vals.dropna())).issubset({0, 1}):
        res.errors.append(f"Column '{col}' must contain only 0 (not green) and 1 (green).")
    if len(gdf) < 30:
        res.warnings.append(
            f"Only {len(gdf)} reference points. Accuracy figures from fewer than ~30-50 points are unreliable."
        )
    return col, res
