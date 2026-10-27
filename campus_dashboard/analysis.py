"""Spatial analysis: land-use areas, green cover and validation against reference points.

Everything here is a *derived estimate*. Accuracy is only reported when the user
supplies independent reference (ground-truth) points; otherwise results carry no
accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from . import config
from .raster_io import RasterView
from .vector_io import find_column, to_metric_crs


# --------------------------------------------------------------------------- land use
def normalise_classes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def landuse_area_table(
    gdf: gpd.GeoDataFrame,
    class_field: str | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Area per land-use class in a local UTM projection, optionally clipped to a boundary."""
    class_field = class_field or find_column(gdf.columns, config.LANDUSE_CLASS_FIELDS)
    if class_field is None:
        raise ValueError("No land-use class field found.")
    metric = to_metric_crs(gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])])
    if boundary is not None and len(boundary):
        clip_geom = boundary.to_crs(metric.crs).union_all()
        metric = gpd.clip(metric, clip_geom)
    metric = metric.assign(_cls=normalise_classes(metric[class_field]), _area=metric.area)

    grouped = metric.groupby("_cls").agg(area_m2=("_area", "sum"), features=("_area", "size")).reset_index()
    grouped = grouped.rename(columns={"_cls": "class"})
    total = grouped["area_m2"].sum()

    def info(c, attr, default):
        return getattr(config.LANDUSE_CLASSES[c], attr) if c in config.LANDUSE_CLASSES else default

    grouped["label"] = [info(c, "label", f"{c} (unknown)") for c in grouped["class"]]
    grouped["color"] = [info(c, "color", config.UNKNOWN_CLASS_COLOR) for c in grouped["class"]]
    grouped["is_green"] = [info(c, "is_green", False) for c in grouped["class"]]
    grouped["area_ha"] = grouped["area_m2"] / 10_000
    grouped["percent"] = np.where(total > 0, grouped["area_m2"] / total * 100, 0.0)
    cols = ["class", "label", "features", "area_m2", "area_ha", "percent", "is_green", "color"]
    return grouped[cols].sort_values("area_m2", ascending=False).reset_index(drop=True)


@dataclass
class GreenCoverSummary:
    green_m2: float
    total_m2: float

    @property
    def percent(self) -> float:
        return 0.0 if self.total_m2 == 0 else self.green_m2 / self.total_m2 * 100


def green_cover_from_table(table: pd.DataFrame) -> GreenCoverSummary:
    return GreenCoverSummary(
        green_m2=float(table.loc[table["is_green"], "area_m2"].sum()),
        total_m2=float(table["area_m2"].sum()),
    )


# --------------------------------------------------------------------------- generic layers
def layer_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Count plus total area (polygons) and length (lines) in metric units."""
    metric = to_metric_crs(gdf)
    gt = metric.geom_type
    poly = gt.isin(["Polygon", "MultiPolygon"])
    line = gt.isin(["LineString", "MultiLineString"])
    point = gt.isin(["Point", "MultiPoint"])
    return {
        "features": int(len(metric)),
        "polygons": int(poly.sum()),
        "lines": int(line.sum()),
        "points": int(point.sum()),
        "area_m2": float(metric[poly].area.sum()),
        "length_m": float(metric[line].length.sum()),
    }


def add_metric_measures(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add area_m2 / length_m columns computed in UTM (non-destructive copy)."""
    metric = to_metric_crs(gdf)
    out = gdf.copy()
    gt = metric.geom_type
    if gt.isin(["Polygon", "MultiPolygon"]).any():
        out["area_m2"] = np.where(gt.isin(["Polygon", "MultiPolygon"]), metric.area, np.nan).round(1)
    if gt.isin(["LineString", "MultiLineString"]).any():
        out["length_m"] = np.where(gt.isin(["LineString", "MultiLineString"]), metric.length, np.nan).round(1)
    return out


# --------------------------------------------------------------------------- raster green cover
INDEX_METHODS = {
    "ExG": "Excess Green, 2g - r - b on chromatic coordinates (RGB). Range about -1..2.",
    "VARI": "Visible Atmospherically Resistant Index, (G - R) / (G + R - B) (RGB). Range about -1..1.",
    "Band 1 as index": "Use band 1 directly, e.g. a pre-computed NDVI raster from a multispectral sensor.",
}


def vegetation_index(view: RasterView, method: str) -> np.ndarray:
    data = view.data
    if method == "Band 1 as index":
        return data[0].astype("float32")
    if data.shape[0] < 3:
        raise ValueError(f"{method} needs an RGB raster (3+ bands); this raster has {data.shape[0]}.")
    r, g, b = (data[i].astype("float64") for i in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "ExG":
            s = r + g + b
            idx = 2 * g / s - r / s - b / s
        elif method == "VARI":
            idx = (g - r) / (g + r - b)
            idx = np.clip(idx, -1, 1)
        else:
            raise ValueError(f"Unknown index method '{method}'.")
    idx[~view.valid] = np.nan
    idx[~np.isfinite(idx)] = np.nan
    return idx.astype("float32")


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's histogram threshold - a starting point, not a validated cut-off."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        raise ValueError("No valid pixels to threshold.")
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = (w0 * w1 * (m0 - m1) ** 2).astype("float64")
    # Empty bins between two modes form a plateau of equal variance; take its middle.
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(centers[best].mean())


def green_mask(index: np.ndarray, threshold: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(index) & (index >= threshold)


def mask_area(mask: np.ndarray, valid: np.ndarray, transform) -> GreenCoverSummary:
    """Pixel areas from the analysis transform (metres, because the view is in UTM)."""
    pixel_m2 = abs(transform.a * transform.e)
    return GreenCoverSummary(green_m2=float(mask.sum() * pixel_m2), total_m2=float(valid.sum() * pixel_m2))


@dataclass
class AccuracyReport:
    n_used: int
    n_outside: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def overall_accuracy(self) -> float:
        return (self.tp + self.tn) / self.n_used if self.n_used else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) and np.isfinite(p) and np.isfinite(r) else float("nan")

    @property
    def kappa(self) -> float:
        n = self.n_used
        if not n:
            return float("nan")
        po = self.overall_accuracy
        pe = ((self.tp + self.fp) * (self.tp + self.fn) + (self.fn + self.tn) * (self.fp + self.tn)) / n**2
        return (po - pe) / (1 - pe) if pe != 1 else float("nan")

    def confusion_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.tp, self.fn], [self.fp, self.tn]],
            index=["Reference: green", "Reference: not green"],
            columns=["Predicted: green", "Predicted: not green"],
        )


def accuracy_against_points(
    mask: np.ndarray, valid: np.ndarray, view: RasterView, points: gpd.GeoDataFrame, label_col: str
) -> AccuracyReport:
    pts = points.to_crs(view.crs)
    rows, cols = rowcol(view.transform, pts.geometry.x.values, pts.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    inside[inside] &= valid[rows[inside], cols[inside]]
    truth = pd.to_numeric(pts[label_col], errors="coerce").values[inside].astype(bool)
    pred = mask[rows[inside], cols[inside]]
    return AccuracyReport(
        n_used=int(inside.sum()),
        n_outside=int((~inside).sum()),
        tp=int((pred & truth).sum()),
        fp=int((pred & ~truth).sum()),
        fn=int((~pred & truth).sum()),
        tn=int((~pred & ~truth).sum()),
    )
