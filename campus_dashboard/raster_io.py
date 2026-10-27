"""GeoTIFF handling: summaries, decimated reads for display and for analysis.

Orthomosaics are often several GB. Nothing here loads a full-resolution raster:
all reads go through a ``WarpedVRT`` with an ``out_shape`` capped by config, so
GDAL decimates on the fly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from . import config


class RasterReadError(Exception):
    """Raised with a message that can be shown directly to the user."""


@dataclass
class RasterSummary:
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str | None
    pixel_size: tuple[float, float] | None
    pixel_unit: str | None
    nodata: float | None
    bounds_wgs84: tuple[float, float, float, float] | None  # (west, south, east, north)
    has_alpha: bool
    tags: dict

    @property
    def size_label(self) -> str:
        return f"{self.width:,} x {self.height:,} px, {self.band_count} band(s), {self.dtype}"


def _open(path: Path):
    try:
        return rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise RasterReadError(f"Could not open '{Path(path).name}' as a raster: {exc}") from exc


def summarize_raster(path: str | Path) -> RasterSummary:
    path = Path(path)
    with _open(path) as src:
        bounds = None
        if src.crs is not None:
            try:
                bounds = tuple(transform_bounds(src.crs, config.DISPLAY_CRS, *src.bounds, densify_pts=21))
            except Exception:
                bounds = None
        unit = None
        if src.crs is not None:
            unit = "degree" if src.crs.is_geographic else (src.crs.linear_units or "unknown")
        return RasterSummary(
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs.to_string() if src.crs else None,
            pixel_size=(abs(src.transform.a), abs(src.transform.e)) if not src.transform.is_identity else None,
            pixel_unit=unit,
            nodata=src.nodata,
            bounds_wgs84=bounds,
            has_alpha=ColorInterp.alpha in src.colorinterp,
            tags=dict(src.tags()),
        )


def _out_shape(width: int, height: int, max_px: int) -> tuple[int, int]:
    scale = min(1.0, max_px / max(width, height))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


@dataclass
class RasterView:
    data: np.ndarray  # (bands, rows, cols), float32, NaN where invalid
    valid: np.ndarray  # (rows, cols) bool
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    bounds: tuple[float, float, float, float]


def read_view(path: str | Path, dst_crs: str | None, max_px: int) -> RasterView:
    """Read a decimated view, optionally reprojected into ``dst_crs``."""
    path = Path(path)
    with _open(path) as src:
        if src.crs is None:
            raise RasterReadError("Raster has no CRS (not georeferenced).")
        vrt_kwargs = {"resampling": Resampling.bilinear}
        if dst_crs:
            vrt_kwargs["crs"] = dst_crs
        if src.nodata is None and ColorInterp.alpha not in src.colorinterp:
            # Without nodata or alpha, warped edges would be filled with 0 and look
            # like black data; an alpha band marks them as outside the image.
            vrt_kwargs["add_alpha"] = True
        with WarpedVRT(src, **vrt_kwargs) as vrt:
            alpha_idx = [i + 1 for i, ci in enumerate(vrt.colorinterp) if ci == ColorInterp.alpha]
            rows, cols = _out_shape(vrt.width, vrt.height, max_px)
            raw = vrt.read(out_shape=(vrt.count, rows, cols), resampling=Resampling.average, masked=True)
            t, sx, sy = vrt.transform, vrt.width / cols, vrt.height / rows
            transform = rasterio.Affine(t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f)
            bounds = tuple(vrt.bounds)
            crs = vrt.crs

    mask = np.ma.getmaskarray(raw)
    data = raw.astype("float32").filled(np.nan)
    valid = ~mask.any(axis=0)
    band_idx = [b for b in range(1, data.shape[0] + 1) if b not in alpha_idx]
    if alpha_idx:
        valid &= data[alpha_idx[0] - 1] > 0
    data = data[[b - 1 for b in band_idx]]
    data[:, ~valid] = np.nan
    return RasterView(data=data, valid=valid, transform=transform, crs=crs, bounds=bounds)


def _stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch to 0..255 for display only."""
    vals = band[valid]
    if vals.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo, hi = np.nanpercentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((band - lo) / (hi - lo), 0, 1) * 255
    return np.nan_to_num(out).astype("uint8")


# Colour ramps for single-band rasters (position 0..1 -> RGB). Index rasters use a
# diverging brown-grey-green ramp; elevation uses a single-hue light-to-dark blue.
RAMPS = {
    "index": [(0.0, (140, 81, 10)), (0.5, (240, 239, 236)), (1.0, (0, 104, 55))],
    "elevation": [(0.0, (205, 226, 251)), (0.5, (85, 152, 231)), (1.0, (13, 54, 107))],
}


def apply_ramp(norm: np.ndarray, ramp: str = "index") -> np.ndarray:
    stops = RAMPS[ramp]
    pos = np.array([p for p, _ in stops])
    rgb = np.stack(
        [np.interp(norm, pos, [c[i] for _, c in stops]) for i in range(3)], axis=-1
    )
    return rgb.astype("uint8")


def render_png(view: RasterView, ramp: str = "index") -> bytes:
    """Encode a view as a transparent PNG. 1-band rasters get a colour ramp."""
    valid = view.valid
    if view.data.shape[0] >= 3:
        rgb3 = view.data[:3]
        vals = rgb3[:, valid]
        if vals.size and np.nanmin(vals) >= 0 and np.nanmax(vals) <= 255:
            # 8-bit RGB (typical orthomosaic export): show the true colours unchanged.
            rgb = np.dstack([np.nan_to_num(b).astype("uint8") for b in rgb3])
        else:
            # 16-bit / float: one stretch for all bands so colour balance is preserved.
            lo, hi = np.nanpercentile(vals, [2, 98]) if vals.size else (0, 1)
            scaled = np.clip((rgb3 - lo) / max(hi - lo, 1e-9), 0, 1) * 255
            rgb = np.dstack([np.nan_to_num(b).astype("uint8") for b in scaled])
    else:
        rgb = apply_ramp(_stretch(view.data[0], valid) / 255.0, ramp)
    alpha = np.where(valid, 255, 0).astype("uint8")
    img = Image.fromarray(np.dstack([rgb, alpha]))  # 4 x uint8 -> RGBA
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def preview_for_map(path: str | Path, max_px: int = config.RASTER_PREVIEW_MAX_PX, ramp: str = "index"):
    """Return (png_bytes, [[south, west], [north, east]]) for a Leaflet image overlay.

    The raster is warped to EPSG:4326. Leaflet stretches the image linearly in Web
    Mercator; across a campus-sized extent (< 2 km) the resulting offset is far
    below one display pixel.
    """
    view = read_view(path, config.DISPLAY_CRS, max_px)
    west, south, east, north = view.bounds
    return render_png(view, ramp), [[south, west], [north, east]]


def read_for_analysis(path: str | Path, max_px: int = config.RASTER_ANALYSIS_MAX_PX) -> RasterView:
    """Decimated view in a metric CRS so that pixel areas are in square metres."""
    with _open(Path(path)) as src:
        if src.crs is None:
            raise RasterReadError("Raster has no CRS (not georeferenced).")
        needs_metric = src.crs.is_geographic or (src.crs.linear_units or "").lower() not in ("metre", "meter")
        dst = None
        if needs_metric:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            lon, lat = (west + east) / 2, (south + north) / 2
            zone = int((lon + 180) // 6) + 1
            dst = f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"
    return read_view(path, dst, max_px)
