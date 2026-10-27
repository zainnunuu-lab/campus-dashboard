"""Shared fixtures. Test data is built in temporary folders; the real data/ tree is not modified."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campus_dashboard import config  # noqa: E402

UTM = "EPSG:32642"


def utm_origin():
    from pyproj import Transformer

    lat, lon = config.CAMPUS_CENTER_LATLON
    return Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(lon, lat)


@pytest.fixture
def origin():
    return utm_origin()


@pytest.fixture
def landuse_gdf(origin):
    """Three non-overlapping 100 m squares (1 ha each) near the campus centre."""
    ox, oy = origin
    return gpd.GeoDataFrame(
        {"class": ["building", "lawn_grass", "trees_shrubs"]},
        geometry=[box(ox, oy, ox + 100, oy + 100), box(ox + 100, oy, ox + 200, oy + 100),
                  box(ox + 200, oy, ox + 300, oy + 100)],
        crs=UTM,
    )


@pytest.fixture
def points_gdf(origin):
    ox, oy = origin
    return gpd.GeoDataFrame({"is_green": [1, 0]}, geometry=[Point(ox, oy), Point(ox + 5, oy + 5)], crs=UTM)


def write_rgb_tif(path: Path, origin, crs=UTM, size=100, pixel=1.0, green_cols=50):
    """Left `green_cols` columns vegetation-green, the rest grey."""
    ox, oy = origin
    data = np.zeros((3, size, size), dtype="uint8")
    data[:, :, :] = 128
    data[0, :, :green_cols] = 60
    data[1, :, :green_cols] = 160
    data[2, :, :green_cols] = 50
    kwargs = {"driver": "GTiff", "width": size, "height": size, "count": 3, "dtype": "uint8", "nodata": 0}
    if crs is not None:
        kwargs.update(crs=crs, transform=from_origin(ox, oy + size * pixel, pixel, pixel))
    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data)
    return path


@pytest.fixture
def rgb_tif(tmp_path, origin):
    return write_rgb_tif(tmp_path / "ortho.tif", origin)


@pytest.fixture
def data_tree(tmp_path, monkeypatch):
    """An isolated data/ folder with demo + user sub-trees, patched into config."""
    data = tmp_path / "data"
    for root in ("demo", "user"):
        for kind in ("vector", "raster", "3d"):
            (data / root / kind).mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "DEMO_DIR", data / "demo")
    monkeypatch.setattr(config, "USER_DIR", data / "user")
    return data
