import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from campus_dashboard.validation import (
    validate_label_points,
    validate_raster,
    validate_upload_file,
    validate_vector,
)
from tests.conftest import write_rgb_tif


def test_valid_landuse_passes(landuse_gdf):
    _, res = validate_vector(landuse_gdf, "landuse")
    assert res.ok and not res.warnings


def test_empty_layer():
    _, res = validate_vector(gpd.GeoDataFrame(geometry=[], crs=4326))
    assert not res.ok


def test_missing_crs(landuse_gdf):
    _, res = validate_vector(landuse_gdf.set_crs(None, allow_override=True))
    assert any("coordinate reference system" in e for e in res.errors)


def test_invalid_geometry_is_repaired(origin):
    ox, oy = origin
    bowtie = Polygon([(ox, oy), (ox + 10, oy + 10), (ox + 10, oy), (ox, oy + 10)])
    gdf = gpd.GeoDataFrame({"class": ["lawn_grass"]}, geometry=[bowtie], crs="EPSG:32642")
    cleaned, res = validate_vector(gdf, "landuse")
    assert res.ok
    assert cleaned.geometry.is_valid.all()
    assert any("repaired" in w for w in res.warnings)


def test_null_geometries_removed(landuse_gdf):
    gdf = landuse_gdf.copy()
    gdf.loc[0, "geometry"] = None
    cleaned, res = validate_vector(gdf, "landuse")
    assert len(cleaned) == 2 and any("removed" in w for w in res.warnings)


def test_wrong_crs_gives_out_of_range_error(landuse_gdf):
    # UTM metres mislabelled as degrees
    mislabelled = landuse_gdf.set_crs(4326, allow_override=True)
    _, res = validate_vector(mislabelled, "landuse")
    assert not res.ok


def test_far_from_campus_warning():
    gdf = gpd.GeoDataFrame({"class": ["lawn_grass"]}, geometry=[box(-0.13, 51.50, -0.12, 51.51)], crs=4326)
    _, res = validate_vector(gdf, "landuse")
    assert res.ok and any("km from the configured campus" in w for w in res.warnings)


def test_landuse_requires_class_field(landuse_gdf):
    _, res = validate_vector(landuse_gdf.rename(columns={"class": "foo"}), "landuse")
    assert any("class attribute" in e for e in res.errors)


def test_unknown_class_warning(landuse_gdf):
    gdf = landuse_gdf.copy()
    gdf.loc[0, "class"] = "helipad"
    _, res = validate_vector(gdf, "landuse")
    assert res.ok and any("helipad" in w for w in res.warnings)


def test_overlap_warning(origin):
    ox, oy = origin
    gdf = gpd.GeoDataFrame({"class": ["lawn_grass", "road"]},
                           geometry=[box(ox, oy, ox + 100, oy + 100), box(ox + 50, oy, ox + 150, oy + 100)],
                           crs="EPSG:32642")
    _, res = validate_vector(gdf, "landuse")
    assert any("overlap" in w for w in res.warnings)


def test_landuse_must_be_polygons(points_gdf):
    _, res = validate_vector(points_gdf.assign(**{"class": "road"}), "landuse")
    assert not res.ok


def test_building_heights(landuse_gdf):
    ok = landuse_gdf.assign(height_m=[10, 12, 9])
    _, res = validate_vector(ok, "buildings")
    assert res.ok and not res.warnings
    neg = landuse_gdf.assign(height_m=[10, -1, 9])
    _, res = validate_vector(neg, "buildings")
    assert not res.ok
    _, res = validate_vector(landuse_gdf, "buildings")
    assert any("height" in w for w in res.warnings)


@pytest.mark.parametrize(
    "name,size,kind,ok",
    [("a.geojson", 10, "vector", True), ("a.exe", 10, "vector", False), ("a.tif", 0, "raster", False),
     ("a.tif", 10**12, "raster", False), ("m.glb", 100, "3d", True), ("m.fbx", 100, "3d", False)],
)
def test_upload_file_checks(name, size, kind, ok):
    assert validate_upload_file(name, size, kind).ok is ok


def test_raster_ok(rgb_tif):
    res = validate_raster(rgb_tif, "orthomosaic")
    assert res.ok and not res.warnings


def test_raster_not_georeferenced(tmp_path, origin):
    path = write_rgb_tif(tmp_path / "plain.tif", origin, crs=None)
    res = validate_raster(path)
    assert any("not georeferenced" in e for e in res.errors)


def test_raster_not_a_raster(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"garbage")
    assert not validate_raster(p).ok


def test_orthomosaic_single_band_warning(tmp_path, origin):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    p = tmp_path / "ndvi.tif"
    with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32", nodata=-9999,
                       crs="EPSG:32642", transform=from_origin(origin[0], origin[1], 1, 1)) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="float32"))
    assert any(">= 3 bands" in w for w in validate_raster(p, "orthomosaic").warnings)
    assert validate_raster(p, "index_raster").ok


def test_label_points(points_gdf, origin):
    col, res = validate_label_points(points_gdf)
    assert col == "is_green" and res.ok
    assert any("unreliable" in w for w in res.warnings)  # fewer than 30 points
    bad = points_gdf.assign(is_green=[1, 5])
    assert not validate_label_points(bad)[1].ok
    assert not validate_label_points(points_gdf.rename(columns={"is_green": "x"}))[1].ok
    poly = gpd.GeoDataFrame({"is_green": [1]}, geometry=[Point(origin).buffer(5)], crs="EPSG:32642")
    assert not validate_label_points(poly)[1].ok
