import zipfile

import geopandas as gpd
import pytest

from campus_dashboard.vector_io import VectorReadError, find_column, read_vector, to_metric_crs, write_geojson


def test_geojson_roundtrip(tmp_path, landuse_gdf):
    path = write_geojson(landuse_gdf, tmp_path / "lu.geojson")
    gdf = read_vector(path)
    assert len(gdf) == 3
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf["class"]) == {"building", "lawn_grass", "trees_shrubs"}


def test_geopackage(tmp_path, landuse_gdf):
    path = tmp_path / "lu.gpkg"
    landuse_gdf.to_file(path, driver="GPKG")
    gdf = read_vector(path)
    assert gdf.crs.to_epsg() == 32642 and len(gdf) == 3


def test_kml(tmp_path, landuse_gdf):
    path = tmp_path / "lu.kml"
    landuse_gdf.to_crs(4326).to_file(path, driver="KML")
    assert len(read_vector(path)) == 3


def test_zipped_shapefile(tmp_path, landuse_gdf):
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    landuse_gdf.to_file(shp_dir / "lu.shp")
    zpath = tmp_path / "lu.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f.name)
    gdf = read_vector(zpath)
    assert len(gdf) == 3 and gdf.crs is not None


def test_incomplete_shapefile_zip(tmp_path, landuse_gdf):
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    landuse_gdf.to_file(shp_dir / "lu.shp")
    zpath = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(shp_dir / "lu.shp", "lu.shp")
    with pytest.raises(VectorReadError, match="incomplete"):
        read_vector(zpath)


def test_corrupt_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(VectorReadError, match="corrupt"):
        read_vector(p)


def test_csv_lat_lon(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text("name,Latitude,Longitude\na,24.9157,67.0920\nb,24.9160,67.0925\n")
    gdf = read_vector(p)
    assert len(gdf) == 2 and gdf.crs.to_epsg() == 4326
    assert gdf.geometry.iloc[0].x == pytest.approx(67.0920)


def test_csv_wkt_with_assumed_crs(tmp_path, origin):
    ox, oy = origin
    p = tmp_path / "w.csv"
    p.write_text(f'id,wkt\n1,"POINT ({ox} {oy})"\n')
    gdf = read_vector(p, assumed_crs="EPSG:32642")
    assert gdf.crs.to_epsg() == 32642


def test_csv_without_coordinates(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(VectorReadError, match="latitude/longitude"):
        read_vector(p)


def test_unsupported_and_missing(tmp_path):
    p = tmp_path / "x.dwg"
    p.write_bytes(b"x")
    with pytest.raises(VectorReadError, match="Unsupported"):
        read_vector(p)
    with pytest.raises(VectorReadError, match="not found"):
        read_vector(tmp_path / "nope.geojson")


def test_garbage_geojson(tmp_path):
    p = tmp_path / "bad.geojson"
    p.write_text("{this is not json")
    with pytest.raises(VectorReadError):
        read_vector(p)


def test_find_column_case_insensitive():
    assert find_column(["ID", "LandUse"], ("class", "landuse")) == "LandUse"
    assert find_column(["a"], ("b",)) is None


def test_metric_crs_from_wgs84(landuse_gdf):
    wgs = landuse_gdf.to_crs(4326)
    metric = to_metric_crs(wgs)
    assert metric.crs.is_projected
    assert metric.area.sum() == pytest.approx(30_000, rel=1e-3)
    assert isinstance(metric, gpd.GeoDataFrame)


def test_csv_excel_encodings(tmp_path):
    bom = tmp_path / "bom.csv"
    bom.write_bytes("﻿name,lat,lon\nCafé,24.9157,67.0920\n".encode("utf-8"))
    gdf = read_vector(bom)
    assert list(gdf.columns)[0] == "name" and gdf["name"].iloc[0] == "Café"
    ansi = tmp_path / "ansi.csv"
    ansi.write_bytes("name,lat,lon\nCafé,24.9157,67.0920\n".encode("cp1252"))
    assert read_vector(ansi)["name"].iloc[0] == "Café"
