import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from campus_dashboard import analysis
from campus_dashboard.raster_io import RasterView, read_for_analysis


def test_landuse_area_table(landuse_gdf):
    t = analysis.landuse_area_table(landuse_gdf.to_crs(4326))
    assert t["area_ha"].sum() == pytest.approx(3.0, rel=1e-3)
    assert t["percent"].sum() == pytest.approx(100)
    gc = analysis.green_cover_from_table(t)
    assert gc.percent == pytest.approx(200 / 3, rel=1e-3)


def test_landuse_unknown_class_kept(landuse_gdf):
    gdf = landuse_gdf.copy()
    gdf.loc[0, "class"] = "Helipad "
    t = analysis.landuse_area_table(gdf)
    row = t[t["class"] == "helipad"].iloc[0]
    assert "unknown" in row["label"] and not row["is_green"]


def test_landuse_clip_to_boundary(landuse_gdf, origin):
    ox, oy = origin
    boundary = gpd.GeoDataFrame(geometry=[box(ox, oy, ox + 150, oy + 100)], crs="EPSG:32642")
    t = analysis.landuse_area_table(landuse_gdf, boundary=boundary)
    assert t["area_m2"].sum() == pytest.approx(15_000, rel=1e-6)


def test_landuse_missing_field(landuse_gdf):
    with pytest.raises(ValueError):
        analysis.landuse_area_table(landuse_gdf.rename(columns={"class": "zzz"}))


def test_layer_summary_mixed(origin):
    from shapely.geometry import LineString

    ox, oy = origin
    gdf = gpd.GeoDataFrame(geometry=[box(ox, oy, ox + 10, oy + 10), LineString([(ox, oy), (ox + 30, oy + 40)]),
                                     Point(ox, oy)], crs="EPSG:32642")
    s = analysis.layer_summary(gdf)
    assert s["area_m2"] == pytest.approx(100)
    assert s["length_m"] == pytest.approx(50)
    assert (s["polygons"], s["lines"], s["points"]) == (1, 1, 1)


def _view(data):
    valid = np.isfinite(data).all(axis=0)
    return RasterView(data=data.astype("float32"), valid=valid, transform=from_origin(0, 10, 1, 1),
                      crs=None, bounds=(0, 0, 10, 10))


def test_exg_and_vari_rank_green_higher():
    green = np.array([[60.0]]), np.array([[160.0]]), np.array([[50.0]])
    grey = np.array([[128.0]]), np.array([[128.0]]), np.array([[128.0]])
    data = np.stack([np.hstack([g, r]) for g, r in zip(green, grey)])[:, :, :]
    view = _view(data)
    for method in ("ExG", "VARI"):
        idx = analysis.vegetation_index(view, method)
        assert idx[0, 0] > idx[0, 1]


def test_index_needs_rgb():
    view = _view(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        analysis.vegetation_index(view, "ExG")
    assert analysis.vegetation_index(view, "Band 1 as index").shape == (2, 2)


def test_otsu_bimodal():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0, 0.05, 5000), rng.normal(1, 0.05, 5000)])
    assert 0.3 < analysis.otsu_threshold(values) < 0.7
    with pytest.raises(ValueError):
        analysis.otsu_threshold(np.array([np.nan]))


def test_raster_green_cover_end_to_end(rgb_tif):
    view = read_for_analysis(rgb_tif)
    idx = analysis.vegetation_index(view, "ExG")
    mask = analysis.green_mask(idx, analysis.otsu_threshold(idx))
    gc = analysis.mask_area(mask, view.valid, view.transform)
    assert gc.total_m2 == pytest.approx(100 * 100, rel=0.01)  # 100 px at 1 m
    assert gc.percent == pytest.approx(50, abs=2)


def test_accuracy_report(rgb_tif, origin):
    view = read_for_analysis(rgb_tif)
    idx = analysis.vegetation_index(view, "ExG")
    mask = analysis.green_mask(idx, analysis.otsu_threshold(idx))
    ox, oy = origin
    # raster spans x: ox..ox+100, y: oy..oy+100; left half green
    pts = gpd.GeoDataFrame(
        {"is_green": [1, 1, 0, 0, 1]},
        geometry=[Point(ox + 10, oy + 50), Point(ox + 20, oy + 20), Point(ox + 80, oy + 50),
                  Point(ox + 90, oy + 10), Point(ox + 75, oy + 75)],  # last one is a deliberate miss
        crs="EPSG:32642",
    )
    rep = analysis.accuracy_against_points(mask, view.valid, view, pts, "is_green")
    assert rep.n_used == 5
    assert (rep.tp, rep.fn, rep.fp, rep.tn) == (2, 1, 0, 2)
    assert rep.overall_accuracy == pytest.approx(0.8)
    assert rep.precision == pytest.approx(1.0)
    assert rep.recall == pytest.approx(2 / 3)
    assert -1 <= rep.kappa <= 1
    outside = pts.assign(geometry=[Point(ox - 500, oy)] * 5)
    assert analysis.accuracy_against_points(mask, view.valid, view, outside, "is_green").n_used == 0


def test_add_metric_measures(landuse_gdf):
    out = analysis.add_metric_measures(landuse_gdf.to_crs(4326))
    assert out["area_m2"].sum() == pytest.approx(30_000, rel=1e-3)
