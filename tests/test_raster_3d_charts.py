import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from campus_dashboard import assets3d, charts, maps, raster_io
from campus_dashboard.analysis import landuse_area_table


def test_summary_and_preview(rgb_tif):
    s = raster_io.summarize_raster(rgb_tif)
    assert (s.width, s.height, s.band_count) == (100, 100, 3)
    assert s.pixel_unit == "metre"
    png, bounds = raster_io.preview_for_map(rgb_tif, max_px=50)
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA" and max(img.size) <= 50
    (south, west), (north, east) = bounds
    assert south < north and west < east and 24 < south < 26 and 66 < west < 68


def test_read_for_analysis_reprojects_geographic(tmp_path, rgb_tif):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject

    dst_path = tmp_path / "wgs.tif"
    with rasterio.open(rgb_tif) as src:
        transform, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        prof = src.profile | {"crs": "EPSG:4326", "transform": transform, "width": w, "height": h}
        with rasterio.open(dst_path, "w", **prof) as dst:
            for b in range(1, 4):
                reproject(rasterio.band(src, b), rasterio.band(dst, b))
    view = raster_io.read_for_analysis(dst_path)
    assert view.crs.is_projected
    assert 0.5 < abs(view.transform.a) < 2  # about 1 m pixels again


def test_colour_ramp_single_band():
    view = raster_io.RasterView(data=np.linspace(0, 1, 16, dtype="float32").reshape(1, 4, 4),
                                valid=np.ones((4, 4), bool), transform=None, crs=None, bounds=(0, 0, 1, 1))
    img = Image.open(io.BytesIO(raster_io.render_png(view, "elevation")))
    assert img.size == (4, 4)


def test_glb_and_point_cloud(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(10, 20, 5))
    glb = tmp_path / "b.glb"
    glb.write_bytes(trimesh.Scene([box]).export(file_type="glb"))
    info = assets3d.mesh_info(glb)
    assert info.vertices == 8 and info.faces == 12 and not info.is_point_cloud
    assert assets3d.to_glb_bytes(glb)[:4] == b"glTF"
    html = assets3d.model_viewer_html(glb.read_bytes(), "test <b>", banner="DEMO")
    assert assets3d.MODEL_VIEWER_JS in html and "&lt;b&gt;" in html and "DEMO" in html

    stl = tmp_path / "b.stl"
    box.export(stl)
    assert assets3d.to_glb_bytes(stl)[:4] == b"glTF"

    ply = tmp_path / "c.ply"
    pts = np.random.default_rng(0).random((500, 3))
    ply.write_bytes(trimesh.PointCloud(pts).export(file_type="ply"))
    xyz, rgb, total = assets3d.read_point_cloud(ply, max_points=100)
    assert total == 500 and xyz.shape == (100, 3)
    assert assets3d.mesh_info(ply).is_point_cloud


def test_bad_3d_file(tmp_path):
    p = tmp_path / "x.glb"
    p.write_bytes(b"nonsense")
    with pytest.raises(assets3d.AssetReadError):
        assets3d.mesh_info(p)
    with pytest.raises(assets3d.AssetReadError):
        assets3d.asset_type(tmp_path / "x.fbx")


def test_building_extrusion(landuse_gdf):
    frame, field = assets3d.building_extrusion_frame(landuse_gdf.assign(height_m=[10, 20, None]))
    assert field == "height_m" and len(frame) == 2
    with pytest.raises(ValueError):
        assets3d.building_extrusion_frame(landuse_gdf)
    deck = assets3d.building_deck(frame, (24.9, 67.1), demo=True, ground=assets3d.context_frame(landuse_gdf))
    assert len(deck.layers) == 2


def test_charts_label_demo(landuse_gdf):
    table = landuse_area_table(landuse_gdf)
    fig = charts.landuse_area_bar(table, demo=True)
    assert "DEMO" in fig.layout.title.text
    assert "DEMO" not in charts.landuse_area_bar(table, demo=False).layout.title.text
    charts.green_split_bar(2, 3, False, "t")
    charts.index_histogram(np.array([0.1, 0.2, np.nan]), 0.15, "ExG", False)
    charts.numeric_histogram(pd.Series([1, 2, 3]), "h", False)
    fig = charts.category_counts_bar(pd.Series(list("abcdefghijklmnopqrst")), "c", False, top=5)
    assert "Other" in list(fig.data[0].y)


def test_map_build_and_escape(landuse_gdf):
    m = maps.base_map("OpenStreetMap")
    gdf = landuse_gdf.to_crs(4326).assign(note=["<script>x</script>", "ok", None])
    maps.add_vector_layer(m, gdf, "LU", "landuse")
    maps.finish(m, maps.landuse_legend_items({"building", "weird"}), [[24.9, 67.0], [24.92, 67.1]])
    html = m.get_root().render()
    assert "<script>x</script>" not in html
    assert "Unknown class" in html and "Legend" in html


@pytest.mark.parametrize("url,ok", [
    ("http://localhost:8080/{z}/{x}/{y}.png", True),
    ("ftp://x/{z}/{x}/{y}", False),
    ("https://x/tiles.png", False),
])
def test_xyz_url_validation(url, ok):
    assert (maps.validate_xyz_url(url) is None) is ok
