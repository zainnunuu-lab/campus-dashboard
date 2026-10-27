import json

import pytest

from campus_dashboard import catalog, config
from campus_dashboard.metadata import DatasetMetadata, load_metadata, save_metadata, sidecar_path


def test_roundtrip(tmp_path):
    f = tmp_path / "x.geojson"
    f.write_text("{}")
    meta = DatasetMetadata(title="Survey", kind="vector", category="landuse", source="Flight 1", gsd_cm=2.5)
    save_metadata(f, meta)
    loaded = load_metadata(f, "vector")
    assert loaded.title == "Survey" and loaded.gsd_cm == 2.5 and not loaded.is_demo
    assert loaded.accuracy_statement == config.DEFAULT_ACCURACY_STATEMENT


def test_missing_and_corrupt_sidecar(tmp_path):
    f = tmp_path / "campus_roads.geojson"
    f.write_text("{}")
    assert load_metadata(f, "vector").title == "Campus Roads"
    sidecar_path(f).write_text("{broken")
    meta = load_metadata(f, "vector")
    assert "No metadata" in meta.notes


def test_unknown_keys_and_category_are_tolerated(tmp_path):
    f = tmp_path / "a.tif"
    f.write_text("x")
    sidecar_path(f).write_text(json.dumps({"title": "A", "kind": "raster", "category": "nonsense", "extra": 1}))
    meta = load_metadata(f, "raster")
    assert meta.category == "other"


def test_demo_folder_forces_demo_flag(data_tree):
    f = data_tree / "demo" / "vector" / "a.geojson"
    f.write_text("{}")
    save_metadata(f, DatasetMetadata(title="Sneaky", kind="vector", is_demo=False))
    meta = load_metadata(f, "vector")
    assert meta.is_demo and meta.display_title.startswith("[DEMO]")


def test_discover_and_filters(data_tree):
    (data_tree / "user" / "vector" / "lu.geojson").write_text("{}")
    (data_tree / "user" / "vector" / "lu.geojson.meta.json").write_text(
        json.dumps({"title": "LU", "kind": "vector", "category": "landuse"}))
    (data_tree / "user" / "vector" / "readme.txt").write_text("ignored")
    (data_tree / "demo" / "raster" / "d.tif").write_text("x")
    found = catalog.discover(include_demo=True)
    assert {d.path.name for d in found} == {"lu.geojson", "d.tif"}
    assert [d.path.name for d in catalog.discover(include_demo=False)] == ["lu.geojson"]
    assert len(catalog.by_category(found, "landuse")) == 1


def test_safe_filename_and_unique_path(tmp_path):
    assert catalog.safe_filename("../../etc/pa ss?wd.GeoJSON") == "pa_ss_wd.geojson"
    assert catalog.safe_filename("....tif") == "dataset.tif"
    (tmp_path / "a.tif").write_text("x")
    assert catalog.unique_path(tmp_path, "a.tif").name == "a_1.tif"


def test_delete_user_but_not_demo(data_tree):
    u = data_tree / "user" / "vector" / "u.geojson"
    u.write_text("{}")
    save_metadata(u, DatasetMetadata(title="U", kind="vector"))
    d = data_tree / "demo" / "vector" / "d.geojson"
    d.write_text("{}")
    ds = {x.path.name: x for x in catalog.discover()}
    with pytest.raises(PermissionError):
        catalog.delete_user_dataset(ds["d.geojson"])
    catalog.delete_user_dataset(ds["u.geojson"])
    assert not u.exists() and not sidecar_path(u).exists() and d.exists()
