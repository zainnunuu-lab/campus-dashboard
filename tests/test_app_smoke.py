"""Smoke tests: every page renders without an exception (uses the bundled demo data)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from campus_dashboard import config
from campus_dashboard.ui import INCLUDE_DEMO_KEY

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p.name for p in (ROOT / "app_pages").glob("*.py"))


@pytest.fixture(scope="session", autouse=True)
def demo_data():
    if not (config.DEMO_DIR / "vector" / "demo_landuse.geojson").exists():
        from scripts.generate_demo_data import main

        main()


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("with_demo", [True, False])
def test_page_renders(page, with_demo):
    at = AppTest.from_file(str(ROOT / "app_pages" / page), default_timeout=180)
    at.session_state[INCLUDE_DEMO_KEY] = with_demo
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_demo_banner_shown_on_overview():
    at = AppTest.from_file(str(ROOT / "app_pages" / "overview.py"), default_timeout=180)
    at.session_state[INCLUDE_DEMO_KEY] = True
    at.run()
    assert any("DEMO DATA ON THIS PAGE" in w.value for w in at.warning)


def test_no_demo_banner_when_demo_hidden():
    at = AppTest.from_file(str(ROOT / "app_pages" / "overview.py"), default_timeout=180)
    at.session_state[INCLUDE_DEMO_KEY] = False
    at.run()
    assert not any("DEMO DATA ON THIS PAGE" in w.value for w in at.warning)


def test_entry_point_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception


def test_delete_user_dataset_through_ui(data_tree, landuse_gdf):
    from campus_dashboard.metadata import DatasetMetadata, save_metadata
    from campus_dashboard.vector_io import write_geojson

    target = write_geojson(landuse_gdf, data_tree / "user" / "vector" / "temp.geojson")
    save_metadata(target, DatasetMetadata(title="Temp layer", kind="vector", category="landuse"))
    at = AppTest.from_file(str(ROOT / "app_pages" / "data_manager.py"), default_timeout=180)
    at.run()
    sel = at.selectbox(key="mgr_ds")
    assert sel.value == "user/vector/temp.geojson"  # project data is listed before demo data
    at.checkbox(key=f"del_{sel.value}").check().run()
    next(b for b in at.button if b.label == "Delete").click().run()
    assert not target.exists()
    assert any("Deleted 'Temp layer'" in s.value for s in at.success)
