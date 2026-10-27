"""Shared Streamlit helpers: data access with caching, banners, messages, selectors."""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components

from . import catalog, config
from .catalog import Dataset
from .metadata import DatasetMetadata
from .validation import ValidationResult
from .vector_io import VectorReadError, read_vector, to_display_crs

INCLUDE_DEMO_KEY = "include_demo"


# --------------------------------------------------------------------------- data access
def include_demo() -> bool:
    return st.session_state.get(INCLUDE_DEMO_KEY, True)


def datasets() -> list[Dataset]:
    """Datasets on disk. Not cached: listing a folder is cheap and must reflect uploads."""
    return catalog.discover(include_demo=include_demo())


def project_first(items: list[Dataset]) -> list[Dataset]:
    """Order so real project data is preferred over demo data in default selections."""
    return sorted(items, key=lambda d: (d.is_demo, d.meta.title.lower()))


@st.cache_data(show_spinner="Reading vector layer...", max_entries=64)
def _load_vector_cached(path: str, mtime: float) -> gpd.GeoDataFrame:
    return read_vector(path)


def load_vector(ds: Dataset) -> gpd.GeoDataFrame | None:
    """Read a layer, showing an error (instead of crashing the page) when it fails."""
    try:
        return _load_vector_cached(str(ds.path), ds.mtime)
    except VectorReadError as exc:
        st.error(f"Could not load **{ds.meta.display_title}**: {exc}")
    except Exception as exc:  # unexpected - still keep the rest of the page usable
        st.error(f"Unexpected error while loading **{ds.meta.display_title}**: {exc}")
    return None


def load_vector_wgs84(ds: Dataset) -> gpd.GeoDataFrame | None:
    gdf = load_vector(ds)
    if gdf is None:
        return None
    try:
        return to_display_crs(gdf)
    except VectorReadError as exc:
        st.error(f"**{ds.meta.display_title}**: {exc}")
        return None


# --------------------------------------------------------------------------- page chrome
def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def demo_banner(used: list[Dataset]) -> None:
    """Show a prominent warning whenever demo data contributes to the page."""
    demo = [d for d in used if d.is_demo]
    if not demo:
        return
    names = ", ".join(d.meta.title for d in demo)
    st.warning(
        f"**DEMO DATA ON THIS PAGE - not survey results.** The following are synthetic: {names}. "
        f"{config.DEMO_DISCLAIMER} Turn off *Include demo data* in the sidebar to hide them.",
        icon=":material/science:",
    )


def no_data(what: str) -> None:
    st.info(
        f"No {what} available yet. Upload one on the **Data Manager** page"
        + ("" if include_demo() else ", or enable *Include demo data* in the sidebar")
        + ".",
        icon=":material/upload_file:",
    )


def show_validation(res: ValidationResult, title: str = "Validation") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for msg in res.errors:
            st.error(msg, icon=":material/error:")
        for msg in res.warnings:
            st.warning(msg, icon=":material/warning:")
        for msg in res.info:
            st.markdown(f"- {msg}")
        if res.ok and not res.warnings:
            st.success("All checks passed.", icon=":material/check_circle:")


def status_badge(meta: DatasetMetadata) -> str:
    return ":red-background[DEMO - synthetic]" if meta.is_demo else ":green-background[Project data]"


def metadata_panel(meta: DatasetMetadata, expanded: bool = False) -> None:
    with st.expander(f"Metadata - {meta.display_title}", expanded=expanded):
        st.markdown(status_badge(meta))
        rows = {
            "Category": config.CATEGORIES.get(meta.category, meta.category),
            "Description": meta.description or "-",
            "Source": meta.source or "Not recorded",
            "Capture date": meta.capture_date or "Not recorded",
            "Sensor": meta.sensor or "Not recorded",
            "Processing software": meta.processing_software or "Not recorded",
            "GSD (cm/px, as reported)": f"{meta.gsd_cm:g}" if meta.gsd_cm else "Not recorded",
            "CRS": meta.crs or "-",
            "Accuracy": meta.accuracy_statement or config.DEFAULT_ACCURACY_STATEMENT,
            "Licence": meta.license or "-",
            "Original file": meta.original_filename or "-",
            "Added": meta.created,
            "Notes": meta.notes or "-",
        }
        st.table({"Field": list(rows), "Value": list(rows.values())})


def dataset_select(options: list[Dataset], label: str, key: str, multi: bool = False):
    labels = {d.id: d.meta.display_title for d in options}
    if multi:
        chosen = st.multiselect(label, list(labels), default=list(labels), format_func=labels.get, key=key)
        return [d for d in options if d.id in chosen]
    chosen = st.selectbox(label, list(labels), format_func=labels.get, key=key)
    return next((d for d in options if d.id == chosen), None)


def embed_html(document: str, height: int) -> None:
    """Embed a self-contained HTML document built by this app (never raw user HTML)."""
    if hasattr(st, "iframe"):  # Streamlit >= 1.52
        st.iframe(document, height=height)
    else:
        components.html(document, height=height)


def render_map(m: folium.Map, height: int = 620) -> None:
    """Embed the full Leaflet document (keeps the custom legend)."""
    embed_html(m.get_root().render(), height=height)


def file_label(path: Path) -> str:
    """Short display path; works even if DATA_DIR was moved outside the project."""
    for base, prefix in ((config.DATA_DIR, "data/"), (config.PROJECT_ROOT, "")):
        try:
            return prefix + path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)
