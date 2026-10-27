"""Orthomosaic & GIS map viewer."""

import streamlit as st

from campus_dashboard import catalog, config, maps, raster_io, ui
from campus_dashboard.vector_io import find_column

ui.page_header(
    "Orthomosaic & GIS Map",
    "Georeferenced orthomosaics and vector GIS layers on an interactive map. "
    "Use the layer control (top right) to toggle layers and the ruler (top left) to measure.",
)


@st.cache_data(show_spinner="Preparing raster preview...", max_entries=16)
def raster_preview(path: str, mtime: float, ramp: str):
    return raster_io.preview_for_map(path, ramp=ramp)


@st.cache_data(max_entries=32)
def raster_summary(path: str, mtime: float):
    return raster_io.summarize_raster(path)


all_ds = ui.datasets()
rasters = ui.project_first([d for d in all_ds if d.kind == "raster"])
vectors = ui.project_first([d for d in all_ds if d.kind == "vector"])

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 3, 3])
    basemap = c1.selectbox("Basemap", list(config.BASEMAPS), index=0)
    sel_rasters = c2.multiselect(
        "Raster layers",
        [d.id for d in rasters],
        default=[d.id for d in rasters[:1]],
        format_func=lambda i: next(d.meta.display_title for d in rasters if d.id == i),
    )
    sel_vectors = c3.multiselect(
        "Vector layers",
        [d.id for d in vectors],
        default=[d.id for d in vectors if d.meta.category in ("boundary", "buildings", "paths")][:3],
        format_func=lambda i: next(d.meta.display_title for d in vectors if d.id == i),
    )
    c4, c5, c6 = st.columns([2, 3, 3])
    opacity = c4.slider("Raster opacity", 0.0, 1.0, 1.0, 0.05)
    height = c5.slider("Map height (px)", 400, 1000, 650, 50)
    xyz = c6.text_input(
        "Optional XYZ tile URL",
        placeholder="http://localhost:8080/tiles/{z}/{x}/{y}.png",
        help="For large tiled orthomosaics served by e.g. WebODM, TiTiler or a GeoServer XYZ endpoint.",
    )

chosen_r = [d for d in rasters if d.id in sel_rasters]
chosen_v = [d for d in vectors if d.id in sel_vectors]
ui.demo_banner(chosen_r + chosen_v)

if not rasters and not vectors:
    ui.no_data("orthomosaic or GIS layer")

m = maps.base_map(basemap)
legend: list[tuple[str, str, str]] = []
extents: list = []  # [[south, west], [north, east]] per layer

for ds in chosen_r:
    ramp = "elevation" if ds.meta.category == "dsm" else "index"
    try:
        png, b = raster_preview(str(ds.path), ds.mtime, ramp)
    except raster_io.RasterReadError as exc:
        st.error(f"**{ds.meta.display_title}**: {exc}")
        continue
    except Exception as exc:
        st.error(f"**{ds.meta.display_title}** could not be displayed: {exc}")
        continue
    maps.add_raster_overlay(m, png, b, ds.meta.display_title, opacity)
    extents.append(b)

if xyz.strip():
    err = maps.validate_xyz_url(xyz)
    if err:
        st.error(err)
    else:
        maps.add_xyz_layer(m, xyz.strip(), "User XYZ tiles")

for ds in chosen_v:
    gdf = ui.load_vector_wgs84(ds)
    if gdf is None or gdf.empty:
        continue
    cat = ds.meta.category
    maps.add_vector_layer(m, gdf, ds.meta.display_title, cat)
    minx, miny, maxx, maxy = gdf.total_bounds
    extents.append([[miny, minx], [maxy, maxx]])
    if cat == "landuse":
        field = find_column(gdf.columns, config.LANDUSE_CLASS_FIELDS)
        present = set(gdf[field].astype(str).str.strip().str.lower()) if field else None
        legend += maps.landuse_legend_items(present)
    else:
        geom = gdf.geom_type.iloc[0]
        shape = "point" if "Point" in geom else "line" if "Line" in geom else "box"
        legend.append((ds.meta.display_title, config.LAYER_COLORS.get(cat, config.LAYER_COLORS["other"]), shape))

bounds = None
if extents:
    bounds = [[min(e[0][0] for e in extents), min(e[0][1] for e in extents)],
              [max(e[1][0] for e in extents), max(e[1][1] for e in extents)]]
maps.finish(m, legend, bounds)
ui.render_map(m, height=height)
st.caption(
    "Raster layers are shown as decimated previews (max "
    f"{config.RASTER_PREVIEW_MAX_PX} px) warped to WGS84; full-resolution files stay unchanged on disk. "
    "Basemap imagery is third-party reference imagery, not project data."
)

if chosen_r or chosen_v:
    st.subheader("Layer details")
for ds in chosen_r:
    try:
        s = raster_summary(str(ds.path), ds.mtime)
        st.markdown(f"**{ds.meta.display_title}** - {s.size_label}, CRS `{s.crs}`")
    except Exception as exc:
        st.error(f"{ds.meta.display_title}: {exc}")
    ui.metadata_panel(ds.meta)
for ds in chosen_v:
    ui.metadata_panel(ds.meta)
