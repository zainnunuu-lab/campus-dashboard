"""Overview: headline figures, a context map and the dataset catalogue."""

import pandas as pd
import streamlit as st

from campus_dashboard import analysis, catalog, charts, config, maps, ui

ui.page_header(
    "Campus Analytics Overview",
    f"Drone Based Campus Analytics - {config.CAMPUS_NAME}",
)

all_ds = ui.datasets()
project_ds = [d for d in all_ds if not d.is_demo]
demo_ds = [d for d in all_ds if d.is_demo]

if not project_ds:
    st.info(
        "**No project survey data has been added yet.** "
        + ("Every figure on this page comes from synthetic DEMO data. " if demo_ds else "")
        + "Add orthomosaics and GIS layers on the **Data Manager** page.",
        icon=":material/info:",
    )

landuse = ui.project_first(catalog.by_category(all_ds, "landuse"))
boundary = ui.project_first(catalog.by_category(all_ds, "boundary"))
buildings = ui.project_first(catalog.by_category(all_ds, "buildings"))
trees = ui.project_first(catalog.by_category(all_ds, "trees"))
rasters = catalog.by_category(all_ds, "orthomosaic")

used = [d for d in (landuse[:1] + boundary[:1] + buildings[:1] + trees[:1]) if d]
ui.demo_banner(used)

# ---------------------------------------------------------------- headline figures
table = None
lu_gdf = ui.load_vector(landuse[0]) if landuse else None
bnd_gdf = ui.load_vector(boundary[0]) if boundary else None
if lu_gdf is not None:
    try:
        table = analysis.landuse_area_table(lu_gdf, boundary=bnd_gdf)
    except Exception as exc:
        st.error(f"Land-use statistics could not be computed: {exc}")

bld_gdf = ui.load_vector(buildings[0]) if buildings else None
tree_gdf = ui.load_vector(trees[0]) if trees else None


def tag(ds_list):
    return " [DEMO]" if ds_list and ds_list[0].is_demo else ""


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Datasets", len(all_ds), help=f"{len(project_ds)} project, {len(demo_ds)} demo")
c2.metric("Orthomosaics", len(rasters), help="Georeferenced raster mosaics available")
if table is not None:
    gc = analysis.green_cover_from_table(table)
    c3.metric("Mapped area" + tag(landuse), f"{gc.total_m2 / 10_000:,.2f} ha")
    c4.metric("Green cover" + tag(landuse), f"{gc.percent:.1f} %",
              help="Share of mapped land-use area in green classes (polygon-based). Not validated.")
else:
    c3.metric("Mapped area", "-")
    c4.metric("Green cover", "-")
c5.metric("Buildings" + tag(buildings), f"{len(bld_gdf):,}" if bld_gdf is not None else "-")

# ---------------------------------------------------------------- map + chart
left, right = st.columns([3, 2])
with left:
    st.subheader("Study area")
    m = maps.base_map("OpenStreetMap")
    bounds = None
    if lu_gdf is not None:
        wgs = lu_gdf.to_crs(config.DISPLAY_CRS)
        maps.add_vector_layer(m, wgs, landuse[0].meta.display_title, "landuse")
        minx, miny, maxx, maxy = wgs.total_bounds
        bounds = [[miny, minx], [maxy, maxx]]
    if bnd_gdf is not None:
        maps.add_vector_layer(m, bnd_gdf.to_crs(config.DISPLAY_CRS), boundary[0].meta.display_title, "boundary")
    legend = None
    if table is not None:
        legend = maps.landuse_legend_items(set(table["class"]))
    maps.finish(m, legend, bounds)
    ui.render_map(m, height=460)
with right:
    st.subheader("Land-use composition")
    if table is not None:
        st.plotly_chart(charts.landuse_area_bar(table, demo=landuse[0].is_demo), width="stretch")
    else:
        ui.no_data("land-use layer")

# ---------------------------------------------------------------- catalogue
st.subheader("Dataset catalogue")
if all_ds:
    rows = [
        {
            "Status": "DEMO (synthetic)" if d.is_demo else "Project data",
            "Title": d.meta.title,
            "Type": d.kind,
            "Category": config.CATEGORIES.get(d.meta.category, d.meta.category),
            "Capture date": d.meta.capture_date or "-",
            "Source": d.meta.source or "Not recorded",
            "Accuracy": d.meta.accuracy_statement,
            "File": ui.file_label(d.path),
            "Size (MB)": round(d.size_mb, 2),
        }
        for d in all_ds
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    ui.no_data("datasets")

with st.expander("How the dashboard fits the project workflow"):
    st.markdown(
        """
1. **Data acquisition** - drone flights over campus (survey team).
2. **Processing** - photogrammetry (e.g. WebODM / Agisoft Metashape / Pix4D) produces a georeferenced
   orthomosaic (GeoTIFF), elevation models and 3D meshes / point clouds.
3. **GIS analysis** - land-use digitising or classification, green-cover mapping (QGIS / Python).
4. **Dashboard (this application)** - validates, catalogues and visualises those outputs, computes
   area statistics, and records provenance (metadata) for each dataset.

Accuracy figures are shown **only** when they are computed against independent reference data
(see *Green Cover - Validation*). No result on this dashboard is validated by default.
"""
    )
