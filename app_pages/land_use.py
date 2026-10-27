"""Land-use / land-cover analysis from classified polygons."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from campus_dashboard import analysis, catalog, charts, config, maps, ui
from campus_dashboard.vector_io import find_column

ui.page_header(
    "Land Use",
    "Area statistics per land-use class, computed in a local UTM projection from classified polygons.",
)

all_ds = ui.datasets()
landuse = ui.project_first(catalog.by_category(all_ds, "landuse"))
boundaries = ui.project_first(catalog.by_category(all_ds, "boundary"))
if not landuse:
    ui.no_data("land-use layer (category: Land-use / land-cover polygons)")
    st.stop()

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        ds = ui.dataset_select(landuse, "Land-use layer", key="lu_ds")
    gdf = ui.load_vector(ds)
    if gdf is None:
        st.stop()
    attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    detected = find_column(attr_cols, config.LANDUSE_CLASS_FIELDS)
    with c2:
        if not attr_cols:
            st.error("This layer has no attribute columns, so no class field is available.")
            st.stop()
        class_field = st.selectbox(
            "Class attribute", attr_cols, index=attr_cols.index(detected) if detected else 0,
            help="Column holding the land-use class code.",
        )
    with c3:
        bnd_opts = ["(no clipping)"] + [d.id for d in boundaries]
        bnd_id = st.selectbox(
            "Clip to boundary", bnd_opts,
            format_func=lambda i: i if i.startswith("(") else next(d.meta.display_title for d in boundaries if d.id == i),
        )
    bnd_ds = next((d for d in boundaries if d.id == bnd_id), None)
    bnd_gdf = ui.load_vector(bnd_ds) if bnd_ds else None

    classes_present = sorted(analysis.normalise_classes(gdf[class_field]).unique())
    label_of = {c: config.LANDUSE_CLASSES[c].label if c in config.LANDUSE_CLASSES else f"{c} (unknown)"
                for c in classes_present}
    chosen = st.multiselect("Filter classes", classes_present, default=classes_present, format_func=label_of.get)

ui.demo_banner([ds] + ([bnd_ds] if bnd_ds else []))

if detected is None:
    st.warning(
        f"No standard class field ({', '.join(config.LANDUSE_CLASS_FIELDS)}) was found; using '{class_field}'. "
        "Values that do not match the configured class codes are shown as 'unknown'.",
    )

filtered = gdf[analysis.normalise_classes(gdf[class_field]).isin(chosen)]
if filtered.empty:
    st.info("No polygons match the selected classes.")
    st.stop()

try:
    table = analysis.landuse_area_table(filtered, class_field=class_field, boundary=bnd_gdf)
except Exception as exc:
    st.error(f"Area statistics could not be computed: {exc}")
    st.stop()

gc = analysis.green_cover_from_table(table)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Area (selected classes)", f"{gc.total_m2 / 10_000:,.2f} ha")
m2.metric("Green share", f"{gc.percent:.1f} %", help="Green classes: "
          + ", ".join(c.label for c in config.LANDUSE_CLASSES.values() if c.is_green))
m3.metric("Polygons", f"{int(table['features'].sum()):,}")
m4.metric("Classes", len(table))
if set(chosen) != set(classes_present):
    st.caption("Percentages are relative to the *selected* classes only.")

left, right = st.columns([3, 2])
with left:
    m = maps.base_map("OpenStreetMap")
    wgs = filtered.to_crs(config.DISPLAY_CRS)
    maps.add_vector_layer(m, wgs, ds.meta.display_title, "landuse", class_field=class_field)
    if bnd_gdf is not None:
        maps.add_vector_layer(m, bnd_gdf.to_crs(config.DISPLAY_CRS), bnd_ds.meta.display_title, "boundary")
    minx, miny, maxx, maxy = wgs.total_bounds
    maps.finish(m, maps.landuse_legend_items(set(table["class"])), [[miny, minx], [maxy, maxx]])
    ui.render_map(m, height=520)
with right:
    st.plotly_chart(charts.landuse_area_bar(table, demo=ds.is_demo), width="stretch")

st.subheader("Class statistics")
show = table.drop(columns=["color"]).rename(columns={
    "class": "Class code", "label": "Class", "features": "Polygons", "area_m2": "Area (m²)",
    "area_ha": "Area (ha)", "percent": "Share (%)", "is_green": "Green class",
})
st.dataframe(
    show, hide_index=True, width="stretch",
    column_config={
        "Area (m²)": st.column_config.NumberColumn(format="%.0f"),
        "Area (ha)": st.column_config.NumberColumn(format="%.3f"),
        "Share (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    },
)
prefix = "DEMO_" if ds.is_demo else ""
st.download_button("Download statistics (CSV)", show.to_csv(index=False).encode("utf-8"),
                   file_name=f"{prefix}landuse_statistics.csv", mime="text/csv")
st.caption(
    "Areas are planimetric (2D) and are computed in a local UTM zone. They are only as accurate as the "
    "polygon digitising / classification, which has not been validated here."
)

# ---------------------------------------------------------------- comparison
if len(landuse) >= 2:
    st.subheader("Compare two land-use layers")
    st.caption("For example two survey dates. Both layers are processed with the same class field detection.")
    cc1, cc2 = st.columns(2)
    with cc1:
        a = ui.dataset_select(landuse, "Layer A", key="cmp_a")
    with cc2:
        b = ui.dataset_select([d for d in landuse if d.id != a.id], "Layer B", key="cmp_b")
    ga, gb = ui.load_vector(a), ui.load_vector(b)
    if ga is not None and gb is not None:
        try:
            ta = analysis.landuse_area_table(ga).set_index("class")
            tb = analysis.landuse_area_table(gb).set_index("class")
        except ValueError as exc:
            st.error(str(exc))
        else:
            cmp = pd.DataFrame({
                "Class": [config.LANDUSE_CLASSES[c].label if c in config.LANDUSE_CLASSES else c
                          for c in ta.index.union(tb.index)],
                "A (ha)": ta["area_ha"].reindex(ta.index.union(tb.index)).fillna(0).values,
                "B (ha)": tb["area_ha"].reindex(ta.index.union(tb.index)).fillna(0).values,
            })
            cmp["Change (ha)"] = cmp["B (ha)"] - cmp["A (ha)"]
            ui.demo_banner([a, b])
            fig = go.Figure()
            for col, color in (("A (ha)", "#2a78d6"), ("B (ha)", "#eb6834")):
                fig.add_bar(y=cmp["Class"], x=cmp[col], name=f"{col[0]}: {(a if col[0] == 'A' else b).meta.display_title}",
                            orientation="h", marker={"color": color, "line": {"color": "white", "width": 2}})
            fig.update_layout(barmode="group", barcornerradius=4, height=max(300, 50 * len(cmp) + 80),
                              margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend={"orientation": "h", "y": -0.15},
                              xaxis_title="Area (ha)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(cmp.round(3), hide_index=True, width="stretch")

ui.metadata_panel(ds.meta)
