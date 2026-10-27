"""Generic explorer for any vector analysis result: filters, summary, charts, export."""

import pandas as pd
import streamlit as st

from campus_dashboard import analysis, charts, config, maps, ui

ui.page_header(
    "Spatial Layers",
    "Explore any available vector result (buildings, paths, trees, survey points, other analysis outputs): "
    "filter by attributes, view summaries and export the selection.",
)

vectors = ui.project_first([d for d in ui.datasets() if d.kind == "vector"])
if not vectors:
    ui.no_data("vector layer")
    st.stop()

ds = ui.dataset_select(vectors, "Layer", key="sl_ds")
ui.demo_banner([ds])
gdf = ui.load_vector(ds)
if gdf is None or gdf.empty:
    st.stop()
try:
    gdf = analysis.add_metric_measures(gdf)
except Exception as exc:
    st.warning(f"Areas/lengths could not be computed: {exc}")

attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
numeric_cols = [c for c in attr_cols if pd.api.types.is_numeric_dtype(gdf[c]) and gdf[c].notna().any()]
cat_cols = [c for c in attr_cols if c not in numeric_cols and 1 < gdf[c].nunique(dropna=True) <= 50]

# ---------------------------------------------------------------- filters
filtered = gdf
with st.container(border=True):
    st.markdown("**Filters**")
    f1, f2 = st.columns(2)
    with f1:
        cat_col = st.selectbox("Filter by category", ["(none)"] + cat_cols, key=f"cat_{ds.id}")
        if cat_col != "(none)":
            values = sorted(gdf[cat_col].dropna().astype(str).unique())
            keep = st.multiselect(f"Values of '{cat_col}'", values, default=values, key=f"vals_{ds.id}_{cat_col}")
            filtered = filtered[filtered[cat_col].astype(str).isin(keep)]
    with f2:
        num_col = st.selectbox("Filter by numeric range", ["(none)"] + numeric_cols, key=f"num_{ds.id}")
        if num_col != "(none)":
            lo, hi = float(gdf[num_col].min()), float(gdf[num_col].max())
            if lo < hi:
                rng = st.slider(f"Range of '{num_col}'", lo, hi, (lo, hi), key=f"rng_{ds.id}_{num_col}")
                filtered = filtered[filtered[num_col].between(*rng)]
            else:
                st.caption(f"'{num_col}' has a single value ({lo:g}); no range to filter.")

if filtered.empty:
    st.info("No features match the current filters.")
    st.stop()

summary = analysis.layer_summary(filtered)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Features", f"{summary['features']:,}", delta=None if len(filtered) == len(gdf)
          else f"of {len(gdf):,}", delta_color="off")
m2.metric("Total area", f"{summary['area_m2'] / 10_000:,.3f} ha" if summary["polygons"] else "-")
m3.metric("Total length", f"{summary['length_m']:,.0f} m" if summary["lines"] else "-")
m4.metric("Points", f"{summary['points']:,}" if summary["points"] else "-")

left, right = st.columns([3, 2])
with left:
    m = maps.base_map("OpenStreetMap")
    wgs = filtered.to_crs(config.DISPLAY_CRS)
    maps.add_vector_layer(m, wgs, ds.meta.display_title, ds.meta.category)
    minx, miny, maxx, maxy = wgs.total_bounds
    pad = 0.0005
    maps.finish(m, None, [[miny - pad, minx - pad], [maxy + pad, maxx + pad]])
    ui.render_map(m, height=480)
with right:
    chart_cols = numeric_cols + cat_cols
    if chart_cols:
        col = st.selectbox("Chart attribute", chart_cols, key=f"chart_{ds.id}")
        if col in numeric_cols:
            st.plotly_chart(charts.numeric_histogram(filtered[col], col, ds.is_demo), width="stretch")
            st.dataframe(filtered[col].describe().to_frame().T.round(2), width="stretch")
        else:
            st.plotly_chart(charts.category_counts_bar(filtered[col], col, ds.is_demo), width="stretch")
    else:
        st.info("This layer has no attributes suitable for charts.")

st.subheader("Attribute table")
st.dataframe(pd.DataFrame(filtered.drop(columns=filtered.geometry.name)), hide_index=True, width="stretch")
prefix = "DEMO_" if ds.is_demo else ""
st.download_button(
    "Download selection (GeoJSON)", wgs.to_json().encode("utf-8"),
    file_name=f"{prefix}{ds.path.stem}_selection.geojson", mime="application/geo+json",
)
ui.metadata_panel(ds.meta)
