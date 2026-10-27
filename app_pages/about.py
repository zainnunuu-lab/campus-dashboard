"""Methodology, data status and limitations."""

import pandas as pd
import streamlit as st

from campus_dashboard import catalog, config, ui

ui.page_header("Methodology & Data Status", "How results are produced, and what they can and cannot claim.")

every = catalog.discover(include_demo=True)
project = [d for d in every if not d.is_demo]
demo = [d for d in every if d.is_demo]

c1, c2, c3 = st.columns(3)
c1.metric("Project datasets", len(project))
c2.metric("Demo datasets (synthetic)", len(demo))
c3.metric("Project datasets with accuracy statement",
          sum(1 for d in project if d.meta.accuracy_statement != config.DEFAULT_ACCURACY_STATEMENT))

if not project:
    st.warning("No project (survey) data has been added. Every analysis shown is a demonstration only.",
               icon=":material/warning:")

st.subheader("Data status policy")
st.markdown(
    f"""
- **Demo data** lives in `data/demo/`, is produced by `scripts/generate_demo_data.py`, and is flagged
  `is_demo` in its metadata. The dashboard *always* treats anything in that folder as demo, labels it
  **[DEMO]** in selectors, map layers, chart titles and file downloads, and shows a warning banner on any
  page where it contributes. {config.DEMO_DISCLAIMER}
- **Project data** is anything uploaded through the Data Manager into `data/user/`. Uploads can also be
  flagged as test/demo data.
- **Accuracy:** the default statement for every dataset is *"{config.DEFAULT_ACCURACY_STATEMENT}"*. Replace it
  only with a measured figure (e.g. horizontal RMSE against GNSS check points, or a classification accuracy
  from independent reference points), stating the method.
"""
)

st.subheader("Methods")
st.markdown(
    f"""
| Output | Method | Limitations |
|---|---|---|
| Orthomosaic display | GeoTIFF read through a GDAL WarpedVRT, decimated to <= {config.RASTER_PREVIEW_MAX_PX} px and warped to WGS84; shown as a Leaflet image overlay. | Preview resolution only. Positional accuracy is that of the photogrammetric product (GCPs / RTK), which the dashboard does not measure. |
| Areas & lengths | Layers reprojected to their local UTM zone (Karachi: zone 42N, EPSG:32642). | Planimetric (2D) values. Overlapping polygons are flagged because they inflate totals. |
| Land-use statistics | Sum of polygon areas per class, optionally clipped to a boundary. | As good as the classification / digitising; not validated by the dashboard. |
| Green cover (polygons) | Share of area in classes configured as green ({", ".join(c.label for c in config.LANDUSE_CLASSES.values() if c.is_green)}). | Depends on class definitions in `config.py`. |
| Green cover (index) | ExG or VARI from RGB bands, or a single-band index (e.g. NDVI); pixels >= threshold counted as green. Otsu's method suggests a starting threshold. | RGB indices measure greenness, not vegetation health; shadows and green-painted surfaces cause errors. Analysis is at <= {config.RASTER_ANALYSIS_MAX_PX} px. |
| Validation | Confusion matrix, overall accuracy, precision, recall, Cohen's kappa against user-supplied reference points. | Needs independent points (ideally >= 50, stratified); points used for threshold tuning give optimistic figures. |
| 3D | Footprint extrusion (pydeck), meshes via model-viewer, point clouds subsampled in Plotly. | Block models have flat roofs; large meshes need decimation. |
"""
)

st.subheader("Software")
st.markdown(
    "Streamlit, GeoPandas / pyogrio (GDAL), Shapely, pyproj, rasterio, Folium (Leaflet), Plotly, pydeck, "
    "trimesh, laspy. Basemaps: OpenStreetMap contributors, CARTO, Esri (reference only)."
)

if every:
    st.subheader("All datasets")
    st.dataframe(pd.DataFrame([{
        "Status": "DEMO" if d.is_demo else "Project",
        "Title": d.meta.title,
        "Category": config.CATEGORIES.get(d.meta.category, d.meta.category),
        "Source": d.meta.source or "Not recorded",
        "Accuracy": d.meta.accuracy_statement,
    } for d in every]), hide_index=True, width="stretch")
