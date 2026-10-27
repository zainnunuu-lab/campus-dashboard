"""Green cover: polygon-based share, RGB/NDVI index estimate, and validation."""

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from campus_dashboard import analysis, catalog, charts, config, raster_io, ui
from campus_dashboard.validation import validate_label_points

ui.page_header(
    "Green Cover",
    "Vegetated share of the study area, from classified polygons or estimated from an orthomosaic "
    "with a vegetation index. Index results are estimates until validated against reference points.",
)


@st.cache_data(show_spinner="Reading raster for analysis...", max_entries=8)
def analysis_view(path: str, mtime: float):
    return raster_io.read_for_analysis(path)


all_ds = ui.datasets()
tab_poly, tab_index, tab_val = st.tabs(
    [":material/pentagon: From land-use polygons", ":material/grid_on: From orthomosaic (index)",
     ":material/fact_check: Validation"]
)

# ======================================================================= polygons
with tab_poly:
    lu = ui.project_first(catalog.by_category(all_ds, "landuse", "green_cover"))
    if not lu:
        ui.no_data("land-use or green-cover polygon layer")
    else:
        ds = ui.dataset_select(lu, "Layer", key="gc_poly_ds")
        ui.demo_banner([ds])
        gdf = ui.load_vector(ds)
        if gdf is not None:
            try:
                if ds.meta.category == "landuse":
                    table = analysis.landuse_area_table(gdf)
                    gc = analysis.green_cover_from_table(table)
                    basis = "Green classes: " + ", ".join(
                        c.label for c in config.LANDUSE_CLASSES.values() if c.is_green)
                else:
                    bnd = ui.project_first(catalog.by_category(all_ds, "boundary"))
                    bnd_gdf = ui.load_vector(bnd[0]) if bnd else None
                    green_area = analysis.layer_summary(gdf)["area_m2"]
                    total = analysis.layer_summary(bnd_gdf)["area_m2"] if bnd_gdf is not None else 0.0
                    gc = analysis.GreenCoverSummary(green_m2=green_area, total_m2=total)
                    basis = f"Green polygons / boundary '{bnd[0].meta.display_title}'" if bnd else ""
                    if not bnd:
                        st.warning("Add a boundary layer to express green cover as a percentage.")
            except Exception as exc:
                st.error(f"Green cover could not be computed: {exc}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Green area", f"{gc.green_m2 / 10_000:,.2f} ha")
                c2.metric("Reference area", f"{gc.total_m2 / 10_000:,.2f} ha" if gc.total_m2 else "-")
                c3.metric("Green cover", f"{gc.percent:.1f} %" if gc.total_m2 else "-")
                if gc.total_m2:
                    st.plotly_chart(charts.green_split_bar(gc.green_m2, gc.total_m2, ds.is_demo,
                                                           "Green vs. other surfaces"), width="stretch")
                st.caption(f"{basis}. Accuracy depends on the polygon classification and is not validated here.")

# ======================================================================= index
mask_result = None
with tab_index:
    rasters = ui.project_first(catalog.by_category(all_ds, "orthomosaic", "index_raster"))
    if not rasters:
        ui.no_data("orthomosaic or index raster")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            rds = ui.dataset_select(rasters, "Raster", key="gc_raster")
        try:
            view = analysis_view(str(rds.path), rds.mtime)
        except Exception as exc:
            st.error(f"Could not read **{rds.meta.display_title}**: {exc}")
            view = None
        if view is not None:
            methods = list(analysis.INDEX_METHODS) if view.data.shape[0] >= 3 else ["Band 1 as index"]
            with c2:
                method = st.radio("Index", methods, horizontal=True, help="\n\n".join(
                    f"**{k}** - {v}" for k, v in analysis.INDEX_METHODS.items()))
            st.caption(analysis.INDEX_METHODS[method])
            ui.demo_banner([rds])
            try:
                idx = analysis.vegetation_index(view, method)
                finite = idx[np.isfinite(idx)]
                if finite.size == 0:
                    raise ValueError("No valid pixels in the raster.")
                suggested = analysis.otsu_threshold(idx)
            except ValueError as exc:
                st.error(str(exc))
            else:
                lo, hi = float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))
                if hi <= lo:
                    hi = lo + 1e-3
                default = float(np.clip(suggested, lo, hi))
                thr = st.slider("Green threshold (pixels with index >= threshold are counted as green)",
                                lo, hi, default, step=(hi - lo) / 200, format="%.3f",
                                key=f"thr_{rds.id}_{method}")
                st.caption(f"Otsu's method suggests {suggested:.3f}. This is a statistical starting point, "
                           "not a validated threshold - tune it and check results on the Validation tab.")
                mask = analysis.green_mask(idx, thr)
                gc = analysis.mask_area(mask, view.valid, view.transform)
                mask_result = (mask, view, rds, method, thr)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Estimated green cover", f"{gc.percent:.1f} %")
                k2.metric("Green area (est.)", f"{gc.green_m2 / 10_000:,.2f} ha")
                k3.metric("Valid raster area", f"{gc.total_m2 / 10_000:,.2f} ha")
                k4.metric("Analysis pixel", f"{abs(view.transform.a):.2f} m",
                          help=f"Raster decimated to <= {config.RASTER_ANALYSIS_MAX_PX} px for analysis.")

                st.plotly_chart(charts.index_histogram(idx, thr, method, rds.is_demo), width="stretch")

                i1, i2 = st.columns(2)
                rgba = np.zeros(mask.shape + (4,), dtype="uint8")
                rgba[mask] = (0, 131, 0, 255)
                rgba[view.valid & ~mask] = (230, 228, 222, 255)
                i1.image(raster_io.render_png(view), caption="Raster (analysis resolution)", width="stretch")
                i2.image(Image.fromarray(rgba), caption="Green mask (green = above threshold, grey = other)",
                         width="stretch")
                buf = io.BytesIO()
                Image.fromarray(rgba).save(buf, format="PNG")
                st.download_button("Download mask preview (PNG)", buf.getvalue(),
                                   file_name=("DEMO_" if rds.is_demo else "") + "green_mask.png")
                st.info(
                    "RGB indices (ExG, VARI) respond to 'greenness', not to plant health: shadows, green roofs "
                    "and painted surfaces can be misclassified, and dry grass may be missed. Report these "
                    "figures as *estimates* unless validated.",
                    icon=":material/info:",
                )

# ======================================================================= validation
with tab_val:
    st.markdown(
        "Validate the index-based green mask against **independent reference points** (e.g. field "
        "observations or careful visual interpretation by someone who did not set the threshold). "
        "Each point needs a 0/1 label column named `is_green`."
    )
    if mask_result is None:
        st.info("First compute a green mask on the *From orthomosaic (index)* tab.")
    else:
        mask, view, rds, method, thr = mask_result
        refs = ui.project_first([d for d in all_ds if d.kind == "vector"])
        point_layers = []
        for d in refs:
            g = ui.load_vector(d)
            if g is not None and len(g) and g.geom_type.eq("Point").all():
                point_layers.append(d)
        if not point_layers:
            ui.no_data("point layer with reference labels")
        else:
            ref_ds = ui.dataset_select(point_layers, "Reference points", key="gc_ref")
            pts = ui.load_vector(ref_ds)
            col, res = validate_label_points(pts)
            ui.show_validation(res, "Reference data checks")
            if res.ok:
                ui.demo_banner([rds, ref_ds])
                rep = analysis.accuracy_against_points(mask, view.valid, view, pts, col)
                if rep.n_used == 0:
                    st.error("None of the reference points fall on valid pixels of the selected raster.")
                else:
                    if rds.is_demo or ref_ds.is_demo:
                        st.error(
                            "These figures use DEMO data (synthetic raster and/or synthetic labels). They show "
                            "how the validation works and say nothing about real accuracy.",
                            icon=":material/science:",
                        )
                    a1, a2, a3, a4, a5 = st.columns(5)
                    a1.metric("Points used", rep.n_used, help=f"{rep.n_outside} outside the valid raster area")
                    a2.metric("Overall accuracy", f"{rep.overall_accuracy:.1%}")
                    a3.metric("Precision (green)", f"{rep.precision:.1%}" if np.isfinite(rep.precision) else "-")
                    a4.metric("Recall (green)", f"{rep.recall:.1%}" if np.isfinite(rep.recall) else "-")
                    a5.metric("Cohen's kappa", f"{rep.kappa:.2f}" if np.isfinite(rep.kappa) else "-")
                    st.markdown("**Confusion matrix** (point counts)")
                    st.dataframe(rep.confusion_frame(), width="content")
                    report = pd.DataFrame([{
                        "raster": rds.meta.title, "raster_is_demo": rds.is_demo, "reference": ref_ds.meta.title,
                        "reference_is_demo": ref_ds.is_demo, "index": method, "threshold": thr,
                        "points_used": rep.n_used, "overall_accuracy": rep.overall_accuracy,
                        "precision": rep.precision, "recall": rep.recall, "f1": rep.f1, "kappa": rep.kappa,
                    }])
                    st.download_button("Download validation report (CSV)", report.to_csv(index=False).encode(),
                                       file_name=("DEMO_" if rds.is_demo or ref_ds.is_demo else "")
                                       + "green_cover_validation.csv")
                    st.caption(
                        "Accuracy is evaluated at the analysis resolution. If the same points were used to "
                        "choose the threshold, the figures are optimistic - keep a separate test set."
                    )
