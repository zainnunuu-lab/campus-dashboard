"""3D campus assets: extruded footprints, mesh models and point clouds."""

from pathlib import Path

import streamlit as st

from campus_dashboard import assets3d, catalog, config, ui

ui.page_header(
    "3D Campus Assets",
    "Building extrusions from footprint heights, photogrammetric meshes (GLB/glTF/OBJ/PLY/STL) and point "
    "clouds (PLY/LAS/LAZ).",
)


@st.cache_data(show_spinner="Preparing 3D model...", max_entries=6)
def glb_bytes(path: str, mtime: float) -> bytes:
    return assets3d.to_glb_bytes(Path(path))


@st.cache_data(show_spinner="Reading 3D file...", max_entries=16)
def mesh_info(path: str, mtime: float):
    return assets3d.mesh_info(Path(path))


@st.cache_data(show_spinner="Reading point cloud...", max_entries=6)
def point_cloud(path: str, mtime: float):
    return assets3d.read_point_cloud(Path(path))


all_ds = ui.datasets()
tab_bld, tab_mesh, tab_pc = st.tabs(
    [":material/apartment: Extruded buildings", ":material/view_in_ar: 3D models", ":material/scatter_plot: Point clouds"]
)

with tab_bld:
    blds = ui.project_first(catalog.by_category(all_ds, "buildings"))
    if not blds:
        ui.no_data("building footprint layer")
    else:
        ds = ui.dataset_select(blds, "Footprint layer", key="b3d_ds")
        ground_opts = ui.project_first(catalog.by_category(all_ds, "landuse", "boundary"))
        ground_ds = ground_opts[0] if ground_opts else None
        ui.demo_banner([ds] + ([ground_ds] if ground_ds else []))
        gdf = ui.load_vector(ds)
        if gdf is not None:
            try:
                frame, hfield = assets3d.building_extrusion_frame(gdf)
            except ValueError as exc:
                st.warning(f"{exc} Add a height column ({', '.join(config.BUILDING_HEIGHT_FIELDS)}) in metres.")
            else:
                if frame.empty:
                    st.warning("No footprints with a valid numeric height.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Buildings extruded", len(frame))
                    c2.metric("Mean height", f"{frame['height'].mean():.1f} m")
                    c3.metric("Max height", f"{frame['height'].max():.1f} m")
                    centre = gdf.to_crs(config.DISPLAY_CRS).union_all().centroid
                    ground = assets3d.context_frame(ui.load_vector(ground_ds)) if ground_ds else None
                    if ground_ds is not None:
                        st.caption(f"Ground context: {ground_ds.meta.display_title}")
                    st.pydeck_chart(assets3d.building_deck(frame, (centre.y, centre.x), ds.is_demo, ground),
                                    height=560)
                    st.caption(
                        f"Heights from attribute '{hfield}'. Extrusion is a block model (LOD1): flat roofs, "
                        "no terrain. Drag to pan, right-drag / Ctrl-drag to rotate."
                    )

with tab_mesh:
    models = ui.project_first([d for d in all_ds if d.kind == "3d" and assets3d.asset_type(d.path) == "mesh"])
    if not models:
        ui.no_data("3D model")
    else:
        ds = ui.dataset_select(models, "Model", key="m3d_ds")
        ui.demo_banner([ds])
        try:
            info = mesh_info(str(ds.path), ds.mtime)
        except assets3d.AssetReadError as exc:
            st.error(str(exc))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vertices", f"{info.vertices:,}")
            c2.metric("Faces", f"{info.faces:,}")
            c3.metric("Parts", info.geometries)
            c4.metric("File size", f"{ds.size_mb:.1f} MB" if ds.size_mb >= 1 else f"{ds.size_mb * 1000:.0f} KB")
            st.caption("Extents (model units): " + " x ".join(f"{e:,.2f}" for e in info.extents))
            if info.is_point_cloud:
                st.info("This file contains only points - open it on the *Point clouds* tab.")
            elif ds.size_mb > config.MODEL_VIEWER_MAX_MB:
                st.warning(
                    f"The model is {ds.size_mb:.0f} MB; in-browser preview is limited to "
                    f"{config.MODEL_VIEWER_MAX_MB} MB. Decimate it (e.g. in Blender or MeshLab) for the dashboard."
                )
            else:
                try:
                    glb = glb_bytes(str(ds.path), ds.mtime)
                except assets3d.AssetReadError as exc:
                    st.error(str(exc))
                else:
                    banner = "DEMO - synthetic model" if ds.is_demo else None
                    ui.embed_html(assets3d.model_viewer_html(glb, ds.meta.title, 560, banner), height=570)
                    st.caption("Rendered with Google model-viewer (needs internet to load the viewer script). "
                               "Non-GLB formats are converted with trimesh; textures/materials may be simplified.")
        ui.metadata_panel(ds.meta)

with tab_pc:
    clouds = []
    for d in all_ds:
        if d.kind != "3d":
            continue
        if assets3d.asset_type(d.path) == "point_cloud" or d.meta.category == "point_cloud":
            clouds.append(d)
    clouds = ui.project_first(clouds)
    if not clouds:
        ui.no_data("point cloud")
    else:
        ds = ui.dataset_select(clouds, "Point cloud", key="pc_ds")
        ui.demo_banner([ds])
        try:
            xyz, rgb, total = point_cloud(str(ds.path), ds.mtime)
        except assets3d.AssetReadError as exc:
            st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Points in file", f"{total:,}")
            c2.metric("Points displayed", f"{len(xyz):,}")
            c3.metric("Height range (z)", f"{xyz[:, 2].max() - xyz[:, 2].min():.1f}")
            if total > len(xyz):
                st.caption(f"Randomly subsampled to {len(xyz):,} points for interactive display.")
            st.plotly_chart(assets3d.point_cloud_figure(xyz, rgb, ds.meta.title, ds.is_demo), width="stretch")
        ui.metadata_panel(ds.meta)
