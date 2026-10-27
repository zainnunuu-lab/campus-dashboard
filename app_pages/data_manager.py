"""Data Manager: upload with validation, metadata entry, and management of saved datasets."""

import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from pyproj import CRS
from pyproj.exceptions import CRSError

from campus_dashboard import assets3d, catalog, config, maps, ui
from campus_dashboard.metadata import DatasetMetadata, save_metadata
from campus_dashboard.raster_io import summarize_raster
from campus_dashboard.validation import (
    ValidationResult,
    validate_raster,
    validate_upload_file,
    validate_vector,
)
from campus_dashboard.vector_io import VectorReadError, read_vector, to_display_crs, write_geojson

ui.page_header(
    "Data Manager",
    "Add orthomosaics, GIS layers and 3D assets. Each upload is validated, and its provenance is stored in "
    "a metadata file next to the data (data/user/...).",
)

KIND_LABELS = {"vector": "Vector GIS layer", "raster": "Raster (GeoTIFF)", "3d": "3D asset"}
KIND_EXTS = {"vector": config.VECTOR_EXTENSIONS, "raster": config.RASTER_EXTENSIONS,
             "3d": config.ASSET_3D_EXTENSIONS}
KIND_CATS = {"vector": config.VECTOR_CATEGORIES, "raster": config.RASTER_CATEGORIES,
             "3d": config.ASSET_3D_CATEGORIES}


def staged_copy(upload) -> Path:
    """Write the upload to a temp file once per upload (GDAL needs a real file path)."""
    staged = st.session_state.setdefault("_staged", {})
    key = f"{upload.file_id}:{upload.name}"
    path = staged.get(key)
    if path is None or not Path(path).exists():
        folder = Path(tempfile.mkdtemp(prefix="campus_upload_"))
        path = folder / catalog.safe_filename(upload.name)
        with open(path, "wb") as fh:
            shutil.copyfileobj(upload, fh)
        staged[key] = str(path)
    return Path(path)


def parse_crs(text: str) -> tuple[str | None, str | None]:
    text = text.strip()
    if not text:
        return None, None
    try:
        crs = CRS.from_user_input(text)
        return crs.to_string(), None
    except CRSError as exc:
        return None, f"'{text}' is not a recognised CRS ({exc})."


tab_up, tab_manage, tab_formats = st.tabs(
    [":material/upload: Upload", ":material/folder_managed: Manage datasets", ":material/description: Formats & rules"]
)

# ======================================================================= upload
with tab_up:
    kind = st.radio("Dataset type", list(KIND_LABELS), format_func=KIND_LABELS.get, horizontal=True)
    upload = st.file_uploader(
        f"Choose a file ({', '.join(KIND_EXTS[kind])})",
        type=[e.lstrip(".") for e in KIND_EXTS[kind]],
        key=f"uploader_{kind}",
        help="Shapefiles must be uploaded as a single .zip containing .shp, .shx, .dbf and .prj.",
    )
    if upload is not None:
        file_check = validate_upload_file(upload.name, upload.size, kind)
        if not file_check.ok:
            ui.show_validation(file_check, "File checks")
            st.stop()
        tmp = staged_copy(upload)

        st.markdown("#### 1. Describe the dataset")
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Title *", value=Path(upload.name).stem.replace("_", " ").title())
            category = st.selectbox("Category *", KIND_CATS[kind], format_func=config.CATEGORIES.get, index=None,
                                    placeholder="Choose what this dataset represents",
                                    help="Decides which pages use the dataset and which checks are run.")
            source = st.text_input("Source", placeholder="e.g. DJI Mavic 3E flight, 80 m AGL, survey team")
            processing = st.text_input("Processing software", placeholder="e.g. WebODM 2.5 / Agisoft Metashape")
            sensor = st.text_input("Sensor / camera", placeholder="e.g. 20 MP RGB")
        with c2:
            date_known = st.checkbox("Capture date known", value=False)
            capture = st.date_input("Capture date", value=date.today(), disabled=not date_known,
                                    max_value=date.today())
            gsd = st.number_input("GSD reported by processing software (cm/px, 0 = unknown)",
                                  min_value=0.0, max_value=500.0, value=0.0, step=0.1)
            accuracy = st.text_area(
                "Accuracy statement",
                value=config.DEFAULT_ACCURACY_STATEMENT,
                help="Only state an accuracy if it was measured (e.g. RMSE against GCP / check points). "
                "Leave the default otherwise.",
            )
            licence = st.text_input("Licence / usage", placeholder="e.g. SSUET FYP internal use")
        description = st.text_area("Description")
        notes = st.text_area("Notes")
        is_demo = st.checkbox(
            "This is test / demo data (label it as DEMO everywhere)", value=False,
            help="Tick for practice files so they can never be mistaken for survey results.",
        )
        assumed_crs_text = ""
        if kind == "vector":
            assumed_crs_text = st.text_input(
                "CRS to assume if the file has none (e.g. EPSG:4326, EPSG:32642)",
                help="CSV files are assumed to be WGS84 (EPSG:4326) unless you enter a CRS here. "
                "This is ignored if the file declares its own CRS.",
            )

        st.markdown("#### 2. Validation")
        result = None
        gdf = None
        assumed_crs, crs_err = parse_crs(assumed_crs_text)
        if category is None:
            st.info("Choose a category above to validate the file.", icon=":material/info:")
        elif crs_err:
            st.error(crs_err)
        elif kind == "vector":
            try:
                raw = read_vector(tmp, assumed_crs=assumed_crs)
                gdf, result = validate_vector(raw, category)
            except VectorReadError as exc:
                st.error(str(exc))
        elif kind == "raster":
            result = validate_raster(tmp, category)
        else:
            result = ValidationResult()
            try:
                if assets3d.asset_type(tmp) == "point_cloud":
                    xyz, _, total = assets3d.read_point_cloud(tmp)
                    result.info.append(f"Point cloud with {total:,} points.")
                    if category != "point_cloud":
                        result.warnings.append("LAS/LAZ files are point clouds; category set to Point cloud.")
                        category = "point_cloud"
                else:
                    info = assets3d.mesh_info(tmp)
                    result.info.append(f"{info.vertices:,} vertices, {info.faces:,} faces, {info.geometries} part(s).")
                    if info.is_point_cloud and category != "point_cloud":
                        result.warnings.append("File contains only points; category set to Point cloud.")
                        category = "point_cloud"
                    if tmp.suffix.lower() == ".gltf":
                        result.warnings.append(
                            ".gltf files often reference external .bin / texture files that were not uploaded. "
                            "Export as a single .glb for reliable display."
                        )
                    if tmp.suffix.lower() == ".obj":
                        result.warnings.append("OBJ textures (.mtl / images) are not uploaded; the model will "
                                               "be shown without them. Prefer .glb.")
            except assets3d.AssetReadError as exc:
                result.errors.append(str(exc))
        if not title.strip() and result is not None:
            result.errors.append("A title is required.")
        if result is not None:
            ui.show_validation(result)

        if result is not None and result.ok:
            if kind == "vector" and gdf is not None:
                with st.expander("Preview", expanded=True):
                    wgs = to_display_crs(gdf)
                    m = maps.base_map("OpenStreetMap")
                    maps.add_vector_layer(m, wgs, title, category)
                    minx, miny, maxx, maxy = wgs.total_bounds
                    maps.finish(m, None, [[miny, minx], [maxy, maxx]])
                    ui.render_map(m, height=380)
                    st.dataframe(pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).head(50),
                                 hide_index=True, width="stretch")

            st.markdown("#### 3. Save")
            if st.button("Save to project data", type="primary", icon=":material/save:"):
                folder = config.USER_DIR / config.KIND_DIRS[kind]
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    if kind == "vector":
                        target = catalog.unique_path(folder, Path(catalog.safe_filename(upload.name)).stem + ".geojson")
                        write_geojson(gdf, target)
                        crs = gdf.crs.to_string()
                    else:
                        target = catalog.unique_path(folder, catalog.safe_filename(upload.name))
                        shutil.copyfile(tmp, target)
                        crs = (summarize_raster(target).crs or "") if kind == "raster" else ""
                    meta = DatasetMetadata(
                        title=title.strip(), kind=kind, category=category, is_demo=is_demo,
                        description=description, source=source, capture_date=capture.isoformat() if date_known else "",
                        processing_software=processing, sensor=sensor, gsd_cm=gsd or None, crs=crs,
                        original_filename=upload.name, accuracy_statement=accuracy.strip()
                        or config.DEFAULT_ACCURACY_STATEMENT, license=licence, notes=notes,
                    )
                    save_metadata(target, meta)
                except Exception as exc:
                    st.error(f"Saving failed: {exc}")
                else:
                    st.success(f"Saved as `{ui.file_label(target)}`. It is now available on the analysis pages.",
                               icon=":material/check_circle:")
                    if kind == "vector":
                        st.caption("Vector layers are stored as GeoJSON in WGS84; the original CRS is recorded "
                                   "in the metadata.")
        elif result is not None:
            st.error("Fix the errors above before the dataset can be saved.")

# ======================================================================= manage
with tab_manage:
    flash = st.session_state.pop("_flash", None)
    if flash:
        st.success(flash, icon=":material/check_circle:")
    every = catalog.discover(include_demo=True)
    if not every:
        ui.no_data("datasets")
    else:
        rows = [{"Status": "DEMO" if d.is_demo else "Project", "Title": d.meta.title, "Type": d.kind,
                 "Category": config.CATEGORIES.get(d.meta.category, d.meta.category),
                 "File": ui.file_label(d.path), "Size (MB)": round(d.size_mb, 2)} for d in every]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ds = ui.dataset_select(ui.project_first(every), "Select a dataset", key="mgr_ds")
        ui.metadata_panel(ds.meta, expanded=True)
        protected = ds.path.is_relative_to(config.DEMO_DIR)
        if protected:
            st.info("Files in data/demo are generated by scripts/generate_demo_data.py and are read-only here.")
        else:
            with st.expander("Edit metadata"):
                with st.form(f"edit_{ds.id}"):
                    m = ds.meta
                    new_title = st.text_input("Title", m.title)
                    new_cat = st.selectbox("Category", KIND_CATS[ds.kind],
                                           index=list(KIND_CATS[ds.kind]).index(m.category)
                                           if m.category in KIND_CATS[ds.kind] else 0,
                                           format_func=config.CATEGORIES.get)
                    new_source = st.text_input("Source", m.source)
                    new_date = st.text_input("Capture date (YYYY-MM-DD)", m.capture_date)
                    new_acc = st.text_area("Accuracy statement", m.accuracy_statement)
                    new_desc = st.text_area("Description", m.description)
                    new_demo = st.checkbox("Test / demo data", m.is_demo)
                    if st.form_submit_button("Save metadata"):
                        err = None
                        if new_date.strip():
                            try:
                                date.fromisoformat(new_date.strip())
                            except ValueError:
                                err = "Capture date must be in YYYY-MM-DD format."
                        if not new_title.strip():
                            err = "Title cannot be empty."
                        if err:
                            st.error(err)
                        else:
                            m.title, m.category, m.source = new_title.strip(), new_cat, new_source
                            m.capture_date, m.accuracy_statement = new_date.strip(), new_acc.strip() or \
                                config.DEFAULT_ACCURACY_STATEMENT
                            m.description, m.is_demo = new_desc, new_demo
                            save_metadata(ds.path, m)
                            st.session_state["_flash"] = f"Metadata of '{m.title}' saved."
                            st.rerun()
            with st.expander("Delete dataset"):
                confirm = st.checkbox(f"I want to permanently delete `{ds.path.name}`", key=f"del_{ds.id}")
                if st.button("Delete", type="secondary", disabled=not confirm, icon=":material/delete:"):
                    try:
                        catalog.delete_user_dataset(ds)
                    except (PermissionError, OSError) as exc:
                        st.error(f"Could not delete: {exc}")
                    else:
                        st.cache_data.clear()
                        st.session_state["_flash"] = f"Deleted '{ds.meta.title}' ({ds.path.name})."
                        st.rerun()

# ======================================================================= formats
with tab_formats:
    st.markdown(
        f"""
| Type | Formats | Notes |
|---|---|---|
| Vector | GeoJSON (`.geojson`, `.json`), GeoPackage (`.gpkg`), zipped Shapefile (`.zip`), KML (`.kml`), CSV | CSV needs `lat`/`lon` columns or a `wkt` column. Stored as GeoJSON (WGS84). |
| Raster | GeoTIFF / Cloud-Optimised GeoTIFF (`.tif`, `.tiff`) | Must contain a CRS and geotransform. RGB orthomosaics, single-band index rasters (NDVI), DSM/DTM. |
| 3D | `.glb` (recommended), `.gltf`, `.obj`, `.ply`, `.stl`, `.off`, `.las`, `.laz` | Meshes shown in model-viewer; point clouds subsampled to {config.POINT_CLOUD_MAX_POINTS:,} points. |

**Land-use layers** need a class attribute named one of `{"`, `".join(config.LANDUSE_CLASS_FIELDS)}` with codes:
"""
    )
    st.dataframe(
        pd.DataFrame([{"Code": k, "Label": c.label, "Counts as green": c.is_green, "Map colour": c.color}
                      for k, c in config.LANDUSE_CLASSES.items()]),
        hide_index=True, width="content",
    )
    st.markdown(
        f"""
**Building footprints** should have a height in metres (`{"`, `".join(config.BUILDING_HEIGHT_FIELDS)}`) for 3D extrusion.

**Reference points** for green-cover validation: Point layer with an `is_green` column (1 = green, 0 = not green).

Size limits: vector {config.MAX_UPLOAD_MB['vector']} MB, raster {config.MAX_UPLOAD_MB['raster']} MB,
3D {config.MAX_UPLOAD_MB['3d']} MB.
"""
    )
