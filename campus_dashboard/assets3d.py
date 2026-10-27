"""3D campus assets: mesh models, point clouds and extruded building footprints."""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config
from .vector_io import find_column, to_display_crs

# Google's model-viewer web component renders glTF/GLB in the browser (WebGL).
# It is loaded from Google's CDN, so the 3D viewer needs an internet connection.
MODEL_VIEWER_JS = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"


class AssetReadError(Exception):
    """Raised with a message that can be shown directly to the user."""


@dataclass
class MeshInfo:
    vertices: int
    faces: int
    geometries: int
    extents: tuple[float, float, float]
    is_point_cloud: bool


def asset_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in config.POINT_CLOUD_EXTENSIONS:
        return "point_cloud"
    if ext in config.MODEL_EXTENSIONS:
        return "mesh"
    raise AssetReadError(f"Unsupported 3D format '{ext}'.")


def _load_trimesh(path: Path):
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - trimesh is in requirements
        raise AssetReadError("The 'trimesh' package is required to read 3D meshes.") from exc
    try:
        return trimesh.load(str(path), force="scene")
    except Exception as exc:
        raise AssetReadError(f"Could not read 3D file '{path.name}': {exc}") from exc


def mesh_info(path: Path) -> MeshInfo:
    import trimesh

    scene = _load_trimesh(path)
    geoms = list(scene.geometry.values())
    if not geoms:
        raise AssetReadError(f"'{path.name}' contains no geometry.")
    verts = sum(len(g.vertices) for g in geoms)
    faces = sum(len(getattr(g, "faces", [])) for g in geoms)
    ext = tuple(float(x) for x in scene.extents) if scene.extents is not None else (0.0, 0.0, 0.0)
    return MeshInfo(
        vertices=int(verts),
        faces=int(faces),
        geometries=len(geoms),
        extents=ext,
        is_point_cloud=all(isinstance(g, trimesh.PointCloud) for g in geoms),
    )


def to_glb_bytes(path: Path) -> bytes:
    """GLB as-is; other mesh formats are converted with trimesh (materials may be simplified)."""
    if path.suffix.lower() == ".glb":
        return path.read_bytes()
    scene = _load_trimesh(path)
    try:
        return scene.export(file_type="glb")
    except Exception as exc:
        raise AssetReadError(f"Could not convert '{path.name}' to GLB for display: {exc}") from exc


def model_viewer_html(glb: bytes, alt: str, height: int = 520, banner: str | None = None) -> str:
    src = "data:model/gltf-binary;base64," + base64.b64encode(glb).decode("ascii")
    banner_html = (
        f'<div style="position:absolute;top:8px;left:8px;z-index:2;background:#d03b3b;color:#fff;'
        f'font:600 12px sans-serif;padding:4px 8px;border-radius:4px;">{html.escape(banner)}</div>'
        if banner
        else ""
    )
    return f"""
<script type="module" src="{MODEL_VIEWER_JS}"></script>
<div style="position:relative;width:100%;height:{height - 10}px;">
  {banner_html}
  <model-viewer src="{src}" alt="{html.escape(alt)}" camera-controls touch-action="pan-y"
      shadow-intensity="0.8" exposure="1.0" environment-image="neutral"
      style="width:100%;height:100%;background:#eef1f4;border-radius:8px;">
    <div slot="poster" style="padding:16px;font:14px sans-serif;color:#52514e;">
      Loading 3D viewer... (requires an internet connection to load model-viewer)
    </div>
  </model-viewer>
</div>
"""


def read_point_cloud(path: Path, max_points: int = config.POINT_CLOUD_MAX_POINTS):
    """Return (xyz Nx3, rgb Nx3 uint8 or None, total_points). Randomly subsampled for display."""
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError as exc:
            raise AssetReadError("Reading LAS/LAZ needs 'laspy' (and 'lazrs' for .laz).") from exc
        try:
            las = laspy.read(str(path))
        except Exception as exc:
            raise AssetReadError(f"Could not read point cloud '{path.name}': {exc}") from exc
        xyz = np.vstack([las.x, las.y, las.z]).T
        rgb = None
        if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype("float64")
            rgb = (rgb / max(rgb.max(), 1) * 255).astype("uint8")
    else:
        import trimesh

        scene = _load_trimesh(path)
        clouds = [g for g in scene.geometry.values() if isinstance(g, trimesh.PointCloud)]
        if not clouds:
            raise AssetReadError(f"'{path.name}' does not contain a point cloud.")
        xyz = np.vstack([c.vertices for c in clouds])
        cols = [c.colors for c in clouds if c.colors is not None and len(c.colors) == len(c.vertices)]
        rgb = np.vstack(cols)[:, :3].astype("uint8") if len(cols) == len(clouds) else None
    total = len(xyz)
    if total == 0:
        raise AssetReadError("The point cloud is empty.")
    if total > max_points:
        idx = np.random.default_rng(0).choice(total, max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
    return xyz, rgb, total


def point_cloud_figure(xyz: np.ndarray, rgb, title: str, demo: bool) -> go.Figure:
    centred = xyz - xyz.mean(axis=0)
    if rgb is not None:
        color = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        marker = {"size": 1.6, "color": color}
    else:
        marker = {"size": 1.6, "color": centred[:, 2], "colorscale": "Blues", "colorbar": {"title": "dz"}}
    fig = go.Figure(go.Scatter3d(x=centred[:, 0], y=centred[:, 1], z=centred[:, 2], mode="markers",
                                 marker=marker, hoverinfo="skip"))
    label = f"{title}  [{config.DEMO_LABEL} - synthetic]" if demo else title
    fig.update_layout(title=label, height=560, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                      scene={"aspectmode": "data", "xaxis_title": "x (local)", "yaxis_title": "y (local)",
                             "zaxis_title": "z"})
    return fig


def building_extrusion_frame(gdf: gpd.GeoDataFrame, height_field: str | None = None) -> tuple[pd.DataFrame, str]:
    """Polygon rings + heights in WGS84 for a pydeck PolygonLayer."""
    height_field = height_field or find_column(gdf.columns, config.BUILDING_HEIGHT_FIELDS)
    if height_field is None:
        raise ValueError("No height attribute found for extrusion.")
    wgs = to_display_crs(gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]).explode(index_parts=False)
    heights = pd.to_numeric(wgs[height_field], errors="coerce")
    rows = []
    name_col = find_column(wgs.columns, ("name", "building", "bldg_name", "id"))
    for geom, h, name in zip(wgs.geometry, heights, wgs[name_col] if name_col else [""] * len(wgs)):
        if geom is None or geom.is_empty or not np.isfinite(h):
            continue
        rows.append({"polygon": [list(c) for c in geom.exterior.coords], "height": float(h), "name": str(name)})
    return pd.DataFrame(rows), height_field


def context_frame(gdf: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Flat ground polygons (land use or boundary) drawn beneath extruded buildings."""
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["polygon", "color"])
    wgs = to_display_crs(gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]).explode(index_parts=False)
    cls_col = find_column(wgs.columns, config.LANDUSE_CLASS_FIELDS)
    rows = []
    for i, geom in enumerate(wgs.geometry):
        if geom is None or geom.is_empty:
            continue
        hexcol = config.UNKNOWN_CLASS_COLOR
        if cls_col is not None:
            info = config.LANDUSE_CLASSES.get(str(wgs[cls_col].iloc[i]).strip().lower())
            hexcol = info.color if info else hexcol
        rgb = [int(hexcol[j:j + 2], 16) for j in (1, 3, 5)]
        rows.append({"polygon": [list(c) for c in geom.exterior.coords], "color": rgb + [150]})
    return pd.DataFrame(rows)


def building_deck(frame: pd.DataFrame, center_latlon, demo: bool, ground: pd.DataFrame | None = None):
    import pydeck as pdk

    layers = []
    if ground is not None and not ground.empty:
        layers.append(pdk.Layer("PolygonLayer", ground, get_polygon="polygon", get_fill_color="color",
                                stroked=False, extruded=False, pickable=False))
    layer = pdk.Layer(
        "PolygonLayer",
        frame,
        get_polygon="polygon",
        get_elevation="height",
        elevation_scale=1,
        extruded=True,
        wireframe=True,
        get_fill_color=[208, 59, 59, 200] if demo else [140, 90, 60, 210],
        get_line_color=[255, 255, 255],
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=center_latlon[0], longitude=center_latlon[1], zoom=16.5, pitch=50, bearing=-20)
    tooltip = {"text": ("[DEMO] " if demo else "") + "{name}\nHeight: {height} m"}
    layers.append(layer)
    # No third-party basemap: the ground context comes from the project's own polygons,
    # so the view also works offline.
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_style=None)
