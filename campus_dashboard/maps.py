"""Folium (Leaflet) map construction: basemaps, vector layers, raster overlays, legends."""

from __future__ import annotations

import base64
import html

import folium
import geopandas as gpd
import pandas as pd
from branca.element import MacroElement, Template
from folium import plugins

from . import config
from .vector_io import find_column

MAX_TOOLTIP_FIELDS = 8


def base_map(basemap: str = "OpenStreetMap", center=None, zoom: int = 17) -> folium.Map:
    spec = config.BASEMAPS.get(basemap, config.BASEMAPS["OpenStreetMap"])
    m = folium.Map(
        location=center or config.CAMPUS_CENTER_LATLON,
        zoom_start=zoom,
        max_zoom=22,
        tiles=None,
        control_scale=True,
    )
    if spec["tiles"]:
        folium.TileLayer(
            tiles=spec["tiles"], attr=spec["attr"], name=f"Basemap: {basemap}", max_zoom=22, max_native_zoom=19
        ).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" | ", prefix="Lat/Lon:", num_digits=6).add_to(m)
    return m


def _jsonable(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Leaflet tooltips need plain JSON types; stringify dates, lists, etc.

    Text is HTML-escaped because Leaflet inserts tooltip values as HTML and the
    attributes come from uploaded files.
    """
    out = gdf.copy()
    for col in out.columns:
        if col == out.geometry.name:
            continue
        if not (pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col])):
            out[col] = out[col].map(lambda v: "" if v is None or (isinstance(v, float) and v != v)
                                    else html.escape(str(v)))
    out.columns = [c if c == out.geometry.name else html.escape(str(c)) for c in out.columns]
    return out.set_geometry(gdf.geometry.name)


def landuse_style(class_field: str):
    def style(feature):
        cls = str(feature["properties"].get(class_field, "")).strip().lower()
        info = config.LANDUSE_CLASSES.get(cls)
        color = info.color if info else config.UNKNOWN_CLASS_COLOR
        return {"fillColor": color, "color": "#ffffff", "weight": 1, "fillOpacity": 0.7}

    return style


def add_vector_layer(
    m: folium.Map,
    gdf_wgs84: gpd.GeoDataFrame,
    name: str,
    category: str,
    show: bool = True,
    class_field: str | None = None,
) -> folium.FeatureGroup:
    group = folium.FeatureGroup(name=name, show=show)
    data = _jsonable(gdf_wgs84)
    fields = [c for c in data.columns if c != data.geometry.name][:MAX_TOOLTIP_FIELDS]
    tooltip = folium.GeoJsonTooltip(fields=fields, sticky=True) if fields else None
    popup = folium.GeoJsonPopup(fields=fields) if fields else None

    if category == "landuse":
        class_field = class_field or find_column(data.columns, config.LANDUSE_CLASS_FIELDS)
    if category == "landuse" and class_field:
        style = landuse_style(class_field)
    else:
        color = config.LAYER_COLORS.get(category, config.LAYER_COLORS["other"])
        is_boundary = category == "boundary"

        def style(_feature, color=color, is_boundary=is_boundary):
            return {
                "color": color,
                "weight": 3 if is_boundary else 2,
                "fillColor": color,
                "fillOpacity": 0.0 if is_boundary else 0.35,
                "dashArray": "6 4" if is_boundary else None,
            }

    point_color = config.LAYER_COLORS.get(category, config.LAYER_COLORS["other"])
    folium.GeoJson(
        data.to_json(),
        name=name,
        style_function=style,
        highlight_function=lambda _f: {"weight": 3, "color": "#2a78d6"},
        tooltip=tooltip,
        popup=popup,
        marker=folium.CircleMarker(radius=5, fill=True, fill_opacity=0.9, color="#ffffff", weight=1,
                                   fill_color=point_color),
    ).add_to(group)
    group.add_to(m)
    return group


def png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def add_raster_overlay(m: folium.Map, png: bytes, bounds, name: str, opacity: float = 1.0, show: bool = True):
    group = folium.FeatureGroup(name=name, show=show)
    folium.raster_layers.ImageOverlay(image=png_data_uri(png), bounds=bounds, opacity=opacity, name=name,
                                      interactive=False, zindex=1).add_to(group)
    group.add_to(m)
    return group


def validate_xyz_url(url: str) -> str | None:
    """Return an error message, or None when the tile URL template looks usable."""
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "Tile URL must start with http:// or https://"
    if not all(tok in url for tok in ("{z}", "{x}", "{y}")):
        return "Tile URL must contain the {z}, {x} and {y} placeholders."
    return None


def add_xyz_layer(m: folium.Map, url: str, name: str, attr: str = "User tile layer"):
    folium.TileLayer(tiles=url, attr=attr, name=name, overlay=True, max_zoom=24, max_native_zoom=22).add_to(m)


_LEGEND_TEMPLATE = """
{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 28px; right: 12px; z-index: 9999; background: rgba(255,255,255,0.94);
     color:#0b0b0b; padding: 8px 10px; border-radius: 6px; font: 12px/1.35 sans-serif;
     box-shadow: 0 1px 4px rgba(0,0,0,0.3); max-height: 45%; overflow-y: auto;">
  <div style="font-weight: 600; margin-bottom: 4px;">{{ this.title }}</div>
  {% for label, color, shape in this.items %}
  <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
    <span style="display:inline-block; width:14px; height:{{ '4px' if shape == 'line' else '12px' }};
          background:{{ color }}; border-radius:{{ '50%' if shape == 'point' else '2px' }};
          border:1px solid rgba(0,0,0,0.25);"></span>{{ label }}
  </div>
  {% endfor %}
</div>
{% endmacro %}
"""


class Legend(MacroElement):
    def __init__(self, title: str, items: list[tuple[str, str, str]]):
        super().__init__()
        self._name = "Legend"
        self.title = html.escape(title)
        self.items = [(html.escape(lbl), color, shape) for lbl, color, shape in items]
        self._template = Template(_LEGEND_TEMPLATE)


def landuse_legend_items(classes_present=None) -> list[tuple[str, str, str]]:
    items = [
        (c.label + (" (green)" if c.is_green else ""), c.color, "box")
        for key, c in config.LANDUSE_CLASSES.items()
        if classes_present is None or key in classes_present
    ]
    if classes_present is not None and set(classes_present) - set(config.LANDUSE_CLASSES):
        items.append(("Unknown class", config.UNKNOWN_CLASS_COLOR, "box"))
    return items


def finish(m: folium.Map, legend_items=None, bounds=None) -> folium.Map:
    if legend_items:
        m.get_root().add_child(Legend("Legend", legend_items))
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    if bounds:
        m.fit_bounds(bounds)
    return m
