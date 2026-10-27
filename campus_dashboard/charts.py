"""Plotly figures. Kept free of Streamlit so they can be unit-tested.

Conventions: one value axis, recessive grid, thin bars with 4px rounded ends,
colour follows the entity (a land-use class keeps its colour in every chart),
classes are always named on the axis so colour is never the only cue, and any
figure built from demo data says so in its title.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config

SEQ_BLUE = "#2a78d6"
GRID = "rgba(128,128,128,0.18)"


def _title(text: str, demo: bool) -> str:
    return f"{text}  <span style='color:#d03b3b'>[{config.DEMO_LABEL} DATA - synthetic]</span>" if demo else text


def _base_layout(fig: go.Figure, title: str, demo: bool, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": _title(title, demo), "x": 0, "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
        showlegend=False,
        barcornerradius=4,
        bargap=0.35,
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def landuse_area_bar(table: pd.DataFrame, demo: bool) -> go.Figure:
    t = table.sort_values("area_m2")
    fig = go.Figure(
        go.Bar(
            x=t["area_ha"],
            y=t["label"],
            orientation="h",
            marker={"color": t["color"], "line": {"width": 0}},
            text=[f"{p:.1f}%" for p in t["percent"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([t["area_m2"], t["percent"], t["features"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]:,.0f} m² (%{x:.3f} ha)"
            "<br>%{customdata[1]:.1f}% of mapped area<br>%{customdata[2]} polygon(s)<extra></extra>",
        )
    )
    _base_layout(fig, "Area by land-use class (ha)", demo, height=max(260, 42 * len(t) + 80))
    fig.update_xaxes(title="Area (ha)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def green_split_bar(green_m2: float, total_m2: float, demo: bool, title: str) -> go.Figure:
    """A single 100% bar: green vs. other. Two named segments with a 2px gap."""
    other = max(total_m2 - green_m2, 0)
    fig = go.Figure()
    for name, val, color in (("Green", green_m2, "#008300"), ("Other surfaces", other, "#a3a19c")):
        pct = val / total_m2 * 100 if total_m2 else 0
        fig.add_bar(
            x=[pct],
            y=[""],
            name=name,
            orientation="h",
            marker={"color": color, "line": {"color": "white", "width": 2}},
            text=f"{name}: {pct:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{name}</b><br>{val:,.0f} m²<br>{pct:.1f}%<extra></extra>",
        )
    _base_layout(fig, title, demo, height=170)
    fig.update_layout(barmode="stack", showlegend=True,
                      legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"})
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(showticklabels=False)
    return fig


def index_histogram(values: np.ndarray, threshold: float, method: str, demo: bool) -> go.Figure:
    v = values[np.isfinite(values)]
    fig = go.Figure(
        go.Histogram(
            x=v,
            nbinsx=120,
            marker={"color": SEQ_BLUE, "line": {"width": 0}},
            hovertemplate="Index %{x}<br>%{y:,} pixels<extra></extra>",
        )
    )
    fig.add_vline(x=threshold, line={"color": "#d03b3b", "width": 2, "dash": "dash"})
    fig.add_annotation(
        x=threshold, y=1, yref="paper", text=f"threshold {threshold:.3f}", showarrow=False,
        xanchor="left", xshift=6, font={"size": 12},
    )
    _base_layout(fig, f"{method} distribution (pixels at analysis resolution)", demo, height=300)
    fig.update_layout(bargap=0.02)
    fig.update_xaxes(title=f"{method} value")
    fig.update_yaxes(title="Pixels")
    return fig


def numeric_histogram(series: pd.Series, name: str, demo: bool) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=series.dropna(),
            marker={"color": SEQ_BLUE, "line": {"width": 0}},
            hovertemplate=f"{name}: %{{x}}<br>%{{y}} feature(s)<extra></extra>",
        )
    )
    _base_layout(fig, f"Distribution of '{name}'", demo, height=300)
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title=name)
    fig.update_yaxes(title="Features")
    return fig


def category_counts_bar(series: pd.Series, name: str, demo: bool, top: int = 15) -> go.Figure:
    counts = series.astype(str).value_counts()
    if len(counts) > top:
        counts = pd.concat([counts.iloc[:top], pd.Series({"Other": counts.iloc[top:].sum()})])
    counts = counts.sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker={"color": SEQ_BLUE},
            hovertemplate="<b>%{y}</b><br>%{x} feature(s)<extra></extra>",
        )
    )
    _base_layout(fig, f"Features by '{name}'", demo, height=max(240, 34 * len(counts) + 80))
    fig.update_xaxes(title="Features")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig
