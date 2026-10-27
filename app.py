"""Drone Based Campus Analytics - Streamlit entry point.

Run from the project folder:

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from campus_dashboard import config
from campus_dashboard.ui import INCLUDE_DEMO_KEY

st.set_page_config(
    page_title="Drone Based Campus Analytics",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Dashboard": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/map_viewer.py", title="Orthomosaic & GIS Map", icon=":material/map:"),
    ],
    "Analysis": [
        st.Page("app_pages/land_use.py", title="Land Use", icon=":material/landscape:"),
        st.Page("app_pages/green_cover.py", title="Green Cover", icon=":material/park:"),
        st.Page("app_pages/spatial_layers.py", title="Spatial Layers", icon=":material/layers:"),
        st.Page("app_pages/assets_3d.py", title="3D Campus Assets", icon=":material/view_in_ar:"),
    ],
    "Data": [
        st.Page("app_pages/data_manager.py", title="Data Manager", icon=":material/upload_file:"),
        st.Page("app_pages/about.py", title="Methodology & Data Status", icon=":material/info:"),
    ],
}

st.session_state.setdefault(INCLUDE_DEMO_KEY, True)

with st.sidebar:
    st.markdown("### Drone Based Campus Analytics")
    st.caption(config.CAMPUS_NAME)
    st.toggle(
        "Include demo data",
        key=INCLUDE_DEMO_KEY,
        help="Show the synthetic demonstration layers. They are labelled [DEMO] everywhere and are "
        "never survey results.",
    )
    if st.session_state[INCLUDE_DEMO_KEY]:
        st.caption(":red[DEMO layers are synthetic and are not survey results.]")

st.navigation(pages).run()

st.sidebar.divider()
st.sidebar.caption("Final Year Project - Dashboard Development, Integration & Testing")
