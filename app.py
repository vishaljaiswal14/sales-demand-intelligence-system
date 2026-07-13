"""End-to-End Sales Forecasting & Demand Intelligence System — Streamlit dashboard.

Entry point only: page registry, global theme, and navigation. All presentation lives in
``src/dashboard/views``; all analytics were precomputed by ``src/dashboard/artifacts.py``
(run ``python3 -m src.dashboard.artifacts`` to rebuild them after a data change).

Launch with:

    streamlit run app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard import theme
from src.dashboard.views import (
    about, anomalies, clusters, forecasting, overview, sales, segments,
)

st.set_page_config(
    page_title="Demand Intelligence · Superstore",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.register_plotly_template()
theme.inject_css()

st.logo(
    str(PROJECT_ROOT / "src" / "dashboard" / "assets" / "logo.svg"),
    size="large",
)

navigation = st.navigation(
    {
        "": [
            # The default page is always served at the app root, so it takes no url_path.
            st.Page(overview.render, title="Executive Overview",
                    icon=":material/space_dashboard:", default=True),
        ],
        "Analytics": [
            st.Page(sales.render, title="Sales Analytics",
                    icon=":material/monitoring:", url_path="sales"),
            st.Page(forecasting.render, title="Forecasting",
                    icon=":material/trending_up:", url_path="forecasting"),
            st.Page(segments.render, title="Segment Forecasts",
                    icon=":material/stacked_line_chart:", url_path="segments"),
        ],
        "Intelligence": [
            st.Page(anomalies.render, title="Anomaly Detection",
                    icon=":material/crisis_alert:", url_path="anomalies"),
            st.Page(clusters.render, title="Product Clustering",
                    icon=":material/bubble_chart:", url_path="clusters"),
        ],
        "Project": [
            st.Page(about.render, title="About",
                    icon=":material/info:", url_path="about"),
        ],
    }
)

navigation.run()
