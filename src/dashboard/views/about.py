"""About: project overview, dataset, methodology, stack, architecture, and author."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content
from src.dashboard.components import callout, page_header, section, spacer

_ARCHITECTURE = """\
SalesForecasting_VishalJaiswal/
├── data/
│   ├── raw/                  # train.csv (Kaggle Superstore)
│   └── processed/            # engineered datasets + dashboard artifacts
├── src/
│   ├── data/                 # loading, quality audit, aggregation
│   ├── features/             # feature engineering & EDA
│   ├── models/               # SARIMA · Prophet · XGBoost · Anomaly Detection · K-Means
│   ├── visualization/        # notebook chart builders
│   └── dashboard/            # Streamlit app, pages, themes & cached artifacts
├── charts/                   # generated figures
├── docs/                     # dashboard screenshots
├── reports/                  # Executive_Business_Report.pdf
├── analysis.ipynb            # complete analytical workflow
├── app.py                    # Streamlit entry point
├── README.md                 # project documentation
└── requirements.txt          # Python dependencies"""


def render() -> None:
    page_header("About This Project", "End-to-End Sales Forecasting & Demand Intelligence System")

    section("Overview")
    st.markdown(
        "A production-style demand-intelligence system built on four years of Superstore "
        "retail data. It answers the question every retailer lives by — *how much of "
        "each product will we sell next quarter?* — through time-series decomposition, "
        "three competing forecasting models, dual-method anomaly detection, K-Means "
        "demand segmentation, and this interactive dashboard."
    )
    callout(
        "<b>Design principle:</b> the dashboard never recomputes analytics. Every result "
        "shown is produced by the same <code>src/</code> modules the analysis notebook "
        "uses, serialized once to versioned artifacts, and served read-only.",
        kind="info",
    )

    spacer(0.3)
    section("Dataset")
    st.markdown(
        "- **Primary:** [Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) "
        "— 9,800 order line items, 2015–2018, across 3 categories, 17 sub-categories, "
        "and 4 US regions.\n"
        "- **Supplementary:** [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) "
        "— evaluated for the anomaly task and deliberately excluded: it shares no "
        "natural join key with Superstore data, so merging it would have been forced "
        "rather than analytical."
    )

    spacer(0.3)
    section("Methodology", "Every modelling decision in the notebook is evidence-backed")
    for i, (title, description) in enumerate(content.METHODOLOGY_STEPS, start=1):
        st.markdown(f"**{i} · {title}** — {description}")

    spacer(0.3)
    section("Technology stack")
    columns = st.columns(2)
    for i, (tool, purpose) in enumerate(content.TECH_STACK):
        with columns[i % 2]:
            st.markdown(f"- **{tool}** — {purpose}")

    spacer(0.3)
    section("Architecture")
    st.code(_ARCHITECTURE, language="text")
    st.markdown(
        "**Data flow:** raw CSV → quality-audited processed datasets → model artifacts "
        "(forecasts, flags, clusters) → cached read-only presentation. Reusable logic "
        "lives in `src/`; the notebook orchestrates analysis; Streamlit handles only "
        "layout and interaction."
    )

    spacer(0.3)
    section("Author")
    st.markdown(
        "**Vishal Jaiswal**  \n"
        "AI & Data Science Intern  \n"
        "jaiswalvishal9694@gmail.com"
    )
