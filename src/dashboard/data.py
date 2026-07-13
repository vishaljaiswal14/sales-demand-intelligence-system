"""Cached data access for the dashboard.

Thin ``st.cache_data`` wrappers around the artifact loaders (and the one processed
dataset the Sales Analytics page filters interactively). Nothing here computes analytics
— artifacts are built offline by ``python3 -m src.dashboard.artifacts``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import artifacts
from src.utils.paths import PROCESSED_DATA_DIR


@st.cache_data(show_spinner=False)
def kpis() -> dict:
    return artifacts.load_kpis()


@st.cache_data(show_spinner=False)
def meta(name: str) -> dict:
    return artifacts.load_json(name)


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    """Load a dashboard artifact CSV, parsing any date-like first column."""
    frame = artifacts.load_table(name)
    first = frame.columns[0]
    if first in ("date", "week"):
        frame[first] = pd.to_datetime(frame[first])
    return frame


@st.cache_data(show_spinner=False)
def line_items() -> pd.DataFrame:
    """The Task 1 processed line-item dataset, used for interactive slicing only."""
    return pd.read_csv(
        PROCESSED_DATA_DIR / "sales_with_features.csv",
        parse_dates=["Order Date", "Ship Date"],
    )


@st.cache_data(show_spinner=False)
def monthly_history() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "monthly_sales.csv", parse_dates=["Order Date"])
    return frame.rename(columns={"Order Date": "date", "Sales": "sales"})
