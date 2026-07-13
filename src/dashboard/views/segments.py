"""Segment Forecasts: category- and region-level SARIMA outlooks compared."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content, data, figures
from src.dashboard.components import callout, page_header, section, spacer
from src.dashboard.theme import PLOTLY_CONFIG

SEGMENTS = ("Furniture", "Technology", "Office Supplies", "West", "East")


def render() -> None:
    page_header(
        "Segment Forecasts",
        "The production SARIMA model applied to three categories and two regions",
    )

    picked = st.pills("Segments", SEGMENTS, selection_mode="multi", default=list(SEGMENTS))
    segments = list(picked) if picked else list(SEGMENTS)

    history = data.table("segment_series")
    forecasts = data.table("segment_forecasts")

    with st.container(border=True):
        section(
            "Twelve-month history and next-quarter forecast",
            "Solid lines are actuals; dashed lines are each segment's 2019 Q1 forecast",
        )
        st.plotly_chart(
            figures.segment_comparison(history, forecasts, segments),
            config=PLOTLY_CONFIG,
        )

    spacer(0.4)
    section("Growth outlook", "Forecast vs. the same quarter one year earlier — "
            "year-over-year comparison removes the Q1 seasonal trough")
    growth = data.table("segment_growth")
    st.dataframe(
        growth.style.format({
            "Same Quarter Last Year": "${:,.0f}",
            "Forecast (Next Quarter)": "${:,.0f}",
            "YoY Growth %": "{:+.1f}%",
            "CI Width % of Forecast": "{:.0f}%",
            "Ljung-Box p-value": "{:.3f}",
        }).background_gradient(subset=["YoY Growth %"], cmap="Blues"),
        hide_index=True, width="stretch",
    )
    st.caption(
        "CI Width % reads confidence: a big growth number on a very wide interval is a "
        "weaker claim than a modest one on a tight interval. All five Ljung-Box p-values "
        "exceed 0.05, so the shared SARIMA specification is adequate per segment."
    )

    spacer(0.3)
    for tone, text in content.SEGMENT_INSIGHTS:
        callout(text, kind=tone)

    spacer(0.4)
    section("Segment detail", "Full history with the 95% forecast interval")
    detail = st.selectbox("Segment", segments, label_visibility="collapsed")
    with st.container(border=True):
        st.plotly_chart(
            figures.segment_detail(history, forecasts, detail),
            config=PLOTLY_CONFIG,
        )
