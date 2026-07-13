"""Anomaly Detection: Z-score and Isolation Forest results with evidence per flagged week."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content, data, figures
from src.dashboard.components import callout, chip, kpi_row, page_header, section, spacer
from src.dashboard.theme import PLOTLY_CONFIG

_DRIVER_TONES = {
    "broad-based (many orders)": "green",
    "concentrated (few large orders)": "amber",
    "mixed / inconclusive": "slate",
}


def render() -> None:
    page_header(
        "Anomaly Detection",
        "Two independent screens over 209 trading weeks — statistical and model-based",
    )

    kpis = data.kpis()
    kpi_row([
        ("Weeks Analyzed", f"{kpis['weeks_analyzed']}", "2015–2018 weekly sales"),
        ("Z-Score Flags", f"{kpis['anomaly_weeks_zscore']}", "beyond ±2σ of 13-week mean"),
        ("Isolation Forest Flags", f"{kpis['anomaly_weeks_iforest']}",
         "top 5% most isolable weeks"),
        ("Both Methods Agree", f"{kpis['anomaly_weeks_both']}",
         "highest-confidence anomalies"),
    ])

    spacer(0.4)
    method = st.segmented_control(
        "Method", ["Z-Score", "Isolation Forest", "Agreement"], default="Z-Score"
    ) or "Z-Score"

    callout(content.ANOMALY_METHOD_EXPLAINERS[method], kind="neutral")

    weekly = data.table("weekly_anomalies")
    with st.container(border=True):
        section(
            f"Weekly sales — {method.lower()} view",
            "Hover any point for the exact value; markers denote flagged weeks",
        )
        st.plotly_chart(figures.anomaly_chart(weekly, method), config=PLOTLY_CONFIG)

    spacer(0.4)
    section(
        "Flagged weeks and business evidence",
        "Order-line counts distinguish broad demand surges from a few large orders — "
        "no explanation is asserted without this evidence",
    )

    evidence = data.table("anomaly_evidence")
    if method == "Z-Score":
        shown = evidence[evidence["Flagged By"].str.contains("Z-Score")]
    elif method == "Isolation Forest":
        shown = evidence[evidence["Flagged By"].str.contains("Isolation Forest")]
    else:
        shown = evidence

    display = shown.copy()
    display["week"] = display["week"].dt.strftime("%d %b %Y")
    st.dataframe(
        display.rename(columns={"week": "Week"}).style.format({
            "Sales": "${:,.0f}", "Avg Order Value": "${:,.0f}",
            "WoW % Change": "{:+.0f}%",
        }),
        hide_index=True, width="stretch", height=380,
    )

    spacer(0.3)
    section("Reading the evidence")
    one, two = st.columns(2)
    with one:
        confidence_chip = chip("High confidence", "rose")
        callout(
            f"{confidence_chip}&nbsp; The three weeks flagged by <b>both</b> methods — "
            "late Jul 2015, mid Sep 2015, and early Dec 2018 — are all broad-based "
            "demand surges (60–126 order lines vs. a 42-line median) aligning with "
            "summer clearance, back-to-school, and the Cyber Monday/holiday peak.",
            kind="neutral",
        )
    with two:
        review_chip = chip("Review", "amber")
        callout(
            f"{review_chip}&nbsp; Single-method flags are lower-priority review items. "
            "Notably the record $37.7K week (22 Mar 2015) is Isolation-Forest-only — "
            "inside Z-score's burn-in — and its $698 average order value against a $206 "
            "median points to bulk purchases, not broad demand.",
            kind="neutral",
        )
