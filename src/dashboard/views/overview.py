"""Executive Overview: headline KPIs, executive summary, and the hero trend."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content, data, figures
from src.dashboard.components import (
    callout, fmt_money, fmt_pct, kpi_row, page_header, section, spacer,
)
from src.dashboard.theme import PLOTLY_CONFIG


def render() -> None:
    kpis = data.kpis()

    page_header(
        "Executive Overview",
        f"Superstore retail performance · {kpis['date_min']} to {kpis['date_max']}",
    )

    kpi_row([
        ("Total Sales", fmt_money(kpis["total_sales"]),
         f"{kpis['total_line_items']:,} line items over 4 years"),
        ("Total Orders", f"{kpis['total_orders']:,}",
         "unique customer orders"),
        ("Avg Order Value", f"${kpis['avg_order_value']:,.0f}",
         "revenue per order"),
        ("Overall Growth", fmt_pct(kpis["growth_pct_last_vs_first_year"]),
         f"<span class='pos'>{kpis['last_year']} vs {kpis['first_year']}</span> annual revenue"),
    ])
    spacer(0.4)
    kpi_row([
        ("Best Category", kpis["best_category"],
         f"{fmt_money(kpis['best_category_sales'])} · "
         f"{kpis['best_category_share_pct']:.1f}% of revenue"),
        ("Best Region", kpis["best_region"],
         f"{fmt_money(kpis['best_region_sales'])} total revenue"),
        ("Forecast Model", kpis["recommended_model"],
         "selected on rolling-origin backtest"),
        ("Anomalies Detected", f"{kpis['anomaly_weeks_total']} weeks",
         f"of {kpis['weeks_analyzed']} analyzed · {kpis['anomaly_weeks_both']} "
         "high-confidence"),
    ])

    spacer()
    section("Executive summary")
    callout(
        content.EXECUTIVE_SUMMARY.format(
            growth=fmt_pct(kpis["growth_pct_last_vs_first_year"]),
            total=fmt_money(kpis["total_sales"]),
            cat_share=f"{kpis['best_category_share_pct']:.0f}%",
            anomalies=kpis["anomaly_weeks_total"],
        ),
        kind="info",
    )

    spacer(0.4)
    with st.container(border=True):
        section(
            "Monthly revenue and next-quarter outlook",
            "Observed monthly sales with the SARIMA 2019 Q1 forecast and 95% interval",
        )
        st.plotly_chart(
            figures.monthly_trend(data.monthly_history(), data.table("future_forecasts")),
            config=PLOTLY_CONFIG,
        )

    spacer(0.4)
    left, right = st.columns(2)
    sales = data.line_items()
    with left, st.container(border=True):
        section("Revenue by category")
        st.plotly_chart(
            figures.category_bar(sales.groupby("Category")["Sales"].sum()),
            config=PLOTLY_CONFIG,
        )
    with right, st.container(border=True):
        section("Annual sales by region")
        st.plotly_chart(
            figures.yearly_region_lines(
                sales.pivot_table(index="Year", columns="Region", values="Sales",
                                  aggfunc="sum")
            ),
            config=PLOTLY_CONFIG,
        )

    spacer(0.4)
    section("Key findings")
    one, two, three = st.columns(3)
    with one:
        callout("<b>Seasonality dominates.</b> Sep, Nov and Dec rank top-3 in every "
                "year; the seasonal swing (~$62K) exceeds four years of trend movement.",
                kind="neutral")
    with two:
        callout("<b>East grows most consistently</b> — the only region up every year, "
                "on a near-perfect straight line (R² ≈ 0.997).", kind="neutral")
    with three:
        callout("<b>Fulfillment is uniform</b>: ~3.96 days from order to ship "
                "nationwide, with negligible regional variation.", kind="neutral")
