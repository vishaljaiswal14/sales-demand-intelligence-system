"""Sales Analytics: interactive slicing of the processed line-item dataset.

Filtering here is display-level pandas aggregation of Task 1's processed data — no
models, no business logic.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import data, figures
from src.dashboard.components import fmt_money, kpi_row, page_header, section, spacer
from src.dashboard.theme import PLOTLY_CONFIG


def render() -> None:
    page_header("Sales Analytics", "Slice four years of line-item sales by year, region, and category")

    sales = data.line_items()
    years = sorted(sales["Year"].unique())
    regions = sorted(sales["Region"].unique())
    categories = sorted(sales["Category"].unique())

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
        with f1:
            picked_years = st.pills("Year", years, selection_mode="multi", default=years)
        with f2:
            picked_regions = st.pills("Region", regions, selection_mode="multi", default=regions)
        with f3:
            picked_categories = st.pills("Category", categories, selection_mode="multi",
                                         default=categories)

    filtered = sales[
        sales["Year"].isin(picked_years or years)
        & sales["Region"].isin(picked_regions or regions)
        & sales["Category"].isin(picked_categories or categories)
    ]
    if filtered.empty:
        st.info("No data matches the current filter combination.")
        return

    spacer(0.4)
    top_subcategory = filtered.groupby("Sub-Category")["Sales"].sum().idxmax()
    kpi_row([
        ("Selected Sales", fmt_money(filtered["Sales"].sum()),
         f"{len(filtered):,} line items"),
        ("Orders", f"{filtered['Order ID'].nunique():,}", "unique orders in selection"),
        ("Avg Order Value",
         f"${filtered['Sales'].sum() / filtered['Order ID'].nunique():,.0f}",
         "revenue per order"),
        ("Top Sub-Category", top_subcategory,
         fmt_money(filtered.groupby('Sub-Category')['Sales'].sum().max())),
    ])

    spacer(0.4)
    with st.container(border=True):
        section("Monthly sales trend", "Aggregated over the current selection")
        monthly = (
            filtered.set_index("Order Date")["Sales"].resample("MS").sum()
            .rename("sales").rename_axis("date").reset_index()
        )
        st.plotly_chart(figures.monthly_trend(monthly), config=PLOTLY_CONFIG)

    spacer(0.4)
    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Revenue by category")
        st.plotly_chart(
            figures.category_bar(filtered.groupby("Category")["Sales"].sum()),
            config=PLOTLY_CONFIG,
        )
    with right, st.container(border=True):
        section("Revenue by region")
        st.plotly_chart(
            figures.category_bar(filtered.groupby("Region")["Sales"].sum()),
            config=PLOTLY_CONFIG,
        )

    spacer(0.4)
    left, right = st.columns([1.1, 1])
    with left, st.container(border=True):
        section("Seasonality heatmap", "Monthly sales by year — darker is stronger")
        heat = filtered.pivot_table(index="Month", columns="Year", values="Sales",
                                    aggfunc="sum").reindex(range(1, 13)).fillna(0)
        st.plotly_chart(figures.seasonality_heatmap(heat), config=PLOTLY_CONFIG)
    with right, st.container(border=True):
        section("Top sub-categories", "Ten largest by revenue in selection")
        st.plotly_chart(
            figures.top_subcategories(filtered.groupby("Sub-Category")["Sales"].sum()),
            config=PLOTLY_CONFIG,
        )
