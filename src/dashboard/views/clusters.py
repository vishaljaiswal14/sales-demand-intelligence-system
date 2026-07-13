"""Product Clustering: K-Means demand segments with per-cluster stocking guidance."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content, data, figures
from src.dashboard.components import callout, kpi_row, page_header, section, spacer
from src.dashboard.theme import PLOTLY_CONFIG

# Human-readable labels for the internal feature columns, applied at display time only.
_COLUMN_LABELS = {
    "TotalSales": "Total Sales",
    "MeanYoYGrowth": "YoY Growth",
    "MonthlyVolatilityCV": "Volatility (CV)",
    "AvgOrderValue": "Avg Order Value",
    "OrderLines": "Order Lines",
    "n": "Products",
}


def render() -> None:
    page_header(
        "Product Clustering",
        "17 sub-categories segmented by volume, growth, volatility, and order value",
    )

    members = data.table("cluster_members")
    profile = data.table("cluster_profile")
    meta = data.meta("clustering_meta")

    with st.container(border=True):
        variance = meta["explained_variance_ratio"]
        section(
            "Demand segments (PCA projection)",
            f"PCA is used for display only — clusters were formed in the full "
            f"4-feature space. The two components retain "
            f"{sum(variance):.0%} of feature variance.",
        )
        st.plotly_chart(figures.pca_scatter(members), config=PLOTLY_CONFIG)

    with st.expander("How k = 4 was chosen (elbow + silhouette)"):
        left, right = st.columns(2)
        elbow, silhouette = figures.elbow_and_silhouette(data.table("cluster_search"))
        left.plotly_chart(elbow, config=PLOTLY_CONFIG)
        right.plotly_chart(silhouette, config=PLOTLY_CONFIG)
        st.caption(
            "Inertia's diminishing returns set in around k=4 while the silhouette score "
            "is nearly flat across k=3–5, so interpretability broke the tie: k=4 cleanly "
            "separates four distinct business profiles, including the Supplies outlier."
        )

    spacer(0.4)
    section("Cluster profiles", "Mean feature values per segment — the statistics every "
            "label below is derived from")
    st.dataframe(
        profile.drop(columns=["Cluster"]).set_index("ClusterName")
        .rename_axis("Segment")[
            ["TotalSales", "MeanYoYGrowth", "MonthlyVolatilityCV", "AvgOrderValue",
             "n", "Sub-Categories"]
        ].rename(columns=_COLUMN_LABELS).style.format({
            "Total Sales": "${:,.0f}", "YoY Growth": "{:+.1f}%",
            "Volatility (CV)": "{:.2f}", "Avg Order Value": "${:,.0f}",
        }),
        width="stretch",
    )

    spacer(0.4)
    section("Segment deep-dive", "Statistics, business meaning, and stocking strategy")
    names = list(profile["ClusterName"])
    selected = st.segmented_control("Cluster", names, default=names[0],
                                    label_visibility="collapsed") or names[0]

    row = profile[profile["ClusterName"] == selected].iloc[0]
    strategy = content.CLUSTER_STRATEGIES[selected]

    kpi_row([
        ("Avg Total Sales", f"${row['TotalSales']:,.0f}", "per sub-category, 4 years"),
        ("Avg YoY Growth", f"{row['MeanYoYGrowth']:+.1f}%", "mean annual change"),
        ("Volatility (CV)", f"{row['MonthlyVolatilityCV']:.2f}",
         "monthly std ÷ mean — scale-free"),
        ("Avg Order Value", f"${row['AvgOrderValue']:,.0f}",
         f"{int(row['n'])} sub-categor{'y' if row['n'] == 1 else 'ies'}"),
    ])
    spacer(0.3)
    callout(f"<b>What this segment is.</b> {strategy['meaning']}", kind="neutral")
    callout(f"<b>Stocking strategy.</b> {strategy['strategy']}", kind="info")

    member_rows = members[members["ClusterName"] == selected]
    with st.container(border=True):
        section(f"Members — {selected}")
        st.dataframe(
            member_rows.set_index("Sub-Category")[
                ["TotalSales", "MeanYoYGrowth", "MonthlyVolatilityCV",
                 "AvgOrderValue", "OrderLines"]
            ].rename(columns=_COLUMN_LABELS).style.format({
                "Total Sales": "${:,.0f}", "YoY Growth": "{:+.1f}%",
                "Volatility (CV)": "{:.2f}", "Avg Order Value": "${:,.0f}",
                "Order Lines": "{:,.0f}",
            }),
            width="stretch",
        )
