"""Forecasting: three-model comparison, hold-out evaluation, and next-quarter outlook."""

from __future__ import annotations

import streamlit as st

from src.dashboard import content, data, figures
from src.dashboard.components import callout, kpi_row, page_header, section, spacer
from src.dashboard.theme import PLOTLY_CONFIG

MODELS = ("SARIMA", "Prophet", "XGBoost")


def render() -> None:
    page_header(
        "Forecasting",
        "SARIMA vs. Prophet vs. XGBoost on monthly sales — evaluated without leakage",
    )

    meta = data.meta("forecasting_meta")
    callout(content.RECOMMENDATION, kind="success")

    spacer(0.3)
    model = st.segmented_control("Model", MODELS, default="SARIMA")
    if model is None:
        model = "SARIMA"

    blurb = content.MODEL_BLURBS[model]
    if model == "SARIMA":
        blurb = blurb.format(ljung_box=meta["sarima_ljung_box_pvalue"])
    callout(blurb, kind="neutral")

    holdout = data.table("holdout_forecasts")
    metrics = data.table("holdout_metrics").set_index("model")
    row = metrics.loc[model]
    kpi_row([
        ("Hold-out MAE", f"${row['MAE']:,.0f}", "mean absolute error, 2018 Q4"),
        ("Hold-out RMSE", f"${row['RMSE']:,.0f}", "penalises large misses"),
        ("Hold-out MAPE", f"{row['MAPE']:.1f}%", "error as % of actual sales"),
        ("Training Window", f"{meta['train_months']} months",
         f"{meta['test_months']} months held out"),
    ])

    spacer(0.4)
    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Hold-out test — 2018 Q4", "Forecast vs. withheld actuals")
        st.plotly_chart(
            figures.holdout_chart(data.monthly_history(), holdout, model),
            config=PLOTLY_CONFIG,
        )
    with right, st.container(border=True):
        section("Production forecast — 2019 Q1", "Refit on all 48 months")
        st.plotly_chart(
            figures.future_chart(data.monthly_history(), data.table("future_forecasts"), model),
            config=PLOTLY_CONFIG,
        )

    tone, note = content.CONFIDENCE_NOTES[model]
    callout(note, kind=tone)

    future = data.table("future_forecasts")
    display_cols = ["date", model] + [
        c for c in (f"{model}_lower", f"{model}_upper") if c in future.columns
    ]
    table = future[display_cols].copy()
    table["date"] = table["date"].dt.strftime("%B %Y")
    table.columns = ["Month", "Forecast"] + (
        ["Lower 95%", "Upper 95%"] if len(display_cols) == 4 else []
    )
    with st.container(border=True):
        section(f"{model} — next-quarter forecast values")
        st.dataframe(
            table.style.format({c: "${:,.0f}" for c in table.columns if c != "Month"}),
            hide_index=True, width="stretch",
        )

    spacer(0.6)
    section("Model comparison", "Same three models, two evaluation protocols")
    tab_backtest, tab_holdout = st.tabs(["Rolling-origin backtest (decisive)", "Single hold-out"])
    with tab_backtest:
        backtest = data.table("backtest_metrics")
        col_chart, col_table = st.columns([1.2, 1])
        with col_chart:
            st.plotly_chart(
                figures.metric_comparison(backtest, highlight="SARIMA"),
                config=PLOTLY_CONFIG,
            )
        with col_table:
            st.dataframe(
                backtest.style.format(
                    {"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "MAPE": "{:.1f}%"}
                ),
                hide_index=True, width="stretch",
            )
            st.caption(
                "18 pooled forecast points from 6 expanding-window refits per model. "
                "Every origin trains only on its own past — no leakage."
            )
    with tab_holdout:
        col_chart, col_table = st.columns([1.2, 1])
        with col_chart:
            st.plotly_chart(
                figures.metric_comparison(data.table("holdout_metrics")),
                config=PLOTLY_CONFIG,
            )
        with col_table:
            st.dataframe(
                data.table("holdout_metrics").style.format(
                    {"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "MAPE": "{:.1f}%"}
                ),
                hide_index=True, width="stretch",
            )
            st.caption(
                "A single 3-month window — noisy on its own. XGBoost's apparent win "
                "here reversed under the backtest, which is why the recommendation "
                "follows the backtest."
            )
