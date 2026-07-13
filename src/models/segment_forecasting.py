"""Category- and region-level forecasting, reusing the Task 3 production SARIMA model.

Task 3 selected SARIMA(1,0,1)(0,1,1,12) as the production model on the strength of a
rolling-origin backtest against Prophet and XGBoost. This module applies that exact
specification to each business segment rather than re-selecting orders per segment: every
segment shares the same underlying retail calendar (the same holidays, the same academic
year, the same Q4 demand surge), and a quick ACF check per segment (see the notebook) shows
the same signature as the aggregate series — modest lag-1 autocorrelation with a materially
larger spike at the seasonal lag of 12 months. Re-running a full order search per segment
would tune to noise on only 45 monthly training points each, not add real signal.
"""

from __future__ import annotations

import pandas as pd

from src.models import sarima

# (dataframe column, value to filter on) for each required segment.
SEGMENT_DEFINITIONS: dict[str, tuple[str, str]] = {
    "Furniture": ("Category", "Furniture"),
    "Technology": ("Category", "Technology"),
    "Office Supplies": ("Category", "Office Supplies"),
    "West": ("Region", "West"),
    "East": ("Region", "East"),
}


def forecast_segment(
    series: pd.Series,
    horizon: int = 3,
    order: tuple = sarima.SELECTED_ORDER,
    seasonal_order: tuple = sarima.SELECTED_SEASONAL_ORDER,
) -> dict:
    """Fit the production SARIMA spec on one segment and forecast ``horizon`` months ahead.

    Returns the forecast alongside the Ljung-Box residual p-value so each segment's model
    adequacy is checked individually rather than assumed to carry over from the aggregate fit.
    """
    fitted = sarima.fit_sarima(series, order=order, seasonal_order=seasonal_order)
    forecast = sarima.sarima_forecast(fitted, horizon=horizon)
    return {"forecast": forecast, "ljung_box_pvalue": sarima.ljung_box_pvalue(fitted)}


def forecast_all_segments(
    segment_series: dict[str, pd.Series], horizon: int = 3
) -> dict[str, dict]:
    """Forecast every segment in ``segment_series`` with the shared production SARIMA spec."""
    return {
        name: forecast_segment(series, horizon) for name, series in segment_series.items()
    }


def summarise_segment_growth(
    segment_series: dict[str, pd.Series], segment_results: dict[str, dict]
) -> pd.DataFrame:
    """Rank segments by year-over-year forecast growth, with a confidence-interval check.

    Growth is measured against the *same calendar months one year earlier* rather than the
    immediately preceding quarter. Comparing to the trailing quarter would conflate genuine
    demand growth with the strong seasonal drop-off out of the Q4 holiday peak (Task 2 found
    January/February are the deepest seasonal troughs); comparing year-over-year isolates
    trend growth from the seasonal cycle.

    ``CI Width % of Forecast`` reports how wide the 95% interval is relative to the point
    forecast, so a large headline growth number can be read alongside how much to trust it —
    a segment with a big point-forecast gain but a very wide interval is a different story
    from one with the same gain and a tight interval.
    """
    rows = []
    for name, series in segment_series.items():
        forecast = segment_results[name]["forecast"]
        prior_year_index = forecast.index - pd.DateOffset(years=1)
        prior_year_actual = series.reindex(prior_year_index)
        if prior_year_actual.isna().any():
            raise ValueError(f"Missing prior-year actuals for segment '{name}'.")

        forecast_total = forecast["forecast"].sum()
        prior_year_total = prior_year_actual.sum()
        ci_width_pct = ((forecast["upper"] - forecast["lower"]) / forecast["forecast"]).mean() * 100

        rows.append(
            {
                "Segment": name,
                "Same Quarter Last Year": round(prior_year_total, 0),
                "Forecast (Next Quarter)": round(forecast_total, 0),
                "YoY Growth %": round((forecast_total / prior_year_total - 1) * 100, 1),
                "CI Width % of Forecast": round(ci_width_pct, 0),
                "Ljung-Box p-value": round(segment_results[name]["ljung_box_pvalue"], 3),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("Segment")
        .sort_values("YoY Growth %", ascending=False)
    )
