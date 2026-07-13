"""Chart builders for the exploration and time-series analysis phases.

Every function takes an already-aggregated frame (or a statsmodels result) and returns a
matplotlib ``Figure``. Keeping the figures free of any file I/O lets the notebook decide
when and where to save them (always via ``style.save_figure``), and makes the charts easy
to unit-test.
"""

from __future__ import annotations

import calendar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import DecomposeResult

from src.visualization.style import CATEGORICAL_PALETTE

_MONTH_LABELS = [calendar.month_abbr[m] for m in range(1, 13)]


def _thousands_formatter(value: float, _pos: int) -> str:
    return f"${value / 1000:,.0f}k"


def plot_monthly_sales_trend(monthly_sales: pd.DataFrame) -> plt.Figure:
    """Line chart of total sales per month across the full history."""
    fig, ax = plt.subplots()
    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        color=CATEGORICAL_PALETTE[0],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_title("Monthly Sales Trend (2015-2018)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    fig.tight_layout()
    return fig


def plot_revenue_by_category(revenue: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of total revenue per product category."""
    ordered = revenue.sort_values("Total Revenue")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(ordered.index, ordered["Total Revenue"], color=CATEGORICAL_PALETTE[:len(ordered)])
    ax.set_title("Total Revenue by Product Category")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("Category")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    for y, (value, share) in enumerate(
        zip(ordered["Total Revenue"], ordered["Revenue Share %"])
    ):
        ax.text(value, y, f"  ${value/1000:,.0f}k ({share:.1f}%)", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_yearly_sales_by_region(yearly_pivot: pd.DataFrame) -> plt.Figure:
    """Line chart of annual sales per region to expose growth trajectories."""
    fig, ax = plt.subplots()
    for i, region in enumerate(yearly_pivot.columns):
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        ax.plot(
            yearly_pivot.index,
            yearly_pivot[region],
            marker="o",
            linewidth=2,
            color=color,
            label=region,
        )
    ax.set_title("Annual Sales by Region")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales")
    ax.set_xticks(yearly_pivot.index)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax.legend(title="Region")
    fig.tight_layout()
    return fig


def plot_fulfillment_by_region(fulfillment: pd.DataFrame) -> plt.Figure:
    """Bar chart of mean order-to-ship days per region with error bars for spread."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        fulfillment.index,
        fulfillment["Mean Days"],
        yerr=fulfillment["Std Days"],
        capsize=5,
        color=CATEGORICAL_PALETTE[1],
    )
    ax.set_title("Average Order-to-Ship Time by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Fulfillment Days")
    for x, value in enumerate(fulfillment["Mean Days"]):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def plot_seasonality_heatmap(seasonality_pivot: pd.DataFrame) -> plt.Figure:
    """Heatmap of monthly sales (rows) across years (columns) to reveal seasonality."""
    matrix = seasonality_pivot.drop(columns="Average", errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=",.0f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Total Sales"},
        ax=ax,
    )
    ax.set_title("Monthly Sales Heatmap by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_yticklabels(_MONTH_LABELS, rotation=0)
    fig.tight_layout()
    return fig


def plot_seasonal_decomposition(result: DecomposeResult) -> plt.Figure:
    """Stacked observed / trend / seasonal / residual panels on a shared time axis.

    We render the four components manually rather than using ``result.plot()`` so the
    figure inherits the project style (sizing, palette, gridlines) and the residual panel
    can be drawn as points around a zero reference line, which reads more clearly than a
    connected line for noise.
    """
    components = [
        ("Observed", result.observed, CATEGORICAL_PALETTE[0], "line"),
        ("Trend", result.trend, CATEGORICAL_PALETTE[1], "line"),
        ("Seasonal", result.seasonal, CATEGORICAL_PALETTE[3], "line"),
        ("Residual", result.resid, CATEGORICAL_PALETTE[4], "points"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (label, values, color, style) in zip(axes, components):
        if style == "points":
            ax.axhline(0, color="#999999", linewidth=1, zorder=1)
            ax.scatter(values.index, values, color=color, s=18, zorder=2)
        else:
            ax.plot(values.index, values, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))

    axes[0].set_title("Additive Decomposition of Monthly Sales")
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    return fig


def plot_first_difference(differenced: pd.Series) -> plt.Figure:
    """Line chart of the first-differenced series around a zero mean line.

    Differencing turns levels into month-over-month change; plotting it around zero makes
    it easy to see whether the mean and variance look stable over time (visual stationarity).
    """
    fig, ax = plt.subplots()
    ax.axhline(0, color="#999999", linewidth=1)
    ax.plot(differenced.index, differenced, color=CATEGORICAL_PALETTE[2], linewidth=1.8, marker="o", markersize=3)
    ax.set_title("First-Differenced Monthly Sales (Month-over-Month Change)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Change in Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 24) -> plt.Figure:
    """Side-by-side ACF and PACF used to justify the SARIMA orders.

    The ACF suggests the moving-average (q, Q) orders and the PACF the autoregressive
    (p, P) orders; spikes at the seasonal lag (12) reveal the annual structure.
    """
    # The PACF is only defined for lags below half the sample size; cap to stay valid on
    # this short series while still reaching well past the seasonal lag of 12.
    lags = min(lags, len(series) // 2 - 1)

    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_acf(series, lags=lags, ax=ax_acf, color=CATEGORICAL_PALETTE[0])
    plot_pacf(series, lags=lags, ax=ax_pacf, method="ywm", color=CATEGORICAL_PALETTE[0])
    ax_acf.set_title("Autocorrelation (ACF)")
    ax_pacf.set_title("Partial Autocorrelation (PACF)")
    for ax in (ax_acf, ax_pacf):
        ax.set_xlabel("Lag (months)")
    fig.tight_layout()
    return fig


def plot_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    actual: pd.Series | None = None,
    title: str = "Forecast",
) -> plt.Figure:
    """Plot observed history, the forecast, an optional confidence band, and optional actuals.

    One reusable figure serves every model: pass a ``forecast`` frame with just a ``forecast``
    column, or with ``lower``/``upper`` to draw the interval; pass ``actual`` to overlay the
    held-out truth for a visual accuracy check.
    """
    fig, ax = plt.subplots()
    ax.plot(history.index, history, color=CATEGORICAL_PALETTE[0], linewidth=2, label="Observed")
    ax.plot(
        forecast.index, forecast["forecast"],
        color=CATEGORICAL_PALETTE[4], linewidth=2, marker="o", markersize=4,
        linestyle="--", label="Forecast",
    )
    if {"lower", "upper"}.issubset(forecast.columns):
        ax.fill_between(
            forecast.index, forecast["lower"], forecast["upper"],
            color=CATEGORICAL_PALETTE[4], alpha=0.18, label="95% interval",
        )
    if actual is not None:
        ax.plot(
            actual.index, actual, color=CATEGORICAL_PALETTE[3],
            linewidth=0, marker="D", markersize=7, label="Actual (held out)",
        )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Compare models on error metrics: dollar errors (MAE, RMSE) and percentage error (MAPE).

    MAE/RMSE and MAPE live on different scales, so they get separate panels rather than being
    forced onto one misleading axis.
    """
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(12, 4.5))

    dollar_metrics = metrics[["MAE", "RMSE"]]
    dollar_metrics.plot(kind="bar", ax=ax_abs, color=CATEGORICAL_PALETTE[:2], rot=0)
    ax_abs.set_title("Absolute Error (lower is better)")
    ax_abs.set_ylabel("Dollars")
    ax_abs.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax_abs.legend(title="Metric")

    ax_pct.bar(metrics.index, metrics["MAPE"], color=CATEGORICAL_PALETTE[2])
    ax_pct.set_title("Mean Absolute Percentage Error (lower is better)")
    ax_pct.set_ylabel("MAPE (%)")
    for x, value in enumerate(metrics["MAPE"]):
        ax_pct.text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return fig


def plot_segment_forecast_comparison(
    segment_series: dict[str, pd.Series],
    segment_forecasts: dict[str, pd.DataFrame],
    history_months: int = 12,
) -> plt.Figure:
    """One chart overlaying recent history and the 3-month forecast for every segment.

    Only the trailing ``history_months`` of actuals are drawn (rather than the full four-year
    history) so the recent trajectory and the forecast it feeds into stay visually legible on
    a single shared axis — five full histories overlaid would be an unreadable tangle.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, name in enumerate(segment_series):
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        recent_history = segment_series[name].iloc[-history_months:]
        forecast_points = segment_forecasts[name]["forecast"]

        ax.plot(recent_history.index, recent_history, color=color, linewidth=2, label=name)
        # Bridge the last actual point into the forecast so the dashed segment reads as a
        # continuation of the solid line rather than a disconnected series.
        bridge_index = recent_history.index[-1:].append(forecast_points.index)
        bridge_values = pd.concat([recent_history.iloc[-1:], forecast_points])
        ax.plot(
            bridge_index, bridge_values, color=color, linewidth=2,
            linestyle="--", marker="o", markersize=4,
        )

    ax.axvline(
        segment_series[next(iter(segment_series))].index[-1],
        color="#999999", linewidth=1, linestyle=":",
    )
    ax.set_title("Category & Region Forecasts: Recent History and 3-Month Outlook")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax.legend(title="Segment (solid = actual, dashed = forecast)", loc="upper left")
    fig.tight_layout()
    return fig


def plot_weekly_anomalies(
    series: pd.Series, flags: pd.Series, method_label: str
) -> plt.Figure:
    """Weekly sales with one method's flagged anomaly weeks marked distinctly.

    ``flags`` may cover fewer weeks than ``series`` (e.g. Isolation Forest's features need
    one week of lag history); it is reindexed onto the full series so unscored weeks are
    simply treated as not flagged rather than raising a misalignment error.
    """
    aligned_flags = flags.reindex(series.index, fill_value=False)
    anomalies = series[aligned_flags]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(series.index, series, color=CATEGORICAL_PALETTE[0], linewidth=1.4, label="Weekly Sales")
    ax.scatter(
        anomalies.index, anomalies, color=CATEGORICAL_PALETTE[4], s=80, zorder=5,
        marker="D", edgecolor="white", linewidth=0.8, label=f"{method_label} anomaly",
    )
    ax.set_title(f"Weekly Sales with {method_label} Anomalies")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_anomaly_method_comparison(
    series: pd.Series, zscore_flags: pd.Series, isolation_forest_flags: pd.Series
) -> plt.Figure:
    """Weekly sales split into three categories: both methods agree, or only one flags it.

    Distinguishing "both agree" from "one method only" turns the comparison into something
    visible at a glance, rather than requiring two separate charts to be mentally overlaid.
    """
    z = zscore_flags.reindex(series.index, fill_value=False)
    iso = isolation_forest_flags.reindex(series.index, fill_value=False)
    both = z & iso
    z_only = z & ~iso
    if_only = iso & ~z

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        series.index, series, color=CATEGORICAL_PALETTE[0],
        linewidth=1.2, alpha=0.8, label="Weekly Sales", zorder=1,
    )
    ax.scatter(
        series.index[z_only], series[z_only], color=CATEGORICAL_PALETTE[2],
        marker="^", s=80, label="Z-Score only", zorder=4,
    )
    ax.scatter(
        series.index[if_only], series[if_only], color=CATEGORICAL_PALETTE[3],
        marker="s", s=80, label="Isolation Forest only", zorder=4,
    )
    ax.scatter(
        series.index[both], series[both], color=CATEGORICAL_PALETTE[4],
        marker="D", s=100, edgecolor="black", linewidth=0.8, label="Both methods agree", zorder=5,
    )
    ax.set_title("Anomaly Detection: Z-Score vs. Isolation Forest Agreement")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Sales")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands_formatter))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cluster_search(search_result: pd.DataFrame) -> plt.Figure:
    """Elbow curve (inertia) and silhouette score side by side across candidate ``k``.

    Showing both in one figure makes it easy to see where the two diagnostics agree or
    disagree, rather than reading the elbow curve's subjective "knee" in isolation.
    """
    fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_elbow.plot(
        search_result.index, search_result["inertia"],
        color=CATEGORICAL_PALETTE[0], linewidth=2, marker="o", markersize=6,
    )
    ax_elbow.set_title("Elbow Method: Inertia by k")
    ax_elbow.set_xlabel("Number of Clusters (k)")
    ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_elbow.set_xticks(search_result.index)

    ax_sil.plot(
        search_result.index, search_result["silhouette_score"],
        color=CATEGORICAL_PALETTE[2], linewidth=2, marker="o", markersize=6,
    )
    ax_sil.set_title("Silhouette Score by k")
    ax_sil.set_xlabel("Number of Clusters (k)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.set_xticks(search_result.index)

    fig.tight_layout()
    return fig


def plot_pca_clusters(
    coords: pd.DataFrame, labels: pd.Series, cluster_names: dict[int, str] | None = None
) -> plt.Figure:
    """2D PCA scatter of every sub-category, coloured by cluster and labelled by name.

    Each point is annotated with its sub-category name — with only 17 products this stays
    readable and lets the chart be read without cross-referencing a separate legend table.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))

    unique_labels = sorted(labels.unique())
    for i, cluster_id in enumerate(unique_labels):
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        members = labels[labels == cluster_id].index
        label_text = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
        ax.scatter(
            coords.loc[members, "PC1"], coords.loc[members, "PC2"],
            color=color, s=110, edgecolor="white", linewidth=0.8, label=label_text, zorder=3,
        )
        for member in members:
            ax.annotate(
                member, (coords.loc[member, "PC1"], coords.loc[member, "PC2"]),
                fontsize=8.5, xytext=(6, 4), textcoords="offset points",
            )

    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.set_title("Product Sub-Category Clusters (PCA Projection, for Visualization Only)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()
    return fig


def plot_cluster_feature_comparison(profile: pd.DataFrame, cluster_names: dict[int, str] | None = None) -> plt.Figure:
    """Small-multiples bar chart comparing each cluster's mean on all four raw features.

    The four features live on very different scales (dollars, a percentage, a ratio, and
    log-dollars), so each gets its own panel rather than being forced onto one shared axis
    where the largest-magnitude feature would visually drown out the others.
    """
    panels = [
        ("TotalSales", "Total Sales ($)", _thousands_formatter),
        ("MeanYoYGrowth", "Mean YoY Growth (%)", None),
        ("MonthlyVolatilityCV", "Monthly Volatility (CV)", None),
        ("AvgOrderValue", "Avg Order Value ($)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    labels = [cluster_names.get(c, f"Cluster {c}") if cluster_names else f"Cluster {c}" for c in profile.index]

    for ax, (column, title, formatter) in zip(axes.flat, panels):
        colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(profile))]
        ax.bar(labels, profile[column], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if formatter:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))

    fig.suptitle("Cluster Feature Comparison (Mean per Cluster)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig
