"""Plotly figure builders for the dashboard.

Presentation only: every function takes already-computed frames (artifacts or simple
display slices) and returns a themed ``go.Figure``. Series colors come from the theme
tokens so meaning stays consistent across pages — indigo for the primary series, cyan for
comparisons, green for held-out actuals, rose for anomalies.
"""

from __future__ import annotations

import calendar

import pandas as pd
import plotly.graph_objects as go

from src.dashboard import theme

_MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _band(fig: go.Figure, index, lower, upper, color: str, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=list(index) + list(index)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(width=0),
            name=name,
            hoverinfo="skip",
        )
    )


def monthly_trend(history: pd.DataFrame, future: pd.DataFrame | None = None) -> go.Figure:
    """Monthly sales area chart, optionally extended with the SARIMA forecast + interval."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"], y=history["sales"], name="Monthly sales",
            mode="lines", line=dict(color=theme.PRIMARY, width=2.2),
            fill="tozeroy", fillcolor=_hex_to_rgba(theme.PRIMARY, 0.06),
            hovertemplate="%{y:$,.0f}<extra></extra>",
        )
    )
    if future is not None:
        _band(fig, future["date"], future["SARIMA_lower"], future["SARIMA_upper"],
              theme.ACCENT, "95% interval")
        bridge_x = [history["date"].iloc[-1], *future["date"]]
        bridge_y = [history["sales"].iloc[-1], *future["SARIMA"]]
        fig.add_trace(
            go.Scatter(
                x=bridge_x, y=bridge_y, name="Forecast (SARIMA)",
                mode="lines+markers", line=dict(color=theme.ACCENT, width=2.2, dash="dash"),
                marker=dict(size=6), hovertemplate="%{y:$,.0f}<extra>Forecast</extra>",
            )
        )
    fig.update_layout(height=340, yaxis_tickprefix="$", yaxis_tickformat="~s")
    return fig


def holdout_chart(history: pd.DataFrame, holdout: pd.DataFrame, model: str) -> go.Figure:
    """Train-tail context, one model's hold-out forecast (with interval if available), actuals."""
    tail = history.iloc[-18:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail["date"], y=tail["sales"], name="Observed (train)",
            mode="lines", line=dict(color=theme.SLATE, width=2),
            hovertemplate="%{y:$,.0f}<extra>Observed</extra>",
        )
    )
    lower, upper = f"{model}_lower", f"{model}_upper"
    if lower in holdout.columns:
        _band(fig, holdout["date"], holdout[lower], holdout[upper], theme.PRIMARY, "95% interval")
    fig.add_trace(
        go.Scatter(
            x=holdout["date"], y=holdout[model], name=f"{model} forecast",
            mode="lines+markers", line=dict(color=theme.PRIMARY, width=2.2, dash="dash"),
            marker=dict(size=7), hovertemplate="%{y:$,.0f}<extra>Forecast</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=holdout["date"], y=holdout["actual"], name="Actual (held out)",
            mode="markers", marker=dict(color=theme.GOOD, size=11, symbol="diamond",
                                        line=dict(color="white", width=1)),
            hovertemplate="%{y:$,.0f}<extra>Actual</extra>",
        )
    )
    fig.update_layout(height=360, yaxis_tickprefix="$", yaxis_tickformat="~s")
    return fig


def future_chart(history: pd.DataFrame, future: pd.DataFrame, model: str) -> go.Figure:
    """Full history plus one model's next-quarter forecast."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"], y=history["sales"], name="Observed",
            mode="lines", line=dict(color=theme.PRIMARY, width=1.8),
            hovertemplate="%{y:$,.0f}<extra>Observed</extra>",
        )
    )
    lower, upper = f"{model}_lower", f"{model}_upper"
    if lower in future.columns:
        _band(fig, future["date"], future[lower], future[upper], theme.ACCENT, "95% interval")
    bridge_x = [history["date"].iloc[-1], *future["date"]]
    bridge_y = [history["sales"].iloc[-1], *future[model]]
    fig.add_trace(
        go.Scatter(
            x=bridge_x, y=bridge_y, name=f"{model} forecast (2019 Q1)",
            mode="lines+markers", line=dict(color=theme.ACCENT, width=2.4, dash="dash"),
            marker=dict(size=7), hovertemplate="%{y:$,.0f}<extra>Forecast</extra>",
        )
    )
    fig.update_layout(height=360, yaxis_tickprefix="$", yaxis_tickformat="~s")
    return fig


def metric_comparison(metrics: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    """Grouped bars of MAE/RMSE with MAPE labels, one group per model."""
    colors = {
        model: theme.PRIMARY if model == highlight else theme.SLATE
        for model in metrics["model"]
    }
    fig = go.Figure()
    for metric, shade in (("MAE", 1.0), ("RMSE", 0.55)):
        fig.add_trace(
            go.Bar(
                x=metrics["model"], y=metrics[metric], name=metric,
                marker_color=[_hex_to_rgba(colors[m], shade) for m in metrics["model"]],
                hovertemplate="%{y:$,.0f}<extra>" + metric + "</extra>",
            )
        )
    fig.update_layout(
        height=300, barmode="group", yaxis_tickprefix="$", yaxis_tickformat="~s",
        xaxis_title=None,
    )
    return fig


def category_bar(revenue: pd.Series) -> go.Figure:
    ordered = revenue.sort_values()
    share = ordered / revenue.sum() * 100
    fig = go.Figure(
        go.Bar(
            x=ordered.values, y=ordered.index, orientation="h",
            marker_color=theme.PRIMARY, marker_line_width=0,
            text=[f"  ${v/1000:,.0f}K · {s:.1f}%" for v, s in zip(ordered.values, share)],
            textposition="outside", textfont=dict(color=theme.MUTED, size=12),
            hovertemplate="%{x:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=250, xaxis_tickprefix="$", xaxis_tickformat="~s",
        xaxis_range=[0, ordered.max() * 1.32],
    )
    return fig


def yearly_region_lines(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, region in enumerate(pivot.columns):
        fig.add_trace(
            go.Scatter(
                x=pivot.index, y=pivot[region], name=region, mode="lines+markers",
                line=dict(width=2.2, color=theme.COLORWAY[i % len(theme.COLORWAY)]),
                marker=dict(size=6),
                hovertemplate="%{y:$,.0f}<extra>" + region + "</extra>",
            )
        )
    fig.update_layout(
        height=250, yaxis_tickprefix="$", yaxis_tickformat="~s",
        xaxis=dict(tickmode="array", tickvals=list(pivot.index)),
    )
    return fig


def seasonality_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Month × year heatmap on a single-hue indigo scale."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns], y=_MONTHS,
            colorscale=[[0, "#F5F6FE"], [1, theme.PRIMARY]],
            hovertemplate="%{y} %{x}: %{z:$,.0f}<extra></extra>",
            colorbar=dict(tickprefix="$", tickformat="~s", thickness=12, outlinewidth=0),
        )
    )
    fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
    return fig


def top_subcategories(revenue: pd.Series, top_n: int = 10) -> go.Figure:
    ordered = revenue.sort_values(ascending=False).head(top_n).sort_values()
    fig = go.Figure(
        go.Bar(
            x=ordered.values, y=ordered.index, orientation="h",
            marker_color=theme.ACCENT, marker_line_width=0,
            hovertemplate="%{x:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(height=330, xaxis_tickprefix="$", xaxis_tickformat="~s")
    return fig


def segment_comparison(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    segments: list[str],
    history_months: int = 12,
) -> go.Figure:
    """Recent history (solid) bridging into each segment's forecast (dashed)."""
    fig = go.Figure()
    for i, segment in enumerate(segments):
        color = theme.COLORWAY[i % len(theme.COLORWAY)]
        past = history[history["segment"] == segment].iloc[-history_months:]
        fc = forecasts[forecasts["segment"] == segment]
        fig.add_trace(
            go.Scatter(
                x=past["date"], y=past["sales"], name=segment, legendgroup=segment,
                mode="lines", line=dict(color=color, width=2.2),
                hovertemplate="%{y:$,.0f}<extra>" + segment + "</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[past["date"].iloc[-1], *fc["date"]],
                y=[past["sales"].iloc[-1], *fc["forecast"]],
                name=segment, legendgroup=segment, showlegend=False,
                mode="lines+markers", line=dict(color=color, width=2.2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="%{y:$,.0f}<extra>" + segment + " · forecast</extra>",
            )
        )
    fig.update_layout(height=400, yaxis_tickprefix="$", yaxis_tickformat="~s")
    return fig


def segment_detail(history: pd.DataFrame, forecast: pd.DataFrame, segment: str) -> go.Figure:
    """One segment's full history with its forecast and confidence interval."""
    past = history[history["segment"] == segment]
    fc = forecast[forecast["segment"] == segment]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=past["date"], y=past["sales"], name="Observed", mode="lines",
            line=dict(color=theme.PRIMARY, width=1.8),
            hovertemplate="%{y:$,.0f}<extra>Observed</extra>",
        )
    )
    _band(fig, fc["date"], fc["lower"], fc["upper"], theme.ACCENT, "95% interval")
    fig.add_trace(
        go.Scatter(
            x=[past["date"].iloc[-1], *fc["date"]],
            y=[past["sales"].iloc[-1], *fc["forecast"]],
            name="Forecast", mode="lines+markers",
            line=dict(color=theme.ACCENT, width=2.2, dash="dash"), marker=dict(size=6),
            hovertemplate="%{y:$,.0f}<extra>Forecast</extra>",
        )
    )
    fig.update_layout(height=320, yaxis_tickprefix="$", yaxis_tickformat="~s")
    return fig


def anomaly_chart(weekly: pd.DataFrame, method: str) -> go.Figure:
    """Weekly sales with one method's anomalies — or the two-method agreement view."""
    fig = go.Figure()

    if method == "Z-Score":
        band_ok = weekly.dropna(subset=["rolling_mean", "rolling_std"])
        upper = band_ok["rolling_mean"] + 2 * band_ok["rolling_std"]
        lower = band_ok["rolling_mean"] - 2 * band_ok["rolling_std"]
        _band(fig, band_ok["date"], lower, upper, theme.SLATE, "±2σ band")
        fig.add_trace(
            go.Scatter(
                x=band_ok["date"], y=band_ok["rolling_mean"], name="13-week mean",
                mode="lines", line=dict(color=theme.SLATE, width=1.4, dash="dot"),
                hovertemplate="%{y:$,.0f}<extra>Rolling mean</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=weekly["date"], y=weekly["Sales"], name="Weekly sales", mode="lines",
            line=dict(color=theme.PRIMARY, width=1.6),
            hovertemplate="%{y:$,.0f}<extra></extra>",
        )
    )

    if method == "Agreement":
        groups = [
            ("Both methods", weekly["z_flag"] & weekly["if_flag"], theme.BAD, "diamond", 12),
            ("Z-Score only", weekly["z_flag"] & ~weekly["if_flag"], theme.WARN, "triangle-up", 10),
            ("Isolation Forest only", ~weekly["z_flag"] & weekly["if_flag"], theme.ACCENT, "square", 9),
        ]
    else:
        flag_col = "z_flag" if method == "Z-Score" else "if_flag"
        groups = [(f"{method} anomaly", weekly[flag_col], theme.BAD, "diamond", 10)]

    for name, mask, color, symbol, size in groups:
        points = weekly[mask]
        fig.add_trace(
            go.Scatter(
                x=points["date"], y=points["Sales"], name=name, mode="markers",
                marker=dict(color=color, size=size, symbol=symbol,
                            line=dict(color="white", width=1)),
                hovertemplate="%{y:$,.0f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(height=380, yaxis_tickprefix="$", yaxis_tickformat="~s",
                      hovermode="closest")
    return fig


# Semantic cluster colors: growth reads green, volatility reads amber, the backbone gets
# the primary indigo, and the low-volume tier the secondary cyan.
_CLUSTER_COLORS = {
    "High Volume, Stable Demand": theme.PRIMARY,
    "Low Volume, Stable Demand": theme.ACCENT,
    "High-Value, Growing Demand": theme.GOOD,
    "Volatile, High-Growth Outlier": theme.WARN,
}


def pca_scatter(members: pd.DataFrame) -> go.Figure:
    """PCA projection of sub-categories, colored by cluster name, labelled by product."""
    fig = go.Figure()
    for i, (name, group) in enumerate(members.groupby("ClusterName", sort=False)):
        fig.add_trace(
            go.Scatter(
                x=group["PC1"], y=group["PC2"], name=name,
                mode="markers+text", text=group["Sub-Category"],
                textposition="top center", textfont=dict(size=10.5, color=theme.MUTED),
                marker=dict(color=_CLUSTER_COLORS.get(name, theme.COLORWAY[i % len(theme.COLORWAY)]),
                            size=13,
                            line=dict(color="white", width=1.5)),
                customdata=group[["TotalSales", "MeanYoYGrowth", "MonthlyVolatilityCV",
                                  "AvgOrderValue"]],
                hovertemplate=(
                    "<b>%{text}</b><br>Total sales: %{customdata[0]:$,.0f}"
                    "<br>YoY growth: %{customdata[1]:.1f}%"
                    "<br>Volatility (CV): %{customdata[2]:.2f}"
                    "<br>Avg order value: %{customdata[3]:$,.0f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=460, hovermode="closest",
        xaxis_title="Principal component 1", yaxis_title="Principal component 2",
    )
    return fig


def elbow_and_silhouette(search: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    elbow = go.Figure(
        go.Scatter(
            x=search["k"], y=search["inertia"], mode="lines+markers",
            line=dict(color=theme.PRIMARY, width=2.2), marker=dict(size=7),
            hovertemplate="k=%{x}: %{y:.1f}<extra></extra>",
        )
    )
    elbow.update_layout(height=260, title="Elbow method (inertia)",
                        xaxis=dict(dtick=1), yaxis_title=None)

    silhouette = go.Figure(
        go.Scatter(
            x=search["k"], y=search["silhouette_score"], mode="lines+markers",
            line=dict(color=theme.ACCENT, width=2.2), marker=dict(size=7),
            hovertemplate="k=%{x}: %{y:.3f}<extra></extra>",
        )
    )
    silhouette.update_layout(height=260, title="Silhouette score",
                             xaxis=dict(dtick=1), yaxis_title=None)
    return elbow, silhouette
