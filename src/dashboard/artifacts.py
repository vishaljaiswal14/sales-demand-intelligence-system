"""Build and load the precomputed artifacts served by the Streamlit dashboard.

The dashboard never recomputes analytics. Every model result it displays is produced
here — by the same ``src/`` modules the analysis notebook uses — and serialized once to
``data/processed/dashboard/``. Streamlit is left with pure presentation: it reads these
files through the loaders at the bottom of this module and renders them.

Rebuild after any upstream data or model change with:

    python3 -m src.dashboard.artifacts

Heavy model libraries (Prophet, XGBoost, statsmodels) are imported inside the build
functions, not at module level, so importing this module for its *loaders* — which is all
the dashboard does — stays fast and dependency-light.
"""

from __future__ import annotations

import json

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

DASHBOARD_DATA_DIR = PROCESSED_DATA_DIR / "dashboard"

FORECAST_HORIZON = 3


# --------------------------------------------------------------------------- build steps


def _load_line_items() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_DIR / "sales_with_features.csv",
        parse_dates=["Order Date", "Ship Date"],
    )


def _build_kpis(sales: pd.DataFrame) -> None:
    """Headline figures for the Executive Overview page."""
    from src.models import anomaly_detection as ad
    from src.models import time_series as ts

    yearly = sales.groupby(sales["Order Date"].dt.year)["Sales"].sum()
    category_revenue = sales.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    region_revenue = sales.groupby("Region")["Sales"].sum().sort_values(ascending=False)

    total_sales = float(sales["Sales"].sum())
    total_orders = int(sales["Order ID"].nunique())

    # Anomaly counts reuse the exact Task 5 pipeline so the KPI matches the notebook.
    weekly = ts.load_weekly_sales_frame()
    zscore = ad.rolling_zscore(weekly["Sales"])
    z_flags = ad.flag_zscore_anomalies(zscore["zscore"])
    if_features = ad.build_isolation_forest_features(weekly["Sales"])
    _, if_flags, _ = ad.fit_isolation_forest(if_features)
    flagged_union = set(z_flags[z_flags].index) | set(if_flags[if_flags].index)
    flagged_both = set(z_flags[z_flags].index) & set(if_flags[if_flags].index)

    kpis = {
        "total_sales": total_sales,
        "total_orders": total_orders,
        "total_line_items": int(len(sales)),
        "avg_order_value": total_sales / total_orders,
        "best_category": category_revenue.index[0],
        "best_category_sales": float(category_revenue.iloc[0]),
        "best_category_share_pct": float(category_revenue.iloc[0] / total_sales * 100),
        "best_region": region_revenue.index[0],
        "best_region_sales": float(region_revenue.iloc[0]),
        "first_year": int(yearly.index.min()),
        "last_year": int(yearly.index.max()),
        "growth_pct_last_vs_first_year": float(
            (yearly.iloc[-1] / yearly.iloc[0] - 1) * 100
        ),
        "recommended_model": "SARIMA",
        "anomaly_weeks_total": len(flagged_union),
        "anomaly_weeks_zscore": int(z_flags.sum()),
        "anomaly_weeks_iforest": int(if_flags.sum()),
        "anomaly_weeks_both": len(flagged_both),
        "weeks_analyzed": int(len(weekly)),
        "date_min": sales["Order Date"].min().strftime("%Y-%m-%d"),
        "date_max": sales["Order Date"].max().strftime("%Y-%m-%d"),
    }
    (DASHBOARD_DATA_DIR / "kpis.json").write_text(json.dumps(kpis, indent=2))


def _build_forecasting_artifacts() -> None:
    """Task 3 outputs: hold-out and future forecasts plus metrics for all three models."""
    from src.models import evaluation as ev
    from src.models import prophet_forecaster as pf
    from src.models import sarima
    from src.models import time_series as ts
    from src.models import xgboost_forecaster as xgb

    series = ts.load_monthly_series()
    train, test = ev.chronological_split(series, horizon=FORECAST_HORIZON)

    # Hold-out fits (train only) — mirrors the notebook's Task 3 evaluation exactly.
    sarima_train = sarima.fit_sarima(train)
    sarima_holdout = sarima.sarima_forecast(sarima_train, FORECAST_HORIZON)

    prophet_train = pf.fit_prophet(train)
    prophet_holdout_full = pf.prophet_forecast(prophet_train, FORECAST_HORIZON)
    prophet_holdout = prophet_holdout_full.set_index("ds").iloc[-FORECAST_HORIZON:]

    X_train, y_train = xgb.make_supervised(train)
    xgb_train = xgb.fit_xgboost(X_train, y_train)
    xgb_holdout = xgb.recursive_forecast(xgb_train, train, FORECAST_HORIZON)

    holdout = pd.DataFrame(
        {
            "actual": test,
            "SARIMA": sarima_holdout["forecast"],
            "SARIMA_lower": sarima_holdout["lower"],
            "SARIMA_upper": sarima_holdout["upper"],
            "Prophet": prophet_holdout["yhat"],
            "Prophet_lower": prophet_holdout["yhat_lower"],
            "Prophet_upper": prophet_holdout["yhat_upper"],
            "XGBoost": xgb_holdout,
        }
    )
    holdout.rename_axis("date").to_csv(DASHBOARD_DATA_DIR / "holdout_forecasts.csv")

    metrics = pd.DataFrame(
        {
            name: ev.forecast_metrics(test, holdout[name])
            for name in ("SARIMA", "Prophet", "XGBoost")
        }
    ).T.rename_axis("model")
    metrics.to_csv(DASHBOARD_DATA_DIR / "holdout_metrics.csv")

    # Future (2019 Q1) forecasts from full-history refits.
    sarima_full = sarima.fit_sarima(series)
    sarima_future = sarima.sarima_forecast(sarima_full, FORECAST_HORIZON)

    prophet_full = pf.fit_prophet(series)
    prophet_future = (
        pf.prophet_forecast(prophet_full, FORECAST_HORIZON)
        .set_index("ds")
        .iloc[-FORECAST_HORIZON:]
    )

    xgb_full = xgb.fit_xgboost(*xgb.make_supervised(series))
    xgb_future = xgb.recursive_forecast(xgb_full, series, FORECAST_HORIZON)

    future = pd.DataFrame(
        {
            "SARIMA": sarima_future["forecast"],
            "SARIMA_lower": sarima_future["lower"],
            "SARIMA_upper": sarima_future["upper"],
            "Prophet": prophet_future["yhat"],
            "Prophet_lower": prophet_future["yhat_lower"],
            "Prophet_upper": prophet_future["yhat_upper"],
            "XGBoost": xgb_future,
        }
    )
    future.rename_axis("date").to_csv(DASHBOARD_DATA_DIR / "future_forecasts.csv")

    # Rolling-origin backtest — the evidence behind the production recommendation.
    backtest = ev.rolling_origin_backtest(
        series,
        {
            "SARIMA": sarima.point_forecast,
            "Prophet": pf.point_forecast,
            "XGBoost": xgb.point_forecast,
        },
        horizon=FORECAST_HORIZON,
        n_origins=6,
    )
    backtest.rename_axis("model").to_csv(DASHBOARD_DATA_DIR / "backtest_metrics.csv")

    xgb.feature_importance(xgb_train).rename_axis("feature").rename("importance").to_csv(
        DASHBOARD_DATA_DIR / "xgb_feature_importance.csv"
    )

    meta = {
        "sarima_order": list(sarima.SELECTED_ORDER),
        "sarima_seasonal_order": list(sarima.SELECTED_SEASONAL_ORDER),
        "sarima_ljung_box_pvalue": sarima.ljung_box_pvalue(sarima_train),
        "train_months": len(train),
        "test_months": len(test),
    }
    (DASHBOARD_DATA_DIR / "forecasting_meta.json").write_text(json.dumps(meta, indent=2))


def _build_segment_artifacts(sales: pd.DataFrame) -> None:
    """Task 4 outputs: per-segment history, forecasts, and the growth summary."""
    from src.data.aggregate import segment_monthly_series
    from src.models import segment_forecasting as seg
    from src.models import time_series as ts

    monthly_index = ts.load_monthly_series().index
    segment_series = {
        name: segment_monthly_series(sales, column, value, full_index=monthly_index)
        for name, (column, value) in seg.SEGMENT_DEFINITIONS.items()
    }
    results = seg.forecast_all_segments(segment_series, horizon=FORECAST_HORIZON)

    history = pd.concat(
        [s.rename("sales").rename_axis("date").reset_index().assign(segment=name)
         for name, s in segment_series.items()]
    )
    history.to_csv(DASHBOARD_DATA_DIR / "segment_series.csv", index=False)

    forecasts = pd.concat(
        [r["forecast"].rename_axis("date").reset_index().assign(segment=name)
         for name, r in results.items()]
    )
    forecasts.to_csv(DASHBOARD_DATA_DIR / "segment_forecasts.csv", index=False)

    growth = seg.summarise_segment_growth(segment_series, results)
    growth.to_csv(DASHBOARD_DATA_DIR / "segment_growth.csv")


def _build_anomaly_artifacts() -> None:
    """Task 5 outputs: weekly series with both methods' flags plus the evidence table."""
    from src.models import anomaly_detection as ad
    from src.models import time_series as ts

    weekly = ts.load_weekly_sales_frame()
    weekly_sales = weekly["Sales"]

    zscore = ad.rolling_zscore(weekly_sales)
    z_flags = ad.flag_zscore_anomalies(zscore["zscore"])

    if_features = ad.build_isolation_forest_features(weekly_sales)
    _, if_flags, if_scores = ad.fit_isolation_forest(if_features)

    combined = weekly.join(zscore)
    combined["z_flag"] = z_flags.reindex(combined.index, fill_value=False)
    combined["if_flag"] = if_flags.reindex(combined.index, fill_value=False)
    combined["if_score"] = if_scores.reindex(combined.index)
    combined.rename_axis("date").to_csv(DASHBOARD_DATA_DIR / "weekly_anomalies.csv")

    evidence = ad.anomaly_evidence_table(
        weekly_sales, weekly["Order Lines"], z_flags, if_flags
    )
    evidence.rename_axis("week").to_csv(DASHBOARD_DATA_DIR / "anomaly_evidence.csv")


def assign_cluster_names(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster ids to the business names derived in the Task 6 analysis.

    Names are assigned by each cluster's standout statistic rather than by hardcoded ids,
    so a re-run that permutes K-Means' label numbering still names every cluster correctly:
    highest average order value → the big-ticket cluster; highest growth of the rest → the
    volatile outlier; highest volume of the rest → the backbone; the remainder → low-volume.
    """
    remaining = profile.copy()
    names: dict[int, str] = {}

    high_value = remaining["AvgOrderValue"].idxmax()
    names[high_value] = "High-Value, Growing Demand"
    remaining = remaining.drop(index=high_value)

    outlier = remaining["MeanYoYGrowth"].idxmax()
    names[outlier] = "Volatile, High-Growth Outlier"
    remaining = remaining.drop(index=outlier)

    backbone = remaining["TotalSales"].idxmax()
    names[backbone] = "High Volume, Stable Demand"
    remaining = remaining.drop(index=backbone)

    for leftover in remaining.index:
        names[leftover] = "Low Volume, Stable Demand"
    return names


def _build_cluster_artifacts(sales: pd.DataFrame) -> None:
    """Task 6 outputs: cluster assignments, profiles, PCA coordinates, and the k search."""
    from src.models import clustering as cl
    from src.models import time_series as ts

    monthly_index = ts.load_monthly_series().index
    features = cl.build_subcategory_features(sales, monthly_index)
    _, X_scaled = cl.prepare_clustering_features(features)

    search = cl.cluster_search(X_scaled)
    search.to_csv(DASHBOARD_DATA_DIR / "cluster_search.csv")

    model = cl.fit_kmeans(X_scaled, n_clusters=4)
    profile = cl.cluster_profile(features, model.labels_)
    names = assign_cluster_names(profile)

    profile["ClusterName"] = [names[c] for c in profile.index]
    profile.to_csv(DASHBOARD_DATA_DIR / "cluster_profile.csv")

    coords, pca = cl.pca_projection(X_scaled, features.index)
    members = features.join(coords)
    members["Cluster"] = model.labels_
    members["ClusterName"] = [names[c] for c in model.labels_]
    members.to_csv(DASHBOARD_DATA_DIR / "cluster_members.csv")

    meta = {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "n_clusters": 4,
    }
    (DASHBOARD_DATA_DIR / "clustering_meta.json").write_text(json.dumps(meta, indent=2))


def build_all() -> None:
    """Produce every dashboard artifact from scratch. Idempotent."""
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sales = _load_line_items()

    print("Building KPI summary...")
    _build_kpis(sales)
    print("Building forecasting artifacts (SARIMA, Prophet, XGBoost + backtest)...")
    _build_forecasting_artifacts()
    print("Building segment forecasting artifacts...")
    _build_segment_artifacts(sales)
    print("Building anomaly detection artifacts...")
    _build_anomaly_artifacts()
    print("Building clustering artifacts...")
    _build_cluster_artifacts(sales)
    print(f"Done. Artifacts written to {DASHBOARD_DATA_DIR}")


# ------------------------------------------------------------------------------ loaders
# The dashboard reads artifacts exclusively through these. They import nothing heavy.


def load_kpis() -> dict:
    return json.loads((DASHBOARD_DATA_DIR / "kpis.json").read_text())


def load_json(name: str) -> dict:
    return json.loads((DASHBOARD_DATA_DIR / f"{name}.json").read_text())


def load_table(name: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(DASHBOARD_DATA_DIR / f"{name}.csv", **read_csv_kwargs)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    build_all()
