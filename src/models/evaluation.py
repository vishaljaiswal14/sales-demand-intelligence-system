"""Shared forecasting-evaluation utilities: chronological splitting, error metrics, and
a rolling-origin backtest.

Time-series evaluation must never shuffle rows — doing so lets a model peek at the future.
Everything here therefore respects temporal order: the split holds out the most recent
months, and the backtest only ever trains on data that precedes the point being forecast.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# A forecaster takes a training series and a horizon, and returns ``horizon`` point
# forecasts. Keeping this signature uniform lets the backtest treat every model the same.
Forecaster = Callable[[pd.Series, int], np.ndarray]

DEFAULT_HORIZON = 3


def chronological_split(
    series: pd.Series, horizon: int = DEFAULT_HORIZON
) -> tuple[pd.Series, pd.Series]:
    """Hold out the final ``horizon`` observations as the test set.

    The most recent months are the test set (not a random sample) because in production we
    always forecast forward from the latest available data.
    """
    if horizon >= len(series):
        raise ValueError("horizon must be smaller than the series length.")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def forecast_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Return MAE, RMSE, and MAPE — the three metrics required for model comparison.

    MAE is in dollars and robust to outliers; RMSE (also dollars) penalises large misses
    more heavily; MAPE expresses error as a percentage of actual sales, which is how a
    business reads accuracy. Reporting all three avoids over-relying on any single view.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return pd.Series(
        {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }
    )


def rolling_origin_backtest(
    series: pd.Series,
    forecasters: dict[str, Forecaster],
    horizon: int = DEFAULT_HORIZON,
    n_origins: int = 6,
) -> pd.DataFrame:
    """Aggregate metrics from an expanding-window (rolling-origin) backtest.

    A single last-``horizon`` hold-out gives only a handful of scored points, so its metrics
    are noisy. Here we slide the forecast origin back over ``n_origins`` positions, refit each
    model on only the data available at that origin, and pool the resulting errors. Because
    every origin trains strictly on its own past, there is no leakage, and pooling many
    forecast points yields a far more stable ranking than any single split.
    """
    origins = range(len(series) - horizon, len(series) - horizon - n_origins, -1)
    pooled: dict[str, list[tuple[float, float]]] = {name: [] for name in forecasters}

    for origin in origins:
        if origin <= horizon:  # need enough history to fit seasonal models
            break
        train = series.iloc[:origin]
        steps = min(horizon, len(series) - origin)
        actual = series.iloc[origin : origin + steps].to_numpy(dtype=float)
        for name, forecaster in forecasters.items():
            prediction = np.asarray(forecaster(train, steps), dtype=float)
            pooled[name].extend(zip(actual, prediction))

    records = {}
    for name, pairs in pooled.items():
        actual = np.array([a for a, _ in pairs])
        predicted = np.array([p for _, p in pairs])
        records[name] = forecast_metrics(actual, predicted)

    summary = pd.DataFrame(records).T
    summary.insert(0, "forecast_points", [len(pooled[name]) for name in summary.index])
    return summary.sort_values("RMSE")
