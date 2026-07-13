"""XGBoost forecasting: the monthly series reframed as a supervised regression problem.

A tree ensemble cannot consume a raw time series, so we turn each month into a feature row
built only from information available *before* that month — lagged sales, a short rolling
mean, and calendar position. Two structural caveats drive the design and are revisited in
the notebook:

* With ~45 monthly observations this is a very small training set, so the model is kept
  deliberately shallow and heavily regularised to avoid memorising the history.
* Trees cannot extrapolate: their predictions are bounded by the target range seen in
  training, so on an upward-trending series XGBoost will tend to under-forecast new highs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.time_features import SEASON_BY_MONTH, SEASON_ORDER

LAGS = (1, 2, 3)
ROLLING_WINDOW = 3

# Ordinal season code (Winter=0 ... Autumn=3). Ordinal rather than one-hot because trees
# split on thresholds and one-hot would only add sparse, low-signal columns on tiny data.
_SEASON_CODE = {season: code for code, season in enumerate(SEASON_ORDER)}

FEATURE_NAMES = [f"lag_{lag}" for lag in LAGS] + [
    f"rolling_mean_{ROLLING_WINDOW}",
    "month",
    "quarter",
    "season",
]

# Hyperparameters tuned for the small-sample regime, not by brute-force search (which would
# overfit a 42-row validation set). Each choice trades capacity for generalisation:
XGB_PARAMS = dict(
    n_estimators=200,        # enough trees to converge at the low learning rate below
    learning_rate=0.05,      # small steps → stable, gradual fit on limited data
    max_depth=3,             # shallow trees prevent memorising individual months
    subsample=0.8,           # row subsampling injects regularisation / reduces variance
    colsample_bytree=0.8,    # feature subsampling decorrelates the trees
    min_child_weight=1,
    reg_lambda=1.0,          # L2 shrinkage on leaf weights
    objective="reg:squarederror",  # L2 loss, consistent with RMSE-based evaluation
    random_state=42,         # reproducible fits
)


def _feature_row(history: pd.Series, target_date: pd.Timestamp) -> dict:
    """Build one feature row for ``target_date`` from sales strictly before it."""
    values = history.to_numpy()
    row = {f"lag_{lag}": values[-lag] for lag in LAGS}
    row[f"rolling_mean_{ROLLING_WINDOW}"] = values[-ROLLING_WINDOW:].mean()
    row["month"] = target_date.month
    row["quarter"] = (target_date.month - 1) // 3 + 1
    row["season"] = _SEASON_CODE[SEASON_BY_MONTH[target_date.month]]
    return row


def make_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the series into an (X, y) table, one row per month that has full lag history.

    The first ``max(LAGS)`` months are dropped because their lag features do not yet exist —
    fabricating them (e.g. with zeros) would feed the model impossible history.
    """
    start = max(LAGS)
    rows, targets, index = [], [], []
    for position in range(start, len(series)):
        target_date = series.index[position]
        rows.append(_feature_row(series.iloc[:position], target_date))
        targets.append(series.iloc[position])
        index.append(target_date)

    X = pd.DataFrame(rows, index=index)[FEATURE_NAMES]
    y = pd.Series(targets, index=index, name="Sales")
    return X, y


def fit_xgboost(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> XGBRegressor:
    """Fit the regularised XGBoost regressor on the supervised table."""
    model = XGBRegressor(**(params or XGB_PARAMS))
    model.fit(X, y)
    return model


def recursive_forecast(
    model: XGBRegressor, history: pd.Series, horizon: int
) -> pd.Series:
    """Forecast ``horizon`` months ahead by feeding each prediction back in as a new lag.

    Multi-step forecasting has no future actuals to lag on, so we predict one month, append
    that prediction to the history, and use it to build the next month's features. This keeps
    the process strictly causal (no leakage) at the cost of compounding any early error.
    """
    working_history = history.copy()
    future_dates = pd.date_range(
        history.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    predictions = []
    for target_date in future_dates:
        features = pd.DataFrame([_feature_row(working_history, target_date)])[FEATURE_NAMES]
        prediction = float(model.predict(features)[0])
        predictions.append(prediction)
        working_history = pd.concat(
            [working_history, pd.Series([prediction], index=[target_date])]
        )
    return pd.Series(predictions, index=future_dates, name="forecast")


def feature_importance(model: XGBRegressor) -> pd.Series:
    """Gain-based feature importances, most influential first."""
    return pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(
        ascending=False
    )


def point_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Refit on ``series`` and recursively forecast — the uniform backtest hook."""
    X, y = make_supervised(series)
    model = fit_xgboost(X, y)
    return recursive_forecast(model, series, horizon).to_numpy()
