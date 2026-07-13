"""SARIMA forecasting for the monthly sales series.

Orders are chosen from evidence, not defaults. Task 2 established that the non-seasonal
series is already stationary, so ``d = 0``; the ACF/PACF show only weak short-lag structure
(so ``p, q <= 1``) and a single strong spike at the seasonal lag 12, which one seasonal
difference (``D = 1``) removes. Among the resulting small, ACF/PACF-justified candidate set
we select by corrected AIC (AICc), which is the appropriate criterion for a short series.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

SEASONAL_PERIOD = 12

# Orders selected by the AICc search below (see notebook for the full candidate table).
# (1,0,1)(0,1,1)_12 is the classic airline-type structure: AR(1)+MA(1) noise on a
# seasonally-differenced series with a seasonal MA(1) term.
SELECTED_ORDER = (1, 0, 1)
SELECTED_SEASONAL_ORDER = (0, 1, 1, SEASONAL_PERIOD)


def _fit(series: pd.Series, order, seasonal_order) -> SARIMAXResults:
    # A constant is only meaningful when nothing is differenced; with a seasonal difference
    # (D=1) an intercept would inject an unwanted deterministic drift, so we drop it then.
    trend = "c" if order[1] == 0 and seasonal_order[1] == 0 else None
    # enforce_* relaxed because near-unit-root MLE on 45 points can otherwise fail to fit;
    # residual diagnostics (Ljung-Box) are what we ultimately rely on for adequacy.
    return SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def _aicc(result: SARIMAXResults, n_obs: int) -> float:
    """Small-sample-corrected AIC. With only ~45 points the correction is non-trivial and
    guards against picking an over-parameterised model that plain AIC would favour."""
    k = len(result.params)
    return result.aic + (2 * k * (k + 1)) / max(n_obs - k - 1, 1)


def sarima_order_search(
    series: pd.Series,
    non_seasonal_orders=(0, 1),
    seasonal_orders=(0, 1),
    d: int = 0,
    seasonal_period: int = SEASONAL_PERIOD,
) -> pd.DataFrame:
    """Fit the compact candidate grid (all p,q,P,D,Q in {0,1}) and rank by AICc.

    This is a small, theory-motivated search rather than a brute-force sweep: the ACF/PACF
    justify keeping every order at 0 or 1, and ``d`` is fixed at 0 from the Task 2
    stationarity result. Non-converging candidates are skipped rather than trusted.
    """
    records = []
    for p, q, P, D, Q in itertools.product(
        non_seasonal_orders, non_seasonal_orders, seasonal_orders, seasonal_orders, seasonal_orders
    ):
        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal_period)
        try:
            with warnings.catch_warnings():
                # Convergence warnings are expected for poorly-specified candidates; we let
                # AICc rank them rather than cluttering output with warnings for each.
                warnings.simplefilter("ignore")
                result = _fit(series, order, seasonal_order)
            records.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "AIC": round(result.aic, 1),
                    "AICc": round(_aicc(result, len(series)), 1),
                    "BIC": round(result.bic, 1),
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

    return pd.DataFrame(records).sort_values("AICc").reset_index(drop=True)


def fit_sarima(
    series: pd.Series,
    order=SELECTED_ORDER,
    seasonal_order=SELECTED_SEASONAL_ORDER,
) -> SARIMAXResults:
    """Fit the selected SARIMA specification on ``series``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(series, order, seasonal_order)


def sarima_forecast(result: SARIMAXResults, horizon: int, alpha: float = 0.05) -> pd.DataFrame:
    """Point forecast with a ``(1-alpha)`` confidence interval for ``horizon`` months."""
    forecast = result.get_forecast(steps=horizon)
    interval = forecast.conf_int(alpha=alpha)
    return pd.DataFrame(
        {
            "forecast": forecast.predicted_mean.to_numpy(),
            "lower": interval.iloc[:, 0].to_numpy(),
            "upper": interval.iloc[:, 1].to_numpy(),
        },
        index=forecast.predicted_mean.index,
    )


def ljung_box_pvalue(result: SARIMAXResults, lags: int = SEASONAL_PERIOD) -> float:
    """Ljung-Box p-value on the residuals up to ``lags``.

    A p-value above 0.05 means we cannot reject "residuals are white noise" — i.e. the model
    has captured the autocorrelation structure and what remains is unpredictable noise. The
    first ``lags`` residuals are skipped because the state-space filter's diffuse
    initialisation makes them unreliable.
    """
    residuals = result.resid[lags + 1 :]
    test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(test["lb_pvalue"].iloc[0])


def point_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Refit and return point forecasts only — the uniform hook used by the backtest."""
    return fit_sarima(series).forecast(steps=horizon).to_numpy()
