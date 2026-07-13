"""Time-series diagnostics for the monthly sales series: decomposition, seasonal
strength, the Augmented Dickey-Fuller stationarity test, and differencing checks.

These are analysis primitives, not forecasting models — they characterise the series so
that later phases can choose an appropriate model. Every function returns structured
values (Series/dict) rather than printing, so the notebook can both display and interpret
them.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller

from src.utils.paths import PROCESSED_DATA_DIR

# Monthly data carries an annual cycle, so one seasonal period spans 12 observations.
SEASONAL_PERIOD = 12

# Conventional 5% threshold for rejecting the ADF null hypothesis of a unit root.
STATIONARITY_ALPHA = 0.05


def load_monthly_series(
    path: str | Path = PROCESSED_DATA_DIR / "monthly_sales.csv",
) -> pd.Series:
    """Load the Phase 2 monthly sales totals as a frequency-aware Series.

    Setting an explicit ``MS`` (month-start) frequency is what lets statsmodels reason
    about seasonality and prevents silent misalignment if a month were ever missing.
    """
    frame = pd.read_csv(path, parse_dates=["Order Date"])
    series = frame.set_index("Order Date")["Sales"].asfreq("MS")
    if series.isna().any():
        raise ValueError("Monthly series has gaps after reindexing to month-start.")
    return series.rename("Monthly Sales")


def load_weekly_sales_frame(
    path: str | Path = PROCESSED_DATA_DIR / "weekly_sales.csv",
) -> pd.DataFrame:
    """Load the Phase 2 weekly Sales and Order Lines together, frequency-aligned to ``W``.

    Order Lines (the order-line count per week) travels alongside Sales because anomaly
    detection needs it to tell a broad-based demand swing from one driven by a handful of
    large orders — a single dollar total cannot make that distinction on its own.
    """
    frame = pd.read_csv(path, parse_dates=["Order Date"])
    indexed = frame.set_index("Order Date")[["Sales", "Order Lines"]].asfreq("W")
    if indexed.isna().any().any():
        raise ValueError("Weekly series has gaps after reindexing to weekly frequency.")
    return indexed


def decompose_monthly_sales(
    series: pd.Series,
    model: str = "additive",
    period: int = SEASONAL_PERIOD,
) -> DecomposeResult:
    """Classical seasonal decomposition into observed/trend/seasonal/residual.

    An *additive* model is the appropriate default here: the seasonal swing stays roughly
    constant in absolute dollar terms across the four years (its year-on-year coefficient
    of variation actually falls as the level rises), so a multiplicative model — which
    assumes the seasonal amplitude scales with the level — is not supported by the data.
    ``period=12`` encodes the expected annual cycle in monthly data.
    """
    return seasonal_decompose(series, model=model, period=period)


def component_strength(result: DecomposeResult) -> pd.Series:
    """Quantify trend and seasonal strength (Hyndman-Athanasopoulos F measures).

    Each strength is ``max(0, 1 - Var(residual) / Var(component + residual))`` on the
    overlap where trend and residual are defined. Values near 1 mean the component
    explains most of the variation left after removing the others; near 0 means it is
    negligible. This turns "seasonality looks strong" into a defensible number.
    """
    components = pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    ).dropna()

    resid_var = components["residual"].var()

    def strength(component: str) -> float:
        combined_var = (components[component] + components["residual"]).var()
        if combined_var == 0:
            return 0.0
        return max(0.0, 1 - resid_var / combined_var)

    return pd.Series(
        {
            "trend_strength": round(strength("trend"), 3),
            "seasonal_strength": round(strength("seasonal"), 3),
        }
    )


def acf_snapshot(series: pd.Series, lags: tuple[int, ...] = (1, SEASONAL_PERIOD)) -> pd.Series:
    """Autocorrelation at specific lags, labelled for direct display.

    Used to check whether a series shows the same short-lag / seasonal-lag signature as the
    aggregate series — e.g. confirming a business segment's autocorrelation profile matches
    before reusing a SARIMA order that was only formally selected on the aggregate series.
    """
    values = acf(series, nlags=max(lags))
    return pd.Series({f"ACF lag {lag}": round(values[lag], 2) for lag in lags})


def seasonal_factors(result: DecomposeResult) -> pd.Series:
    """Average seasonal effect (in dollars) for each calendar month.

    The additive seasonal component repeats every year, so one value per month fully
    describes it. Positive means the month runs above the deseasonalised level, negative
    below — the concrete evidence behind any claim about which months spike.
    """
    seasonal = result.seasonal
    monthly = seasonal.groupby(seasonal.index.month).first()
    monthly.index = [calendar.month_abbr[m] for m in monthly.index]
    return monthly.round(0).rename("Seasonal Factor")


def adf_test(series: pd.Series, regression: str = "c") -> pd.Series:
    """Run the Augmented Dickey-Fuller test and return an interpretable summary.

    ``regression='c'`` tests for stationarity around a constant mean (the standard
    choice); pass ``'ct'`` to allow a deterministic trend. Lag length is selected by AIC
    so the test adapts to the data rather than using an arbitrary fixed lag.
    """
    stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(
        series.dropna(), regression=regression, autolag="AIC"
    )
    summary = {
        "adf_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "used_lag": int(used_lag),
        "n_observations": int(n_obs),
        "is_stationary": bool(p_value < STATIONARITY_ALPHA),
    }
    for label, value in critical_values.items():
        summary[f"critical_{label}"] = round(value, 3)
    return pd.Series(summary)


def difference_series(series: pd.Series, order: int = 1) -> pd.Series:
    """Return the ``order``-th successive difference, with leading NaNs dropped.

    First-order differencing (``order=1``) models period-over-period *change* rather than
    level, which removes a linear trend; higher orders difference repeatedly.
    """
    if order < 1:
        raise ValueError("order must be a positive integer.")
    differenced = series
    for _ in range(order):
        differenced = differenced.diff()
    return differenced.dropna()


def over_differencing_diagnostics(
    level: pd.Series, differenced: pd.Series
) -> pd.Series:
    """Evidence for whether differencing helped or over-differenced the series.

    Two well-known tells of over-differencing: the differenced series has a *larger*
    variance than the level series, and its lag-1 autocorrelation is pushed strongly
    negative (toward -0.5). Both indicate a difference was applied to an already-
    stationary series and should be avoided.
    """
    return pd.Series(
        {
            "variance_level": round(level.var(), 2),
            "variance_differenced": round(differenced.var(), 2),
            "variance_ratio": round(differenced.var() / level.var(), 3),
            "lag1_autocorr_level": round(acf(level.dropna(), nlags=1)[1], 3),
            "lag1_autocorr_differenced": round(acf(differenced, nlags=1)[1], 3),
        }
    )
