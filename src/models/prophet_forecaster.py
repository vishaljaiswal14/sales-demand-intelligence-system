"""Facebook Prophet forecasting for the monthly sales series.

Prophet decomposes a series into trend + seasonal (+ holiday) terms with a Bayesian
additive model. The configuration below is chosen for *monthly* data, which drives several
non-default decisions documented on ``PROPHET_CONFIG``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

# Prophet pulls in tqdm's notebook widget, which warns when ipywidgets is absent; silence
# that import-time warning before importing Prophet so notebook output stays clean.
warnings.filterwarnings("ignore", message="IProgress not found.*")

from prophet import Prophet  # noqa: E402  (import after the warning filter above)

# Prophet and its Stan backend are chatty (INFO "Chain [1] start processing" lines). Set the
# levels high and stop propagation so these never reach the notebook's captured output.
for _logger_name in ("prophet", "cmdstanpy"):
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False
    _logger.addHandler(logging.NullHandler())

MONTH_START_FREQ = "MS"

PROPHET_CONFIG = dict(
    growth="linear",
    # Monthly totals cannot resolve within-week or within-day patterns, so those
    # seasonalities are switched off — enabling them would only fit noise.
    weekly_seasonality=False,
    daily_seasonality=False,
    # Yearly seasonality is the signal that matters here. We cap the Fourier order at 4
    # (Prophet's default is 10): with only 12 observations per cycle, 10 harmonics would
    # overfit the annual shape, whereas 4 captures the smooth peak/trough without chasing noise.
    yearly_seasonality=4,
    # Additive, matching the Task 2 finding that the seasonal swing is roughly constant in
    # absolute dollars rather than scaling with the level.
    seasonality_mode="additive",
    # Trend flexibility. The default 0.05 is deliberately kept: on a 45-point series a larger
    # value lets the trend bend to noise, hurting out-of-sample forecasts.
    changepoint_prior_scale=0.05,
    # Report a 95% interval so uncertainty is comparable to the SARIMA confidence interval.
    interval_width=0.95,
)


def to_prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Convert an indexed sales Series to Prophet's required ``ds`` / ``y`` columns."""
    frame = series.reset_index()
    frame.columns = ["ds", "y"]
    return frame


def fit_prophet(series: pd.Series, **overrides) -> Prophet:
    """Fit Prophet on ``series`` using the documented monthly configuration."""
    config = {**PROPHET_CONFIG, **overrides}
    model = Prophet(**config)
    model.fit(to_prophet_frame(series))
    return model


def prophet_forecast(model: Prophet, horizon: int) -> pd.DataFrame:
    """Return Prophet's full forecast frame extended ``horizon`` months into the future."""
    future = model.make_future_dataframe(periods=horizon, freq=MONTH_START_FREQ)
    return model.predict(future)


def point_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Refit and return the ``horizon`` future point forecasts — the backtest hook."""
    model = fit_prophet(series)
    forecast = prophet_forecast(model, horizon)
    return forecast["yhat"].to_numpy()[-horizon:]
