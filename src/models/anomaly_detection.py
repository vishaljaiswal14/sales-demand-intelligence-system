"""Anomaly detection for weekly sales: rolling Z-score and Isolation Forest.

Both methods answer the same question — "which weeks were unusually high or low compared
to the expected pattern?" — from different angles. Z-score is a transparent, univariate
statistical rule anchored to a local rolling baseline. Isolation Forest is a multivariate,
model-based method that isolates unusual combinations of level and week-over-week change
without assuming any particular distribution. Running both and comparing where they agree
and disagree (see the notebook) is more informative than trusting either one alone.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# One retail quarter. Chosen over 8/26/52 weeks by comparing candidates directly (see
# ``zscore_window_sensitivity`` and the notebook): 13 weeks gives a large enough sample for
# a stable rolling standard deviation, recovers the September-2015 sales spike that Task 2's
# decomposition already flagged as the series' single largest residual (a 52-week window
# cannot — its burn-in period runs past that date), and produces an empirical flag rate close
# to the theoretical two-sigma tail probability under a Normal distribution, which is the
# best evidence that the window is neither too reactive nor too sluggish.
ZSCORE_WINDOW = 13

# The internship specification mandates flagging weeks that deviate more than two standard
# deviations from the rolling mean; see the notebook for the Normal-tail cross-check that
# supports this threshold rather than treating it as an arbitrary round number.
ZSCORE_THRESHOLD = 2.0

# ~ the theoretical two-sided tail probability for a 2-sigma threshold under Normality
# (2 * (1 - Phi(2)) = 4.55%), rounded to a business-friendly 5% "flag about 1 week in 20"
# budget. Using the same order of magnitude as the Z-score threshold keeps the two methods'
# anomaly *rates* comparable, so differences in *which* weeks get flagged reflect genuine
# methodological disagreement rather than one method simply being more permissive.
IF_CONTAMINATION = 0.05

# Verified empirically (see notebook): the flagged week set is identical at 100, 200, and
# 500 trees for this 208-row dataset at a fixed random_state, so the sklearn default is kept
# rather than paying extra compute for no change in output.
IF_N_ESTIMATORS = 100

# Fixes the forest's random feature/split-point selection so results are reproducible run to
# run, consistent with the seed used for XGBoost in Task 3.
IF_RANDOM_STATE = 42

# A reusable, month-based note on the US retail calendar. This is not a lookup keyed to any
# specific anomaly date — it is a generic mapping from calendar month to the retail season
# typically associated with it, applied to whichever weeks the statistical methods below
# happen to flag; no anomaly date is hardcoded anywhere in this module.
_RETAIL_CALENDAR_CONTEXT_BY_MONTH = {
    1: "January — post-holiday lull and New Year clearance",
    2: "February — Presidents' Day promotions",
    3: "March — spring / end-of-Q1 corporate purchasing push",
    4: "April — spring restocking",
    5: "May — Memorial Day weekend sales",
    6: "June — early-summer promotions",
    7: "July — mid-summer clearance and Independence Day sales",
    8: "August — back-to-school shopping ramp-up",
    9: "September — Labor Day sales and continued back-to-school demand",
    10: "October — pre-holiday inventory build-up",
    11: "November — Black Friday / Thanksgiving shopping peak",
    12: "December — Cyber Monday and Christmas holiday shopping peak",
}


def normal_tail_probability(z: float) -> float:
    """Two-sided tail probability P(|Z| > z) under a standard Normal distribution.

    Used to sanity-check the mandated Z-score threshold (is 2 standard deviations a
    reasonable cutoff, or arbitrary?) and to motivate the Isolation Forest contamination
    rate as a comparable anomaly "budget" between the two methods.
    """
    return 2 * (1 - norm.cdf(z))


def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Rolling mean, rolling standard deviation, and z-score for every week.

    ``min_periods`` is pinned to the full window rather than allowed to shrink: a standard
    deviation computed from only a handful of points is unstable and would produce
    spuriously extreme z-scores in the first few weeks of the series.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return pd.DataFrame(
        {"rolling_mean": rolling_mean, "rolling_std": rolling_std, "zscore": zscore}
    )


def flag_zscore_anomalies(
    zscore: pd.Series, threshold: float = ZSCORE_THRESHOLD
) -> pd.Series:
    """Boolean flag for weeks where the absolute z-score exceeds ``threshold``."""
    return (zscore.abs() > threshold).fillna(False).rename("is_anomaly")


def zscore_window_sensitivity(
    series: pd.Series,
    windows: tuple[int, ...] = (8, 13, 26, 52),
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Flagged-week count and rate for each candidate rolling window.

    A window that is too short bases its baseline on very few points (a noisy standard
    deviation estimate); one that is too long burns a large fraction of the series before
    any week can be scored at all. Comparing candidates side by side — including how many
    weeks each one can even evaluate — is what supports the window choice, instead of
    picking one arbitrarily.
    """
    rows = []
    for window in windows:
        result = rolling_zscore(series, window=window)
        flags = flag_zscore_anomalies(result["zscore"], threshold=threshold)
        scored = int(result["zscore"].notna().sum())
        rows.append(
            {
                "window_weeks": window,
                "first_scorable_week": series.index[window] if window < len(series) else pd.NaT,
                "scored_weeks": scored,
                "flagged_weeks": int(flags.sum()),
                "flagged_pct": round(flags.sum() / scored * 100, 1) if scored else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("window_weeks")


def build_isolation_forest_features(series: pd.Series) -> pd.DataFrame:
    """Level and week-over-week percentage change: two weakly-correlated views of "unusual".

    Level alone would just re-flag the biggest weeks in dollar terms. Percentage change
    alone over-reacts to swings off a tiny base (the business's low-volume 2015 ramp-up
    weeks show percentage jumps in the hundreds or thousands). Combining both — empirically
    only ~0.2 correlated on this series — lets Isolation Forest isolate weeks that are
    unusual on either axis, or both, rather than double-counting one signal.
    """
    return pd.DataFrame({"sales": series, "pct_change_1w": series.pct_change()}).dropna()


def fit_isolation_forest(
    features: pd.DataFrame,
    contamination: float = IF_CONTAMINATION,
    n_estimators: int = IF_N_ESTIMATORS,
    random_state: int = IF_RANDOM_STATE,
) -> tuple[IsolationForest, pd.Series, pd.Series]:
    """Fit Isolation Forest on standardised features; return the model, flags, and scores.

    Isolation Forest's random split thresholds are drawn per feature, so it is fairly
    scale-tolerant compared to distance-based methods, but standardising still keeps two
    differently-scaled features (dollars vs. a ratio) contributing comparably and keeps the
    resulting anomaly scores on a consistent footing — cheap to do and standard practice.

    ``max_samples`` is left at ``"auto"``: with 208 training rows (below its 256-row cap),
    every tree already sees the full dataset, so there is nothing to tune there.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
    )
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    flags = pd.Series(predictions == -1, index=features.index, name="is_anomaly")
    anomaly_scores = pd.Series(scores, index=features.index, name="anomaly_score")
    return model, flags, anomaly_scores


def classify_anomaly_driver(
    sales: float,
    order_lines: float,
    median_order_lines: float,
    median_avg_order_value: float,
    ratio_threshold: float = 1.3,
) -> str:
    """Classify an anomalous week as broad-based demand or a handful of large orders.

    Order-line counts and average order value are cheap, always-available signals for
    telling apart two very different anomalies: many customers buying at once (broad
    demand, e.g. a seasonal promotion) versus a small number of large transactions
    inflating the week's total despite ordinary customer traffic (e.g. bulk or enterprise
    orders). The two call for a different business response, so the distinction matters.
    """
    avg_order_value = sales / order_lines if order_lines else 0.0
    if order_lines > median_order_lines * ratio_threshold:
        return "broad-based (many orders)"
    if order_lines < median_order_lines and avg_order_value > median_avg_order_value * ratio_threshold:
        return "concentrated (few large orders)"
    return "mixed / inconclusive"


def retail_calendar_context(date: pd.Timestamp) -> str:
    """A generic, month-based note on the US retail calendar for any given date."""
    return _RETAIL_CALENDAR_CONTEXT_BY_MONTH[date.month]


def anomaly_evidence_table(
    weekly_sales: pd.Series,
    order_lines: pd.Series,
    zscore_flags: pd.Series,
    isolation_forest_flags: pd.Series,
) -> pd.DataFrame:
    """One evidence row per week flagged by either method.

    The flagged weeks themselves come entirely from the two detectors above — nothing here
    hardcodes which dates are anomalous. This only assembles the context (z-score, WoW %
    change, order-line count vs. the series median, demand-driver classification, and
    retail-calendar note) needed to write an explanation that is actually supported by data.
    """
    z_flagged = set(zscore_flags[zscore_flags].index)
    if_flagged = set(isolation_forest_flags[isolation_forest_flags].index)
    flagged_dates = sorted(z_flagged | if_flagged)

    median_order_lines = order_lines.median()
    median_avg_order_value = (weekly_sales / order_lines).median()
    week_over_week_pct = weekly_sales.pct_change()

    rows = []
    for date in flagged_dates:
        lines = order_lines.loc[date]
        sales = weekly_sales.loc[date]
        flagged_by = ", ".join(
            label
            for label, member in (("Z-Score", z_flagged), ("Isolation Forest", if_flagged))
            if date in member
        )
        rows.append(
            {
                "Week": date,
                "Sales": round(sales, 0),
                "Order Lines": int(lines),
                "Avg Order Value": round(sales / lines, 0),
                "WoW % Change": round(week_over_week_pct.loc[date] * 100, 0)
                if date in week_over_week_pct.index and pd.notna(week_over_week_pct.loc[date])
                else float("nan"),
                "Flagged By": flagged_by,
                "Demand Driver": classify_anomaly_driver(
                    sales, lines, median_order_lines, median_avg_order_value
                ),
                "Retail Calendar Context": retail_calendar_context(date),
            }
        )

    return pd.DataFrame(rows).set_index("Week")
