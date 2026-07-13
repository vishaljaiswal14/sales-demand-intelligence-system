"""Aggregation of line-item sales into daily, weekly, and monthly time series.

Each line item in the raw data is one product on one order. To model demand over time
we need those line items rolled up to regular calendar intervals. Resampling (rather
than a plain ``groupby`` on the date) is used deliberately: it inserts zero-sales rows
for intervals with no orders, giving a continuous, gap-free series that the forecasting
models in later phases require.
"""

from __future__ import annotations

import pandas as pd

# pandas frequency aliases: calendar-day, week ending Sunday, and month-start.
_FREQUENCIES = {"daily": "D", "weekly": "W", "monthly": "MS"}


def aggregate_sales(
    df: pd.DataFrame,
    granularity: str,
    date_column: str = "Order Date",
    value_column: str = "Sales",
) -> pd.DataFrame:
    """Roll line-item sales up to ``daily``, ``weekly`` or ``monthly`` totals.

    Returns a frame indexed by period start with total sales and the number of order
    line items in each period. Empty periods are preserved as zero-sales rows so the
    resulting series is continuous.
    """
    if granularity not in _FREQUENCIES:
        raise ValueError(
            f"granularity must be one of {list(_FREQUENCIES)}, got '{granularity}'."
        )

    resampled = (
        df.set_index(date_column)
        .resample(_FREQUENCIES[granularity])
        .agg(sales=(value_column, "sum"), order_lines=(value_column, "size"))
    )
    # size() counts calendar rows including inserted empty periods; make the intent
    # explicit by naming the columns and keeping the date as a real column.
    resampled = resampled.rename(columns={"sales": "Sales", "order_lines": "Order Lines"})
    return resampled.reset_index()


def build_sales_time_series(
    df: pd.DataFrame,
    date_column: str = "Order Date",
    value_column: str = "Sales",
) -> dict[str, pd.DataFrame]:
    """Return the daily, weekly, and monthly sales series in one call."""
    return {
        granularity: aggregate_sales(df, granularity, date_column, value_column)
        for granularity in _FREQUENCIES
    }


def segment_monthly_series(
    df: pd.DataFrame,
    column: str,
    value: str,
    date_column: str = "Order Date",
    value_column: str = "Sales",
    full_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Monthly sales for the subset of ``df`` where ``column`` equals ``value``.

    Built on top of ``aggregate_sales`` so segment-level series get the same zero-filled,
    gap-free resampling as the top-level series. When ``full_index`` is supplied (typically
    the overall monthly series' index), the segment series is reindexed onto it with missing
    months filled at zero — a segment can genuinely have no orders in a month even when the
    business as a whole does, and without this a segment's own min/max order date could
    silently produce a shorter series than its peers.
    """
    subset = df[df[column] == value]
    if subset.empty:
        raise ValueError(f"No rows found where {column} == '{value}'.")

    monthly = aggregate_sales(subset, "monthly", date_column, value_column)
    series = monthly.set_index(date_column)["Sales"].asfreq("MS")

    if full_index is not None:
        series = series.reindex(full_index, fill_value=0.0)

    return series.rename(f"{column}={value}")
