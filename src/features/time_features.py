"""Time-based feature engineering derived from the order timeline.

Seasons follow the meteorological Northern-Hemisphere convention because every order
in this dataset is shipped within the United States:

    December, January, February  -> Winter
    March, April, May            -> Spring
    June, July, August           -> Summer
    September, October, November -> Autumn

Meteorological (rather than astronomical) boundaries are used so that each season maps
cleanly onto whole calendar months, which is what the monthly sales aggregations need.
"""

from __future__ import annotations

import pandas as pd

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def add_time_features(df: pd.DataFrame, date_column: str = "Order Date") -> pd.DataFrame:
    """Return a copy of ``df`` enriched with calendar features from ``date_column``.

    A copy is returned so the caller's frame is never mutated in place — important when
    the same raw frame feeds several independent steps in a notebook.
    """
    if date_column not in df.columns:
        raise KeyError(f"'{date_column}' not found; parse dates before feature engineering.")

    result = df.copy()
    dates = result[date_column]

    result["Year"] = dates.dt.year
    result["Month"] = dates.dt.month
    result["Month Name"] = dates.dt.month_name()
    # ISO week keeps week 1 consistent across years, which matters once we compare
    # the same trading week across the four years of history.
    result["Week Number"] = dates.dt.isocalendar().week.astype("int64")
    result["Day of Week"] = dates.dt.day_name()
    result["Quarter"] = dates.dt.quarter
    result["Season"] = result["Month"].map(SEASON_BY_MONTH)

    return result


def add_fulfillment_time(
    df: pd.DataFrame,
    order_column: str = "Order Date",
    ship_column: str = "Ship Date",
) -> pd.DataFrame:
    """Add the number of days between order and shipment as ``Fulfillment Days``."""
    result = df.copy()
    result["Fulfillment Days"] = (result[ship_column] - result[order_column]).dt.days
    return result
