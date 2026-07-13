"""Loading and data-quality auditing for the Superstore sales dataset.

The raw file ships with dates formatted as ``DD/MM/YYYY`` and a handful of columns
that need explicit typing. Parsing is pinned to an explicit format rather than relying
on pandas' inference so that ambiguous days (e.g. ``08/11/2017``) are never silently
interpreted as month-first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_SALES_FILE

DATE_COLUMNS = ["Order Date", "Ship Date"]

# The source encodes dates day-first; pinning the format guards against the parser
# flipping day/month for values that happen to be valid under both interpretations.
_DATE_FORMAT = "%d/%m/%Y"


def load_superstore_sales(path: str | Path = RAW_SALES_FILE) -> pd.DataFrame:
    """Load the raw Superstore CSV with the date columns parsed as datetimes.

    Raises a clear error if the expected file or date columns are missing, which is
    friendlier than the downstream ``KeyError`` an analyst would otherwise hit.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the Superstore dataset at '{path}'. Download 'train.csv' from "
            "Kaggle and place it in data/raw/ (see README)."
        )

    df = pd.read_csv(path)

    missing_date_cols = [col for col in DATE_COLUMNS if col not in df.columns]
    if missing_date_cols:
        raise KeyError(f"Missing expected date columns: {missing_date_cols}")

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format=_DATE_FORMAT)

    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages, worst offenders first."""
    missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    )
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def duplicate_summary(df: pd.DataFrame, id_column: str = "Row ID") -> pd.Series:
    """Summarise fully-duplicated rows and duplicated identifier values.

    ``Order ID`` is deliberately *not* treated as a uniqueness key: a single order can
    legitimately contain several product line items, so duplicates there are expected
    rather than a defect.
    """
    result = {"fully_duplicated_rows": int(df.duplicated().sum())}
    if id_column in df.columns:
        result[f"duplicate_{id_column}"] = int(df[id_column].duplicated().sum())
    return pd.Series(result)


def date_integrity_summary(df: pd.DataFrame) -> pd.Series:
    """Flag logically invalid dates (e.g. an order shipped before it was placed)."""
    order, ship = df["Order Date"], df["Ship Date"]
    return pd.Series(
        {
            "order_date_min": order.min(),
            "order_date_max": order.max(),
            "ship_date_min": ship.min(),
            "ship_date_max": ship.max(),
            "ship_before_order": int((ship < order).sum()),
            "unparsed_order_dates": int(order.isna().sum()),
            "unparsed_ship_dates": int(ship.isna().sum()),
        }
    )
