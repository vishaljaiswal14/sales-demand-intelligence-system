"""Analytical helpers that answer the Task 1 business questions with evidence.

Each function returns a tidy frame or scalar summary rather than printing, so the
notebook can both display the numbers and reuse them in prose. No conclusion in the
notebook is asserted without a value produced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Total and share of revenue per product category, highest first."""
    revenue = (
        df.groupby("Category")["Sales"].sum().sort_values(ascending=False).to_frame()
    )
    revenue["Revenue Share %"] = (revenue["Sales"] / revenue["Sales"].sum() * 100).round(2)
    return revenue.rename(columns={"Sales": "Total Revenue"})


def yearly_sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot of total sales with one row per year and one column per region."""
    return df.pivot_table(
        index="Year", columns="Region", values="Sales", aggfunc="sum"
    ).round(2)


def regional_growth_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Score each region on how *consistently* its annual sales grow.

    Consistency is captured with three complementary pieces of evidence:

    * ``Mean YoY Growth %`` - the average year-over-year change.
    * ``YoY Growth Std %``  - volatility of that change; lower is steadier.
    * ``Trend R2``          - fit of a straight line to annual sales; closer to 1
      means the growth follows a stable, near-linear path rather than swinging around.

    A region is only a candidate for "most consistent growth" if it never contracts
    year over year (``Grew Every Year``).
    """
    yearly = yearly_sales_by_region(df).sort_index()
    years = yearly.index.to_numpy(dtype=float)

    records = []
    for region in yearly.columns:
        annual = yearly[region].to_numpy(dtype=float)
        yoy_growth = pd.Series(annual).pct_change().dropna() * 100

        slope, intercept = np.polyfit(years, annual, 1)
        predicted = slope * years + intercept
        ss_res = np.sum((annual - predicted) ** 2)
        ss_tot = np.sum((annual - annual.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan

        records.append(
            {
                "Region": region,
                "Mean YoY Growth %": round(yoy_growth.mean(), 2),
                "YoY Growth Std %": round(yoy_growth.std(ddof=0), 2),
                "Trend R2": round(r_squared, 3),
                "Grew Every Year": bool((yoy_growth > 0).all()),
            }
        )

    summary = pd.DataFrame(records).set_index("Region")
    # Rank steadiest-growing first: consistent growers, then tightest fit, then
    # lowest volatility as a tie-breaker.
    return summary.sort_values(
        ["Grew Every Year", "Trend R2", "YoY Growth Std %"],
        ascending=[False, False, True],
    )


def fulfillment_time_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Average, median, and spread of order-to-ship days per region.

    Requires the ``Fulfillment Days`` feature (see ``add_fulfillment_time``).
    """
    if "Fulfillment Days" not in df.columns:
        raise KeyError("'Fulfillment Days' missing; call add_fulfillment_time first.")

    summary = (
        df.groupby("Region")["Fulfillment Days"]
        .agg(["mean", "median", "std", "count"])
        .round(2)
    )
    summary.columns = ["Mean Days", "Median Days", "Std Days", "Order Lines"]
    return summary.sort_values("Mean Days")


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Sales per calendar month for every year, plus a cross-year average.

    Used to judge whether particular months spike *consistently* rather than in a
    single strong year.
    """
    pivot = df.pivot_table(
        index="Month", columns="Year", values="Sales", aggfunc="sum"
    ).round(2)
    pivot["Average"] = pivot.mean(axis=1).round(2)
    return pivot


def consistently_strong_months(df: pd.DataFrame, top_n: int = 3) -> pd.Series:
    """Count how often each month ranks in the yearly top ``top_n`` by sales.

    A month that appears in the top ``top_n`` in every year is strong evidence of
    genuine, repeatable seasonality rather than a one-off surge.
    """
    monthly = df.pivot_table(
        index="Month", columns="Year", values="Sales", aggfunc="sum"
    )
    top_flags = monthly.rank(ascending=False) <= top_n
    return top_flags.sum(axis=1).astype(int).rename(f"Years in Top {top_n}")
