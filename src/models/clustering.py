"""Product demand segmentation: sub-category feature engineering and K-Means clustering.

Products are aggregated to the Sub-Category level (17 groups) and described by four
features — total sales volume, year-over-year growth, monthly sales volatility, and
average order value — chosen to capture size, trend direction, stability, and price
positioning as four largely distinct dimensions of demand behaviour. K-Means groups
sub-categories on these dimensions; PCA is used only to render the result in 2D, never as
an input to the clustering itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.aggregate import aggregate_sales

# The four engineered features fed to K-Means, after the log transform below is applied to
# AvgOrderValue. Order matters only for readability of downstream tables.
FEATURE_COLUMNS = ["TotalSales", "MeanYoYGrowth", "MonthlyVolatilityCV", "AvgOrderValueLog"]

# Reproducibility, consistent with the seed already used for XGBoost and Isolation Forest
# elsewhere in this project.
KMEANS_RANDOM_STATE = 42

# With only 17 sub-categories, K-Means' inertia surface can have several comparably-good
# local minima; 10 random restarts (keeping the lowest-inertia run) is cheap here and
# meaningfully more robust than relying on a single initialisation.
KMEANS_N_INIT = 10


def build_subcategory_features(
    df: pd.DataFrame,
    full_month_index: pd.DatetimeIndex,
    date_column: str = "Order Date",
    value_column: str = "Sales",
) -> pd.DataFrame:
    """One feature row per product sub-category, aggregated from line-item sales.

    Each engineered feature answers a distinct business question:

    * ``TotalSales`` — how big is this sub-category (four-year total revenue)?
    * ``MeanYoYGrowth`` — is it growing or shrinking? The mean of the three year-over-year
      percentage changes available in four years of data (2015→16, 16→17, 17→18).
    * ``MonthlyVolatilityCV`` — how erratic is its month-to-month demand, *relative to its
      own scale*? This is the coefficient of variation (std / mean) of its monthly sales,
      not the raw dollar standard deviation — see ``prepare_clustering_features`` for why.
    * ``AvgOrderValue`` — Sales divided by order-line count: the typical revenue per line
      item, i.e. is this a high-ticket or low-ticket product line?

    ``full_month_index`` should be the same month-start index used for the top-level
    monthly series, so a sub-category with no orders in a given month is scored as a
    genuine zero rather than silently shortening its series.
    """
    rows = []
    for sub_category, group in df.groupby("Sub-Category"):
        monthly = aggregate_sales(group, "monthly", date_column, value_column)
        monthly_series = (
            monthly.set_index(date_column)[value_column]
            .asfreq("MS")
            .reindex(full_month_index, fill_value=0.0)
        )

        annual_sales = group.groupby(group[date_column].dt.year)[value_column].sum()
        yoy_growth_pct = annual_sales.pct_change().dropna() * 100

        total_sales = group[value_column].sum()
        order_lines = len(group)

        rows.append(
            {
                "Sub-Category": sub_category,
                "TotalSales": total_sales,
                "MeanYoYGrowth": yoy_growth_pct.mean(),
                "MonthlyVolatilityCV": monthly_series.std() / monthly_series.mean(),
                "AvgOrderValue": total_sales / order_lines,
                "OrderLines": order_lines,
            }
        )

    return pd.DataFrame(rows).set_index("Sub-Category").sort_index()


def raw_monthly_std_by_subcategory(
    df: pd.DataFrame,
    full_month_index: pd.DatetimeIndex,
    date_column: str = "Order Date",
    value_column: str = "Sales",
) -> pd.Series:
    """The raw dollar standard deviation of monthly sales, per sub-category.

    Exists only to support the decision in ``build_subcategory_features`` to use the
    coefficient of variation instead of this raw figure: comparing this against
    ``TotalSales`` shows how strongly a sub-category's absolute dollar swings are just a
    restatement of its size, rather than genuine relative volatility.
    """
    result = {}
    for sub_category, group in df.groupby("Sub-Category"):
        monthly = aggregate_sales(group, "monthly", date_column, value_column)
        monthly_series = (
            monthly.set_index(date_column)[value_column]
            .asfreq("MS")
            .reindex(full_month_index, fill_value=0.0)
        )
        result[sub_category] = monthly_series.std()
    return pd.Series(result, name="RawMonthlyStd").sort_index()


def skewness_before_after_log(series: pd.Series) -> tuple[float, float]:
    """Skewness of ``series`` before and after a ``log1p`` transform.

    Used to decide, per feature, whether a log transform genuinely reduces skew (and
    should be applied) or would make it worse (and should be left alone) — a transform is
    only applied where this check supports it, never applied uniformly to every monetary
    feature by default.
    """
    return float(skew(series)), float(skew(np.log1p(series)))


def prepare_clustering_features(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Log-transform the skewed monetary feature and standardise everything for K-Means.

    Two preprocessing decisions, both checked against the data rather than assumed:

    * ``AvgOrderValue`` is right-skewed (skew ≈ 2.1, driven by Copiers' $2,216 average
      versus a $206 median) — a single high-ticket sub-category would otherwise dominate
      Euclidean distance in that dimension. ``log1p`` brings its skew to ≈ 0. ``TotalSales``
      is only mildly skewed (≈ 0.5) and log-transforming it *increases* skew (to ≈ -1.2), so
      it is deliberately left untransformed.
    * All four features are standardised (zero mean, unit variance) because they live on
      incomparable scales — dollars in the hundreds of thousands, a percentage, a ratio
      near 1, and log-dollars — and K-Means' Euclidean distance would otherwise be
      dominated by whichever feature happens to have the largest raw numeric range.

    Returns the (untransformed, unscaled) feature matrix used for interpretation alongside
    the scaled array actually passed to K-Means, so cluster profiles can be reported in
    business-readable units.
    """
    engineered = feature_table.copy()
    engineered["AvgOrderValueLog"] = np.log1p(engineered["AvgOrderValue"])

    matrix = engineered[FEATURE_COLUMNS]
    scaled = StandardScaler().fit_transform(matrix)
    return matrix, scaled


def cluster_search(X: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    """Inertia (for the elbow method) and silhouette score for each candidate ``k``.

    Both are computed rather than just one: inertia alone always decreases with ``k`` and
    needs a subjective "elbow" read, while the silhouette score gives an independent,
    single-number check on cluster separation that can confirm or challenge that read.
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(X, model.labels_),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def fit_kmeans(X: np.ndarray, n_clusters: int) -> KMeans:
    """Fit the final K-Means model with the project's standard reproducibility settings."""
    return KMeans(
        n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT
    ).fit(X)


def cluster_profile(feature_table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean feature values, member list, and size.

    This is the statistical profile a business-meaningful cluster name and stocking
    recommendation should be derived from — every label assigned in the notebook traces
    back to a number in this table, not to an assumption.
    """
    profiled = feature_table.copy()
    profiled["Cluster"] = labels

    summary = profiled.groupby("Cluster")[
        ["TotalSales", "MeanYoYGrowth", "MonthlyVolatilityCV", "AvgOrderValue"]
    ].mean().round(2)
    summary["Sub-Categories"] = (
        profiled.reset_index().groupby("Cluster")["Sub-Category"].agg(", ".join)
    )
    summary["n"] = profiled.groupby("Cluster").size()
    return summary


def pca_projection(X: np.ndarray, index: pd.Index) -> tuple[pd.DataFrame, PCA]:
    """2D PCA projection of the standardised clustering features, for visualisation only.

    PCA plays no role in forming the clusters — K-Means above is fit on the full
    standardised four-feature space. This projection exists purely so the resulting
    clusters can be plotted in two dimensions; every interpretation and business
    recommendation in the notebook is drawn from the original features, never from the
    PCA components.
    """
    pca = PCA(n_components=2, random_state=KMEANS_RANDOM_STATE)
    coords = pca.fit_transform(X)
    return pd.DataFrame(coords, columns=["PC1", "PC2"], index=index), pca
