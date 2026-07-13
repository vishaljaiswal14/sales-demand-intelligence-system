"""Shared plotting configuration and a single figure-saving entry point.

Centralising the style keeps every chart in the project visually consistent, and routing
all saves through one helper guarantees a uniform resolution and that figures always land
in ``charts/`` with tight bounding boxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import CHARTS_DIR

# A calm, professional qualitative palette used across categorical charts.
CATEGORICAL_PALETTE = ["#2E5E8C", "#3C8DAD", "#E08E45", "#6B8E5A", "#A65959", "#8C6BAE"]

SAVE_DPI = 150


def configure_plot_style() -> None:
    """Apply the project-wide matplotlib/seaborn styling. Idempotent."""
    sns.set_theme(style="whitegrid", palette=CATEGORICAL_PALETTE)
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "figure.titlesize": 15,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": "#444444",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save ``fig`` into ``charts/`` at the project resolution and return its path."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / filename
    fig.savefig(output_path, dpi=SAVE_DPI)
    return output_path
