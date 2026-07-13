"""Centralised project paths.

Keeping every filesystem location in one module means notebooks and source code
never hardcode relative paths, which would otherwise break depending on the working
directory a notebook or script happens to be launched from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHARTS_DIR = PROJECT_ROOT / "charts"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_SALES_FILE = RAW_DATA_DIR / "train.csv"


def ensure_directories() -> None:
    """Create the output directories if they do not already exist.

    Raw inputs are expected to be supplied by the user, so only the directories the
    pipeline writes to are created here.
    """
    for directory in (PROCESSED_DATA_DIR, CHARTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
