from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"

AVAILABLE_DATASETS = {
    "delhi_aqi": {
        "label": "Delhi Air Quality (PM2.5, 2015-2024)",
        "description": "Daily PM2.5 concentrations in Delhi. Test the 2020 lockdown effect.",
        "date_col": "date",
        "metric_col": "pm25",
        "intervention_date": "2020-03-25",
    },
    "gst_revenue": {
        "label": "India GST Revenue (Monthly, 2017-2024)",
        "description": "Monthly GST collections. Test the GST implementation effect.",
        "date_col": "month",
        "metric_col": "revenue_cr",
        "intervention_date": "2017-07-01",
    },
}


def get_available_datasets() -> dict[str, dict]:
    return AVAILABLE_DATASETS


def load_dataset(name: str) -> pd.DataFrame:
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(AVAILABLE_DATASETS.keys())}"
        )

    csv_path = DATASETS_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Generate it by running scripts/generate_datasets.py"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset '{name}': {len(df)} rows, {len(df.columns)} cols")
    return df


def load_user_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        raise ValueError("CSV file is empty")

    return df
