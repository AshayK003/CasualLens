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
        "label": "India GST Revenue (Monthly, 2016-2023)",
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


def load_uploaded_file(file) -> pd.DataFrame:
    filename = file.name.lower()

    try:
        if filename.endswith(".csv"):
            file.seek(0)
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            file.seek(0)
            df = pd.read_excel(file, engine="openpyxl")
        else:
            raise ValueError(
                f"Unsupported file format: '{file.name}'. "
                "Please upload a CSV or Excel file (.csv, .xlsx, .xls)."
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    if df.empty:
        raise ValueError("File is empty — no data rows found.")

    if len(df.columns) < 2:
        raise ValueError(
            "File must have at least 2 columns (a date column and a metric column)."
        )

    logger.info(f"Loaded uploaded file: {len(df)} rows, {len(df.columns)} cols")
    return df
