from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_dataset", "load_uploaded_file", "get_available_datasets"]

DATASETS_DIR = Path(__file__).parent / "datasets"
MAX_FILE_SIZE_BYTES = 50_000_000  # 50MB
MAX_ROWS = 200_000

AVAILABLE_DATASETS = {
    "delhi_aqi": {
        "label": "Delhi Air Quality (PM2.5, 2015-2024)",
        "description": "Daily PM2.5 concentrations. Test the 2020 lockdown effect.",
        "date_col": "date",
        "metric_col": "pm25",
        "intervention_date": "2020-03-25",
        "frequency": "daily",
    },
    "gst_revenue": {
        "label": "India GST Revenue (Monthly, 2016-2023)",
        "description": "Monthly GST collections. Test the GST implementation effect.",
        "date_col": "month",
        "metric_col": "revenue_cr",
        "intervention_date": "2017-07-01",
        "frequency": "monthly",
    },
    "hospital_admissions": {
        "label": "Hospital Admissions — Influenza (Weekly, 2018-2023)",
        "description": "Weekly flu admissions. Test the vaccine rollout effect.",
        "date_col": "date",
        "metric_col": "admissions",
        "intervention_date": "2021-01-01",
        "frequency": "weekly",
    },
    "electricity_demand": {
        "label": "Electricity Demand (Hourly, 2022-2023)",
        "description": "Hourly electricity demand. Test time-of-use pricing effect.",
        "date_col": "datetime",
        "metric_col": "demand_mwh",
        "intervention_date": "2023-06-01",
        "frequency": "hourly",
    },
    "crime_rates": {
        "label": "Crime Rates — Monthly (2015-2023)",
        "description": "Monthly crime incidents. Test the community policing effect.",
        "date_col": "date",
        "metric_col": "incidents",
        "intervention_date": "2020-01-01",
        "frequency": "monthly",
    },
    "student_scores": {
        "label": "Student Test Scores (Annual, 1984-2023)",
        "description": "Annual average test scores. Test the new curriculum effect.",
        "date_col": "year",
        "metric_col": "score",
        "intervention_date": "2019-01-01",
        "frequency": "yearly",
    },
    "traffic_accidents": {
        "label": "Traffic Accidents (Daily, 2019-2023)",
        "description": "Daily traffic accidents. Test the roundabout installation effect.",
        "date_col": "date",
        "metric_col": "accidents",
        "intervention_date": "2021-06-01",
        "frequency": "daily",
    },
    "website_sessions": {
        "label": "Website Sessions (Daily, 2022-2023)",
        "description": "Daily website sessions. Test the marketing campaign effect.",
        "date_col": "date",
        "metric_col": "sessions",
        "intervention_date": "2023-03-01",
        "frequency": "daily",
    },
    "mask_mandates": {
        "label": "Respiratory Illness Admissions (Monthly, 2016-2023)",
        "description": "Monthly hospital admissions for respiratory illness. Test mask mandate effect.",
        "date_col": "date",
        "metric_col": "admissions",
        "intervention_date": "2020-04-01",
        "frequency": "monthly",
    },
    "interest_rates": {
        "label": "Mortgage Applications (Monthly, 2019-2023)",
        "description": "Monthly mortgage applications. Test interest rate hike effect.",
        "date_col": "date",
        "metric_col": "applications",
        "intervention_date": "2022-03-01",
        "frequency": "monthly",
    },
    "carbon_tax": {
        "label": "Industrial CO2 Emissions (Quarterly, 2016-2023)",
        "description": "Quarterly industrial CO2 emissions (kilotons). Test carbon tax effect.",
        "date_col": "date",
        "metric_col": "emissions_kt",
        "intervention_date": "2019-01-01",
        "frequency": "quarterly",
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

    if hasattr(file, "size") and file.size is not None and file.size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {file.size / 1_000_000:.1f}MB. "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES / 1_000_000:.0f}MB."
        )

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

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"File has {len(df):,} rows. Maximum allowed is {MAX_ROWS:,} rows. "
            "Try filtering your data before uploading."
        )

    logger.info(f"Loaded uploaded file: {len(df)} rows, {len(df.columns)} cols")
    return df
