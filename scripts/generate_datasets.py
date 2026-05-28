from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASETS_DIR = ROOT / "src" / "data" / "datasets"


def generate_delhi_aqi() -> pd.DataFrame:
    np.random.seed(42)
    n_days = 365 * 10
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")

    trend = np.linspace(120, 100, n_days)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 15, n_days)

    pm25 = trend + seasonal + noise

    lockdown_start = pd.Timestamp("2020-03-25")
    lockdown_idx = np.searchsorted(dates, lockdown_start)
    pm25[lockdown_idx:lockdown_idx + 60] -= 40

    pm25 = np.clip(pm25, 10, 500)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pm25": np.round(pm25, 1),
    })
    return df


def generate_gst_revenue() -> pd.DataFrame:
    np.random.seed(42)
    n_months = 12 * 8
    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")

    trend = np.linspace(80000, 180000, n_months)
    seasonal = 15000 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise = np.random.normal(0, 5000, n_months)

    revenue = trend + seasonal + noise

    gst_start = pd.Timestamp("2017-07-01")
    gst_idx = np.searchsorted(dates, gst_start)
    revenue[gst_idx:gst_idx + 6] += 8000

    revenue = np.clip(revenue, 50000, 300000)

    df = pd.DataFrame({
        "month": dates.strftime("%Y-%m-%d"),
        "revenue_cr": np.round(revenue, 0).astype(int),
    })
    return df


if __name__ == "__main__":
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Delhi AQI dataset...")
    df_aqi = generate_delhi_aqi()
    df_aqi.to_csv(DATASETS_DIR / "delhi_aqi.csv", index=False)
    print(f"  -> {len(df_aqi)} rows saved to delhi_aqi.csv")

    print("Generating GST Revenue dataset...")
    df_gst = generate_gst_revenue()
    df_gst.to_csv(DATASETS_DIR / "gst_revenue.csv", index=False)
    print(f"  -> {len(df_gst)} rows saved to gst_revenue.csv")

    print("Done.")
