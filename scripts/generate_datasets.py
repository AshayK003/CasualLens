from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASETS_DIR = ROOT / "src" / "data" / "datasets"


class TimeSeriesBuilder:
    """Compose realistic time series from additive components."""

    def __init__(self, n_points: int, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n = n_points
        self.components = np.zeros(n_points)
        self.noise = np.zeros(n_points)
        self.intervention_mask = np.zeros(n_points, dtype=bool)

    def add_trend(self, slope: float, intercept: float = 0) -> TimeSeriesBuilder:
        t = np.arange(self.n)
        self.components += slope * t + intercept
        return self

    def add_seasonality(
        self, period: float, amplitude: float, phase: float = 0
    ) -> TimeSeriesBuilder:
        t = np.arange(self.n)
        self.components += amplitude * np.sin(2 * np.pi * t / period + phase)
        return self

    def add_double_seasonality(
        self,
        period1: float,
        amp1: float,
        period2: float,
        amp2: float,
        phase: float = 0,
    ) -> TimeSeriesBuilder:
        t = np.arange(self.n)
        self.components += amp1 * np.sin(2 * np.pi * t / period1 + phase)
        self.components += amp2 * np.sin(2 * np.pi * t / period2 + phase)
        return self

    def add_ar_noise(self, ar_params: list[float], sigma: float) -> TimeSeriesBuilder:
        p = len(ar_params)
        errors = self.rng.normal(0, sigma, self.n)
        ar = np.zeros(self.n)
        for i in range(p, self.n):
            ar[i] = sum(ar_params[j] * ar[i - j - 1] for j in range(p))
            ar[i] += errors[i]
        self.noise += ar
        return self

    def add_gaussian_noise(self, sigma: float) -> TimeSeriesBuilder:
        self.noise += self.rng.normal(0, sigma, self.n)
        return self

    def add_level_shift(
        self, start_idx: int, effect: float, ramp_up: int = 0
    ) -> TimeSeriesBuilder:
        if ramp_up <= 0:
            self.components[start_idx:] += effect
        else:
            ramp = np.linspace(0, effect, ramp_up)
            self.components[start_idx : start_idx + ramp_up] += ramp
            self.components[start_idx + ramp_up :] += effect
        return self

    def add_transient_effect(
        self, start_idx: int, effect: float, duration: int
    ) -> TimeSeriesBuilder:
        end_idx = min(start_idx + duration, self.n)
        self.components[start_idx:end_idx] += effect
        return self

    def add_regressor_effect(
        self, regressor: np.ndarray, coef: float
    ) -> TimeSeriesBuilder:
        self.components += regressor[: self.n] * coef
        return self

    def clip(self, lower: float, upper: float) -> TimeSeriesBuilder:
        self.components = np.clip(self.components, lower, upper)
        return self

    def build(self) -> np.ndarray:
        return self.components + self.noise


def _apply_weekly_pattern(
    values: np.ndarray, n_days: int, weekday_effect: float, weekend_effect: float
) -> np.ndarray:
    """Add weekday/weekend pattern to daily data."""
    result = values.copy()
    for i in range(n_days):
        day_of_week = i % 7
        if day_of_week >= 5:  # Saturday=5, Sunday=6
            result[i] += weekend_effect
        else:
            result[i] += weekday_effect
    return result


def generate_delhi_aqi() -> tuple[pd.DataFrame, dict]:
    n_days = 365 * 10
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")

    ts = TimeSeriesBuilder(n_days, seed=42)
    ts.add_trend(slope=-0.005, intercept=120)
    ts.add_seasonality(period=365.25, amplitude=30, phase=np.pi / 2)
    ts.add_ar_noise(ar_params=[0.3], sigma=8)
    values = ts.build()

    lockdown_idx = np.searchsorted(dates, pd.Timestamp("2020-03-25"))
    values = _apply_weekly_pattern(values, n_days, weekday_effect=0, weekend_effect=-2)
    values[lockdown_idx : lockdown_idx + 60] -= 40

    values = np.clip(values, 10, 500)

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "pm25": np.round(values, 1)})

    meta = {
        "label": "Delhi Air Quality (PM2.5, 2015-2024)",
        "description": "Daily PM2.5 concentrations. Test the 2020 lockdown effect.",
        "date_col": "date",
        "metric_col": "pm25",
        "intervention_date": "2020-03-25",
        "frequency": "daily",
        "ground_truth": {
            "effect_size": -40.0,
            "effect_direction": "decrease",
            "effect_duration": "60 days (transient)",
            "expected_effect_range": [-55, -25],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "COVID lockdown caused immediate PM2.5 drop, recovered after 60 days",
        },
    }
    return df, meta


def generate_gst_revenue() -> tuple[pd.DataFrame, dict]:
    n_months = 12 * 8
    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")

    ts = TimeSeriesBuilder(n_months, seed=42)
    ts.add_trend(slope=1040, intercept=80000)
    ts.add_seasonality(period=12, amplitude=15000, phase=np.pi / 2)
    ts.add_ar_noise(ar_params=[0.4], sigma=3000)
    values = ts.build()

    gst_idx = np.searchsorted(dates, pd.Timestamp("2017-07-01"))
    values[gst_idx:] += 8000

    values = np.clip(values, 50000, 300000)

    df = pd.DataFrame(
        {"month": dates.strftime("%Y-%m-%d"), "revenue_cr": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "India GST Revenue (Monthly, 2016-2023)",
        "description": "Monthly GST collections. Test the GST implementation effect.",
        "date_col": "month",
        "metric_col": "revenue_cr",
        "intervention_date": "2017-07-01",
        "frequency": "monthly",
        "ground_truth": {
            "effect_size": 8000.0,
            "effect_direction": "increase",
            "effect_duration": "permanent",
            "expected_effect_range": [4000, 15000],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "GST implementation caused permanent revenue level shift",
        },
    }
    return df, meta


def generate_hospital_admissions() -> tuple[pd.DataFrame, dict]:
    n_weeks = 52 * 6  # 6 years
    dates = pd.date_range("2018-01-01", periods=n_weeks, freq="W")

    ts = TimeSeriesBuilder(n_weeks, seed=42)
    ts.add_trend(slope=0.1, intercept=200)
    ts.add_seasonality(period=52, amplitude=40, phase=np.pi)
    ts.add_ar_noise(ar_params=[0.5, 0.2], sigma=12)
    values = ts.build()

    vaccine_idx = np.searchsorted(dates, pd.Timestamp("2021-01-01"))
    ramp_up = 8
    effect = -30  # -15% of ~200 baseline
    ramp = np.linspace(0, effect, ramp_up)
    values[vaccine_idx : vaccine_idx + ramp_up] += ramp
    values[vaccine_idx + ramp_up :] += effect

    values = np.clip(values, 20, 500)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "admissions": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Hospital Admissions — Influenza (Weekly, 2018-2023)",
        "description": "Weekly flu admissions. Test the vaccine rollout effect.",
        "date_col": "date",
        "metric_col": "admissions",
        "intervention_date": "2021-01-01",
        "frequency": "weekly",
        "ground_truth": {
            "effect_size": -30.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent (gradual 8-week ramp)",
            "expected_effect_range": [-50, -10],
            "expected_p_value_range": [0.0, 0.10],
            "notes": "Vaccine rollout gradually reduced admissions over 8 weeks",
        },
    }
    return df, meta


def generate_electricity_demand() -> tuple[pd.DataFrame, dict]:
    n_hours = 24 * 365 * 2  # 2 years
    dates = pd.date_range("2022-01-01", periods=n_hours, freq="h")

    ts = TimeSeriesBuilder(n_hours, seed=42)
    ts.add_trend(slope=0.002, intercept=500)
    ts.add_double_seasonality(
        period1=24, amp1=80, period2=24 * 7, amp2=30
    )
    ts.add_ar_noise(ar_params=[0.6], sigma=25)
    values = ts.build()

    pricing_idx = np.searchsorted(dates, pd.Timestamp("2023-06-01"))
    for i in range(pricing_idx, n_hours):
        hour = dates[i].hour
        if 14 <= hour < 20:
            values[i] *= 0.92

    values = np.clip(values, 100, 2000)

    df = pd.DataFrame(
        {"datetime": dates.strftime("%Y-%m-%d %H:%M:%S"), "demand_mwh": np.round(values, 1)}
    )

    meta = {
        "label": "Electricity Demand (Hourly, 2022-2023)",
        "description": "Hourly electricity demand. Test time-of-use pricing effect.",
        "date_col": "datetime",
        "metric_col": "demand_mwh",
        "intervention_date": "2023-06-01",
        "frequency": "hourly",
        "ground_truth": {
            "effect_size": -40.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent (peak hours only)",
            "expected_effect_range": [-80, -10],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "Time-of-use pricing reduced peak-hour demand by 8%",
        },
    }
    return df, meta


def generate_crime_rates() -> tuple[pd.DataFrame, dict]:
    n_months = 12 * 9
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")

    ts = TimeSeriesBuilder(n_months, seed=42)
    ts.add_trend(slope=0.3, intercept=180)
    ts.add_seasonality(period=12, amplitude=15, phase=0)
    ts.add_ar_noise(ar_params=[0.3], sigma=8)
    values = ts.build()

    policing_idx = np.searchsorted(dates, pd.Timestamp("2020-01-01"))
    values[policing_idx:] += -12

    values = np.clip(values, 50, 400)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "incidents": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Crime Rates — Monthly (2015-2023)",
        "description": "Monthly crime incidents. Test the community policing effect.",
        "date_col": "date",
        "metric_col": "incidents",
        "intervention_date": "2020-01-01",
        "frequency": "monthly",
        "ground_truth": {
            "effect_size": -12.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent",
            "expected_effect_range": [-25, -3],
            "expected_p_value_range": [0.0, 0.10],
            "notes": "Community policing program reduced monthly incidents",
        },
    }
    return df, meta


def generate_student_scores() -> tuple[pd.DataFrame, dict]:
    n_years = 40
    dates = pd.date_range("1984-01-01", periods=n_years, freq="YS")

    ts = TimeSeriesBuilder(n_years, seed=42)
    ts.add_trend(slope=0.3, intercept=62)
    ts.add_gaussian_noise(sigma=1.5)
    values = ts.build()

    curriculum_idx = np.searchsorted(dates, pd.Timestamp("2019-01-01"))
    values[curriculum_idx:] += 5

    values = np.clip(values, 40, 100)

    df = pd.DataFrame(
        {"year": dates.strftime("%Y-%m-%d"), "score": np.round(values, 1)}
    )

    meta = {
        "label": "Student Test Scores (Annual, 1984-2023)",
        "description": "Annual average test scores. Test the new curriculum effect.",
        "date_col": "year",
        "metric_col": "score",
        "intervention_date": "2019-01-01",
        "frequency": "yearly",
        "ground_truth": {
            "effect_size": 5.0,
            "effect_direction": "increase",
            "effect_duration": "permanent",
            "expected_effect_range": [1, 10],
            "expected_p_value_range": [0.0, 0.20],
            "notes": "New curriculum raised average scores by 5 points",
        },
    }
    return df, meta


def generate_traffic_accidents() -> tuple[pd.DataFrame, dict]:
    n_days = 365 * 5
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")

    ts = TimeSeriesBuilder(n_days, seed=42)
    ts.add_trend(slope=0.001, intercept=25)
    ts.add_seasonality(period=365.25, amplitude=4, phase=0)
    ts.add_ar_noise(ar_params=[0.2], sigma=3)
    values = ts.build()

    values = _apply_weekly_pattern(values, n_days, weekday_effect=0, weekend_effect=5)

    roundabout_idx = np.searchsorted(dates, pd.Timestamp("2021-06-01"))
    values[roundabout_idx:] += -3

    values = np.clip(values, 5, 60)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "accidents": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Traffic Accidents (Daily, 2019-2023)",
        "description": "Daily traffic accidents. Test the roundabout installation effect.",
        "date_col": "date",
        "metric_col": "accidents",
        "intervention_date": "2021-06-01",
        "frequency": "daily",
        "ground_truth": {
            "effect_size": -3.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent",
            "expected_effect_range": [-6, -1],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "Roundabout installation reduced daily accidents",
        },
    }
    return df, meta


def generate_website_sessions() -> tuple[pd.DataFrame, dict]:
    n_days = 365 * 2
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    ts = TimeSeriesBuilder(n_days, seed=42)
    ts.add_trend(slope=1.5, intercept=1200)
    ts.add_seasonality(period=365.25, amplitude=150, phase=0)
    ts.add_ar_noise(ar_params=[0.4], sigma=80)
    values = ts.build()

    values = _apply_weekly_pattern(values, n_days, weekday_effect=50, weekend_effect=-100)

    campaign_idx = np.searchsorted(dates, pd.Timestamp("2023-03-01"))
    ramp_up = 30
    effect = 500
    ramp = np.linspace(0, effect, ramp_up)
    values[campaign_idx : campaign_idx + ramp_up] += ramp
    values[campaign_idx + ramp_up :] += effect

    values = np.clip(values, 200, 5000)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "sessions": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Website Sessions (Daily, 2022-2023)",
        "description": "Daily website sessions. Test the marketing campaign effect.",
        "date_col": "date",
        "metric_col": "sessions",
        "intervention_date": "2023-03-01",
        "frequency": "daily",
        "ground_truth": {
            "effect_size": 500.0,
            "effect_direction": "increase",
            "effect_duration": "permanent (gradual 30-day ramp)",
            "expected_effect_range": [200, 800],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "Marketing campaign gradually increased sessions over 30 days",
        },
    }
    return df, meta


def generate_mask_mandates() -> tuple[pd.DataFrame, dict]:
    n_months = 12 * 7
    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")

    ts = TimeSeriesBuilder(n_months, seed=42)
    ts.add_trend(slope=0.5, intercept=350)
    ts.add_seasonality(period=12, amplitude=60, phase=np.pi)
    ts.add_ar_noise(ar_params=[0.4], sigma=15)
    values = ts.build()

    mandate_idx = np.searchsorted(dates, pd.Timestamp("2020-04-01"))
    ramp_up = 3
    effect = -55
    ramp = np.linspace(0, effect, ramp_up)
    values[mandate_idx : mandate_idx + ramp_up] += ramp
    values[mandate_idx + ramp_up :] += effect

    values = np.clip(values, 100, 600)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "admissions": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Respiratory Illness Admissions (Monthly, 2016-2023)",
        "description": "Monthly hospital admissions for respiratory illness. Test mask mandate effect.",
        "date_col": "date",
        "metric_col": "admissions",
        "intervention_date": "2020-04-01",
        "frequency": "monthly",
        "ground_truth": {
            "effect_size": -55.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent (gradual 3-month ramp)",
            "expected_effect_range": [-80, -30],
            "expected_p_value_range": [0.0, 0.10],
            "notes": "Mask mandates reduced respiratory admissions by ~15%",
        },
    }
    return df, meta


def generate_interest_rates() -> tuple[pd.DataFrame, dict]:
    n_months = 12 * 5
    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    ts = TimeSeriesBuilder(n_months, seed=42)
    ts.add_trend(slope=20, intercept=6000)
    ts.add_seasonality(period=12, amplitude=400, phase=0)
    ts.add_ar_noise(ar_params=[0.3], sigma=150)
    values = ts.build()

    hike_idx = np.searchsorted(dates, pd.Timestamp("2022-03-01"))
    ramp_up = 6
    effect = -1200
    ramp = np.linspace(0, effect, ramp_up)
    values[hike_idx : hike_idx + ramp_up] += ramp
    values[hike_idx + ramp_up :] += effect

    values = np.clip(values, 3000, 12000)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "applications": np.round(values, 0).astype(int)}
    )

    meta = {
        "label": "Mortgage Applications (Monthly, 2019-2023)",
        "description": "Monthly mortgage applications. Test interest rate hike effect.",
        "date_col": "date",
        "metric_col": "applications",
        "intervention_date": "2022-03-01",
        "frequency": "monthly",
        "ground_truth": {
            "effect_size": -1200.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent (gradual 6-month ramp)",
            "expected_effect_range": [-2000, -500],
            "expected_p_value_range": [0.0, 0.05],
            "notes": "Interest rate hike reduced mortgage applications by ~20%",
        },
    }
    return df, meta


def generate_carbon_tax() -> tuple[pd.DataFrame, dict]:
    n_quarters = 4 * 8
    dates = pd.date_range("2016-01-01", periods=n_quarters, freq="QS")

    ts = TimeSeriesBuilder(n_quarters, seed=42)
    ts.add_trend(slope=-2, intercept=500)
    ts.add_seasonality(period=4, amplitude=20, phase=0)
    ts.add_ar_noise(ar_params=[0.2], sigma=10)
    values = ts.build()

    tax_idx = np.searchsorted(dates, pd.Timestamp("2019-01-01"))
    ramp_up = 4
    effect = -45
    ramp = np.linspace(0, effect, ramp_up)
    values[tax_idx : tax_idx + ramp_up] += ramp
    values[tax_idx + ramp_up :] += effect

    values = np.clip(values, 300, 600)

    df = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "emissions_kt": np.round(values, 1)}
    )

    meta = {
        "label": "Industrial CO2 Emissions (Quarterly, 2016-2023)",
        "description": "Quarterly industrial CO2 emissions (kilotons). Test carbon tax effect.",
        "date_col": "date",
        "metric_col": "emissions_kt",
        "intervention_date": "2019-01-01",
        "frequency": "quarterly",
        "ground_truth": {
            "effect_size": -45.0,
            "effect_direction": "decrease",
            "effect_duration": "permanent (gradual 4-quarter ramp)",
            "expected_effect_range": [-70, -20],
            "expected_p_value_range": [0.0, 0.10],
            "notes": "Carbon tax reduced industrial emissions by ~10%",
        },
    }
    return df, meta


ALL_GENERATORS = [
    ("delhi_aqi", generate_delhi_aqi),
    ("gst_revenue", generate_gst_revenue),
    ("hospital_admissions", generate_hospital_admissions),
    ("electricity_demand", generate_electricity_demand),
    ("crime_rates", generate_crime_rates),
    ("student_scores", generate_student_scores),
    ("traffic_accidents", generate_traffic_accidents),
    ("website_sessions", generate_website_sessions),
    ("mask_mandates", generate_mask_mandates),
    ("interest_rates", generate_interest_rates),
    ("carbon_tax", generate_carbon_tax),
]


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    all_metadata = {}

    for name, generator in ALL_GENERATORS:
        print(f"Generating {name}...")
        df, meta = generator()
        csv_path = DATASETS_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        all_metadata[name] = meta
        print(f"  -> {len(df):,} rows, {meta['frequency']} frequency")

    meta_path = DATASETS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDone. {len(ALL_GENERATORS)} datasets saved to {DATASETS_DIR}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
