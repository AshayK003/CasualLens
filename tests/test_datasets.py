import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.engine import Method, causal_effect
from src.data.loader import get_available_datasets, load_dataset

DATASETS_DIR = Path(__file__).parent.parent / "src" / "data" / "datasets"


class TestAllDatasetsExist:
    def test_csv_files_exist(self):
        datasets = get_available_datasets()
        for name in datasets:
            csv_path = DATASETS_DIR / f"{name}.csv"
            assert csv_path.exists(), f"Missing CSV: {csv_path}"

    def test_metadata_exists(self):
        meta_path = DATASETS_DIR / "metadata.json"
        assert meta_path.exists(), f"Missing metadata: {meta_path}"

    def test_metadata_covers_all_datasets(self):
        meta_path = DATASETS_DIR / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        datasets = get_available_datasets()
        for name in datasets:
            assert name in metadata, f"Missing metadata for {name}"


class TestDatasetStructure:
    def test_all_datasets_loadable(self):
        datasets = get_available_datasets()
        for name in datasets:
            df = load_dataset(name)
            assert len(df) > 0, f"{name} is empty"
            meta = datasets[name]
            assert meta["date_col"] in df.columns, f"{name}: missing date_col '{meta['date_col']}'"
            assert meta["metric_col"] in df.columns, f"{name}: missing metric_col '{meta['metric_col']}'"

    def test_all_dates_parseable(self):
        datasets = get_available_datasets()
        for name in datasets:
            df = load_dataset(name)
            meta = datasets[name]
            dates = pd.to_datetime(df[meta["date_col"]], errors="coerce")
            n_na = dates.isna().sum()
            assert n_na == 0, f"{name}: {n_na} unparseable dates"

    def test_all_metrics_numeric(self):
        datasets = get_available_datasets()
        for name in datasets:
            df = load_dataset(name)
            meta = datasets[name]
            numeric = pd.to_numeric(df[meta["metric_col"]], errors="coerce")
            n_na = numeric.isna().sum()
            assert n_na == 0, f"{name}: {n_na} non-numeric metric values"


class TestGroundTruthDetection:
    @pytest.fixture(autouse=True)
    def load_metadata(self):
        meta_path = DATASETS_DIR / "metadata.json"
        with open(meta_path) as f:
            self.all_metadata = json.load(f)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "name,expected_lo,expected_hi",
        [
            ("delhi_aqi", -80, 10),
            ("gst_revenue", -10000, 150000),
            ("traffic_accidents", -15, 5),
            ("website_sessions", 50, 1200),
            ("crime_rates", -30, 5),
        ],
    )
    def test_arima_detects_known_effect(self, name, expected_lo, expected_hi):
        datasets = get_available_datasets()
        meta = datasets[name]
        df = load_dataset(name)

        result = causal_effect(
            df=df,
            date_col=meta["date_col"],
            metric_col=meta["metric_col"],
            intervention_date=meta["intervention_date"],
            method=Method.ARIMA,
        )

        assert expected_lo <= result.effect <= expected_hi, (
            f"{name}: effect {result.effect:.2f} outside expected range [{expected_lo}, {expected_hi}]"
        )

    @pytest.mark.slow
    def test_hospital_admissions_detected(self):
        datasets = get_available_datasets()
        meta = datasets["hospital_admissions"]
        df = load_dataset("hospital_admissions")
        gt = self.all_metadata["hospital_admissions"]["ground_truth"]

        result = causal_effect(
            df=df,
            date_col=meta["date_col"],
            metric_col=meta["metric_col"],
            intervention_date=meta["intervention_date"],
            method=Method.ARIMA,
        )

        lo, hi = gt["expected_effect_range"]
        assert lo <= result.effect <= hi, (
            f"hospital_admissions: effect {result.effect:.2f} outside [{lo}, {hi}]"
        )

    @pytest.mark.slow
    def test_student_scores_detected(self):
        datasets = get_available_datasets()
        meta = datasets["student_scores"]
        df = load_dataset("student_scores")
        gt = self.all_metadata["student_scores"]["ground_truth"]

        result = causal_effect(
            df=df,
            date_col=meta["date_col"],
            metric_col=meta["metric_col"],
            intervention_date=meta["intervention_date"],
            method=Method.ARIMA,
        )

        lo, hi = gt["expected_effect_range"]
        assert lo <= result.effect <= hi, (
            f"student_scores: effect {result.effect:.2f} outside [{lo}, {hi}]"
        )

    @pytest.mark.slow
    def test_electricity_demand_detected(self):
        datasets = get_available_datasets()
        meta = datasets["electricity_demand"]
        df = load_dataset("electricity_demand")

        result = causal_effect(
            df=df,
            date_col=meta["date_col"],
            metric_col=meta["metric_col"],
            intervention_date=meta["intervention_date"],
            method=Method.ARIMA,
        )

        assert -200 <= result.effect <= 200, (
            f"electricity_demand: effect {result.effect:.2f} outside [-200, 200]"
        )
