import numpy as np
import pandas as pd
import pytest

from src.core.arima_its import run_arima_its
from src.core.engine import causal_effect, Method
from src.utils.validators import (
    validate_dataframe,
    validate_intervention_date,
    validate_series_length,
)


class TestValidators:
    def test_validate_dataframe_valid(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100),
            "value": np.random.randn(100),
        })
        date_col, metric_col = validate_dataframe(df)
        assert date_col == "date"
        assert metric_col == "value"

    def test_validate_dataframe_empty(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)

    def test_validate_dataframe_single_column(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 columns"):
            validate_dataframe(df)

    def test_validate_dataframe_no_date(self):
        df = pd.DataFrame({"text_a": ["hello", "world", "foo"], "text_b": ["bar", "baz", "qux"]})
        with pytest.raises(ValueError, match="No date column"):
            validate_dataframe(df)

    def test_validate_dataframe_no_numeric(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "text": ["a", "b", "c"],
        })
        with pytest.raises(ValueError, match="No numeric metric"):
            validate_dataframe(df)

    def test_validate_intervention_date_valid(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        idx = validate_intervention_date(dates, "2020-03-15")
        assert idx == 74

    def test_validate_intervention_date_too_early(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="too early"):
            validate_intervention_date(dates, "2020-01-02")

    def test_validate_intervention_date_too_late(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="too late"):
            validate_intervention_date(dates, "2020-12-29")

    def test_validate_intervention_date_outside_range(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="too late|outside the data range"):
            validate_intervention_date(dates, "2025-01-01")

    def test_validate_intervention_date_invalid(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="Cannot parse"):
            validate_intervention_date(dates, "not-a-date")

    def test_validate_series_length_ok(self):
        y = np.random.randn(50)
        validate_series_length(y, min_length=30)

    def test_validate_series_length_too_short(self):
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="too short"):
            validate_series_length(y, min_length=30)


class TestARIMAITS:
    def test_detects_known_positive_effect(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_arima_its(y, intervention_idx=70)

        assert result.effect > 5
        assert result.p_value < 0.05
        assert result.intervention_idx == 70
        assert len(result.counterfactual) == n

    def test_detects_known_negative_effect(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 100 + 0.3 * t + np.random.normal(0, 2, n)
        y[70:] -= 15

        result = run_arima_its(y, intervention_idx=70)

        assert result.effect < -5
        assert result.p_value < 0.05

    def test_no_effect_returns_insignificant(self):
        np.random.seed(42)
        n = 100
        y = 50 + np.random.normal(0, 2, n)

        result = run_arima_its(y, intervention_idx=70)

        assert abs(result.effect) < 10
        assert result.p_value > 0.05

    def test_counterfactual_length_matches_input(self):
        np.random.seed(42)
        y = np.random.randn(80)
        result = run_arima_its(y, intervention_idx=60)
        assert len(result.counterfactual) == 80
        assert len(result.fitted_values) == 80
        assert len(result.observed) == 80

    def test_result_fields_are_numeric(self):
        np.random.seed(42)
        y = np.random.randn(80)
        result = run_arima_its(y, intervention_idx=60)
        assert isinstance(result.effect, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.ci_lower, float)
        assert isinstance(result.ci_upper, float)


class TestEngine:
    def _make_df(self, n=100, effect=10):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += effect
        return pd.DataFrame({"date": dates, "value": y})

    def test_causal_effect_basic(self):
        df = self._make_df()
        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert result.method == "arima"
        assert result.effect > 5
        assert result.significant is True
        assert result.direction == "increase"
        assert result.n_pre == 70
        assert result.n_post == 30

    def test_causal_effect_decrease(self):
        df = self._make_df(effect=-15)
        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert result.effect < -5
        assert result.direction == "decrease"

    def test_causal_effect_invalid_method(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="Unknown method"):
            causal_effect(
                df=df,
                date_col="date",
                metric_col="value",
                intervention_date="2020-03-11",
                method="invalid",
            )

    def test_causal_effect_dates_are_strings(self):
        df = self._make_df()
        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert all(isinstance(d, str) for d in result.dates)
        assert len(result.dates) == 100
