import numpy as np
import pandas as pd
import pytest

from src.core.arima_its import _check_residuals, _select_arima_order, _select_arima_order_fast
from src.core.engine import CausalResult, Method, causal_effect
from src.utils.validators import validate_intervention_date


class TestSelectArimaOrder:
    def test_returns_tuple_of_three_ints(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100)) + 50
        order = _select_arima_order(y, max_p=1, max_d=1, max_q=1)
        assert isinstance(order, tuple)
        assert len(order) == 3
        assert all(isinstance(v, int) for v in order)

    def test_default_range(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(80)) + 50
        order = _select_arima_order(y, max_p=2, max_d=1, max_q=2)
        assert 0 <= order[0] <= 2
        assert 0 <= order[1] <= 1
        assert 0 <= order[2] <= 2
        assert not (order[0] == 0 and order[2] == 0)


class TestSelectArimaOrderFast:
    def test_large_dataset_reduces_search(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(1500)) + 50
        order = _select_arima_order_fast(y)
        assert isinstance(order, tuple)
        assert len(order) == 3

    def test_medium_dataset(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(500)) + 50
        order = _select_arima_order_fast(y)
        assert isinstance(order, tuple)

    def test_small_dataset_full_search(self):
        np.random.seed(42)
        y = np.cumsum(np.random.randn(50)) + 50
        order = _select_arima_order_fast(y)
        assert isinstance(order, tuple)


class TestCheckResiduals:
    def test_white_noise(self):
        np.random.seed(42)
        residuals = np.random.randn(200)
        lb_pval, ok = _check_residuals(residuals)
        assert isinstance(lb_pval, float)
        assert isinstance(ok, bool)
        assert 0 <= lb_pval <= 1

    def test_too_short_returns_default(self):
        residuals = np.array([1.0])
        lb_pval, ok = _check_residuals(residuals)
        assert lb_pval == 1.0
        assert ok is True

    def test_autocorrelated(self):
        np.random.seed(42)
        n = 200
        residuals = np.zeros(n)
        for i in range(1, n):
            residuals[i] = 0.9 * residuals[i - 1] + np.random.randn() * 0.1
        lb_pval, ok = _check_residuals(residuals)
        assert lb_pval < 0.05 or ok is False


class TestEngineEdgeCases:
    def _make_df(self, n=100, effect=10):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += effect
        return pd.DataFrame({"date": dates, "value": y})

    def test_duplicate_dates_are_aggregated(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.random.randn(n)
        df = pd.DataFrame({"date": dates, "value": y})
        dup = df.iloc[:10].copy()
        df = pd.concat([df, dup], ignore_index=True)
        assert len(df) == 110

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert len(result.dates) == 100

    def test_nonexistent_date_col_falls_back_to_auto_detect(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="nonexistent_col",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert isinstance(result, CausalResult)

    def test_string_dates_are_parsed(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "value": y,
        })

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert all(isinstance(d, str) for d in result.dates)

    def test_invalid_date_col_raises(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="totally_invalid",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert isinstance(result, CausalResult)

    def test_numeric_value_column_with_commas(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = (50 + np.random.randn(n)) * 1000
        df = pd.DataFrame({
            "date": dates,
            "value": [f"{v:,.0f}" for v in y],
        })

        from src.data.preprocessor import preprocess_data
        cleaned, report = preprocess_data(df, date_col="date", metric_col="value")

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_effect_percentage_near_zero_denominator(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.full(n, 0.00001)
        y[40:] += 0.000001
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-02-10",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect_pct, float)
        assert np.isfinite(result.effect_pct)

    def test_intervention_at_nearest_date(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-14T12:00:00",
            method=Method.ARIMA,
        )
        assert isinstance(result.intervention_idx, int)
        assert 0 < result.intervention_idx < n


class TestValidateInterventionDateEdgeCases:
    def test_intervention_date_at_exact_boundary_start(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="too early"):
            validate_intervention_date(dates, "2020-01-01")

    def test_intervention_date_at_exact_boundary_end(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        with pytest.raises(ValueError, match="too late"):
            validate_intervention_date(dates, "2020-04-09")

    def test_duplicate_dates_are_handled(self):
        unique_dates = pd.date_range("2020-01-01", periods=20, freq="D")
        dates = pd.DatetimeIndex(list(unique_dates) * 5)
        idx = validate_intervention_date(dates, "2020-01-10")
        assert isinstance(idx, int)
        assert idx >= 0

    def test_string_date_with_time_component(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        idx = validate_intervention_date(dates, "2020-03-15 10:30:00")
        assert isinstance(idx, int)
