import numpy as np
import pandas as pd
import pytest

from src.core.engine import Method, causal_effect
from src.data.loader import load_uploaded_file
from src.data.preprocessor import preprocess_data


class TestFullPipeline:
    def _make_df(self, n=100, effect=10):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += effect
        return pd.DataFrame({"date": dates, "value": y})

    def test_upload_preprocess_analyze_export(self):
        df = self._make_df()
        csv_bytes = df.to_csv(index=False).encode()
        import io
        f = io.BytesIO(csv_bytes)
        f.name = "test.csv"

        loaded = load_uploaded_file(f)
        assert len(loaded) == 100

        cleaned, report = preprocess_data(
            loaded, date_col="date", metric_col="value"
        )
        assert len(cleaned) == 100

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert result.significant is True
        assert result.effect > 5
        assert len(result.dates) == 100
        assert all(isinstance(d, str) for d in result.dates)

    def test_pipeline_with_missing_values(self):
        np.random.seed(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[10] = np.nan
        y[20] = np.nan
        df = pd.DataFrame({"date": dates, "value": y})

        cleaned, report = preprocess_data(df, missing_strategy="drop")
        assert len(cleaned) == 78

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-01",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_pipeline_with_dirty_column_names(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({" My Date ": dates, " My Value ": y})

        cleaned, report = preprocess_data(df)
        date_col = [c for c in cleaned.columns if "date" in c.lower()][0]
        metric_col = [c for c in cleaned.columns if "value" in c.lower()][0]

        result = causal_effect(
            df=cleaned,
            date_col=date_col,
            metric_col=metric_col,
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)


class TestEdgeCases:
    def test_all_same_values(self):
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.full(n, 50.0)
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )
        assert result.effect == 0.0
        assert result.p_value >= 0.05

    def test_very_short_series_just_above_minimum(self):
        np.random.seed(42)
        n = 32
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[25:] += 10
        df = pd.DataFrame({"date": dates, "value": y})

        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-01-26",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_upload_size_limit(self):
        import io
        f = io.BytesIO(b"col1,col2\n")
        f.name = "empty.csv"
        f.size = 0
        with pytest.raises(ValueError, match="empty"):
            load_uploaded_file(f)
