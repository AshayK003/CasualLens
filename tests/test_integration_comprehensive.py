import io

import numpy as np
import pandas as pd

from src.core.engine import CausalResult, Method, causal_effect
from src.data.loader import load_uploaded_file
from src.data.preprocessor import preprocess_data
from src.reports.pdf_export import generate_pdf_report
from src.reports.summary import generate_summary


class TestFullPipelineVariations:
    def _make_df(self, n=100, effect=10):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += effect
        return pd.DataFrame({"date": dates, "value": y})

    def test_full_pipeline_with_arima(self):
        df = self._make_df()
        csv_bytes = df.to_csv(index=False).encode()
        f = io.BytesIO(csv_bytes)
        f.name = "test.csv"

        loaded = load_uploaded_file(f)
        cleaned, report = preprocess_data(loaded, date_col="date", metric_col="value")
        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )

        assert isinstance(result, CausalResult)
        assert result.method == "arima"
        assert result.significant is True
        assert result.effect > 5

        summary = generate_summary(
            effect=result.effect,
            effect_pct=result.effect_pct,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            p_value=result.p_value,
            significant=result.significant,
            direction=result.direction,
            metric_name="value",
        )
        assert "statistically significant" in summary

        pdf_bytes = generate_pdf_report(
            dates=result.dates,
            observed=result.observed.tolist(),
            counterfactual=result.counterfactual.tolist(),
            intervention_idx=result.intervention_idx,
            effect=result.effect,
            effect_pct=result.effect_pct,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            p_value=result.p_value,
            significant=result.significant,
            direction=result.direction,
            metric_name="value",
            method=result.method,
        )
        assert pdf_bytes[:4] == b"%PDF"

    def test_pipeline_with_mean_imputation(self):
        np.random.seed(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[10] = np.nan
        y[20] = np.nan
        y[30] = np.nan
        df = pd.DataFrame({"date": dates, "value": y})

        cleaned, report = preprocess_data(df, missing_strategy="mean")
        assert cleaned["value"].isna().sum() == 0

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-01",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_pipeline_with_forward_fill(self):
        np.random.seed(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[10:15] = np.nan
        df = pd.DataFrame({"date": dates, "value": y})

        cleaned, report = preprocess_data(df, missing_strategy="forward_fill")
        assert cleaned["value"].isna().sum() == 0

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-01",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_pipeline_with_outlier_removal(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[50] = 200
        y[60] = -200
        df = pd.DataFrame({"date": dates, "value": y})

        cleaned, report = preprocess_data(
            df, remove_outliers_flag=True, outlier_method="iqr"
        )
        assert len(cleaned) < n

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="value",
            intervention_date="2020-03-11",
            method=Method.ARIMA,
        )
        assert isinstance(result.effect, float)

    def test_pipeline_with_year_month_columns(self):
        np.random.seed(42)
        n = 60
        months = pd.date_range("2018-01-01", periods=n, freq="MS")
        y = 50 + np.random.randn(n)
        y[40:] += 10
        df = pd.DataFrame({
            "year": months.year,
            "month": months.month,
            "value": y,
        })

        cleaned, report = preprocess_data(df)
        date_cols = [c for c in cleaned.columns if "date" in c.lower() or "constructed" in c.lower()]
        assert len(date_cols) > 0

    def test_pipeline_returns_correct_metadata(self):
        df = self._make_df(n=60)
        result = causal_effect(
            df=df,
            date_col="date",
            metric_col="value",
            intervention_date="2020-02-15",
            method=Method.ARIMA,
        )

        assert len(result.dates) == 60
        assert all(isinstance(d, str) for d in result.dates)
        assert result.n_pre + result.n_post == 60
        assert isinstance(result.arima_order, tuple)
        assert isinstance(result.aic, float)
        assert isinstance(result.ljung_box_pvalue, float)
        assert isinstance(result.residuals_ok, bool)


class TestEndToEndWithPreloadedDatasets:
    def test_delhi_aqi_dataset(self):
        from src.data.loader import load_dataset
        df = load_dataset("delhi_aqi")
        assert len(df) > 0

        cleaned, report = preprocess_data(
            df, date_col="date", metric_col="pm25"
        )
        assert len(cleaned) > 0

        result = causal_effect(
            df=cleaned,
            date_col="date",
            metric_col="pm25",
            intervention_date="2020-03-25",
            method=Method.ARIMA,
        )
        assert isinstance(result, CausalResult)
        assert result.n_pre > 0
        assert result.n_post > 0


class TestMultipleMethodsComparison:
    def test_arima_vs_bsts_same_input(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "value": y})

        arima_result = causal_effect(
            df=df, date_col="date", metric_col="value",
            intervention_date="2020-03-11", method=Method.ARIMA,
        )

        assert arima_result.method == "arima"
        assert isinstance(arima_result.effect, float)
