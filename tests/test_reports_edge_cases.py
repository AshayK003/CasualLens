
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.reports.pdf_export import generate_pdf_report
from src.reports.plots import build_counterfactual_plot
from src.reports.summary import generate_summary


class TestGenerateSummaryEdgeCases:
    def test_insignificant_with_increase_direction(self):
        result = generate_summary(
            effect=0.5,
            effect_pct=1.0,
            ci_lower=-0.3,
            ci_upper=1.3,
            p_value=0.45,
            significant=False,
            direction="increase",
            metric_name="revenue",
        )
        assert "not produce a statistically significant" in result
        assert "increase" in result
        assert "revenue" in result

    def test_significant_decrease(self):
        result = generate_summary(
            effect=-5.0,
            effect_pct=-10.0,
            ci_lower=-8.0,
            ci_upper=-2.0,
            p_value=0.01,
            significant=True,
            direction="decrease",
            metric_name="crime_rate",
        )
        assert "decrease" in result
        assert "statistically significant" in result
        assert "crime_rate" in result

    def test_default_metric_name(self):
        result = generate_summary(
            effect=1.0,
            effect_pct=5.0,
            ci_lower=0.0,
            ci_upper=2.0,
            p_value=0.03,
            significant=True,
            direction="increase",
        )
        assert "the metric" in result

    def test_very_small_p_value(self):
        result = generate_summary(
            effect=10.0,
            effect_pct=20.0,
            ci_lower=5.0,
            ci_upper=15.0,
            p_value=0.0001,
            significant=True,
            direction="increase",
            metric_name="score",
        )
        assert "0.0001" in result

    def test_ci_crosses_zero_insignificant(self):
        result = generate_summary(
            effect=0.1,
            effect_pct=0.5,
            ci_lower=-0.5,
            ci_upper=0.7,
            p_value=0.7,
            significant=False,
            direction="increase",
            metric_name="sessions",
        )
        assert "not produce a statistically significant" in result


class TestBuildCounterfactualPlot:
    def test_returns_figure(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        observed = list(range(10))
        counterfactual = list(range(10))
        fig = build_counterfactual_plot(dates, observed, counterfactual, 5)
        assert isinstance(fig, go.Figure)

    def test_has_three_traces(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        observed = list(range(10))
        counterfactual = list(range(10))
        fig = build_counterfactual_plot(dates, observed, counterfactual, 5)
        assert len(fig.data) == 3

    def test_intervention_line_present(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        observed = list(range(10))
        counterfactual = list(range(10))
        fig = build_counterfactual_plot(dates, observed, counterfactual, 5)
        assert len(fig.layout.shapes) >= 1

    def test_annotation_present(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        observed = list(range(10))
        counterfactual = list(range(10))
        fig = build_counterfactual_plot(dates, observed, counterfactual, 5)
        assert len(fig.layout.annotations) >= 1

    def test_trace_names(self):
        dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
        observed = list(range(10))
        counterfactual = list(range(10))
        fig = build_counterfactual_plot(dates, observed, counterfactual, 5)
        names = [trace.name for trace in fig.data]
        assert "Observed" in names
        assert "Fitted (pre-intervention)" in names
        assert "Counterfactual (without policy)" in names


class TestPDFExport:
    def _make_args(self, significant=True):
        n = 50
        dates = [f"2020-01-{i:02d}" for i in range(1, min(n, 31))]
        dates += [f"2020-02-{i:02d}" for i in range(1, n - len(dates) + 1)]
        observed = list(np.random.randn(n) + 50)
        counterfactual = list(np.random.randn(n) + 48)
        return dict(
            dates=dates,
            observed=observed,
            counterfactual=counterfactual,
            intervention_idx=30,
            effect=10.0 if significant else 0.5,
            effect_pct=15.0 if significant else 0.5,
            ci_lower=5.0 if significant else -1.0,
            ci_upper=15.0 if significant else 2.0,
            p_value=0.01 if significant else 0.6,
            significant=significant,
            direction="increase" if significant else "increase",
            metric_name="test_metric",
            method="arima",
        )

    def test_significant_report(self):
        pdf_bytes = generate_pdf_report(**self._make_args(significant=True))
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_insignificant_report(self):
        pdf_bytes = generate_pdf_report(**self._make_args(significant=False))
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_decrease_report(self):
        args = self._make_args(significant=True)
        args["direction"] = "decrease"
        args["effect"] = -10.0
        pdf_bytes = generate_pdf_report(**args)
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_contains_text(self):
        pdf_bytes = generate_pdf_report(**self._make_args())
        assert b"PDF" in pdf_bytes[:10]
        assert len(pdf_bytes) > 1000

    def test_large_dataset(self):
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
        observed = (np.random.randn(n) + 50).tolist()
        counterfactual = (np.random.randn(n) + 48).tolist()
        pdf_bytes = generate_pdf_report(
            dates=dates,
            observed=observed,
            counterfactual=counterfactual,
            intervention_idx=400,
            effect=10.0,
            effect_pct=15.0,
            ci_lower=5.0,
            ci_upper=15.0,
            p_value=0.01,
            significant=True,
            direction="increase",
            metric_name="large_metric",
            method="arima",
        )
        assert pdf_bytes[:4] == b"%PDF"
