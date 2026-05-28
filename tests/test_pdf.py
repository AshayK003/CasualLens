import numpy as np

from src.reports.pdf_export import generate_pdf_report


class TestPDFExport:
    def test_generates_pdf_bytes(self):
        np.random.seed(42)
        n = 100
        dates = [f"2020-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n)]
        observed = (50 + np.random.randn(n)).tolist()
        counterfactual = (50 + np.random.randn(n)).tolist()

        pdf_bytes = generate_pdf_report(
            dates=dates,
            observed=observed,
            counterfactual=counterfactual,
            intervention_idx=70,
            effect=8.5,
            effect_pct=15.3,
            ci_lower=3.2,
            ci_upper=13.8,
            p_value=0.003,
            significant=True,
            direction="increase",
            metric_name="PM2.5",
            method="arima",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes[:4] == b"%PDF"

    def test_generates_insignificant_report(self):
        np.random.seed(42)
        n = 50
        dates = [f"2020-01-{i+1:02d}" for i in range(n)]
        observed = (100 + np.random.randn(n)).tolist()
        counterfactual = (100 + np.random.randn(n)).tolist()

        pdf_bytes = generate_pdf_report(
            dates=dates,
            observed=observed,
            counterfactual=counterfactual,
            intervention_idx=30,
            effect=2.1,
            effect_pct=3.0,
            ci_lower=-5.0,
            ci_upper=9.2,
            p_value=0.35,
            significant=False,
            direction="increase",
            metric_name="revenue",
            method="arima",
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"
