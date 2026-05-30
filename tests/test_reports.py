from src.reports.plots import build_counterfactual_plot
from src.reports.summary import generate_summary


class TestSummary:
    def test_significant_increase(self):
        text = generate_summary(
            effect=10.5, effect_pct=15.3, ci_lower=5.0, ci_upper=16.0,
            p_value=0.003, significant=True, direction="increase",
            metric_name="PM2.5",
        )
        assert "15.3% increase" in text
        assert "statistically significant" in text
        assert "PM2.5" in text

    def test_significant_decrease(self):
        text = generate_summary(
            effect=-8.2, effect_pct=-12.1, ci_lower=-15.0, ci_upper=-1.4,
            p_value=0.01, significant=True, direction="decrease",
            metric_name="revenue",
        )
        assert "12.1% decrease" in text
        assert "statistically significant" in text

    def test_insignificant(self):
        text = generate_summary(
            effect=2.1, effect_pct=3.0, ci_lower=-5.0, ci_upper=9.2,
            p_value=0.35, significant=False, direction="increase",
            metric_name="GDP",
        )
        assert "not produce a statistically significant" in text
        assert "p=0.3500" in text
        assert "GDP" in text

    def test_default_metric_name(self):
        text = generate_summary(
            effect=5.0, effect_pct=10.0, ci_lower=1.0, ci_upper=9.0,
            p_value=0.02, significant=True, direction="increase",
        )
        assert "the metric" in text


class TestPlots:
    def test_build_counterfactual_plot(self):
        import numpy as np

        n = 100
        dates = [f"2020-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n)]
        observed = np.random.randn(n).tolist()
        counterfactual = np.random.randn(n).tolist()

        fig = build_counterfactual_plot(dates, observed, counterfactual, 70)

        assert len(fig.data) == 3
        assert fig.data[0].name == "Observed"
        assert fig.data[1].name == "Fitted (pre-intervention)"
        assert fig.data[2].name == "Counterfactual (without policy)"
