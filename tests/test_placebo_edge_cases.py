
import numpy as np

from src.core.placebo import run_placebo_test


class TestPlaceboEdgeCases:
    def test_progress_callback_is_called(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        progress_calls = []
        def on_progress(current, total):
            progress_calls.append((current, total))

        run_placebo_test(y, real_intervention_idx=70, n_placebos=5, on_progress=on_progress)
        assert len(progress_calls) > 0
        assert all(c[1] == 5 for c in progress_calls)

    def test_very_short_series_returns_default(self):
        np.random.seed(42)
        y = np.random.randn(5)
        result = run_placebo_test(y, real_intervention_idx=3, n_placebos=2)
        assert result["p_value"] == 1.0
        assert result["placebo_effects"] == []
        assert result["is_real_effect_extreme"] is False

    def test_placebo_indices_in_result(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=10)
        assert "placebo_indices" in result
        assert len(result["placebo_indices"]) <= 10

    def test_p_value_bounded(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=20)
        assert 0 <= result["p_value"] <= 1

    def test_real_effect_matches_direct_calculation(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=5)
        from src.core.arima_its import run_arima_its
        direct = run_arima_its(y, 70)
        assert abs(result["real_effect"] - direct.effect) < 0.01

    def test_n_placebos_larger_than_possible(self):
        np.random.seed(42)
        y = np.random.randn(30)
        result = run_placebo_test(y, real_intervention_idx=15, n_placebos=100)
        assert len(result["placebo_effects"]) <= 100

    def test_all_same_values(self):
        np.random.seed(42)
        y = np.full(100, 50.0)
        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=5)
        assert result["p_value"] >= 0.5
        assert result["is_real_effect_extreme"] is False

    def test_extreme_outlier_effect_detected(self):
        np.random.seed(42)
        n = 100
        y = 50 + np.random.normal(0, 0.1, n)
        y[70:] += 100

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=10)
        assert result["is_real_effect_extreme"] is True
        assert result["p_value"] < 0.2
