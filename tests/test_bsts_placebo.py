import numpy as np
import pytest

from src.core.bsts import run_bsts
from src.core.placebo import run_placebo_test


def _bsts_works() -> bool:
    try:
        from causalimpact import CausalImpact
        import pandas as pd

        y = np.concatenate([np.ones(50) * 10, np.ones(50) * 15])
        data = pd.DataFrame({"y": y})
        data.index = pd.RangeIndex(100)
        ci = CausalImpact(data, [0, 49], [50, 99], model_args={"niter": 500})
        predicted = ci.model.predicted_mean
        return predicted is not None and np.any(np.asarray(predicted) != 0)
    except Exception:
        return False


BSTS_AVAILABLE = _bsts_works()


class TestBSTS:
    @pytest.mark.skipif(not BSTS_AVAILABLE, reason="causalimpact model fitting broken on this system")
    def test_bsts_runs(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_bsts(y, intervention_idx=70)

        assert result.effect > 0
        assert len(result.counterfactual) == n
        assert len(result.observed) == n
        assert result.intervention_idx == 70

    def test_bsts_gracefully_handles_fitting_failure(self):
        np.random.seed(42)
        n = 100
        y = 50 + np.random.randn(n)

        result = run_bsts(y, intervention_idx=70)

        assert len(result.counterfactual) == n
        assert len(result.observed) == n
        assert isinstance(result.effect, float)
        assert isinstance(result.p_value, float)


class TestPlacebo:
    def test_placebo_returns_valid_result(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        y = 50 + 0.5 * t + np.random.normal(0, 2, n)
        y[70:] += 10

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=3)

        assert "real_effect" in result
        assert "placebo_effects" in result
        assert "p_value" in result
        assert len(result["placebo_effects"]) <= 3

    def test_placebo_with_short_series(self):
        np.random.seed(42)
        y = np.random.randn(15)
        result = run_placebo_test(y, real_intervention_idx=10, n_placebos=3)
        assert result["p_value"] == 1.0

    def test_placebo_effect_extreme_detection(self):
        np.random.seed(42)
        n = 100
        y = 50 + np.random.normal(0, 0.1, n)
        y[70:] += 50

        result = run_placebo_test(y, real_intervention_idx=70, n_placebos=5)
        assert result["is_real_effect_extreme"] is True
