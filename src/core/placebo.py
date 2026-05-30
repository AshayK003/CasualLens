from __future__ import annotations

import logging

import numpy as np

from .arima_its import run_arima_its

logger = logging.getLogger(__name__)


def run_placebo_test(
    y: np.ndarray,
    real_intervention_idx: int,
    n_placebos: int = 50,
    seed: int = 42,
    on_progress=None,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    min_idx = max(10, n // 5)
    max_idx = min(n - 10, n - n // 5)

    if max_idx <= min_idx:
        return {
            "real_effect": 0,
            "placebo_effects": [],
            "p_value": 1.0,
            "is_real_effect_extreme": False,
        }

    actual_n = min(n_placebos, max_idx - min_idx)
    placebo_indices = rng.choice(
        range(min_idx, max_idx), size=actual_n, replace=False
    )
    placebo_indices = sorted(placebo_indices)

    real_result = run_arima_its(y, real_intervention_idx)
    real_effect = real_result.effect

    placebo_effects = []
    for i, idx in enumerate(placebo_indices):
        if on_progress:
            on_progress(i + 1, actual_n)
        try:
            result = run_arima_its(y, idx)
            placebo_effects.append(result.effect)
        except Exception as e:
            logger.warning(f"Placebo test at index {idx} failed: {e}")

    if not placebo_effects:
        return {
            "real_effect": real_effect,
            "placebo_effects": [],
            "p_value": 1.0,
            "is_real_effect_extreme": False,
        }

    placebo_arr = np.array(placebo_effects)
    n_extreme = np.sum(np.abs(placebo_arr) >= np.abs(real_effect))
    p_value = (n_extreme + 1) / (len(placebo_effects) + 1)

    return {
        "real_effect": float(real_effect),
        "placebo_effects": [float(x) for x in placebo_effects],
        "p_value": float(p_value),
        "is_real_effect_extreme": bool(np.abs(real_effect) > np.percentile(np.abs(placebo_arr), 95)),
        "placebo_indices": [int(x) for x in placebo_indices],
    }
