from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


@dataclass
class ITSResult:
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    counterfactual: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    observed: np.ndarray
    intervention_idx: int


def run_arima_its(
    y: np.ndarray,
    intervention_idx: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> ITSResult:
    y = np.asarray(y, dtype=float)
    n = len(y)

    pre = y[:intervention_idx]
    post = y[intervention_idx:]

    model = ARIMA(pre, order=order)
    fitted = model.fit()

    forecast = fitted.get_forecast(steps=len(post))
    predicted_mean = np.asarray(forecast.predicted_mean).flatten()
    raw_ci = forecast.conf_int(alpha=0.05)
    forecast_ci = np.asarray(raw_ci)

    if forecast_ci.ndim == 1:
        forecast_ci = forecast_ci.reshape(-1, 2)

    counterfactual_full = np.concatenate([
        np.asarray(fitted.fittedvalues).flatten(),
        predicted_mean,
    ])

    post_actual = y[intervention_idx:]
    post_predicted = predicted_mean

    effect = float(np.mean(post_actual - post_predicted))
    effect_pct = float(
        np.mean((post_actual - post_predicted) / (np.abs(post_predicted) + 1e-10)) * 100
    )

    ci_lower = float(np.mean(forecast_ci[:, 0]))
    ci_upper = float(np.mean(forecast_ci[:, 1]))

    se = (ci_upper - ci_lower) / (2 * stats.norm.ppf(0.975))
    if se > 0:
        t_stat = effect / se
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(post) - 1)))
    else:
        p_value = 1.0

    fitted_all = np.zeros(n)
    fitted_values_arr = np.asarray(fitted.fittedvalues).flatten()
    fitted_all[: len(fitted_values_arr)] = fitted_values_arr
    fitted_all[intervention_idx:] = post_predicted

    residuals = y - fitted_all

    return ITSResult(
        effect=effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        counterfactual=counterfactual_full,
        fitted_values=fitted_all,
        residuals=residuals,
        observed=y,
        intervention_idx=intervention_idx,
    )
