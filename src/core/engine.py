from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .arima_its import ITSResult, run_arima_its
from ..utils.validators import (
    validate_dataframe,
    validate_intervention_date,
    validate_series_length,
)

logger = logging.getLogger(__name__)


class Method(str, Enum):
    ARIMA = "arima"
    BSTS = "bsts"


@dataclass
class CausalResult:
    method: str
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    direction: str
    counterfactual: np.ndarray
    fitted_values: np.ndarray
    observed: np.ndarray
    intervention_idx: int
    dates: list[str]
    n_pre: int
    n_post: int


def causal_effect(
    df: "pd.DataFrame",
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: Method = Method.ARIMA,
) -> CausalResult:
    import pandas as pd

    date_col, metric_col = validate_dataframe(df)

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str), errors="coerce", format="mixed"
        )
        n_failed = df[date_col].isna().sum()
        if n_failed > 0:
            df = df.dropna(subset=[date_col])

    if df.empty:
        raise ValueError("No valid data after date parsing.")

    df = df.sort_values(date_col).reset_index(drop=True)
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    if df.empty:
        raise ValueError("No valid data after numeric conversion.")

    dates = df[date_col].values
    y = df[metric_col].values

    intervention_idx = validate_intervention_date(
        pd.DatetimeIndex(dates), intervention_date
    )
    validate_series_length(y)

    if method == Method.ARIMA:
        arima_result = run_arima_its(y, intervention_idx)
        result = CausalResult(
            method="arima",
            effect=arima_result.effect,
            effect_pct=arima_result.effect_pct,
            ci_lower=arima_result.ci_lower,
            ci_upper=arima_result.ci_upper,
            p_value=arima_result.p_value,
            significant=arima_result.p_value < 0.05,
            direction="increase" if arima_result.effect > 0 else "decrease",
            counterfactual=arima_result.counterfactual,
            fitted_values=arima_result.fitted_values,
            observed=arima_result.observed,
            intervention_idx=arima_result.intervention_idx,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
        )
    elif method == Method.BSTS:
        from .bsts import run_bsts
        bsts_result = run_bsts(y, intervention_idx)
        result = CausalResult(
            method="bsts",
            effect=bsts_result.effect,
            effect_pct=bsts_result.effect_pct,
            ci_lower=bsts_result.ci_lower,
            ci_upper=bsts_result.ci_upper,
            p_value=bsts_result.p_value,
            significant=bsts_result.p_value < 0.05,
            direction="increase" if bsts_result.effect > 0 else "decrease",
            counterfactual=bsts_result.counterfactual,
            fitted_values=bsts_result.fitted_values,
            observed=bsts_result.observed,
            intervention_idx=bsts_result.intervention_idx,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(
        f"Analysis complete: method={result.method}, effect={result.effect:.4f}, "
        f"p={result.p_value:.4f}, significant={result.significant}"
    )

    return result
