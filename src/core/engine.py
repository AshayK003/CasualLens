from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..utils.constants import SIGNIFICANCE_LEVEL
from ..utils.validators import (
    validate_dataframe,
    validate_intervention_date,
    validate_series_length,
)
from .arima_its import run_arima_its

__all__ = ["causal_effect", "Method", "CausalResult"]

logger = logging.getLogger(__name__)


class Method(str, Enum):
    ARIMA = "arima"
    BSTS = "bsts"
    SARIMAX = "sarimax"
    DID = "did"
    SYNTHETIC_CONTROL = "synthetic_control"


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
    arima_order: tuple[int, int, int] | None = None
    aic: float | None = None
    ljung_box_pvalue: float | None = None
    residuals_ok: bool | None = None
    seasonal_order: tuple[int, int, int, int] | None = None


def causal_effect(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: Method = Method.ARIMA,
    group_col: str | None = None,
    treatment_unit: str | None = None,
) -> CausalResult:
    if date_col not in df.columns:
        # fall back to auto-detection if passed column doesn't exist
        detected_date, detected_metric = validate_dataframe(df)
        date_col = detected_date
        metric_col = detected_metric

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

    dates_pd = pd.to_datetime(dates)

    is_panel = method in (Method.DID, Method.SYNTHETIC_CONTROL)

    if not is_panel and not dates_pd.is_unique:
        logger.warning("Duplicate dates detected. Aggregating data by date.")
        df = df.groupby(date_col)[metric_col].mean().reset_index()
        dates_pd = pd.DatetimeIndex(df[date_col]).sort_values()
        dates = df[date_col].values
        y = df[metric_col].values

    if is_panel:
        dates_pd = pd.DatetimeIndex(dates).sort_values()
        intervention_idx = validate_intervention_date(dates_pd, intervention_date)
    else:
        intervention_idx = validate_intervention_date(dates_pd, intervention_date)
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
            significant=arima_result.p_value < SIGNIFICANCE_LEVEL,
            direction="increase" if arima_result.effect > 0 else "decrease",
            counterfactual=arima_result.counterfactual,
            fitted_values=arima_result.fitted_values,
            observed=arima_result.observed,
            intervention_idx=arima_result.intervention_idx,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
            arima_order=arima_result.arima_order,
            aic=arima_result.aic,
            ljung_box_pvalue=arima_result.ljung_box_pvalue,
            residuals_ok=arima_result.residuals_ok,
        )
    elif method == Method.BSTS:
        from .bsts import run_bsts
        try:
            bsts_result = run_bsts(y, intervention_idx)
        except RuntimeError as e:
            raise ValueError(f"BSTS analysis failed: {e}")
        result = CausalResult(
            method="bsts",
            effect=bsts_result.effect,
            effect_pct=bsts_result.effect_pct,
            ci_lower=bsts_result.ci_lower,
            ci_upper=bsts_result.ci_upper,
            p_value=bsts_result.p_value,
            significant=bsts_result.p_value < SIGNIFICANCE_LEVEL,
            direction="increase" if bsts_result.effect > 0 else "decrease",
            counterfactual=bsts_result.counterfactual,
            fitted_values=bsts_result.fitted_values,
            observed=bsts_result.observed,
            intervention_idx=bsts_result.intervention_idx,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
        )
    elif method == Method.SARIMAX:
        sarimax_result = run_arima_its(y, intervention_idx, seasonal=True)
        result = CausalResult(
            method="sarimax",
            effect=sarimax_result.effect,
            effect_pct=sarimax_result.effect_pct,
            ci_lower=sarimax_result.ci_lower,
            ci_upper=sarimax_result.ci_upper,
            p_value=sarimax_result.p_value,
            significant=sarimax_result.p_value < SIGNIFICANCE_LEVEL,
            direction="increase" if sarimax_result.effect > 0 else "decrease",
            counterfactual=sarimax_result.counterfactual,
            fitted_values=sarimax_result.fitted_values,
            observed=sarimax_result.observed,
            intervention_idx=sarimax_result.intervention_idx,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
            arima_order=sarimax_result.arima_order,
            aic=sarimax_result.aic,
            ljung_box_pvalue=sarimax_result.ljung_box_pvalue,
            residuals_ok=sarimax_result.residuals_ok,
            seasonal_order=sarimax_result.seasonal_order,
        )
    elif method == Method.DID:
        if group_col is None or treatment_unit is None:
            raise ValueError(
                "DiD method requires group_col and treatment_unit parameters."
            )
        from .did import run_did
        did_result = run_did(
            df=df,
            time_col=date_col,
            outcome_col=metric_col,
            group_col=group_col,
            treatment_unit=treatment_unit,
            intervention_date=intervention_date,
        )
        result = CausalResult(
            method="did",
            effect=did_result.effect,
            effect_pct=did_result.effect_pct,
            ci_lower=did_result.ci_lower,
            ci_upper=did_result.ci_upper,
            p_value=did_result.p_value,
            significant=did_result.significant,
            direction=did_result.direction,
            counterfactual=did_result.counterfactual,
            fitted_values=did_result.counterfactual,
            observed=did_result.observed,
            intervention_idx=did_result.intervention_idx,
            dates=did_result.dates,
            n_pre=did_result.n_treated,
            n_post=did_result.n_control,
        )
    elif method == Method.SYNTHETIC_CONTROL:
        if group_col is None or treatment_unit is None:
            raise ValueError(
                "Synthetic Control method requires group_col and treatment_unit parameters."
            )
        from .synthetic_control import run_synthetic_control
        sc_result = run_synthetic_control(
            df=df,
            time_col=date_col,
            outcome_col=metric_col,
            unit_col=group_col,
            treated_unit=treatment_unit,
            intervention_date=intervention_date,
        )
        result = CausalResult(
            method="synthetic_control",
            effect=sc_result.effect,
            effect_pct=sc_result.effect_pct,
            ci_lower=sc_result.ci_lower,
            ci_upper=sc_result.ci_upper,
            p_value=sc_result.p_value,
            significant=sc_result.significant,
            direction=sc_result.direction,
            counterfactual=sc_result.synth_outcome,
            fitted_values=sc_result.synth_outcome,
            observed=sc_result.treated_outcome,
            intervention_idx=sc_result.intervention_idx,
            dates=sc_result.dates,
            n_pre=sc_result.intervention_idx,
            n_post=len(sc_result.dates) - sc_result.intervention_idx,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(
        f"Analysis complete: method={result.method}, effect={result.effect:.4f}, "
        f"p={result.p_value:.4f}, significant={result.significant}"
    )

    return result
