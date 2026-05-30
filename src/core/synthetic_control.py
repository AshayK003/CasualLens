from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SyntheticControlResult", "run_synthetic_control"]

logger = logging.getLogger(__name__)


@dataclass
class SyntheticControlResult:
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    direction: str
    unit_weights: dict[str, float]
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    treated_outcome: np.ndarray
    synth_outcome: np.ndarray
    dates: list[str]
    intervention_idx: int


def run_synthetic_control(
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    unit_col: str,
    treated_unit: str,
    intervention_date: str,
) -> SyntheticControlResult:
    from pysyncon import Dataprep, Synth

    if unit_col not in df.columns:
        raise ValueError(
            f"Unit column '{unit_col}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    if treated_unit not in df[unit_col].unique():
        raise ValueError(
            f"Treatment unit '{treated_unit}' not found in column '{unit_col}'. "
            f"Available units: {sorted(df[unit_col].unique())}"
        )

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    units = df[unit_col].unique()
    control_units = [u for u in units if u != treated_unit]

    if len(control_units) < 1:
        raise ValueError(
            f"Need at least one control unit for synthetic control. "
            f"Found only the treated unit '{treated_unit}'. "
            f"Upload a dataset with multiple groups/units."
        )

    intervention_ts = pd.Timestamp(intervention_date)
    all_times = sorted(df[time_col].unique())
    intervention_idx = next(i for i, t in enumerate(all_times) if t >= intervention_ts)

    pre_times = [t for t in all_times if t < intervention_ts]

    wide_df = df.pivot_table(
        index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean"
    )
    wide_df = wide_df.sort_index()

    treated_series = wide_df[treated_unit].values
    synth_series = np.zeros(len(all_times))

    dataprep = Dataprep(
        foo=df,
        predictors=[outcome_col],
        predictors_op="mean",
        dependent=outcome_col,
        unit_variable=unit_col,
        time_variable=time_col,
        treatment_identifier=treated_unit,
        controls_identifier=control_units,
        time_predictors_prior=[t for t in pre_times],
        time_optimize_ssr=[t for t in pre_times],
    )

    model = Synth()
    model.fit(dataprep=dataprep)

    weights_dict = model.weights().to_dict()

    for i, t in enumerate(all_times):
        if t in wide_df.index:
            synth_val = 0.0
            for ctrl, w in weights_dict.items():
                if ctrl in wide_df.columns:
                    synth_val += w * wide_df.loc[t, ctrl]
            synth_series[i] = synth_val

    pre_actual = treated_series[:intervention_idx]
    pre_synth = synth_series[:intervention_idx]
    post_actual = treated_series[intervention_idx:]
    post_synth = synth_series[intervention_idx:]

    pre_rmspe = float(np.sqrt(np.mean((pre_actual - pre_synth) ** 2)))
    post_rmspe = float(np.sqrt(np.mean((post_actual - post_synth) ** 2)))
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else 0.0

    pointwise_effects = post_actual - post_synth
    effect = float(np.mean(pointwise_effects))

    pre_mean = float(np.mean(pre_actual))
    effect_pct = effect / (abs(pre_mean) + 1e-10) * 100

    n_post = len(post_actual)
    se = float(np.std(pointwise_effects, ddof=1) / np.sqrt(n_post)) if n_post > 1 else 0.0

    ci_lower = effect - 1.96 * se
    ci_upper = effect + 1.96 * se

    if se > 0:
        from scipy import stats
        t_stat = effect / se
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_post - 1)))
    else:
        p_value = 1.0

    significant = p_value < 0.05
    direction = "increase" if effect > 0 else "decrease"

    date_strs = [str(d)[:10] for d in all_times]

    return SyntheticControlResult(
        effect=effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=significant,
        direction=direction,
        unit_weights=weights_dict,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        rmspe_ratio=rmspe_ratio,
        treated_outcome=treated_series,
        synth_outcome=synth_series,
        dates=date_strs,
        intervention_idx=intervention_idx,
    )
