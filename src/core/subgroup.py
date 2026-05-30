from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .engine import Method, causal_effect

__all__ = ["SubgroupResult", "run_subgroup_analysis"]

logger = logging.getLogger(__name__)

MIN_SEGMENT_SIZE = 30


@dataclass
class SubgroupResult:
    segment: str
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    n_points: int


def _create_segments(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    segment_by: str,
) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if segment_by == "quarter":
        df["_segment"] = df[date_col].dt.to_period("Q").astype(str)
    elif segment_by == "month":
        df["_segment"] = df[date_col].dt.to_period("M").astype(str)
    elif segment_by == "weekday":
        df["_segment"] = df[date_col].dt.day_name()
    elif segment_by == "value_bin":
        try:
            df["_segment"] = pd.qcut(
                df[metric_col], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
            ).astype(str)
        except ValueError:
            df["_segment"] = pd.cut(
                df[metric_col], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
            ).astype(str)
    else:
        raise ValueError(f"Unknown segment_by: {segment_by}")

    return df


def run_subgroup_analysis(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: Method = Method.ARIMA,
    segment_by: str = "quarter",
) -> list[SubgroupResult]:
    df = _create_segments(df, date_col, metric_col, segment_by)

    results = []
    for segment_name, group_df in df.groupby("_segment"):
        if len(group_df) < MIN_SEGMENT_SIZE:
            logger.warning(
                f"Segment '{segment_name}' has only {len(group_df)} points, "
                f"need {MIN_SEGMENT_SIZE}. Skipping."
            )
            continue

        try:
            result = causal_effect(
                group_df,
                date_col=date_col,
                metric_col=metric_col,
                intervention_date=intervention_date,
                method=method,
            )
            results.append(SubgroupResult(
                segment=segment_name,
                effect=result.effect,
                effect_pct=result.effect_pct,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                p_value=result.p_value,
                significant=result.significant,
                n_points=len(group_df),
            ))
        except Exception as e:
            logger.warning(f"Analysis failed for segment '{segment_name}': {e}")

    results.sort(key=lambda r: abs(r.effect), reverse=True)
    return results
