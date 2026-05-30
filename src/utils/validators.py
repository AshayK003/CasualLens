from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MIN_DATA_POINTS_RATIO


def validate_dataframe(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns (date and metric)")

    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                break
        except (ValueError, TypeError):
            continue

    if date_col is None:
        raise ValueError(
            "No date column found. Ensure one column has parseable dates."
        )

    numeric_cols = [c for c in df.columns if c != date_col]
    metric_col = None
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_col = col
            break
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                metric_col = col
                break
        except (ValueError, TypeError):
            continue

    if metric_col is None:
        raise ValueError(
            "No numeric metric column found. Ensure one column has numeric values."
        )

    return date_col, metric_col


def validate_intervention_date(
    dates: pd.DatetimeIndex, intervention_date: str
) -> int:
    try:
        intervention_dt = pd.to_datetime(str(intervention_date))
    except Exception:
        raise ValueError(f"Cannot parse intervention date: {intervention_date}")

    if pd.isna(intervention_dt):
        raise ValueError(f"Cannot parse intervention date: {intervention_date}")

    # Ensure dates are unique before indexing
    if not dates.is_unique:
        dates = pd.DatetimeIndex(dates.unique())

    idx = dates.get_indexer([intervention_dt], method="nearest")[0]

    n = len(dates)

    if idx < 0 or idx >= n:
        raise ValueError(
            f"Intervention date {intervention_date} is outside the data range "
            f"({dates[0].date()} to {dates[-1].date()})."
        )

    min_before = max(3, int(n * MIN_DATA_POINTS_RATIO))
    min_after = max(3, int(n * MIN_DATA_POINTS_RATIO))
    if idx < min_before:
        raise ValueError(
            f"Intervention date is too early ({idx}/{n} data points before). "
            f"Need at least {min_before} data points before intervention."
        )
    if n - idx - 1 < min_after:
        raise ValueError(
            f"Intervention date is too late ({n - idx}/{n} data points after). "
            f"Need at least {min_after} data points after intervention."
        )

    return int(idx)


def validate_series_length(y: np.ndarray, min_length: int = 30) -> None:
    if len(y) < min_length:
        raise ValueError(
            f"Time series has only {len(y)} data points, but at least {min_length} are needed. "
            f"Upload a dataset with more rows, or use a pre-loaded dataset."
        )
