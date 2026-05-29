from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    original_rows: int
    original_cols: int
    final_rows: int = 0
    final_cols: int = 0
    steps_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    return col
            except Exception:
                continue
    return None


def detect_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > len(df) * 0.3:
                    numeric_cols.append(col)
            except Exception:
                continue
    return numeric_cols


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    new_cols = []
    for col in df.columns:
        new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
        new_name = "".join(c for c in new_name if c.isalnum() or c == "_")
        new_name = new_name.strip("_")
        if not new_name:
            new_name = f"col_{len(new_cols)}"
        if new_name != col:
            steps.append(f"Renamed '{col}' → '{new_name}'")
        new_cols.append(new_name)

    if steps:
        df = df.copy()
        df.columns = new_cols
    return df, steps


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} duplicate rows")
    return df, steps


def drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    n_before = len(df)
    df = df.dropna(how="all")
    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} completely empty rows")
    return df, steps


def drop_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        steps.append(f"Dropped empty columns: {', '.join(empty_cols)}")
        df = df.drop(columns=empty_cols)
    return df, steps


def parse_dates(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df, steps

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        n_failed = df[date_col].isna().sum()
        if n_failed > 0:
            steps.append(f"Parsed dates: {n_failed} values failed, dropped")
            df = df.dropna(subset=[date_col])
        else:
            steps.append(f"Parsed '{date_col}' as datetime")
    except Exception as e:
        steps.append(f"Date parsing failed: {e}")

    return df, steps


def parse_numeric(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if pd.api.types.is_numeric_dtype(df[col]):
        return df, steps

    try:
        df = df.copy()
        original = df[col].copy()
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce",
        )
        n_failed = df[col].isna().sum()
        n_original = original.notna().sum()
        if n_failed > 0 and n_original > 0:
            steps.append(
                f"Converted '{col}' to numeric: {n_failed}/{n_original} values became NaN"
            )
    except Exception as e:
        steps.append(f"Numeric conversion failed for '{col}': {e}")

    return df, steps


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if strategy == "drop":
        n_before = len(df)
        df = df.dropna()
        n_removed = n_before - len(df)
        if n_removed > 0:
            steps.append(f"Dropped {n_removed} rows with missing values")
    elif strategy == "forward_fill":
        n_filled = df.isna().sum().sum()
        df = df.ffill()
        df = df.bfill()
        steps.append(f"Forward-filled {int(n_filled)} missing values")
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                steps.append(f"Filled {n_missing} missing values in '{col}' with mean ({mean_val:.2f})")
    return df, steps


def remove_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df, steps

    n_before = len(df)

    if method == "iqr":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif method == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df = df[np.abs((df[col] - mean) / std) <= threshold]

    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} outliers from '{col}' ({method} method)")
    return df, steps


def preprocess_data(
    df: pd.DataFrame,
    date_col: str | None = None,
    metric_col: str | None = None,
    missing_strategy: str = "drop",
    remove_outliers_flag: bool = False,
    outlier_method: str = "iqr",
) -> tuple[pd.DataFrame, PreprocessReport]:
    report = PreprocessReport(
        original_rows=len(df),
        original_cols=len(df.columns),
    )

    df, steps = clean_column_names(df)
    report.steps_applied.extend(steps)

    df, steps = drop_empty_columns(df)
    report.steps_applied.extend(steps)

    df, steps = drop_empty_rows(df)
    report.steps_applied.extend(steps)

    df, steps = drop_duplicates(df)
    report.steps_applied.extend(steps)

    if date_col is None:
        date_col = detect_date_column(df)
        if date_col:
            report.warnings.append(f"Auto-detected date column: '{date_col}'")

    if date_col and date_col in df.columns:
        df, steps = parse_dates(df, date_col)
        report.steps_applied.extend(steps)

    if metric_col is None:
        exclude = [date_col] if date_col else []
        numeric_cols = detect_numeric_columns(df, exclude=exclude)
        if numeric_cols:
            metric_col = numeric_cols[0]
            report.warnings.append(f"Auto-detected metric column: '{metric_col}'")

    if metric_col and metric_col in df.columns:
        df, steps = parse_numeric(df, metric_col)
        report.steps_applied.extend(steps)

    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    df, steps = handle_missing_values(df, strategy=missing_strategy)
    report.steps_applied.extend(steps)

    if remove_outliers_flag and metric_col and metric_col in df.columns:
        df, steps = remove_outliers(df, metric_col, method=outlier_method)
        report.steps_applied.extend(steps)

    report.final_rows = len(df)
    report.final_cols = len(df.columns)

    logger.info(
        f"Preprocessing complete: {report.original_rows} → {report.final_rows} rows, "
        f"{len(report.steps_applied)} steps applied"
    )

    return df, report
