from __future__ import annotations

SIGNIFICANCE_LEVEL: float = 0.05
"""P-value threshold for determining statistical significance."""

DEFAULT_CI_ALPHA: float = 0.05
"""Alpha level used to compute confidence intervals (95% CI)."""

MIN_DATA_POINTS_RATIO: float = 0.05
"""Minimum fraction of data points required before/after intervention."""

LARGE_DATASET_ROWS: int = 5_000
"""Row count above which analysis may be slow; UI shows a warning."""
