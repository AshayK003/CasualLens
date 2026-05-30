from __future__ import annotations


def format_effect(value: float) -> str:
    """Format effect size with sign (e.g. +1.23, -0.45)."""
    return f"{value:+.2f}"


def format_effect_pct(value: float) -> str:
    """Format effect percentage with sign (e.g. +12.3%).

    Note: Callers should pass absolute values if they want unsigned display.
    This function always adds a +/- sign.
    """
    return f"{value:+.1f}%"


def format_p_value(value: float, *, decimals: int = 4) -> str:
    """Format p-value to given decimal places."""
    return f"{value:.{decimals}f}"


def format_ci(lower: float, upper: float) -> str:
    """Format a 95% confidence interval (e.g. [1.23, 4.56])."""
    return f"[{lower:.2f}, {upper:.2f}]"


def format_sample_size(n_pre: int, n_post: int) -> str:
    """Format pre/post sample sizes (e.g. 80 / 20)."""
    return f"{n_pre} / {n_post}"


def format_stat_summary(
    effect: float,
    effect_pct: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
) -> str:
    """One-line statistical summary for logging or display."""
    return (
        f"Effect: {format_effect(effect)} ({format_effect_pct(effect_pct)}), "
        f"95% CI {format_ci(ci_lower, ci_upper)}, "
        f"p={format_p_value(p_value)}"
    )
