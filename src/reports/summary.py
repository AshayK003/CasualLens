from __future__ import annotations

from ..utils.formatters import format_ci, format_effect_pct, format_p_value


def generate_summary(
    effect: float,
    effect_pct: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    significant: bool,
    direction: str,
    metric_name: str = "the metric",
) -> str:
    ci_str = format_ci(ci_lower, ci_upper)
    p_str = format_p_value(p_value)
    pct_str = format_effect_pct(abs(effect_pct))

    if significant:
        verb = "increase" if direction == "increase" else "decrease"
        return (
            f"The intervention caused a **{pct_str} {verb}** "
            f"in {metric_name} (95% CI {ci_str}, "
            f"p={p_str}). This effect is **statistically significant**."
        )
    return (
        f"The intervention did not produce a statistically significant "
        f"effect on {metric_name} (p={p_str}). "
        f"The observed change ({direction}) may be due to random variation."
    )
