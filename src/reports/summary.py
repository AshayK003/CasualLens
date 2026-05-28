from __future__ import annotations


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
    if significant:
        if direction == "increase":
            return (
                f"The intervention caused a **{abs(effect_pct):.1f}% increase** "
                f"in {metric_name} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
                f"p={p_value:.4f}). This effect is **statistically significant**."
            )
        else:
            return (
                f"The intervention caused a **{abs(effect_pct):.1f}% decrease** "
                f"in {metric_name} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
                f"p={p_value:.4f}). This effect is **statistically significant**."
            )
    else:
        return (
            f"The intervention did not produce a statistically significant "
            f"effect on {metric_name} (p={p_value:.4f}). "
            f"The observed change ({direction}) may be due to random variation."
        )
