from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio

from ..utils.formatters import format_ci, format_effect, format_effect_pct, format_p_value


def _build_counterfactual_chart(
    dates: list[str],
    observed: list[float],
    counterfactual: list[float],
    intervention_idx: int,
    ci_lower: list[float] | None = None,
    ci_upper: list[float] | None = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=observed,
        mode="lines", name="Observed",
        line=dict(color="#e2e8f0", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=dates[:intervention_idx], y=counterfactual[:intervention_idx],
        mode="lines", name="Fitted (pre-intervention)",
        line=dict(color="#818cf8", width=1, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=dates[intervention_idx:], y=counterfactual[intervention_idx:],
        mode="lines", name="Counterfactual",
        line=dict(color="#f87171", width=2, dash="dash"),
    ))

    if ci_lower is not None and ci_upper is not None:
        post_dates = dates[intervention_idx:]
        post_lower = ci_lower[intervention_idx:]
        post_upper = ci_upper[intervention_idx:]
        fig.add_trace(go.Scatter(
            x=post_dates + post_dates[::-1],
            y=post_upper + post_lower[::-1],
            fill="toself",
            fillcolor="rgba(248,113,113,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

    intervention_date = dates[intervention_idx]
    fig.add_shape(
        type="line",
        x0=intervention_date, x1=intervention_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="#fbbf24", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=intervention_date, y=1, yref="paper",
        text="Intervention", showarrow=False,
        font=dict(size=11, color="#fbbf24"),
        yshift=10,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", tickfont=dict(color="#94a3b8")),
    )
    return fig


def _build_effect_chart(
    dates: list[str],
    observed: list[float],
    counterfactual: list[float],
) -> go.Figure:
    import numpy as np

    effects = np.array(observed) - np.array(counterfactual)
    colors = ["#22c55e" if e > 0 else "#ef4444" for e in effects]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=effects.tolist(),
        name="Effect",
        marker_color=colors,
        opacity=0.7,
    ))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Effect (Observed - Counterfactual)",
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", tickfont=dict(color="#94a3b8")),
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CausalLens Report — {metric_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0e1117; color: #e2e8f0;
    padding: 2rem; line-height: 1.6;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; color: #fafafa; margin-bottom: 0.25rem; letter-spacing: -0.03em; }}
  h2 {{ font-size: 1.2rem; color: #94a3b8; margin: 2rem 0 0.75rem; font-weight: 500; }}
  .subtitle {{ color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .summary {{
    background: #1a1d24; border: 1px solid rgba(148,163,184,0.2);
    border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;
  }}
  .summary table {{ width: 100%; border-collapse: collapse; }}
  .summary td, .summary th {{
    padding: 0.5rem 0.75rem; text-align: left;
    border-bottom: 1px solid rgba(148,163,184,0.1);
  }}
  .summary th {{ color: #94a3b8; font-weight: 500; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary td {{ color: #e2e8f0; }}
  .sig {{ color: #22c55e; font-weight: 600; }}
  .not-sig {{ color: #f59e0b; font-weight: 600; }}
  .chart {{ margin: 1.5rem 0; }}
  .interpretation {{
    background: #1a1d24; border: 1px solid rgba(148,163,184,0.2);
    border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;
    font-size: 0.95rem;
  }}
  footer {{
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid rgba(148,163,184,0.15);
    color: #64748b; font-size: 0.8rem;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>CausalLens — Causal Impact Report</h1>
  <p class="subtitle">Generated {report_date}</p>

  <div class="summary">
    <table>
      <tr><th>Metric</th><td>{metric_name}</td></tr>
      <tr><th>Method</th><td>{method}</td></tr>
      <tr><th>Intervention</th><td>{intervention_date} (index {intervention_idx})</td></tr>
      <tr><th>Result</th><td class="{sig_class}">{sig_text}</td></tr>
      <tr><th>Effect</th><td>{effect_str} ({effect_pct_str})</td></tr>
      <tr><th>95% CI</th><td>{ci_str}</td></tr>
      <tr><th>p-value</th><td>{p_value_str}</td></tr>
      <tr><th>Sample</th><td>{n_pre} pre / {n_post} post</td></tr>
    </table>
  </div>

  <h2>Counterfactual Analysis</h2>
  <div class="chart">
    {counterfactual_chart}
  </div>

  <h2>Pointwise Effect</h2>
  <div class="chart">
    {effect_chart}
  </div>

  <h2>Interpretation</h2>
  <div class="interpretation">
    <p>{interpretation}</p>
  </div>

  <footer>
    Generated by <strong>CausalLens</strong> &mdash; Causal Impact Calculator
  </footer>
</div>
</body>
</html>
"""


def generate_html_report(
    dates: list[str],
    observed: list[float],
    counterfactual: list[float],
    intervention_idx: int,
    effect: float,
    effect_pct: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    significant: bool,
    direction: str,
    metric_name: str,
    method: str,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> str:

    if n_pre is None:
        n_pre = intervention_idx
    if n_post is None:
        n_post = len(dates) - intervention_idx

    ci_lower_list = [ci_lower] * len(dates)
    ci_upper_list = [ci_upper] * len(dates)

    cf_fig = _build_counterfactual_chart(dates, observed, counterfactual, intervention_idx, ci_lower_list, ci_upper_list)
    eff_fig = _build_effect_chart(dates, observed, counterfactual)

    cf_html = pio.to_html(cf_fig, include_plotlyjs="cdn", full_html=False)
    eff_html = pio.to_html(eff_fig, include_plotlyjs=False, full_html=False)

    verb = "increase" if direction == "increase" else "decrease"
    if significant:
        interpretation = (
            f"The intervention caused a <strong>{format_effect_pct(abs(effect_pct))} {verb}</strong> "
            f"in {metric_name}. This effect is <strong>statistically significant</strong> "
            f"(p={format_p_value(p_value)}, 95% CI {format_ci(ci_lower, ci_upper)})."
        )
        sig_class = "sig"
        sig_text = "Statistically Significant"
    else:
        interpretation = (
            f"The intervention did not produce a statistically significant effect on {metric_name} "
            f"(p={format_p_value(p_value)}). The observed change ({direction}) may be due to random variation."
        )
        sig_class = "not-sig"
        sig_text = "Not Statistically Significant"

    return HTML_TEMPLATE.format(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        metric_name=metric_name,
        method=method.upper(),
        intervention_date=dates[intervention_idx],
        intervention_idx=intervention_idx,
        sig_class=sig_class,
        sig_text=sig_text,
        effect_str=format_effect(effect),
        effect_pct_str=format_effect_pct(effect_pct),
        ci_str=format_ci(ci_lower, ci_upper),
        p_value_str=format_p_value(p_value),
        n_pre=n_pre,
        n_post=n_post,
        counterfactual_chart=cf_html,
        effect_chart=eff_html,
        interpretation=interpretation,
    )
