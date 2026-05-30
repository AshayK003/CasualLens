from __future__ import annotations

import plotly.graph_objects as go

# Dark mode color palette
COLORS = {
    "observed": "#e2e8f0",
    "fitted": "#818cf8",
    "counterfactual": "#f87171",
    "intervention": "#fbbf24",
    "ci": "rgba(248,113,113,0.12)",
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(148,163,184,0.1)",
    "text": "#94a3b8",
}


def build_counterfactual_plot(
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
        line=dict(color=COLORS["observed"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=dates[:intervention_idx], y=counterfactual[:intervention_idx],
        mode="lines", name="Fitted (pre-intervention)",
        line=dict(color=COLORS["fitted"], width=1, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=dates[intervention_idx:], y=counterfactual[intervention_idx:],
        mode="lines", name="Counterfactual (without policy)",
        line=dict(color=COLORS["counterfactual"], width=2, dash="dash"),
    ))

    # Confidence interval shading on post-intervention period
    if ci_lower is not None and ci_upper is not None:
        post_dates = dates[intervention_idx:]
        post_lower = ci_lower[intervention_idx:]
        post_upper = ci_upper[intervention_idx:]
        fig.add_trace(go.Scatter(
            x=post_dates + post_dates[::-1],
            y=post_upper + post_lower[::-1],
            fill="toself",
            fillcolor=COLORS["ci"],
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
        line=dict(color=COLORS["intervention"], width=2, dash="dot"),
    )
    fig.add_annotation(
        x=intervention_date, y=1, yref="paper",
        text="Intervention", showarrow=False,
        font=dict(size=11, color=COLORS["intervention"]),
        yshift=10,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["text"]),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text"]),
        ),
    )

    return fig
