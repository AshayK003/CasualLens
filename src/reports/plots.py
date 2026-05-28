from __future__ import annotations

import plotly.graph_objects as go


def build_counterfactual_plot(
    dates: list[str],
    observed: list[float],
    counterfactual: list[float],
    intervention_idx: int,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=observed,
        mode="lines", name="Observed",
        line=dict(color="#1e293b", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=dates[:intervention_idx], y=counterfactual[:intervention_idx],
        mode="lines", name="Fitted (pre-intervention)",
        line=dict(color="#6366f1", width=1, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=dates[intervention_idx:], y=counterfactual[intervention_idx:],
        mode="lines", name="Counterfactual (without policy)",
        line=dict(color="#ef4444", width=2, dash="dash"),
    ))

    intervention_date = dates[intervention_idx]
    fig.add_shape(
        type="line",
        x0=intervention_date, x1=intervention_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=intervention_date, y=1, yref="paper",
        text="Intervention", showarrow=False,
        font=dict(size=11, color="#f59e0b"),
        yshift=10,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
    )

    return fig
