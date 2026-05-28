from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.core.engine import CausalResult, causal_effect, Method
from src.data.loader import get_available_datasets, load_dataset, load_user_csv
from src.utils.validators import validate_dataframe

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="CausalLens",
    page_icon="",
    layout="wide",
)

st.title("CausalLens")
st.caption("Causal Impact Calculator — Did this policy actually work?")


@st.cache_data
def run_cached_analysis(
    df_json: str,
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: str,
) -> dict:
    df = pd.read_json(df_json, orient="split")
    result = causal_effect(
        df=df,
        date_col=date_col,
        metric_col=metric_col,
        intervention_date=intervention_date,
        method=Method(method),
    )
    return {
        "method": result.method,
        "effect": result.effect,
        "effect_pct": result.effect_pct,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_value": result.p_value,
        "significant": result.significant,
        "direction": result.direction,
        "counterfactual": result.counterfactual.tolist(),
        "fitted_values": result.fitted_values.tolist(),
        "observed": result.observed.tolist(),
        "intervention_idx": result.intervention_idx,
        "dates": result.dates,
        "n_pre": result.n_pre,
        "n_post": result.n_post,
    }


def build_plot(result_dict: dict) -> go.Figure:
    dates = result_dict["dates"]
    observed = result_dict["observed"]
    counterfactual = result_dict["counterfactual"]
    idx = result_dict["intervention_idx"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=observed,
        mode="lines", name="Observed",
        line=dict(color="#1e293b", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=dates[:idx], y=counterfactual[:idx],
        mode="lines", name="Fitted (pre-intervention)",
        line=dict(color="#6366f1", width=1, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=dates[idx:], y=counterfactual[idx:],
        mode="lines", name="Counterfactual (predicted without policy)",
        line=dict(color="#ef4444", width=2, dash="dash"),
    ))

    intervention_date = dates[idx]
    fig.add_vline(
        x=intervention_date, line_dash="dot", line_color="#f59e0b",
        annotation_text="Intervention", annotation_position="top",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
    )

    return fig


def main():
    st.sidebar.header("Data Source")

    source = st.sidebar.radio(
        "Choose data source",
        ["Upload CSV", "Pre-loaded dataset"],
    )

    df = None
    date_col = None
    metric_col = None
    default_intervention = None

    if source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file", type=["csv"]
        )
        if uploaded_file is not None:
            try:
                df = load_user_csv(uploaded_file)
                st.sidebar.success(f"Loaded {len(df)} rows")
            except ValueError as e:
                st.sidebar.error(str(e))
                return

        if df is not None:
            try:
                date_col, metric_col = validate_dataframe(df)
                st.sidebar.info(f"Date column: `{date_col}` | Metric column: `{metric_col}`")
            except ValueError as e:
                st.sidebar.error(str(e))
                return

    else:
        datasets = get_available_datasets()
        selected = st.sidebar.selectbox(
            "Select dataset",
            options=list(datasets.keys()),
            format_func=lambda k: datasets[k]["label"],
        )

        if selected:
            try:
                df = load_dataset(selected)
                meta = datasets[selected]
                date_col = meta["date_col"]
                metric_col = meta["metric_col"]
                default_intervention = meta["intervention_date"]
                st.sidebar.info(meta["description"])
            except (ValueError, FileNotFoundError) as e:
                st.sidebar.error(str(e))
                return

    if df is None:
        st.info(
            "Upload a CSV or select a pre-loaded dataset from the sidebar to begin."
        )
        return

    st.sidebar.header("Analysis Settings")

    min_date = pd.to_datetime(df[date_col]).min().date()
    max_date = pd.to_datetime(df[date_col]).max().date()

    default_date = (
        pd.to_datetime(default_intervention).date()
        if default_intervention
        else min_date + (max_date - min_date) // 2
    )

    intervention_date = st.sidebar.date_input(
        "Intervention date (policy change)",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
    )

    method = st.sidebar.selectbox(
        "Method",
        options=["arima"],
        format_func=lambda x: {"arima": "ARIMA ITS (fast)"}.get(x, x),
    )

    if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
        intervention_str = intervention_date.strftime("%Y-%m-%d")

        with st.spinner("Running causal analysis..."):
            try:
                df_json = df.to_json(orient="split")
                result_dict = run_cached_analysis(
                    df_json, date_col, metric_col, intervention_str, method
                )
                st.session_state["result"] = result_dict
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    if "result" in st.session_state:
        result_dict = st.session_state["result"]
        show_results(result_dict, df, date_col, metric_col)


def show_results(result_dict: dict, df: pd.DataFrame, date_col: str, metric_col: str):
    st.header("Results")

    col1, col2, col3, col4 = st.columns(4)

    effect = result_dict["effect"]
    effect_pct = result_dict["effect_pct"]
    p_value = result_dict["p_value"]
    significant = result_dict["significant"]

    with col1:
        st.metric(
            "Effect",
            f"{effect:+.2f}",
            delta=f"{effect_pct:+.1f}%",
            delta_color="normal" if effect > 0 else "inverse",
        )
    with col2:
        st.metric("p-value", f"{p_value:.4f}")
    with col3:
        st.metric(
            "Significant",
            "Yes" if significant else "No",
            delta="p < 0.05" if significant else "p >= 0.05",
            delta_color="normal" if significant else "off",
        )
    with col4:
        st.metric(
            "Data Points",
            f"{result_dict['n_pre']} pre + {result_dict['n_post']} post",
        )

    st.subheader("Counterfactual Visualization")
    fig = build_plot(result_dict)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretation")
    direction = result_dict["direction"]
    if significant:
        if direction == "increase":
            st.success(
                f"The intervention caused a **{abs(effect_pct):.1f}% increase** "
                f"in {metric_col} (95% CI [{result_dict['ci_lower']:.2f}, "
                f"{result_dict['ci_upper']:.2f}], p={p_value:.4f}). "
                "This effect is **statistically significant**."
            )
        else:
            st.success(
                f"The intervention caused a **{abs(effect_pct):.1f}% decrease** "
                f"in {metric_col} (95% CI [{result_dict['ci_lower']:.2f}, "
                f"{result_dict['ci_upper']:.2f}], p={p_value:.4f}). "
                "This effect is **statistically significant**."
            )
    else:
        st.warning(
            f"The intervention did not produce a statistically significant "
            f"effect on {metric_col} (p={p_value:.4f}). "
            "The observed change may be due to random variation."
        )

    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
