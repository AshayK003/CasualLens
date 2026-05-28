from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.core.engine import causal_effect, Method
from src.core.placebo import run_placebo_test
from src.data.loader import get_available_datasets, load_dataset, load_user_csv
from src.reports.pdf_export import generate_pdf_report
from src.reports.plots import build_counterfactual_plot
from src.reports.summary import generate_summary
from src.utils.validators import validate_dataframe

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="CausalLens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
        font-size: 1.1rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.08) 100%);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        opacity: 0.7;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.25);
    }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cache ─────────────────────────────────────────────────────
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


# ─── Results Display ───────────────────────────────────────────
def show_results(result_dict: dict, metric_col: str):
    effect = result_dict["effect"]
    effect_pct = result_dict["effect_pct"]
    p_value = result_dict["p_value"]
    significant = result_dict["significant"]

    # ── Metric cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_color = "normal" if effect > 0 else "inverse"
        st.metric(
            "Effect Size",
            f"{effect:+.2f}",
            delta=f"{effect_pct:+.1f}%",
            delta_color=delta_color,
        )
    with col2:
        st.metric("p-value", f"{p_value:.4f}")
    with col3:
        sig_color = "normal" if significant else "off"
        st.metric(
            "Statistical Significance",
            "Yes" if significant else "No",
            delta="p < 0.05" if significant else "p >= 0.05",
            delta_color=sig_color,
        )
    with col4:
        st.metric(
            "Sample Size",
            f"{result_dict['n_pre']} / {result_dict['n_post']}",
            delta="pre / post",
            delta_color="off",
        )

    # ── Visualization + Summary tabs ──
    tab_chart, tab_summary, tab_download = st.tabs(
        [" Chart", " Interpretation", " Export"]
    )

    with tab_chart:
        fig = build_counterfactual_plot(
            dates=result_dict["dates"],
            observed=result_dict["observed"],
            counterfactual=result_dict["counterfactual"],
            intervention_idx=result_dict["intervention_idx"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f" **Black line** = observed data  |  "
            f" **Red dashed** = counterfactual (what would have happened without the policy)  |  "
            f" **Orange line** = intervention date"
        )

    with tab_summary:
        summary_text = generate_summary(
            effect=result_dict["effect"],
            effect_pct=result_dict["effect_pct"],
            ci_lower=result_dict["ci_lower"],
            ci_upper=result_dict["ci_upper"],
            p_value=result_dict["p_value"],
            significant=result_dict["significant"],
            direction=result_dict["direction"],
            metric_name=metric_col,
        )

        if significant:
            st.success(summary_text)
        else:
            st.warning(summary_text)

        with st.expander("Statistical Details", expanded=False):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Method | `{result_dict['method'].upper()}` |
| Effect | `{effect:+.4f}` |
| Effect % | `{effect_pct:+.2f}%` |
| 95% CI | `[{result_dict['ci_lower']:.4f}, {result_dict['ci_upper']:.4f}]` |
""")
            with detail_col2:
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| p-value | `{p_value:.6f}` |
| Significant | {'Yes' if significant else 'No'} |
| Direction | {result_dict['direction']} |
| Pre/Post | {result_dict['n_pre']} / {result_dict['n_post']} |
""")

    with tab_download:
        pdf_bytes = generate_pdf_report(
            dates=result_dict["dates"],
            observed=result_dict["observed"],
            counterfactual=result_dict["counterfactual"],
            intervention_idx=result_dict["intervention_idx"],
            effect=result_dict["effect"],
            effect_pct=result_dict["effect_pct"],
            ci_lower=result_dict["ci_lower"],
            ci_upper=result_dict["ci_upper"],
            p_value=result_dict["p_value"],
            significant=result_dict["significant"],
            direction=result_dict["direction"],
            metric_name=metric_col,
            method=result_dict["method"],
        )

        st.markdown("### Download Report")
        st.markdown(
            "Export a PDF report with charts, statistics, and interpretation."
        )
        st.download_button(
            label="  Download PDF Report",
            data=pdf_bytes,
            file_name="causal_impact_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        csv_data = pd.DataFrame({
            "date": result_dict["dates"],
            "observed": result_dict["observed"],
            "counterfactual": result_dict["counterfactual"],
        })
        st.download_button(
            label="  Download CSV Data",
            data=csv_data.to_csv(index=False),
            file_name="causal_impact_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─── Sidebar ───────────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("#  CausalLens")
        st.caption("Causal Impact Calculator")

        st.divider()

        # ── Data Source ──
        st.markdown("###  1. Data Source")
        source = st.radio(
            "Choose data source",
            ["Upload CSV", "Pre-loaded dataset"],
            label_visibility="collapsed",
        )

        df = None
        date_col = None
        metric_col = None
        default_intervention = None

        if source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="CSV must have at least a date column and a numeric column",
            )
            if uploaded_file is not None:
                try:
                    df = load_user_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                except ValueError as e:
                    st.error(str(e))

            if df is not None:
                try:
                    date_col, metric_col = validate_dataframe(df)
                    st.info(f" **Date:** `{date_col}` | **Metric:** `{metric_col}`")
                except ValueError as e:
                    st.error(str(e))
                    df = None
        else:
            datasets = get_available_datasets()
            selected = st.selectbox(
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
                    st.info(meta["description"])
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))

        st.divider()

        # ── Analysis Settings ──
        if df is not None:
            st.markdown("###  2. Analysis Settings")

            min_date = pd.to_datetime(df[date_col]).min().date()
            max_date = pd.to_datetime(df[date_col]).max().date()

            default_date = (
                pd.to_datetime(default_intervention).date()
                if default_intervention
                else min_date + (max_date - min_date) // 2
            )

            intervention_date = st.date_input(
                "Intervention date",
                value=default_date,
                min_value=min_date,
                max_value=max_date,
                help="The date when the policy/intervention was implemented",
            )

            method = st.selectbox(
                "Analysis method",
                options=["arima", "bsts"],
                format_func=lambda x: {
                    "arima": "ARIMA ITS (fast, recommended)",
                    "bsts": "Bayesian STS (slower, experimental)",
                }.get(x, x),
            )

            run_placebo = st.checkbox(
                "Run placebo sensitivity test",
                value=False,
                help="Tests if the effect is robust by comparing to fake intervention dates",
            )

            st.divider()

            st.markdown("###  3. Run")
            run_clicked = st.button(
                "▶  Run Analysis",
                type="primary",
                use_container_width=True,
            )

            return df, date_col, metric_col, intervention_date, method, run_placebo, run_clicked

        return None, None, None, None, None, None, False


# ─── Main ──────────────────────────────────────────────────────
def main():
    (
        df,
        date_col,
        metric_col,
        intervention_date,
        method,
        run_placebo,
        run_clicked,
    ) = build_sidebar()

    # ── Empty state ──
    if df is None:
        st.markdown("""
        #  Welcome to CausalLens

        **Did this policy actually work?**

        CausalLens estimates the causal effect of any policy intervention
        on a time series using counterfactual analysis.

        ---

        ### How it works
        1. **Upload data** or pick a pre-loaded dataset from the sidebar
        2. **Pick the intervention date** — when the policy changed
        3. **Run the analysis** — get a counterfactual chart and statistics
        4. **Download the report** — PDF with charts and interpretation

        ---

        ### Quick Start
        - Select **Pre-loaded dataset** in the sidebar
        - Choose **Delhi Air Quality** (tests the 2020 lockdown effect)
        - Click **Run Analysis**
        """)

        # Feature cards
        c1, c2, c3 = st.columns(3)
        card_style = "padding:20px; border-radius:12px; border:1px solid rgba(148,163,184,0.25); background:rgba(148,163,184,0.08);"
        with c1:
            st.markdown(f"""
            <div style="{card_style}">
                <h3 style="margin:0; font-size:1.1rem;"> </h3>
                <p style="margin:8px 0 0; font-size:0.85rem; opacity:0.7;">
                    See what would have happened without the policy
                </p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="{card_style}">
                <h3 style="margin:0; font-size:1.1rem;"> </h3>
                <p style="margin:8px 0 0; font-size:0.85rem; opacity:0.7;">
                    Confidence intervals, p-values, effect sizes
                </p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="{card_style}">
                <h3 style="margin:0; font-size:1.1rem;"> </h3>
                <p style="margin:8px 0 0; font-size:0.85rem; opacity:0.7;">
                    Download PDF reports and CSV data
                </p>
            </div>
            """, unsafe_allow_html=True)

        return

    # ── Data loaded — show preview ──
    st.markdown(f"###  Data Preview: `{metric_col}` over time")

    preview_fig = go.Figure()
    preview_fig.add_trace(go.Scatter(
        x=df[date_col].astype(str).tolist(),
        y=df[metric_col].tolist(),
        mode="lines",
        name=metric_col,
        line=dict(color="#6366f1", width=1.5),
    ))
    preview_fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric_col,
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(preview_fig, use_container_width=True)

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Rows", f"{len(df):,}")
    with stats_col2:
        min_date_str = pd.to_datetime(df[date_col]).min().date()
        max_date_str = pd.to_datetime(df[date_col]).max().date()
        st.metric("Date Range", f"{min_date_str} → {max_date_str}")
    with stats_col3:
        st.metric("Mean", f"{df[metric_col].mean():.2f}")
    with stats_col4:
        st.metric("Std Dev", f"{df[metric_col].std():.2f}")

    # ── Run analysis ──
    if run_clicked:
        intervention_str = intervention_date.strftime("%Y-%m-%d")

        with st.spinner("Running causal analysis..."):
            try:
                df_json = df.to_json(orient="split")
                result_dict = run_cached_analysis(
                    df_json, date_col, metric_col, intervention_str, method
                )
                st.session_state["result"] = result_dict
                st.session_state["metric_col"] = metric_col
                st.session_state["df_raw"] = df
            except Exception as e:
                st.error(f"Analysis failed: `{e}`")
                return

        if run_placebo:
            with st.spinner("Running placebo sensitivity test..."):
                try:
                    y = df[metric_col].values
                    idx = result_dict["intervention_idx"]
                    placebo_result = run_placebo_test(y, idx, n_placebos=5)
                    st.session_state["placebo"] = placebo_result
                except Exception as e:
                    st.warning(f"Placebo test failed: `{e}`")

    # ── Show results ──
    if "result" in st.session_state:
        st.divider()
        show_results(st.session_state["result"], st.session_state["metric_col"])

    # ── Placebo results ──
    if "placebo" in st.session_state:
        st.divider()
        st.markdown("###  Sensitivity Analysis (Placebo Test)")
        pb = st.session_state["placebo"]

        pb_col1, pb_col2, pb_col3 = st.columns(3)
        with pb_col1:
            st.metric("Real Effect", f"{pb['real_effect']:.2f}")
        with pb_col2:
            st.metric("Placebo p-value", f"{pb['p_value']:.3f}")
        with pb_col3:
            st.metric(
                "Robust",
                "Yes" if pb["is_real_effect_extreme"] else "Uncertain",
            )

        placebo_fig = go.Figure()
        placebo_fig.add_trace(go.Bar(
            x=[f"Placebo {i+1}" for i in range(len(pb["placebo_effects"]))],
            y=pb["placebo_effects"],
            name="Placebo Effects",
            marker_color="#94a3b8",
        ))
        placebo_fig.add_trace(go.Bar(
            x=["Real"],
            y=[pb["real_effect"]],
            name="Real Effect",
            marker_color="#6366f1",
        ))
        placebo_fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
        )
        st.plotly_chart(placebo_fig, use_container_width=True)

        if pb["is_real_effect_extreme"]:
            st.success(
                "The real effect is extreme compared to placebo effects — result is **robust**."
            )
        else:
            st.warning(
                "The real effect is not extreme compared to placebo effects — **interpret with caution**."
            )

    # ── Raw Data ──
    st.divider()
    with st.expander("  View Raw Data", expanded=False):
        raw_df = st.session_state.get("df_raw")
        if raw_df is not None:
            tab_view, tab_stats, tab_download = st.tabs(["Data", "Statistics", "Download"])

            with tab_view:
                st.dataframe(
                    raw_df,
                    use_container_width=True,
                    height=min(400, 35 * len(raw_df) + 40),
                )

            with tab_stats:
                st.markdown("#### Descriptive Statistics")
                st.dataframe(
                    raw_df.describe(),
                    use_container_width=True,
                )

                st.markdown("#### Data Types")
                dtypes_df = pd.DataFrame({
                    "Column": raw_df.columns,
                    "Type": [str(t) for t in raw_df.dtypes],
                    "Non-Null": [raw_df[c].notna().sum() for c in raw_df.columns],
                    "Null": [raw_df[c].isna().sum() for c in raw_df.columns],
                })
                st.dataframe(dtypes_df, use_container_width=True)

            with tab_download:
                csv = raw_df.to_csv(index=False)
                st.download_button(
                    label="  Download Raw Data as CSV",
                    data=csv,
                    file_name="raw_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
