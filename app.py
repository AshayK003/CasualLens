from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
# Required for Streamlit Cloud which runs without pip install.
# Remove after migrating to a proper package build.
sys.path.insert(0, str(ROOT))

from src.core.engine import Method, causal_effect
from src.core.placebo import run_placebo_test
from src.data.loader import get_available_datasets, load_dataset, load_uploaded_file
from src.data.preprocessor import detect_date_column, detect_numeric_columns, preprocess_data
from src.reports.pdf_export import generate_pdf_report
from src.reports.plots import build_counterfactual_plot
from src.reports.summary import generate_summary
from src.utils.constants import LARGE_DATASET_ROWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CausalLens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SECURITY: unsafe_allow_html is used ONLY for static CSS below.
# NEVER interpolate user-controlled strings into HTML blocks.
st.markdown("""
<style>
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
        font-size: 1.3rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(168,85,247,0.06) 100%);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99,102,241,0.4);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ── DataFrames ── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.2);
    }

    /* ── Download buttons ── */
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* ── Focus indicators for keyboard navigation ── */
    button:focus-visible,
    [data-baseweb="tab"]:focus-visible,
    [data-baseweb="select"] input:focus-visible,
    [data-baseweb="radio"] input:focus-visible,
    [data-baseweb="checkbox"] input:focus-visible {
        outline: 2px solid #6366f1;
        outline-offset: 2px;
    }

    /* ── Tighter top padding ── */
    .block-container {
        padding-top: 2rem !important;
    }

    /* ── Hero heading ── */
    .stMarkdown h1 {
        letter-spacing: -0.03em;
    }

    /* ── Narrative section ── */
    .stMarkdown strong {
        color: #e2e8f0;
    }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        font-weight: 500;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        div[data-testid="stMetric"] {
            padding: 8px 12px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─── Cache ─────────────────────────────────────────────────────
@st.cache_data
def run_cached_analysis(
    df_hash: int,
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: str,
) -> dict:
    _df = st.session_state.get("_df_cache")
    if _df is None:
        return {}
    result = causal_effect(
        df=_df,
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
        "arima_order": result.arima_order,
        "aic": result.aic,
        "ljung_box_pvalue": result.ljung_box_pvalue,
        "residuals_ok": result.residuals_ok,
    }


@st.cache_data
def _cached_pdf_report(
    result_hash: int,
    dates: tuple[str, ...],
    observed: tuple[float, ...],
    counterfactual: tuple[float, ...],
    intervention_idx: int,
    effect: float,
    effect_pct: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    significant: bool,
    direction: str,
    metric_col: str,
    method: str,
) -> bytes:
    return generate_pdf_report(
        dates=list(dates),
        observed=list(observed),
        counterfactual=list(counterfactual),
        intervention_idx=intervention_idx,
        effect=effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=significant,
        direction=direction,
        metric_name=metric_col,
        method=method,
    )


def _format_optional_float(value: float | None, fmt: str = ".1f") -> str:
    if value is None:
        return "N/A"
    return format(value, fmt)


def _format_arima_order(order: tuple | None) -> str:
    if order is None:
        return "N/A"
    return str(order)


def _format_residuals_status(residuals_ok: bool | None) -> str:
    if residuals_ok is None:
        return "N/A"
    return "White noise" if residuals_ok else "Check model"


def _generate_narrative(result_dict: dict, metric_col: str) -> str:
    """Generate a step-by-step narrative interpretation of the results."""
    effect = result_dict["effect"]
    effect_pct = result_dict["effect_pct"]
    p_value = result_dict["p_value"]
    significant = result_dict["significant"]
    direction = result_dict["direction"]
    n_pre = result_dict["n_pre"]
    n_post = result_dict["n_post"]
    ci_lower = result_dict["ci_lower"]
    ci_upper = result_dict["ci_upper"]
    intervention_date = result_dict["dates"][result_dict["intervention_idx"]]

    verb = "increased" if direction == "increase" else "decreased"
    abs_pct = abs(effect_pct)

    lines = [
        f"**On {intervention_date}**, an intervention was applied.",
        "",
    ]

    if significant:
        lines.append(
            f"The {metric_col} **{verb} by {abs_pct:.1f}%** "
            f"compared to what would have happened without the intervention."
        )
    else:
        lines.append(
            f"The observed change in {metric_col} ({direction} of {abs(effect):.2f}) "
            f"**was not statistically significant** (p={p_value:.4f})."
        )

    lines.extend([
        "",
        "**Statistical details:**",
        f"- Effect size: {effect:+.4f} ({effect_pct:+.1f}%)",
        f"- 95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]",
        f"- p-value: {p_value:.4f} {'(significant)' if significant else '(not significant)'}",
        f"- Sample: {n_pre} pre-intervention, {n_post} post-intervention points",
    ])

    if result_dict.get("residuals_ok") is False:
        lines.append("")
        lines.append(
            "**Note:** Residual diagnostics suggest the model fit may not be ideal. "
            "Consider trying a different method or intervention date."
        )

    return "\n".join(lines)


# ─── Results Display ───────────────────────────────────────────
def show_results(result_dict: dict, metric_col: str):
    effect = result_dict["effect"]
    effect_pct = result_dict["effect_pct"]
    p_value = result_dict["p_value"]
    significant = result_dict["significant"]

    # ── Metric cards: Effect Size gets hero treatment, others are compact ──
    hero_col, stat_col1, stat_col2, stat_col3 = st.columns([3, 2, 2, 2])
    with hero_col:
        delta_color = "normal" if effect > 0 else "inverse"
        st.metric(
            "Effect Size",
            f"{effect:+.2f}",
            delta=f"{effect_pct:+.1f}%",
            delta_color=delta_color,
            help="Average difference between observed and counterfactual after intervention.",
        )
    with stat_col1:
        st.metric(
            "p-value",
            f"{p_value:.4f}",
            help="Probability of observing this effect by chance if there were no real impact.",
        )
    with stat_col2:
        sig_color = "normal" if significant else "off"
        st.metric(
            "Significant",
            "Yes" if significant else "No",
            delta="p < 0.05" if significant else "p >= 0.05",
            delta_color=sig_color,
            help="Whether the effect is statistically significant at the 5% level.",
        )
    with stat_col3:
        st.metric(
            "Sample Size",
            f"{result_dict['n_pre']} / {result_dict['n_post']}",
            delta="pre / post",
            delta_color="off",
            help="Number of data points before and after the intervention.",
        )

    # ── Visualization + Summary tabs ──
    tab_chart, tab_narrative, tab_summary, tab_export = st.tabs(
        ["Chart", "Summary", "Statistical Details", "Export"]
    )

    with tab_chart:
        fig = build_counterfactual_plot(
            dates=result_dict["dates"],
            observed=result_dict["observed"],
            counterfactual=result_dict["counterfactual"],
            intervention_idx=result_dict["intervention_idx"],
            ci_lower=[result_dict["ci_lower"]] * len(result_dict["dates"]),
            ci_upper=[result_dict["ci_upper"]] * len(result_dict["dates"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Solid line = observed data. "
            "Dashed red = counterfactual (what would have happened without the policy). "
            "Shaded area = 95% confidence interval. "
            "Dotted orange = intervention date."
        )

        with st.expander("How to read this chart", expanded=False):
            st.markdown("""
**Observed (solid line):** The actual measured values over time.

**Counterfactual (dashed red):** A statistical prediction of what would have happened *without* the policy. This is the "what if" scenario.

**The gap** between the observed and counterfactual lines is the policy's estimated effect. If the observed line is above the counterfactual, the policy increased the metric. If below, it decreased it.

**95% CI (shaded area):** The range within which the true counterfactual likely falls. A wider band means more uncertainty.

**Intervention line:** The date when the policy took effect. Data to the left is used to train the model; data to the right is where we measure the impact.
            """)

    with tab_narrative:
        narrative = _generate_narrative(result_dict, metric_col)
        st.markdown(narrative)

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
            aic_str = _format_optional_float(result_dict.get("aic"))
            lb_pval = result_dict.get("ljung_box_pvalue")
            lb_pval_str = _format_optional_float(lb_pval, ".4f")
            with detail_col1:
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Method | `{result_dict['method'].upper()}` |
| ARIMA Order | `{_format_arima_order(result_dict.get('arima_order'))}` |
| AIC | `{aic_str}` |
| Effect | `{effect:+.4f}` |
| Effect % | `{effect_pct:+.2f}%` |
| 95% CI | `[{result_dict['ci_lower']:.4f}, {result_dict['ci_upper']:.4f}]` |
""")
            with detail_col2:
                residuals_ok = result_dict.get("residuals_ok")
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| p-value | `{p_value:.6f}` |
| Significant | {'Yes' if significant else 'No'} |
| Direction | {result_dict['direction']} |
| Pre/Post | {result_dict['n_pre']} / {result_dict['n_post']} |
| Residuals | {_format_residuals_status(residuals_ok)} |
| Ljung-Box p | `{lb_pval_str}` |
""")

    with tab_export:
        st.markdown("### Download Report")
        st.markdown(
            "Export a PDF report with charts, statistics, and interpretation."
        )

        if st.button("Generate PDF Report", use_container_width=True, key="generate_pdf_btn"):
            st.session_state["pdf_requested"] = True

        if st.session_state.get("pdf_requested"):
            result_hash = hash((
                tuple(result_dict["dates"]),
                tuple(result_dict["observed"]),
                tuple(result_dict["counterfactual"]),
                result_dict["intervention_idx"],
                result_dict["effect"],
                result_dict["method"],
            ))
            with st.spinner("Generating PDF..."):
                pdf_bytes = _cached_pdf_report(
                    result_hash,
                    tuple(result_dict["dates"]),
                    tuple(result_dict["observed"]),
                    tuple(result_dict["counterfactual"]),
                    result_dict["intervention_idx"],
                    result_dict["effect"],
                    result_dict["effect_pct"],
                    result_dict["ci_lower"],
                    result_dict["ci_upper"],
                    result_dict["p_value"],
                    result_dict["significant"],
                    result_dict["direction"],
                    metric_col,
                    result_dict["method"],
                )

            st.download_button(
                label="Download PDF Report",
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
            label="Download CSV Data",
            data=csv_data.to_csv(index=False),
            file_name="causal_impact_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─── Sidebar ───────────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("## CausalLens")

        st.divider()

        # ── Data Source ──
        st.markdown("### Data Source")
        source_options = ["Upload File", "Pre-loaded dataset"]
        source_override = st.session_state.pop("_source_override", None)
        source_index = source_options.index(source_override) if source_override in source_options else 0
        source = st.radio(
            "Choose data source",
            source_options,
            index=source_index,
            label_visibility="collapsed",
            horizontal=True,
        )

        df = None
        date_col = None
        metric_col = None
        default_intervention = None
        preprocess_report = None

        if source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a data file",
                type=["csv", "xlsx", "xls"],
                help="CSV or Excel with at least a date column and a numeric column.",
            )
            if uploaded_file is not None:
                try:
                    df = load_uploaded_file(uploaded_file)
                    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                except ValueError as e:
                    st.error(str(e))

            if df is not None:
                with st.expander("Preprocessing options", expanded=False):
                    missing_strategy = st.selectbox(
                        "Handle missing values",
                        options=["drop", "forward_fill", "mean"],
                        index=2,
                        format_func=lambda x: {
                            "drop": "Drop rows with NaN",
                            "forward_fill": "Forward-fill gaps",
                            "mean": "Fill with column mean",
                        }.get(x, x),
                    )
                    remove_outliers = st.checkbox("Remove outliers (IQR)", value=False)

                df, preprocess_report = preprocess_data(
                    df,
                    missing_strategy=missing_strategy,
                    remove_outliers_flag=remove_outliers,
                )

                if preprocess_report.steps_applied:
                    with st.expander("Preprocessing applied", expanded=False):
                        st.markdown(
                            f"**Before:** {preprocess_report.original_rows:,} rows, "
                            f"{preprocess_report.original_cols} columns"
                        )
                        st.markdown(
                            f"**After:** {preprocess_report.final_rows:,} rows, "
                            f"{preprocess_report.final_cols} columns"
                        )
                        st.markdown("---")
                        for step in preprocess_report.steps_applied:
                            st.markdown(f"- {step}")
                        if preprocess_report.warnings:
                            for warn in preprocess_report.warnings:
                                st.warning(warn)

                detected_date = detect_date_column(df)
                detected_metrics = detect_numeric_columns(df, exclude=[detected_date] if detected_date else [])

                all_cols = list(df.columns)
                date_index = all_cols.index(detected_date) if detected_date and detected_date in all_cols else 0
                metric_index = all_cols.index(detected_metrics[0]) if detected_metrics else 0

                date_col = st.selectbox(
                    "Date column",
                    options=all_cols,
                    index=date_index,
                )
                metric_col = st.selectbox(
                    "Metric column",
                    options=all_cols,
                    index=metric_index,
                )

                if not date_col or not metric_col:
                    st.error("Select both a date column and a metric column.")
                    df = None
        else:
            datasets = get_available_datasets()
            auto_load = st.session_state.pop("_auto_load", None)
            dataset_keys = list(datasets.keys())
            default_index = dataset_keys.index(auto_load) if auto_load and auto_load in dataset_keys else 0
            selected = st.selectbox(
                "Select dataset",
                options=dataset_keys,
                index=default_index,
                format_func=lambda k: f"{datasets[k]['label']} [{datasets[k].get('frequency', '')}]",
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
            st.markdown("### Analysis Settings")

            min_date = pd.to_datetime(df[date_col], errors="coerce").min().date()
            max_date = pd.to_datetime(df[date_col], errors="coerce").max().date()

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
                help="The date when the policy or event took effect. "
                     "Pick the midpoint if you're unsure.",
            )

            method = st.selectbox(
                "Method",
                options=["arima", "bsts"],
                format_func=lambda x: {
                    "arima": "ARIMA ITS (fast)",
                    "bsts": "Bayesian STS (slow, experimental)",
                }.get(x, x),
            )

            with st.expander("Which method should I use?", expanded=False):
                st.markdown("""
**ARIMA ITS** (recommended) uses traditional time series forecasting. Best for most use cases. Results in seconds.

**Bayesian STS** is more flexible and handles complex seasonal patterns, but is slower. Results in 1-2 minutes. Experimental — may not work on all platforms.

When in doubt, use **ARIMA ITS**.
                """)

            run_placebo = st.checkbox(
                "Placebo sensitivity test",
                value=False,
                help="Runs the analysis at fake dates to check if the result is robust.",
            )

            st.divider()

            run_clicked = st.button(
                "Run Analysis",
                type="primary",
                use_container_width=True,
            )

            return df, date_col, metric_col, intervention_date, method, run_placebo, run_clicked, preprocess_report

        # ── Developer support link ──
        st.divider()
        st.markdown(
            '<div style="text-align:center;padding:8px 0;">'
            '<a href="https://chai4.me/darkcharon3301" target="_blank" '
            'title="Support darkcharon3301 on Chai4Me" '
            'style="display:inline-flex;flex-direction:column;align-items:center;'
            'justify-content:center;background:#ffffff;padding:8px 32px;'
            'border-radius:16px;text-decoration:none;border:1px solid #e5e7eb;'
            'box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);'
            'transition:transform 0.2s;">'
            '<img src="https://chai4.me/icons/wordmark.png" alt="Chai4Me" '
            'style="height:32px;object-fit:contain;"/>'
            '</a></div>',
            unsafe_allow_html=True,
        )

        return None, None, None, None, None, None, False, None


# ─── Main ──────────────────────────────────────────────────────
def main():
    # ── Handle quick-start demo request ──
    if st.session_state.get("_demo_requested"):
        st.session_state.pop("_demo_requested", None)
        st.session_state["_auto_load"] = "delhi_aqi"
        st.session_state["_source_override"] = "Pre-loaded dataset"
        st.rerun()

    (
        df,
        date_col,
        metric_col,
        intervention_date,
        method,
        run_placebo,
        run_clicked,
        preprocess_report,
    ) = build_sidebar()

    # ── Empty state — Hero landing ──
    if df is None:
        st.markdown("""
        # Did this policy actually work?

        CausalLens estimates the causal effect of any policy intervention
        on a time series using rigorous statistical methods.

        ---
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**ARIMA ITS**")
                st.caption("Fast, reliable interrupted time series analysis.")
        with c2:
            with st.container(border=True):
                st.markdown("**Bayesian STS**")
                st.caption("Flexible Bayesian structural time series.")
        with c3:
            with st.container(border=True):
                st.markdown("**Placebo Tests**")
                st.caption("Sensitivity analysis to validate robustness.")

        st.markdown("")

        # Quick-start demo button
        if st.button(
            "Try with Delhi Air Quality data  \u2192",
            use_container_width=False,
            type="secondary",
            key="demo_btn",
        ):
            st.session_state["_demo_requested"] = True
            st.rerun()

        st.info(
            "Or select a data source in the sidebar to upload your own data."
        )

        return

    # ── Data loaded — show preview ──
    if len(df) > LARGE_DATASET_ROWS:
        st.warning(
            f"This dataset has {len(df):,} rows. Analysis may take several minutes. "
            f"Consider filtering to daily or weekly aggregates if results are slow."
        )

    st.markdown(f"### Data Preview: `{metric_col}` over time")

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
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(preview_fig, use_container_width=True)

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Rows", f"{len(df):,}")
    with stats_col2:
        min_date_str = pd.to_datetime(df[date_col], errors="coerce").min().date()
        max_date_str = pd.to_datetime(df[date_col], errors="coerce").max().date()
        st.metric("Date Range", f"{min_date_str} to {max_date_str}")
    with stats_col3:
        st.metric("Mean", f"{df[metric_col].mean():.2f}")
    with stats_col4:
        st.metric("Std Dev", f"{df[metric_col].std():.2f}")

    st.session_state["df_raw"] = df

    # ── Run analysis ──
    if run_clicked:
        st.session_state.pop("result", None)
        st.session_state.pop("placebo", None)
        st.session_state.pop("pdf_requested", None)
        intervention_str = intervention_date.strftime("%Y-%m-%d")

        with st.spinner("Running causal analysis..."):
            try:
                df_hash = int(pd.util.hash_pandas_object(df).sum())
                st.session_state["_df_cache"] = df
                result_dict = run_cached_analysis(
                    df_hash, date_col, metric_col, intervention_str, method
                )
                if not result_dict:
                    return
                st.session_state["result"] = result_dict
                st.session_state["metric_col"] = metric_col
                st.session_state["df_raw"] = df
            except ValueError as e:
                logger.error("Analysis failed: %s", e, exc_info=True)
                st.error(f"Analysis failed: {e}")
                st.caption(
                    "Try a different intervention date, method, or check your data for issues."
                )
                return
            except Exception as e:
                logger.error("Analysis failed: %s", e, exc_info=True)
                st.error("An unexpected error occurred during analysis.")
                st.caption("Please try again. If the problem persists, check your data format.")
                return

        if run_placebo:
            n_placebos = 50
            progress_bar = st.progress(0, text=f"Running placebo test (0/{n_placebos})...")
            try:
                y = df[metric_col].values
                idx = result_dict["intervention_idx"]

                def _placebo_progress(current: int, total: int) -> None:
                    progress_bar.progress(
                        current / total,
                        text=f"Running placebo test ({current}/{total})...",
                    )

                placebo_result = run_placebo_test(
                    y, idx, n_placebos=n_placebos, on_progress=_placebo_progress
                )
                progress_bar.progress(1.0, text="Placebo test complete!")
                st.session_state["placebo"] = placebo_result
            except Exception as e:
                logger.warning("Placebo test failed: %s", e)
                st.warning("Placebo test failed. The main results are still valid.")
            finally:
                progress_bar.empty()

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
        placebo_fig.add_trace(go.Histogram(
            x=pb["placebo_effects"],
            name="Placebo Effects",
            marker_color="#64748b",
            opacity=0.7,
            nbinsx=20,
        ))
        real_effect = pb["real_effect"]
        placebo_fig.add_vline(
            x=real_effect,
            line=dict(color="#818cf8", width=3, dash="dash"),
            annotation_text=f"Real: {real_effect:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#818cf8"),
        )
        placebo_fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#94a3b8")),
            xaxis=dict(
                title="Effect Size",
                tickfont=dict(color="#94a3b8"),
                gridcolor="rgba(148,163,184,0.1)",
            ),
            yaxis=dict(
                title="Count",
                tickfont=dict(color="#94a3b8"),
                gridcolor="rgba(148,163,184,0.1)",
            ),
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
    with st.expander("View Raw Data", expanded=False):
        raw_df = st.session_state.get("df_raw")
        if raw_df is not None:
            tab_view, tab_stats = st.tabs(["Data", "Statistics"])

            with tab_view:
                st.dataframe(
                    raw_df,
                    use_container_width=True,
                    height=min(400, max(200, 35 * min(len(raw_df), 15) + 40)),
                )

            with tab_stats:
                st.dataframe(
                    raw_df.describe(),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
