# CausalLens — Project Plan

> An open-source Python tool that takes any policy date + metric time series and outputs the causal effect of that policy with confidence intervals, counterfactual visualization, and statistical significance tests.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 1 — Idea Validation](#2-phase-1--idea-validation)
3. [Phase 2 — Research & State of the Art](#3-phase-2--research--state-of-the-art)
4. [Phase 3 — System Architecture](#4-phase-3--system-architecture)
5. [Phase 4 — AI/ML Stack Selection](#5-phase-4--aiml-stack-selection)
6. [Phase 5 — Open Source Stack Analysis](#6-phase-5--open-source-stack-analysis)
7. [Phase 6 — Project Structure](#7-phase-6--project-structure)
8. [Phase 7 — MVP Planning](#8-phase-7--mvp-planning)
9. [Phase 8 — Agentic System Design](#9-phase-8--agentic-system-design)
10. [Phase 9 — Execution Plan](#10-phase-9--execution-plan)
11. [Phase 10 — Code Generation Prompts](#11-phase-10--code-generation-prompts)
12. [Phase 11 — Evaluation & Benchmarking](#12-phase-11--evaluation--benchmarking)
13. [Phase 12 — Productionization](#13-phase-12--productionization)
14. [Risk Analysis](#14-risk-analysis)
15. [Scaling Strategy](#15-scaling-strategy)

---

## 1. Executive Summary

### What We Are Building
A Python library + Streamlit dashboard that answers: "Did this policy actually work?" Users upload a time series, pick an intervention date, and get a causal effect estimate with counterfactual visualization.

### Key Decisions
- **No LLMs needed** — this is a statistical computing project, not AI
- **No database needed** — Streamlit Community Cloud + in-memory computation
- **No authentication needed** — public tool, no user accounts
- **No vector DB** — no embeddings, no RAG, no semantic search
- **Core library:** `causalimpact` (Python port) + `statsmodels` for ARIMA ITS
- **Deployment:** Streamlit Community Cloud (free, 1GB RAM)

### Constraints
- **Budget:** $0 (free tiers only)
- **Timeline:** 1 month
- **Team:** Solo developer
- **Deployment:** Streamlit Community Cloud
- **Target users:** Policy analysts, economists, researchers

### What Makes This Defensible
1. Nobody has built a general-purpose Python causal impact tool with a dashboard
2. Pre-loaded Indian policy datasets are unique
3. The combination of BSTS + ARIMA + placebo tests in one interface doesn't exist
4. Plain-language summary feature is novel

---

## 2. Phase 1 — Idea Validation

### Problem Statement
Policy analysts, economists, and researchers need to measure "Did this policy actually work?" but existing tools are either R-only (Google CausalImpact), require TensorFlow (tfp-causalimpact), or are code-only libraries with no dashboard. No tool integrates Indian economic data. No tool generates a plain-language report.

### Highest-Value Use Case
A policy analyst at a think tank receives a dataset with monthly tax revenue. They want to know: "Did GST increase revenue?" They open the tool, upload the CSV, pick July 2017, and get a one-page report with a counterfactual graph, effect estimate, and confidence interval. Total time: 2 minutes.

### Target Users
1. **Primary:** Policy analysts at Indian think tanks (NITI Aayog, ICRIER, NCAER, CPR)
2. **Secondary:** Economics PhD students running causal inference for dissertations
3. **Tertiary:** Data scientists in business who need to measure campaign/policy impact

### Painful Workflows This Solves
- Currently: analyst writes 200+ lines of R/Python code, gets a number, manually creates charts, writes a report. Takes 2-4 hours.
- With this tool: analyst uploads CSV, picks date, gets a full report in 2 minutes.

### Why Existing Solutions Fail
| Solution | Failure mode |
|----------|-------------|
| Google CausalImpact (R) | R-only, most data scientists use Python |
| tfp-causalimpact (Python) | TensorFlow dependency, heavy, poorly maintained |
| causalimpact (PyPI) | Thin wrapper, no dashboard, no Indian data |
| CausalPy | Library only, no user-facing tool |
| Academic papers | Describe methods but provide no software |
| Custom ARIMA in statsmodels | Analyst writes 200+ lines, no counterfactual visualization |

### Competitors
- **Direct:** None that combine dashboard + pre-loaded data + multiple methods + plain-language output
- **Indirect:** Google CausalImpact (R), CausalPy (Python library), econometrics textbooks

### Market Opportunity
- India has 500+ think tanks and policy research organizations
- Every economics PhD student needs causal inference
- No open-source tool dominates this space
- Natural virality: "Did X policy work?" is a question every journalist asks

### Technical Feasibility Risks
| Risk | Severity | Mitigation |
|------|----------|------------|
| Bayesian MCMC too slow for Streamlit (1GB RAM) | High | Use `causalimpact` (faster) + ARIMA (fast) as fallback. Cap data at 5 years daily. |
| Streamlit Cloud sleeps after inactivity | Medium | Acceptable for demo/portfolio. Add "wake up" prompt. |
| PyMC dependency is heavy | Medium | Don't use PyMC in MVP. Use `causalimpact` + `statsmodels` only. |
| Pre-loaded datasets become stale | Low | Document data sources. Users can refresh. |

### Implementation Complexity: 6/10

---

## 3. Phase 2 — Research & State of the Art

### Essential Papers

| # | Title | Authors | Year | Why it matters | Difficulty | Production Viability |
|---|-------|---------|------|---------------|------------|---------------------|
| 1 | "Inferring Causal Impact Using Bayesian Structural Time-Series Models" | Brodersen et al. (Google) | 2015 | Foundational paper. Defines BSTS method. 2,500+ citations. | Low | High — implemented in R and Python |
| 2 | "Interrupted time series regression for the evaluation of public health interventions: a tutorial" | Lopez Bernal et al. | 2017 | Gold-standard ITS tutorial. 3,246 citations. | Low | High — step-by-step methodology |
| 3 | "Causal Inference for Time Series" | Runge et al. | 2023 | Comprehensive review in Nature Reviews. | Medium | High — defines the field |
| 4 | "Causal Inference for Time Series Analysis: Problems, Methods and Evaluation" | Moraffah et al. | 2021 | Full survey with benchmarks. arXiv:2102.05829. | Medium | High — evaluation metrics |

### Practical Papers

| # | Title | Year | Why it matters | Difficulty | Viability |
|---|-------|------|---------------|------------|-----------|
| 5 | "A Bayesian Interrupted Time Series framework for evaluating policy change" (Gascoigne et al.) | 2024 | Bayesian ITS applied to welfare policy. Directly applicable. | Medium | High |
| 6 | "Interrupted Time Series Design and Analyses in Health Policy Assessment" | 2024 | Compares ARIMA vs GAM for ITS. | Medium | High |
| 7 | "Estimating the effect of annual PM2.5 exposure on mortality in India" | 2024 | Causal methods on Indian air pollution data. | Low | High |
| 8 | "Ambient air pollution and daily mortality in ten cities of India" (Lancet) | 2024 | Instrumental variables on Indian AQI. | Medium | High |

### Experimental Papers

| # | Title | Year | Why it matters | Difficulty | Viability |
|---|-------|------|---------------|------------|-----------|
| 9 | "Dynamic Structural Causal Models" (Boeken & Mooij) | 2024 | Cutting-edge causal discovery. CI4TS 2024. | High | Low — too complex for MVP |
| 10 | "Signature Kernel CI Tests for Causal Discovery" (Manten et al.) | 2024 | Novel method for stochastic processes. | High | Low — research-only |

### Technique Comparison

| Method | Pros | Cons | Use in MVP? |
|--------|------|------|-------------|
| **BSTS** | Gold standard, handles seasonality, uncertainty quantification | Slow MCMC, heavy dependencies | Yes — primary |
| **ARIMA-based ITS** | Fast, well-understood, statsmodels has it | No Bayesian uncertainty, less flexible | Yes — fast alternative |
| **CausalPy** | Research-grade, flexible | Heavy PyMC dependency, slow | No — too heavy |
| **Difference-in-Differences** | Powerful for comparative cases | Requires control group | No — different use case |
| **Synthetic Control** | Powerful for comparative cases | Requires multiple units | No — single time series |

### Recommended Production Approach
- **Primary:** `causalimpact` Python package (BSTS)
- **Alternative:** `statsmodels` ARIMA ITS
- **Sensitivity:** Placebo tests at fake dates

---

## 4. Phase 3 — System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────┐
│            STREAMLIT CLOUD              │
│            (1GB RAM, free)              │
│                                         │
│  ┌─────────────┐    ┌───────────────┐  │
│  │  Streamlit   │    │   Core Engine  │  │
│  │  Dashboard   │───▶│               │  │
│  │  (app.py)    │    │ causalimpact  │  │
│  └─────────────┘    │ statsmodels   │  │
│                      │ plotly        │  │
│                      └───────┬───────┘  │
│                              │          │
│                      ┌───────▼───────┐  │
│                      │  Pre-loaded   │  │
│                      │  Datasets     │  │
│                      │  (CSV files)  │  │
│                      └───────────────┘  │
│                                         │
│  ┌─────────────┐    ┌───────────────┐  │
│  │  PDF Export  │    │  Report Gen   │  │
│  │  (reportlab) │◀───│  (summary.py) │  │
│  └─────────────┘    └───────────────┘  │
│                                         │
└─────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   ┌──────────┐      ┌──────────────┐
   │  GitHub   │      │  PyPI (pip   │
   │  (source) │      │  install)    │
   └──────────┘      └──────────────┘
```

### Frontend
- **Framework:** Streamlit (single `app.py`)
- **No React/Next.js** — Streamlit handles all UI
- **Charts:** Plotly (interactive, embeddable in Streamlit)
- **State:** Streamlit session state

### Backend
- **No separate backend** — Streamlit runs Python directly
- **Core engine:** Pure Python library (`src/core/`)
- **No API** — Streamlit IS the interface

### Database
**None.** All data loaded from CSV into pandas DataFrames in memory.

### Vector Database
**None.** No embeddings, no RAG needed.

### Authentication
**None.** Public tool, no user accounts.

### Caching
```python
@st.cache_data
def load_dataset(name: str) -> pd.DataFrame: ...

@st.cache_data
def run_analysis(data, intervention_date, method): ...
```

### Deployment
```
GitHub repo → Push to main → Streamlit Cloud auto-deploys
```

---

## 5. Phase 4 — AI/ML Stack Selection

### Critical Insight: This Is NOT an AI Project

This project does **not** need:
- LLMs (no text generation)
- Embeddings (no semantic search)
- RAG (no retrieval)
- Vision models (no image processing)
- Agent frameworks (no autonomous agents)
- VRAM (no GPU needed)

This is a **statistical computing** project.

### What You Actually Need

| Task | Tool | Why |
|------|------|-----|
| Bayesian causal inference | `causalimpact` (PyPI) | Google's BSTS, lightweight |
| ARIMA time series | `statsmodels` | Standard, fast |
| Data manipulation | `pandas` | Standard |
| Interactive charts | `plotly` | Best for Streamlit |
| Statistical summaries | `scipy.stats` | p-values, confidence intervals |
| PDF generation | `matplotlib` + `reportlab` | Downloadable reports |
| Dashboard | `streamlit` | Free hosting, fast dev |

### Dependencies (requirements.txt)
```
streamlit>=1.35.0
causalimpact>=0.2.6
statsmodels>=0.14.0
pandas>=2.0.0
plotly>=5.18.0
scipy>=1.11.0
matplotlib>=3.7.0
reportlab>=4.0.0
numpy>=1.24.0
```

Total download: ~50MB. Fits in Streamlit Cloud's 1GB limit.

---

## 6. Phase 5 — Open Source Stack Analysis

| Layer | Choice | Maturity | Community | License | Complexity | Alternative |
|-------|--------|----------|-----------|---------|------------|-------------|
| Dashboard | Streamlit | High | 40K+ stars | Apache 2.0 | Low | Gradio, Panel |
| Causal engine | causalimpact | Medium (beta) | Low stars | MIT | Low | CausalPy, custom ARIMA |
| Time series | statsmodels | Very high | 10K+ stars | BSD | Low | Prophet |
| Data | pandas | Very high | 43K+ stars | BSD | Low | polars |
| Charts | plotly | Very high | 15K+ stars | MIT | Low | altair |
| PDF | reportlab | Very high | 3K+ stars | BSD | Medium | fpdf2 |
| Stats | scipy | Very high | 13K+ stars | BSD | Low | — |

**Risk:** `causalimpact` (PyPI) is the weakest link — beta status, low community.
**Mitigation:** If it breaks, fall back to pure `statsmodels` ARIMA ITS.

---

## 7. Phase 6 — Project Structure

```
CausalLens/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT
├── PROJECT_PLAN.md             # This file
├── AGENTS.md                   # AI agent instructions
├── ENGINEERING_MEMORY.md       # Lessons learned log
├── .streamlit/
│   └── config.toml             # Streamlit config
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py           # Main causal_effect() function
│   │   ├── bsts.py             # Bayesian Structural Time Series
│   │   ├── arima_its.py        # ARIMA-based ITS
│   │   └── placebo.py          # Sensitivity/placebo tests
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Load user CSV or pre-loaded data
│   │   └── datasets/
│   │       ├── delhi_aqi.csv
│   │       ├── gst_revenue.csv
│   │       ├── rbi_policy.csv
│   │       └── gdp_quarterly.csv
│   ├── reports/
│   │   ├── __init__.py
│   │   ├── summary.py          # Plain-language summary
│   │   ├── plots.py            # Plotly chart generators
│   │   └── pdf_export.py       # PDF report generator
│   └── utils/
│       ├── __init__.py
│       ├── validators.py       # Input validation
│       └── formatters.py       # Number/date formatting
├── notebooks/
│   ├── demo_delhi_lockdown.ipynb
│   ├── demo_gst_impact.ipynb
│   └── user_guide.ipynb
├── tests/
│   ├── test_engine.py
│   ├── test_arima.py
│   └── test_validators.py
└── docs/
    ├── METHODOLOGY.md
    └── DATASETS.md
```

### Naming Conventions
- Files: `snake_case.py`
- Functions: `snake_case()`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Environment Variables
**None needed.** No API keys, no secrets.

### Docker Strategy
**None.** Streamlit Cloud handles deployment.

### Branching Strategy
- `main` — deployed to Streamlit Cloud
- `dev` — local development

---

## 8. Phase 7 — MVP Planning

### Smallest Viable MVP (Week 1-2)
1. Upload a CSV with date + metric columns
2. User picks an intervention date
3. Tool runs ARIMA-based ITS
4. Shows counterfactual plot
5. Shows effect estimate + 95% confidence interval
6. Shows p-value

**That's it.** No BSTS, no PDF, no pre-loaded datasets, no placebo tests.

### Core Differentiator
The counterfactual visualization — "here's what would have happened without the policy."

### Non-Essential for MVP
- BSTS method (Week 3)
- PDF export (Week 3)
- Pre-loaded datasets (Week 2-3)
- Placebo tests (Week 3-4)
- Plain-language summary (Week 3)

### 30-Day Roadmap

| Week | Milestone | Days | Tasks |
|------|-----------|------|-------|
| **1** | Core engine + basic UI | 1-7 | ARIMA ITS engine, Streamlit skeleton, CSV upload, date picker, counterfactual plot |
| **2** | Polish + pre-loaded data | 8-14 | Effect summary, p-value, confidence interval, 4 pre-loaded datasets, dataset selector |
| **3** | BSTS + reports | 15-21 | Add BSTS method, plain-language summary, PDF export, placebo tests |
| **4** | Documentation + deploy | 22-30 | README, demo notebooks, methodology doc, deploy to Streamlit Cloud, GitHub release |

---

## 9. Phase 8 — Agentic System Design

### Verdict: No Agents Needed

This project does not benefit from agentic architecture:
- No multi-step reasoning required
- No tool calling needed
- No retrieval needed
- No planning needed
- No hallucination risk

The workflow is: load data → run model → show results. This is a pipeline, not an agent.

---

## 10. Phase 9 — Execution Plan

### First 20 Implementation Tasks

| # | Task | Files | Edge Cases | Validation |
|---|------|-------|------------|------------|
| 1 | Setup repo + deps | `requirements.txt`, `.streamlit/config.toml` | — | `pip install -r requirements.txt` works |
| 2 | Create Streamlit skeleton | `app.py` | Empty file, wrong format | App loads without error |
| 3 | Implement CSV loader | `src/data/loader.py` | No date column, multiple formats, missing values | Loads test CSV correctly |
| 4 | Implement ARIMA ITS engine | `src/core/arima_its.py` | Short series (<30), non-stationary, missing values | Effect matches manual calc |
| 5 | Implement main engine interface | `src/core/engine.py` | Invalid date, date outside range | Returns structured result |
| 6 | Build counterfactual plot | `src/reports/plots.py` | No data, single point | Chart renders in Streamlit |
| 7 | Build effect summary | `src/reports/summary.py` | Zero effect, negative effect, insignificant | Summary is human-readable |
| 8 | Add p-value and significance | `src/core/engine.py` | p=0.05 exactly | Correct statistical interpretation |
| 9 | Add input validation | `src/utils/validators.py` | Wrong types, empty data, NaN-heavy | Clear error messages |
| 10 | Add pre-loaded datasets | `src/data/datasets/`, `src/data/loader.py` | File missing, stale data | All 4 datasets load |
| 11 | Build dataset selector UI | `app.py` | Both selected, neither selected | UI flow is clear |
| 12 | Add BSTS method | `src/core/bsts.py` | Package not installed, MCMC timeout | Same result structure |
| 13 | Add method selector | `app.py` | — | Both methods selectable |
| 14 | Implement placebo tests | `src/core/placebo.py` | Too few data points | Real effect outside placebo dist |
| 15 | Build PDF export | `src/reports/pdf_export.py` | Empty result, long title | PDF downloads correctly |
| 16 | Add methodology doc | `docs/METHODOLOGY.md` | — | Readable by non-statistician |
| 17 | Write demo notebook 1 | `notebooks/demo_delhi_lockdown.ipynb` | — | Runs end-to-end |
| 18 | Write demo notebook 2 | `notebooks/demo_gst_impact.ipynb` | — | Runs end-to-end |
| 19 | Write README | `README.md` | — | Complete and accurate |
| 20 | Deploy to Streamlit Cloud | Push to GitHub | Sleep/wake behavior | App loads in browser |

---

## 11. Phase 10 — Code Generation Prompts

### Prompt 1: Core ARIMA ITS Engine
```
Create `src/core/arima_its.py` implementing an ARIMA-based Interrupted Time Series analysis.

Requirements:
- Function: `run_arima_its(y: np.ndarray, intervention_idx: int) -> dict`
- Fit ARIMA(p,d,q) on pre-intervention data
- Forecast post-intervention period
- Compute: effect estimate, confidence interval, p-value, counterfactual array
- Use statsmodels ARIMA
- Handle: short series, non-stationary data, missing values
- Return dict with: effect, ci_lower, ci_upper, p_value, counterfactual, fitted_values
- No comments beyond docstrings. PEP 8 naming.
```

### Prompt 2: Streamlit Dashboard
```
Create `app.py` — a Streamlit dashboard for CausalLens.

Requirements:
- Title: "CausalLens — Causal Impact Calculator"
- Sidebar: file uploader (CSV), date picker, method selector (ARIMA/Bayesian)
- Main area: counterfactual chart (Plotly), effect summary, confidence interval
- Download button for PDF report
- Error handling for invalid inputs
- Clean, minimal design
- Use st.session_state for computed results
- Cache with @st.cache_data
```

### Prompt 3: Plain-Language Summary
```
Create `src/reports/summary.py` generating human-readable causal impact summaries.

Requirements:
- Function: `generate_summary(result: dict) -> str`
- Input: dict with effect, ci_lower, ci_upper, p_value, direction
- Output: 2-3 sentence summary
- Handle: zero effect, negative effect, insignificant result
- No jargon. Write for a policy analyst.
```

---

## 12. Phase 11 — Evaluation & Benchmarking

### KPIs

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Time to first result | < 5 seconds | Upload → result |
| Accuracy | Within 10% of known value | Synthetic data test |
| False positive rate | < 5% | Placebo tests |
| Dashboard load | < 3 seconds | Cold start |
| PDF generation | < 2 seconds | Download button |

### Latency Targets
| Operation | Target | Acceptable |
|-----------|--------|------------|
| CSV parsing | < 500ms | < 1s |
| ARIMA ITS | < 2s | < 5s |
| BSTS | < 10s | < 20s |
| Plot rendering | < 1s | < 2s |
| PDF export | < 2s | < 3s |

### Evaluation Datasets
1. **Synthetic data with known effect** — verify recovery within CI
2. **Placebo test data** — no intervention, verify no significant effect
3. **Delhi AQI 2020** — real lockdown data, verify matches published research
4. **US unemployment 2020** — CARES Act, verify matches published research

### Failure Modes
| Failure | Detection | Mitigation |
|---------|-----------|------------|
| ARIMA doesn't converge | statsmodels exception | Fall back, show warning |
| Series too short | length < 30 | Error message |
| No trend | ADF test | Warning |
| Intervention at boundaries | idx check | Error |
| Streamlit RAM limit | App crash | Cap at 1800 points |

---

## 13. Phase 12 — Productionization

### Deployment Checklist
1. [ ] All tests passing
2. [ ] `requirements.txt` pinned
3. [ ] `.streamlit/config.toml` configured
4. [ ] README complete with screenshots
5. [ ] Demo notebooks run end-to-end
6. [ ] Deployed to Streamlit Community Cloud
7. [ ] Public URL works
8. [ ] PDF export works in cloud
9. [ ] All 4 pre-loaded datasets load
10. [ ] GitHub repo public with MIT license

### Monitoring
- Streamlit Cloud provides basic health
- `st.error()` for user-facing errors
- `logging` for debug output

### Security
- No user input goes to a database
- No SQL queries, no shell commands
- Input validation on all user data

### Disaster Recovery
- Source on GitHub (version controlled)
- Streamlit Cloud redeploys from GitHub
- No database to back up

---

## 14. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| `causalimpact` breaks | Medium | High | ARIMA ITS as fallback |
| Streamlit Cloud down | Low | Medium | Acceptable for demo |
| 1GB RAM limit hit | Medium | Medium | Cap at 1800 points |
| Bayesian too slow | Medium | Medium | ARIMA as default, BSTS as option |
| Judge doesn't understand | Low | High | Counterfactual plot is self-explanatory |
| Indian datasets stale | Low | Low | Document sources |

---

## 15. Scaling Strategy

This is a **library + demo**, not a SaaS. Scaling means:
1. GitHub stars — good README, demo notebooks
2. PyPI downloads — `pip install causallens`
3. Citations — academics cite the tool
4. Community — open-source, accept PRs

If popular:
- Self-host on $5/mo VPS
- Add user accounts for saving analyses
- Add REST API for programmatic access
- Consider Streamlit in Snowflake for enterprise

---

## 16. Immediate Next Actions

| # | Action | Time | Why first |
|---|--------|------|-----------|
| 1 | `pip install streamlit causalimpact statsmodels pandas plotly scipy` | 5 min | Confirms stack works |
| 2 | Create `app.py` with Streamlit skeleton | 15 min | Visible progress |
| 3 | Implement `run_arima_its()` with synthetic test | 1 hour | Core engine works |

After these 3 tasks, you have a working prototype. Everything else is incremental.
