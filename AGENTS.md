# AGENTS.md — AI Coding Agent Instructions

> This file tells AI coding tools (Cursor, Claude, Copilot) everything they need to know about this project. Reference this file in every prompt.

---

## Project Overview

**CausalLens** is an open-source Python tool that estimates the causal effect of a policy intervention on a time series. Users upload a CSV, pick an intervention date, and get a counterfactual analysis with confidence intervals.

**Stack:** Python + Streamlit + statsmodels + causalimpact + plotly
**Deployment:** Streamlit Community Cloud (free, 1GB RAM)
**Target users:** Policy analysts, economists, researchers

---

## Architecture Constraints

- **NO separate backend** — Streamlit runs Python directly
- **NO database** — all computation is in-memory
- **NO authentication** — public tool, no user accounts
- **NO API routes** — Streamlit IS the interface
- **NO React/Next.js** — Streamlit handles all UI
- **NO LLMs** — this is pure statistics, not AI
- **NO GPU required** — CPU-only computation
- **NO external API calls** — everything runs locally
- **NO file system writes** except PDF download

---

## File Map

```
app.py                      → Streamlit entry point (ALL UI lives here)
src/core/engine.py          → Main causal_effect() function (entry point for all methods)
src/core/bsts.py            → Bayesian Structural Time Series wrapper
src/core/arima_its.py       → ARIMA-based ITS implementation
src/core/placebo.py         → Sensitivity/placebo tests
src/data/loader.py          → Load user CSV or pre-loaded datasets
src/data/datasets/*.csv     → Pre-loaded Indian policy datasets
src/reports/summary.py      → Plain-language summary generator
src/reports/plots.py        → Plotly chart generators
src/reports/pdf_export.py   → PDF report generator
src/utils/validators.py     → Input validation
src/utils/formatters.py     → Number/date formatting
```

---

## Allowed Dependencies (Whitelist)

```txt
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

**Do NOT add any other dependencies.** If you think you need something else, ask first.

---

## Forbidden Patterns

- Do NOT use `st.cache` (deprecated) — use `@st.cache_data`
- Do NOT import PyMC, TensorFlow, PyTorch — too heavy for Streamlit Cloud
- Do NOT create API routes (`/api/...`) — no FastAPI, no Flask
- Do NOT use SQLAlchemy or any ORM — no database
- Do NOT use `os.system()` or `subprocess` — security risk
- Do NOT hardcode file paths — use relative paths from project root
- Do NOT use `print()` for debugging — use `logging` module
- Do NOT add comments unless the code is genuinely cryptic
- Do NOT use `type: ignore` — fix the type error instead
- Do NOT create new files without updating this file

---

## Coding Standards

- **Naming:** `snake_case` for files, functions, variables. `PascalCase` for classes.
- **Imports:** Group: stdlib → third-party → local. One blank line between groups.
- **Type hints:** Optional in MVP, but preferred for public functions.
- **Docstrings:** Required for all public functions. Google style.
- **Error handling:** Always catch specific exceptions, log them, show user-friendly message.
- **No comments** unless the algorithm is non-obvious. Code should be self-documenting.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

---

## How to Deploy

1. Push to `main` branch on GitHub
2. Streamlit Cloud auto-deploys from GitHub
3. No CI/CD pipeline needed

---

## Testing Requirements

- Every function in `src/core/` must have at least one test
- Test with synthetic data (known effect size)
- Test edge cases: short series, missing values, no trend
- Use `pytest` with `@pytest.mark.parametrize` for parameterized tests

---

## Common Mistakes to Avoid

1. **Using `st.cache`** — deprecated, use `@st.cache_data`
2. **Loading heavy models** — PyMC/PyTorch won't fit in 1GB RAM
3. **Forgetting `@st.cache_data`** — expensive computations run on every rerun
4. **Not handling empty DataFrames** — always check `df.empty` before processing
5. **Hardcoding intervention date as datetime** — use string parsing, let user pick
6. **Ignoring autocorrelation in residuals** — use Newey-West or ARIMA errors
7. **Not capping data size** — Streamlit Cloud has 1GB RAM limit
8. **Using `st.set_page_config()` after other Streamlit commands** — must be first call
