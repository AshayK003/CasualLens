# ENGINEERING_MEMORY.md — Lessons Learned Log

> Living document. Every mistake, decision, and lesson goes here. Check this before repeating work.

---

## How to Use This File

Each entry follows this format:

```
### [YYYY-MM-DD] — [Category] — [Title]
- **What happened:**
- **Why it happened:**
- **What to do instead:**
- **Files affected:**
```

Categories: `Decision` | `Bug` | `Performance` | `Deployment` | `Dependency` | `Architecture` | `Testing`

---

## Pre-Populated Known Risks (Before Coding Starts)

### [2026-05-28] — Decision — Use ARIMA as primary, BSTS as secondary
- **What happened:** Chose ARIMA ITS as the primary causal engine, with BSTS as an advanced option.
- **Why it happened:** `causalimpact` Python package is beta-quality with low community. ARIMA via statsmodels is rock-solid. BSTS is slower and heavier. ARIMA fits in Streamlit Cloud's 1GB RAM.
- **What to do instead:** Always build the ARIMA path first. Add BSTS only after ARIMA works end-to-end. If BSTS breaks, the tool still works.
- **Files affected:** `src/core/arima_its.py`, `src/core/bsts.py`, `src/core/engine.py`

### [2026-05-28] — Dependency — causalimpact PyPI package is beta
- **What happened:** The `causalimpact` Python package (PyPI) is maintained by a single person, last updated Jan 2023, beta status.
- **Why it happened:** Google's original is R-only. The Python port is a community effort.
- **What to do instead:** Pin the version in requirements.txt. Test after every dependency update. Have a fallback: if `causalimpact` fails to import, run only ARIMA ITS and show a warning.
- **Files affected:** `requirements.txt`, `src/core/bsts.py`

### [2026-05-28] — Performance — Streamlit Cloud has 1GB RAM limit
- **What happened:** Streamlit Community Cloud gives each app 1GB of RAM. Bayesian MCMC can be memory-hungry.
- **Why it happened:** Free tier constraint.
- **What to do instead:** Cap user data at 1800 points (~5 years of daily data). Use `@st.cache_data` to avoid recomputation. Profile memory usage before deploying. If BSTS uses too much RAM, disable it in the Streamlit deployment and offer only ARIMA.
- **Files affected:** `app.py`, `src/data/loader.py`, `src/core/bsts.py`

### [2026-05-28] — Deployment — Streamlit Cloud apps sleep after inactivity
- **What happened:** Apps on Streamlit Community Cloud go to sleep if not visited for a while. First visit after sleep takes 30-60 seconds to wake up.
- **Why it happened:** Free tier resource management.
- **What to do instead:** Accept this for MVP. Add a visible "Waking up, please wait..." message. In README, note that the app may take a moment on first load.
- **Files affected:** `app.py`

### [2026-05-28] — Architecture — No database, no auth, no API
- **What happened:** Deliberately chose the simplest architecture: in-memory computation, no user accounts, no REST API.
- **Why it happened:** This is a tool/library, not a SaaS. Adding auth/DB/API would triple the development time for zero user value in MVP.
- **What to do instead:** If someone asks for "user accounts" or "save analyses", say it's a v2 feature. Keep v1 dead simple.
- **Files affected:** All files (by absence — none of these exist)

### [2026-05-28] — Decision — Pre-loaded datasets are CSV files in repo
- **What happened:** Bundled 4 Indian policy datasets as CSV files in `src/data/datasets/`.
- **Why it happened:** No database, no external API. CSVs are simple, version-controlled, and work offline.
- **What to do instead:** Keep datasets small (< 1MB each). Document the source and last-updated date in `docs/DATASETS.md`. If a dataset becomes stale, note it in this memory file.
- **Files affected:** `src/data/datasets/`, `src/data/loader.py`, `docs/DATASETS.md`

---

## Bugs & Fixes

*(Populate as you encounter bugs)*

---

## Performance Lessons

*(Populate as you profile and optimize)*

---

## Deployment Issues

*(Populate as you deploy and encounter issues)*

---

## Dependency Issues

*(Populate as you hit dependency conflicts or breaking changes)*

---

## Architecture Decisions

*(Populate as you make architectural choices)*

---

## Testing Lessons

*(Populate as you write and run tests)*

---

## User Feedback

*(Populate as you get feedback from users or hackathon judges)*

---

## Lessons from Building (Phases 1-3)

### [2026-05-28] — Bug — `forecast_ci` returns numpy array, not DataFrame
- **What happened:** `forecast.conf_int()` returns a numpy array in newer statsmodels, not a DataFrame. Using `.iloc[:, 0]` throws `AttributeError`.
- **Why it happened:** Different statsmodels versions return different types.
- **What to do instead:** Always wrap with `np.asarray()` and handle both DataFrame and ndarray. Use `forecast_ci[:, 0]` instead of `.iloc[:, 0]`.
- **Files affected:** `src/core/arima_its.py`

### [2026-05-28] — Bug — `pd.to_datetime` with `infer_datetime_format` deprecated
- **What happened:** `pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)` causes warnings and unexpected behavior in pandas 3.x.
- **Why it happened:** `infer_datetime_format` was deprecated in pandas 2.0 and removed in 3.0.
- **What to do instead:** Use `pd.to_datetime(df[col], errors="coerce")` without `infer_datetime_format`. Check `pd.api.types.is_datetime64_any_dtype()` first.
- **Files affected:** `src/utils/validators.py`

### [2026-05-28] — Bug — `np.searchsorted` with datetime64 type mismatch
- **What happened:** `np.searchsorted(dates.values, intervention_dt.value)` throws TypeError because `dates.values` is datetime64 but `intervention_dt.value` is int64.
- **Why it happened:** numpy datetime64 comparison requires same types.
- **What to do instead:** Use `pd.DatetimeIndex.get_indexer()` with `method="nearest"` — it handles datetime comparison internally.
- **Files affected:** `src/utils/validators.py`

### [2026-05-28] — Bug — Plotly `add_vline` with string x-values
- **What happened:** `fig.add_vline(x="2020-03-11")` throws TypeError because plotly tries to compute mean of string dates.
- **Why it happened:** `add_vline` expects numeric x-values for the annotation calculation.
- **What to do instead:** Use `fig.add_shape()` + `fig.add_annotation()` for vertical lines on categorical/string axes.
- **Files affected:** `src/reports/plots.py`

### [2026-05-28] — Dependency — `causalimpact` Python package broken on Windows
- **What happened:** The `causalimpact` Python package's model fitting returns None/inferences, making BSTS results unreliable.
- **Why it happened:** The package is beta quality, last updated Jan 2023, depends on PyMC which has complex installation.
- **What to do instead:** ARIMA ITS is the reliable primary method. BSTS is optional and gracefully degrades. Always test BSTS with a simple dataset before claiming it works.
- **Files affected:** `src/core/bsts.py`, `tests/test_bsts_placebo.py`

### [2026-05-28] — Dependency — statsmodels 0.14.2 incompatible with numpy 1.26.4
- **What happened:** `from statsmodels.tsa.arima.model import ARIMA` throws TypeError about `deprecate_kwarg`.
- **Why it happened:** statsmodels 0.14.2 has a bug with newer numpy versions.
- **What to do instead:** Upgrade to `statsmodels>=0.14.3` (0.14.6 works). Pin in requirements.txt.
- **Files affected:** `requirements.txt`

### [2026-05-28] — Performance — BSTS takes 5-20 seconds per analysis
- **What happened:** Bayesian STS with 2000 MCMC iterations is slow on CPU.
- **Why it happened:** MCMC sampling is computationally expensive.
- **What to do instead:** Use ARIMA ITS as default (fast). Offer BSTS as "advanced" option. Cache results with `@st.cache_data`.
- **Files affected:** `app.py`, `src/core/bsts.py`

### [2026-05-28] — Architecture — Separate concerns into modules
- **What happened:** Initially put plots, summary, and analysis all in `app.py`. Hard to test.
- **Why it happened:** Streamlit encourages monolithic apps.
- **What to do instead:** Extract business logic into `src/core/`, `src/reports/`. Keep `app.py` as thin UI layer only. This makes unit testing possible without Streamlit.
- **Files affected:** `app.py`, `src/core/engine.py`, `src/reports/plots.py`, `src/reports/summary.py`

### [2026-05-28] — Testing — Use synthetic data with known effects
- **What happened:** Tests that check "effect > 5" pass reliably because we generate data with a known +10 effect.
- **Why it happened:** Synthetic data with controlled parameters is the only way to test statistical methods.
- **What to do instead:** Always generate test data with known effect sizes. Test: positive effect detected, negative effect detected, no effect insignificant.
- **Files affected:** `tests/test_engine.py`

### [2026-05-28] — Deployment — Streamlit Community Cloud sleeps
- **What happened:** App goes to sleep after inactivity, first visit takes 30-60s.
- **Why it happened:** Free tier resource management.
- **What to do instead:** Accept for MVP. If demo is critical, keep browser tab open. For production, use Streamlit in Snowflake or self-host.
- **Files affected:** `app.py`

---

## Template for New Entries

```markdown
### [YYYY-MM-DD] — [Category] — [Short Title]
- **What happened:**
- **Why it happened:**
- **What to do instead:**
- **Files affected:**
```
