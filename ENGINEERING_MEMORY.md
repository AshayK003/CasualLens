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

## Template for New Entries

```markdown
### [YYYY-MM-DD] — [Category] — [Short Title]
- **What happened:**
- **Why it happened:**
- **What to do instead:**
- **Files affected:**
```
