# Contributing to CausalLens

## Project Overview

CausalLens estimates the causal effect of policy interventions on time series. It answers the question: **"Did this policy actually work?"**

Given a time series with a known intervention date (e.g., a new law, a lockdown, a marketing campaign), CausalLens builds a statistical model of what *would have happened* without the intervention (the counterfactual), then compares it to what actually happened.

### Why it exists

Most causal impact tools require statistical expertise. CausalLens wraps ARIMA ITS and Bayesian STS in a Streamlit UI so non-technical users can run rigorous analyses, interpret results in plain language, and export PDF reports.

### Key design decisions

- **ARIMA ITS as default** — fast, reliable, works on all platforms. BSTS is optional and experimental.
- **Streamlit over Flask/Django** — zero-config UI, fast iteration, native dark mode.
- **Pre-loaded datasets** — users can try the tool immediately without finding their own data.
- **Placebo tests** — sensitivity analysis to build confidence in results.
- **CPU-only, 1GB RAM** — designed for Streamlit Community Cloud free tier.

---

## Architecture

```
app.py                        # Streamlit entry point (UI, state, layout)
src/
├── core/
│   ├── engine.py             # causal_effect() — main entry point for analysis
│   ├── arima_its.py          # ARIMA interrupted time series implementation
│   ├── bsts.py               # Bayesian STS wrapper (Google CausalImpact)
│   └── placebo.py            # Sensitivity tests (run at fake intervention dates)
├── data/
│   ├── loader.py             # Load CSV/Excel uploads + pre-loaded datasets
│   ├── preprocessor.py       # Clean data, detect columns, handle missing values
│   └── datasets/             # CSV files for pre-loaded examples
├── reports/
│   ├── plots.py              # Plotly counterfactual charts
│   ├── summary.py            # Plain-language result summaries
│   └── pdf_export.py         # Matplotlib + ReportLab PDF generation
└── utils/
    ├── constants.py          # SIGNIFICANCE_LEVEL, LARGE_DATASET_ROWS, etc.
    ├── formatters.py         # Number/date formatting helpers
    └── validators.py         # Input validation (dataframe, dates, series length)
tests/                        # 204 tests across 17 files
```

### Data flow

```
User uploads CSV / selects dataset
        │
        ▼
loader.py → raw DataFrame
        │
        ▼
preprocessor.py → clean DataFrame + PreprocessReport
        │
        ▼
engine.py → CausalResult (effect, p-value, CI, counterfactual array)
        │
        ├──► plots.py → Plotly figure (shown in Streamlit)
        ├──► summary.py → Plain-language text
        ├──► pdf_export.py → PDF bytes (download button)
        └──► placebo.py → Placebo test results (optional)
```

### Why this structure

- `src/core/` is pure Python, no Streamlit dependency — testable independently.
- `src/data/` handles all I/O — file uploads, CSV reads, pre-processing.
- `src/reports/` generates all outputs — charts, summaries, PDFs.
- `app.py` is the thin UI layer — it wires everything together but contains no business logic.

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
# Clone the repo
git clone https://github.com/AshayK003/CausalLens.git
cd CausalLens

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The app auto-detects date and metric columns from uploaded data.

### Regenerate datasets (if needed)

```bash
python scripts/generate_datasets.py
```

---

## Environment Variables

CausalLens uses **no environment variables** for core functionality. Everything runs locally.

Streamlit-specific:
- `.streamlit/secrets.toml` — optional, for Streamlit Community Cloud deployment secrets (gitignored)
- `.streamlit/config.toml` — theme config (dark mode, primary color `#6366f1`)

---

## Local Development Flow

1. Create a feature branch: `git checkout -b feature/my-change`
2. Make changes in `src/` (not `app.py` unless UI-only)
3. Run tests: `python -m pytest tests/ -v -m "not slow"`
4. Run linter: `ruff check src/ tests/ app.py`
5. Run the app: `streamlit run app.py`
6. Test your change in the browser
7. Commit and push

### Code style

- **Formatter/Linter**: ruff (line length 100, target Python 3.10)
- **Imports**: sorted by ruff (isort rules)
- **Type hints**: required for new code
- **Docstrings**: not required but encouraged for public functions
- **No print()** — use `logging` module

### Ruff config (in pyproject.toml)

```
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = ["E501", "B904"]
```

---

## Testing

### Run tests

```bash
# Fast (~30s) — skips dataset generation tests
python -m pytest tests/ -v -m "not slow"

# Full suite (~4 min) — includes dataset ground-truth checks
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test structure

| File | What it tests |
|------|--------------|
| `test_engine.py` | `causal_effect()` with ARIMA and BSTS |
| `test_engine_edge_cases.py` | Short series, single-value, monotonic |
| `test_preprocessor.py` | Date detection, numeric detection, missing values |
| `test_preprocessor_edge_cases.py` | Year columns, empty data, edge formats |
| `test_loader.py` | CSV loading, file validation |
| `test_loader_edge_cases.py` | Empty files, wrong formats, size limits |
| `test_placebo_edge_cases.py` | Short series, extreme indices |
| `test_reports.py` | Summary generation, PDF export |
| `test_reports_edge_cases.py` | Plot building, formatting edge cases |
| `test_formatters.py` | Number/date formatting |
| `test_datasets.py` | Pre-loaded dataset ground truth values |
| `test_integration.py` | Full pipeline: load → preprocess → analyze |
| `test_integration_comprehensive.py` | Multiple methods, all datasets |
| `test_bsts_placebo.py` | BSTS method and placebo interaction |
| `test_pdf.py` | PDF generation and structure |

### Writing new tests

- Put them in `tests/`
- Name them `test_<module>_<behavior>.py`
- Use `@pytest.fixture` in `conftest.py` for shared fixtures
- Mark slow tests with `@pytest.mark.slow`
- Test both happy path and error cases

---

## Deployment

### Streamlit Community Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the `AshayK003/CausalLens` repo
4. Set main file path: `app.py`
5. Deploy

The app will auto-install `requirements.txt` on each deploy.

### Local / Self-hosted

```bash
streamlit run app.py --server.port 8501
```

For background execution on Windows:
```powershell
Start-Process -WindowStyle Hidden -FilePath "streamlit" -ArgumentList "run app.py --server.port 8501"
```

---

## Troubleshooting

### "Analysis failed" error

- **Cause**: Intervention date is outside the data range, or too few data points.
- **Fix**: Pick a date within the data range. Ensure at least 10 points before and after.

### BSTS fails on Windows

- **Cause**: The `causalimpact` package depends on `tensorflow` or has compatibility issues.
- **Fix**: Use ARIMA ITS method instead. BSTS is experimental.

### Streamlit won't start

- **Cause**: Port 8501 is in use.
- **Fix**: Kill the existing process or use a different port: `streamlit run app.py --server.port 8502`

### Tests fail after pulling

- **Cause**: Dependencies changed.
- **Fix**: `pip install -r requirements-dev.txt`

### "No valid data after date parsing"

- **Cause**: The date column has unparseable values.
- **Fix**: Check your CSV. Dates should be in a standard format (YYYY-MM-DD, MM/DD/YYYY, etc.).

---

## Contribution Guidelines

### Before you start

1. Open an issue describing what you want to change
2. Wait for feedback (especially for architectural changes)
3. Fork and create a branch

### What to work on

- **Bug fixes** — always welcome
- **New datasets** — add CSV files to `src/data/datasets/` and register in `loader.py`
- **Test coverage** — edge cases, integration tests
- **UI improvements** — keep changes minimal and accessible
- **Documentation** — fix typos, clarify explanations

### What to avoid

- Don't add new dependencies without discussion
- Don't change the statistical engine without understanding the methodology (read `docs/METHODOLOGY.md`)
- Don't break the Streamlit Community Cloud deployment
- Don't add secrets, API keys, or credentials

### Pull request checklist

- [ ] Tests pass locally (`pytest -v -m "not slow"`)
- [ ] Lint passes (`ruff check`)
- [ ] No new warnings
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

---

## Key Concepts

### Counterfactual

The counterfactual is a prediction of what *would have happened* if the intervention never occurred. It's estimated by fitting a model on pre-intervention data and forecasting forward.

### Effect size

The average difference between the observed values and the counterfactual after the intervention. Positive = the intervention increased the metric. Negative = it decreased it.

### Statistical significance (p < 0.05)

If the p-value is below 0.05, we reject the null hypothesis that the intervention had no effect. This means the observed effect is unlikely to be due to chance alone.

### 95% Confidence Interval

The range within which the true effect likely falls. If the interval includes zero, the effect may not be real.

### Placebo test

Run the same analysis at multiple fake intervention dates. If the real effect is larger than 95% of placebo effects, the result is robust.

---

## References

- Brodersen et al. "Inferring Causal Impact Using Bayesian Structural Time-Series Models." *Annals of Applied Statistics*, 2015.
- Lopez Bernal et al. "Interrupted time series regression for the evaluation of public health interventions." *International Journal of Epidemiology*, 2017.
- [Google CausalImpact](https://google.github.io/CausalImpact/)
- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
