# CausalLens

**Causal Impact Calculator** — Did this policy actually work?

CausalLens estimates the causal effect of policy interventions on time series. Given an intervention date, it builds a counterfactual (what *would have happened* without the intervention), then compares it to observed data. Results include effect size, p-values, 95% confidence intervals, and a downloadable PDF report.

Designed for Streamlit Community Cloud free tier: CPU-only, 1GB RAM, zero-config deployment.

## Architecture

```
app.py                        # Streamlit UI (thin — no business logic)
src/
├── core/
│   ├── engine.py             # causal_effect() — orchestrates analysis
│   ├── arima_its.py          # ARIMA interrupted time series
│   ├── bsts.py               # Bayesian STS wrapper (Google CausalImpact)
│   └── placebo.py            # Sensitivity: reruns at fake intervention dates
├── data/
│   ├── loader.py             # CSV/Excel upload, pre-loaded dataset registry
│   ├── preprocessor.py       # Date detection, missing values, outlier removal
│   └── datasets/             # 8 pre-loaded policy datasets (Delhi AQI, GST, etc.)
├── reports/
│   ├── plots.py              # Plotly counterfactual chart (with optional CI shading)
│   ├── summary.py            # Plain-language result summary
│   └── pdf_export.py         # Matplotlib + ReportLab PDF
└── utils/
    ├── constants.py          # SIGNIFICANCE_LEVEL=0.05, MIN_DATA_POINTS, etc.
    ├── formatters.py         # Number/CI/p-value formatting helpers
    └── validators.py         # DataFrame validation, intervention date checks
tests/                        # 204 tests across 17 files
docs/
├── CONTRIBUTING.md           # Full developer guide
├── METHODOLOGY.md            # Statistical methods (plain English)
└── DATASETS.md               # Pre-loaded dataset reference
```

### Data flow

```
Upload CSV / select dataset
  → loader.py → raw DataFrame
    → preprocessor.py → clean DataFrame + PreprocessReport
      → engine.py → CausalResult (effect, p-value, CI, counterfactual)
        ├── replots.py → Plotly chart (displayed in UI)
        ├── summary.py → plain-text (displayed in UI)
        ├── pdf_export.py → PDF bytes (downloadable)
        └── placebo.py → sensitivity histogram (optional)
```

### Why this structure

- `src/core/` is pure Python with zero Streamlit dependency — testable in isolation, importable from scripts or notebooks.
- `src/data/` owns all I/O and preprocessing — file uploads, CSV parsing, column detection.
- `src/reports/` generates all outputs — charts, summaries, PDFs.
- `app.py` is intentionally thin — it wires components together and manages Streamlit state. Business logic lives in `src/`.

## Setup

```bash
git clone https://github.com/AshayK003/CausalLens.git
cd CausalLens
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, ruff
```

To regenerate pre-loaded datasets (if deleted):
```bash
python scripts/generate_datasets.py
```

## Environment Variables

CausalLens uses **none** for core functionality.

| File | Purpose | Gitignored |
|------|---------|------------|
| `.streamlit/secrets.toml` | Streamlit Cloud secrets | Yes |
| `.streamlit/config.toml` | Theme (dark mode, `#6366f1` primary) | No |

## Local Development

```bash
streamlit run app.py                    # → http://localhost:8501
streamlit run app.py --server.port 8502 # if 8501 is in use
```

For Windows background execution:
```powershell
Start-Process -WindowStyle Hidden -FilePath "streamlit" -ArgumentList "run app.py --server.port 8501"
```

### Code conventions

- **Linter**: ruff (line-length 100, target Python 3.10)
- **Imports**: ruff auto-sorts (isort rules)
- **Type hints**: required on all new public functions
- **Logging**: use `logging.getLogger(__name__)`, never `print()`
- **No comments** unless explaining *why*, not *what*

Run lint before committing:
```bash
ruff check src/ tests/ app.py
```

## Testing

```bash
# Fast (~30s) — skips slow dataset ground-truth checks
python -m pytest tests/ -v -m "not slow"

# Full suite (~4 min)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Test file map

| File | What it covers |
|------|---------------|
| `test_engine.py` | `causal_effect()` — ARIMA + BSTS |
| `test_engine_edge_cases.py` | Short series, single values, monotonic data |
| `test_preprocessor.py` | Date/numeric detection, missing value strategies |
| `test_preprocessor_edge_cases.py` | Year columns, empty data, edge formats |
| `test_loader.py` | CSV loading, file size/format validation |
| `test_loader_edge_cases.py` | Empty files, wrong formats, boundary sizes |
| `test_placebo_edge_cases.py` | Short series, extreme indices, single-placebo edge |
| `test_reports.py` | Summary text, PDF generation |
| `test_reports_edge_cases.py` | Plot edge cases, formatting tails |
| `test_formatters.py` | Number/CI/p-value formatting |
| `test_datasets.py` | Pre-loaded dataset ground truth |
| `test_integration.py` | Full pipeline: load → preprocess → analyze |
| `test_integration_comprehensive.py` | All methods × all datasets |
| `test_bsts_placebo.py` | BSTS + placebo interaction |
| `test_pdf.py` | PDF structural integrity |

Add new tests to the existing files or create new `test_<module>.py` files. Mark slow tests with `@pytest.mark.slow`.

## Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect `AshayK003/CausalLens`, set entry point to `app.py`
4. Deploy — auto-installs `requirements.txt` on each push

### Self-hosted

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| "Analysis failed" | Intervention date outside data range | Pick a date within the data bounds |
| BSTS error on Windows | `causalimpact` dependency issue | Use ARIMA ITS method (default) |
| Port 8501 already in use | Previous process didn't die | `streamlit run app.py --server.port 8502` |
| "No valid data after date parsing" | Unparseable date column | Format dates as YYYY-MM-DD |
| Tests fail after pull | Dependencies changed | `pip install -r requirements-dev.txt` |
| Push rejected (workflow) | PAT lacks `workflow` scope | Remove `.github/` or update token at github.com/settings/tokens |

## Contribution Guidelines

1. Open an issue before starting architectural work
2. Branch from `main`: `git checkout -b feature/my-change`
3. Make changes in `src/`, not `app.py` unless UI-only
4. Ensure `ruff check` passes and `pytest -v -m "not slow"` is green
5. Push and open a PR

### What's welcome

- Bug fixes, edge cases, test coverage
- New pre-loaded datasets (add CSV to `src/data/datasets/`, register in `loader.py`)
- UI polish that doesn't break the existing flow
- Documentation fixes

### What to avoid

- New dependencies without discussion (1GB RAM budget)
- Changes to `src/core/` statistical logic without understanding `docs/METHODOLOGY.md`
- API keys, secrets, or credentials in any file
- Breaking Streamlit Cloud compatibility

Full guide in `docs/CONTRIBUTING.md`.

## Methods

| Method | Speed | Use case |
|--------|-------|----------|
| **ARIMA ITS** | Seconds | Default. Works on any dataset with 30+ points. |
| **Bayesian STS** | 1-2 min | Seasonal patterns, probabilistic CIs. Experimental. |

### Key Concepts

- **Counterfactual**: A statistical prediction of what would have happened without the intervention.
- **Effect size**: Average difference between observed and counterfactual post-intervention.
- **p < 0.05**: The effect is unlikely to be due to random chance.
- **95% CI**: Range containing the true effect with 95% probability. If it spans zero, the effect is uncertain.
- **Placebo test**: Reruns analysis at fake dates. If the real effect exceeds 95% of placebo effects, the result is robust.

## Dependencies

| Package | Role |
|---------|------|
| streamlit | UI framework |
| statsmodels | ARIMA ITS model |
| causalimpact | Bayesian STS |
| pandas | Data manipulation |
| plotly | Interactive charts |
| scipy | Statistical tests |
| reportlab | PDF generation |
| matplotlib | PDF chart rendering |
| openpyxl | Excel file support |

## License

MIT

## Citation

```bibtex
@software{causallens2026,
  title={CausalLens: Causal Impact Calculator},
  year={2026},
  url={https://github.com/AshayK003/CausalLens}
}
```
