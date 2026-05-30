# CausalLens

**Causal Impact Calculator** — Did this policy actually work?

An open-source Python tool that estimates the causal effect of a policy intervention on a time series. Upload a CSV, pick an intervention date, and get a counterfactual analysis with confidence intervals, statistical significance, and a downloadable PDF report.

## Features

- **Counterfactual visualization** — see what would have happened without the policy
- **Statistical significance** — p-values, confidence intervals, effect estimates
- **Multiple methods** — ARIMA ITS (fast) and Bayesian STS (advanced)
- **Pre-loaded datasets** — Delhi AQI, India GST Revenue, and more
- **PDF export** — download a formatted report with charts and interpretation
- **Sensitivity analysis** — placebo tests to verify robustness
- **Plain-language summaries** — no jargon, written for policy analysts

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

### Use

1. Open http://localhost:8501
2. Select a pre-loaded dataset or upload your own CSV
3. Pick an intervention date (policy change)
4. Click "Run Analysis"
5. View the counterfactual chart and download the PDF report

## Example: Delhi Lockdown Effect on Air Quality

1. Select "Delhi Air Quality (PM2.5)" from the sidebar
2. Intervention date: 2020-03-25 (India lockdown)
3. Click "Run Analysis"
4. Result: The lockdown caused a ~40% decrease in PM2.5, statistically significant (p < 0.001)

## Methods

### ARIMA ITS (Primary)
- Fast, reliable, works on all datasets
- Uses statsmodels ARIMA model
- Recommended for most use cases

### Bayesian STS (Experimental)
- Uses Google's CausalImpact methodology
- Provides posterior distributions
- May fail on some systems (beta dependency)

## Project Structure

```
CausalLens/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── core/
│   │   ├── engine.py       # Main causal_effect() function
│   │   ├── arima_its.py    # ARIMA ITS implementation
│   │   ├── bsts.py         # Bayesian STS wrapper
│   │   └── placebo.py      # Sensitivity tests
│   ├── data/
│   │   ├── loader.py       # CSV and dataset loading
│   │   ├── preprocessor.py # Data cleaning and column detection
│   │   └── datasets/       # Pre-loaded policy datasets
│   ├── reports/
│   │   ├── plots.py        # Counterfactual charts
│   │   ├── summary.py      # Plain-language summaries
│   │   └── pdf_export.py   # PDF report generator
│   └── utils/
│       ├── validators.py   # Input validation
│       ├── formatters.py   # Number/date formatting
│       └── constants.py    # Shared thresholds
├── tests/                  # 204 tests
├── docs/
│   ├── CONTRIBUTING.md     # Developer guide
│   ├── METHODOLOGY.md      # Statistical methods explained
│   └── DATASETS.md         # Dataset documentation
└── requirements.txt
```

## Testing

```bash
# Fast tests only (~30s)
python -m pytest tests/ -v -m "not slow"

# Full suite including dataset ground-truth checks (~4 min)
python -m pytest tests/ -v
```

## Deployment

### Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo
4. Deploy

### Local
```bash
streamlit run app.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Dashboard UI |
| causalimpact | Bayesian STS |
| statsmodels | ARIMA ITS |
| pandas | Data manipulation |
| plotly | Interactive charts |
| scipy | Statistical tests |
| reportlab | PDF generation |
| matplotlib | PDF chart rendering |
| openpyxl | Excel file upload support |

## Methodology

See `docs/METHODOLOGY.md` for a plain-English explanation of the statistical methods.

## Contributing

See `docs/CONTRIBUTING.md` for setup, testing, and contribution guidelines.

## License

MIT

## Citation

If you use CausalLens in research, please cite:

```bibtex
@software{causallens2026,
  title={CausalLens: Causal Impact Calculator},
  year={2026},
  url={https://github.com/AshayK003/CausalLens}
}
```
