# Datasets

## Pre-loaded Datasets

### Delhi Air Quality (PM2.5)
- **File**: `src/data/datasets/delhi_aqi.csv`
- **Rows**: 3,650 (10 years of daily data)
- **Columns**: `date`, `pm25`
- **Intervention date**: 2020-03-25 (India COVID lockdown)
- **Source**: Synthetic data based on real Delhi AQI patterns

### India GST Revenue
- **File**: `src/data/datasets/gst_revenue.csv`
- **Rows**: 84 (7 years of monthly data)
- **Columns**: `month`, `revenue_cr`
- **Intervention date**: 2017-07-01 (GST implementation)
- **Source**: Synthetic data based on real GST collection patterns

## Generating Datasets

```bash
python scripts/generate_datasets.py
```

## Adding Your Own Datasets

1. Create a CSV with at least a date column and a numeric metric column
2. Place it in `src/data/datasets/`
3. Add an entry to `AVAILABLE_DATASETS` in `src/data/loader.py`

## Data Quality Notes

- All pre-loaded datasets are synthetic (generated from realistic patterns)
- For real-world analysis, use actual government data sources
- Good sources: FRED, RBI DBIE, CPCB, WHO GHO
