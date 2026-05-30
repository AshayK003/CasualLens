# Datasets

CausalLens includes 8 synthetic datasets with **known ground truth** for testing causal inference methods.

## Regenerating Datasets

```bash
python scripts/generate_datasets.py
```

This generates all CSVs in `src/data/datasets/` and a `metadata.json` with ground truth.

## Dataset Summary

| Dataset | Frequency | Rows | Intervention | True Effect |
|---------|-----------|------|-------------|-------------|
| Delhi Air Quality | Daily | 3,650 | 2020-03-25 (Lockdown) | -40 PM2.5 (60 days) |
| GST Revenue | Monthly | 96 | 2017-07-01 (GST) | +80,000 Cr (permanent) |
| Hospital Admissions | Weekly | 312 | 2021-01-01 (Vaccine) | -30 admissions (gradual) |
| Electricity Demand | Hourly | 17,520 | 2023-06-01 (Pricing) | -8% peak hours |
| Crime Rates | Monthly | 108 | 2020-01-01 (Policing) | -12 incidents/month |
| Student Scores | Yearly | 14 | 2019-01-01 (Curriculum) | +5 points |
| Traffic Accidents | Daily | 1,825 | 2021-06-01 (Roundabout) | -3 accidents/day |
| Website Sessions | Daily | 730 | 2023-03-01 (Campaign) | +500 sessions (gradual) |

## Ground Truth

Each dataset has documented ground truth in `src/data/datasets/metadata.json`:

```json
{
  "delhi_aqi": {
    "ground_truth": {
      "effect_size": -40.0,
      "effect_direction": "decrease",
      "effect_duration": "60 days (transient)",
      "expected_effect_range": [-55, -25],
      "expected_p_value_range": [0.0, 0.05]
    }
  }
}
```

## Generator Architecture

`scripts/generate_datasets.py` uses a `TimeSeriesBuilder` class that composes time series from additive components:

- **Trend**: Linear slope + intercept
- **Seasonality**: Single or double sine waves
- **AR noise**: Autoregressive errors (AR(p))
- **Interventions**: Level shifts, transient effects, gradual ramp-ups
- **Weekly patterns**: Weekday/weekend effects

Each generator is seeded for deterministic output.

## Test Validation

`tests/test_datasets.py` validates that:
1. All CSVs exist and are loadable
2. All dates parse correctly
3. All metrics are numeric
4. ARIMA detects the known effect within expected ranges
