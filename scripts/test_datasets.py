import sys
sys.path.insert(0, ".")

import pandas as pd
from src.data.loader import load_dataset, get_available_datasets
from src.core.engine import causal_effect, Method
from src.utils.validators import validate_dataframe

print("=== Testing Pre-loaded Datasets ===\n")

datasets = get_available_datasets()
for name, meta in datasets.items():
    print(f"Dataset: {name}")
    print(f"  Label: {meta['label']}")
    print(f"  Date col: {meta['date_col']}")
    print(f"  Metric col: {meta['metric_col']}")
    print(f"  Intervention: {meta['intervention_date']}")

    try:
        df = load_dataset(name)
        print(f"  Rows: {len(df)}, Cols: {list(df.columns)}")
        print(f"  Date dtype: {df[meta['date_col']].dtype}")
        print(f"  Metric dtype: {df[meta['metric_col']].dtype}")

        date_col, metric_col = validate_dataframe(df)
        print(f"  Validation OK: date={date_col}, metric={metric_col}")

        result = causal_effect(df, date_col, metric_col, meta["intervention_date"], Method.ARIMA)
        print(f"  Effect: {result.effect:+.2f}")
        print(f"  p-value: {result.p_value:.4f}")
        print(f"  Significant: {result.significant}")
        print(f"  PASSED\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")

print("=== Done ===")
