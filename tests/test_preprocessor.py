import numpy as np
import pandas as pd

from src.data.preprocessor import (
    clean_column_names,
    detect_date_column,
    detect_numeric_columns,
    drop_duplicates,
    handle_missing_values,
    parse_dates,
    parse_numeric,
    preprocess_data,
    remove_outliers,
)


class TestDetectDateColumn:
    def test_detects_datetime_column(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })
        assert detect_date_column(df) == "date"

    def test_detects_string_date_column(self):
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3],
        })
        assert detect_date_column(df) == "date"

    def test_returns_none_for_no_dates(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "value": [1, 2, 3],
        })
        assert detect_date_column(df) is None


class TestDetectNumericColumns:
    def test_detects_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert detect_numeric_columns(df) == ["a"]

    def test_detects_string_numeric(self):
        df = pd.DataFrame({"a": ["1.5", "2.5", "3.5"], "b": ["x", "y", "z"]})
        assert detect_numeric_columns(df) == ["a"]

    def test_excludes_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert detect_numeric_columns(df, exclude=["a"]) == ["b"]


class TestCleanColumnNames:
    def test_cleans_names(self):
        df = pd.DataFrame({"My Column ": [1], " another-col ": [2], "UPPER": [3]})
        cleaned, steps = clean_column_names(df)
        assert list(cleaned.columns) == ["my_column", "another_col", "upper"]
        assert len(steps) == 3

    def test_no_change_needed(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        _, steps = clean_column_names(df)
        assert len(steps) == 0


class TestDropDuplicates:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        cleaned, steps = drop_duplicates(df)
        assert len(cleaned) == 2
        assert any("duplicate" in s.lower() for s in steps)


class TestParseDates:
    def test_parses_string_dates(self):
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "v": [1, 2]})
        cleaned, steps = parse_dates(df, "date")
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_handles_already_datetime(self):
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "v": [1, 2]})
        _, steps = parse_dates(df, "date")
        assert len(steps) == 0


class TestParseNumeric:
    def test_converts_string_numeric(self):
        df = pd.DataFrame({"val": ["1.5", "2.5", "bad", "4.5"]})
        cleaned, steps = parse_numeric(df, "val")
        assert pd.api.types.is_numeric_dtype(cleaned["val"])
        assert cleaned["val"].iloc[0] == 1.5
        assert any("NaN" in s for s in steps)

    def test_handles_clean_numeric(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        _, steps = parse_numeric(df, "val")
        assert len(steps) == 0


class TestHandleMissingValues:
    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        cleaned, steps = handle_missing_values(df, strategy="drop")
        assert len(cleaned) == 2
        assert any("Dropped" in s for s in steps)

    def test_forward_fill_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cleaned, steps = handle_missing_values(df, strategy="forward_fill")
        assert cleaned["a"].iloc[1] == 1.0
        assert any("Forward-filled" in s for s in steps)

    def test_mean_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cleaned, steps = handle_missing_values(df, strategy="mean")
        assert cleaned["a"].iloc[1] == 2.0
        assert any("mean" in s for s in steps)


class TestRemoveOutliers:
    def test_removes_iqr_outliers(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        cleaned, steps = remove_outliers(df, "val", method="iqr")
        assert len(cleaned) < 6
        assert any("outlier" in s.lower() for s in steps)

    def test_no_change_for_normal_data(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.randn(200)})
        cleaned, steps = remove_outliers(df, "val", method="iqr")
        assert len(cleaned) >= 190


class TestPreprocessData:
    def test_full_pipeline_messy_data(self):
        df = pd.DataFrame({
            " Date ": ["2020-01-01", "2020-01-02", "2020-01-02", "", "2020-01-04"],
            " Value ": ["1.5", "2.5", "2.5", "bad", "4.5"],
            "Empty": [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        cleaned, report = preprocess_data(df, missing_strategy="drop")

        assert report.original_rows == 5
        assert report.original_cols == 3
        assert report.final_rows > 0
        assert report.final_cols == 2
        assert len(report.steps_applied) > 0
        assert "Empty" not in cleaned.columns

    def test_preserves_clean_data(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "value": range(10),
        })

        cleaned, report = preprocess_data(df, date_col="date", metric_col="value")

        assert len(cleaned) == 10
        assert report.final_cols == 2

    def test_auto_detects_columns(self):
        df = pd.DataFrame({
            "my_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "my_value": [1, 2, 3],
        })

        cleaned, report = preprocess_data(df)
        assert any("date" in w.lower() for w in report.warnings)
        assert any("metric" in w.lower() for w in report.warnings)
