import numpy as np
import pandas as pd

from src.data.preprocessor import (
    PreprocessReport,
    clean_column_names,
    detect_date_column,
    detect_numeric_columns,
    drop_empty_columns,
    drop_empty_rows,
    handle_missing_values,
    parse_dates,
    parse_numeric,
    preprocess_data,
    remove_outliers,
    try_construct_date_from_year_month,
)


class TestDetectDateColumnEdgeCases:
    def test_all_string_dates(self):
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3],
        })
        result = detect_date_column(df)
        assert result == "date"

    def test_mixed_formats(self):
        df = pd.DataFrame({
            "date": ["01/01/2020", "01/02/2020", "01/03/2020"],
            "value": [1, 2, 3],
        })
        result = detect_date_column(df)
        assert result == "date"

    def test_no_dates_at_all(self):
        df = pd.DataFrame({
            "text_a": ["hello", "world", "foo"],
            "text_b": ["bar", "baz", "qux"],
        })
        result = detect_date_column(df)
        assert result is None

    def test_numeric_not_date(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        result = detect_date_column(df)
        assert result is None


class TestDetectNumericColumnsEdgeCases:
    def test_currency_strings(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "price": ["$10.50", "$20.00", "$30.75"],
        })
        result = detect_numeric_columns(df, exclude=["date"])
        assert "price" in result

    def test_percentage_strings(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "rate": ["5.5%", "10.2%", "15.0%"],
        })
        result = detect_numeric_columns(df, exclude=["date"])
        assert "rate" in result

    def test_all_text_columns(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "city": ["NYC", "LA", "SF"],
        })
        result = detect_numeric_columns(df)
        assert result == []

    def test_exclude_multiple_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "category": ["A", "B", "C"],
            "value": [1, 2, 3],
        })
        result = detect_numeric_columns(df, exclude=["date", "value"])
        assert result == []


class TestCleanColumnNamesEdgeCases:
    def test_special_characters(self):
        df = pd.DataFrame({
            "Col!@#$%": [1, 2],
            "Col Name": [3, 4],
        })
        result_df, steps = clean_column_names(df)
        assert "col_name" in result_df.columns
        assert len(steps) > 0

    def test_all_empty_after_clean(self):
        df = pd.DataFrame({
            "!!!": [1, 2],
            "@@@": [3, 4],
        })
        result_df, steps = clean_column_names(df)
        assert all(c.startswith("col_") for c in result_df.columns)

    def test_no_changes_needed(self):
        df = pd.DataFrame({
            "already_clean": [1, 2],
            "also_clean": [3, 4],
        })
        result_df, steps = clean_column_names(df)
        assert list(result_df.columns) == ["already_clean", "also_clean"]
        assert steps == []


class TestDropEmptyColumns:
    def test_removes_all_nan_columns(self):
        df = pd.DataFrame({
            "good": [1, 2, 3],
            "empty": [np.nan, np.nan, np.nan],
        })
        result_df, steps = drop_empty_columns(df)
        assert "empty" not in result_df.columns
        assert "good" in result_df.columns
        assert len(steps) > 0

    def test_no_empty_columns(self):
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
        })
        result_df, steps = drop_empty_columns(df)
        assert list(result_df.columns) == ["a", "b"]
        assert steps == []


class TestDropEmptyRows:
    def test_removes_all_nan_rows(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [4, np.nan, 6],
        })
        result_df, steps = drop_empty_rows(df)
        assert len(result_df) == 2
        assert len(steps) > 0

    def test_no_empty_rows(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        result_df, steps = drop_empty_rows(df)
        assert len(result_df) == 3
        assert steps == []


class TestParseDatesEdgeCases:
    def test_iso_format(self):
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "date")
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

    def test_us_format(self):
        df = pd.DataFrame({
            "date": ["01/15/2020", "02/15/2020", "03/15/2020"],
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "date")
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])

    def test_numeric_year_out_of_range(self):
        df = pd.DataFrame({
            "year": [99999999, 100000000, 100000001],
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "year")
        assert any("out of datetime range" in s or "failed" in s.lower() or len(steps) > 0 for s in steps)

    def test_already_datetime(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3),
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "date")
        assert steps == []

    def test_partial_parse_failure(self):
        df = pd.DataFrame({
            "date": ["2020-01-01", "not-a-date", "2020-01-03"],
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "date")
        assert any("dropped" in s.lower() or "invalid" in s.lower() for s in steps)

    def test_all_invalid_dates(self):
        df = pd.DataFrame({
            "date": ["not", "a", "date"],
            "value": [1, 2, 3],
        })
        result_df, steps = parse_dates(df, "date")
        assert any("failed" in s.lower() for s in steps)


class TestTryConstructDateFromYearMonth:
    def test_year_and_month_columns(self):
        df = pd.DataFrame({
            "year": [2020, 2020, 2020],
            "month": [1, 2, 3],
            "value": [1, 2, 3],
        })
        result_df, date_col, steps = try_construct_date_from_year_month(df)
        assert date_col is not None
        assert pd.api.types.is_datetime64_any_dtype(result_df[date_col])

    def test_year_only_column(self):
        df = pd.DataFrame({
            "year": [2020, 2021, 2022],
            "value": [1, 2, 3],
        })
        result_df, date_col, steps = try_construct_date_from_year_month(df)
        assert date_col is not None

    def test_no_year_column(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [1, 2, 3],
        })
        result_df, date_col, steps = try_construct_date_from_year_month(df)
        assert date_col is None

    def test_fiscal_year_column(self):
        df = pd.DataFrame({
            "fiscal_year": [2020, 2021, 2022],
            "value": [1, 2, 3],
        })
        result_df, date_col, steps = try_construct_date_from_year_month(df)
        assert date_col is not None


class TestParseNumericEdgeCases:
    def test_with_commas(self):
        df = pd.DataFrame({"value": ["1,000", "2,000", "3,000"]})
        result_df, steps = parse_numeric(df, "value")
        assert pd.api.types.is_numeric_dtype(result_df["value"])

    def test_with_currency_symbol(self):
        df = pd.DataFrame({"price": ["$10", "$20", "$30"]})
        result_df, steps = parse_numeric(df, "price")
        assert pd.api.types.is_numeric_dtype(result_df["price"])

    def test_already_numeric(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result_df, steps = parse_numeric(df, "value")
        assert steps == []

    def test_all_non_numeric(self):
        df = pd.DataFrame({"text": ["hello", "world", "foo"]})
        result_df, steps = parse_numeric(df, "text")
        assert len(result_df) == 3
        assert result_df["text"].isna().all()


class TestHandleMissingValuesEdgeCases:
    def test_drop_with_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result_df, steps = handle_missing_values(df, strategy="drop")
        assert len(result_df) == 3

    def test_forward_fill_preserves_length(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        result_df, steps = handle_missing_values(df, strategy="forward_fill")
        assert len(result_df) == 3
        assert result_df["a"].iloc[1] == 1.0

    def test_mean_fill(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result_df, steps = handle_missing_values(df, strategy="mean")
        assert result_df["a"].iloc[1] == 2.0


class TestRemoveOutliersEdgeCases:
    def test_iqr_removes_outliers(self):
        np.random.seed(42)
        data = np.concatenate([np.random.randn(100) * 2, [100, -100]])
        df = pd.DataFrame({"value": data})
        result_df, steps = remove_outliers(df, "value", method="iqr")
        assert len(result_df) < len(df)

    def test_zscore_removes_outliers(self):
        np.random.seed(42)
        data = np.concatenate([np.random.randn(100) * 2, [100, -100]])
        df = pd.DataFrame({"value": data})
        result_df, steps = remove_outliers(df, "value", method="zscore", threshold=3.0)
        assert len(result_df) < len(df)

    def test_non_numeric_column(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result_df, steps = remove_outliers(df, "text")
        assert len(result_df) == 3
        assert steps == []

    def test_no_outliers_in_normal_data(self):
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(100)})
        result_df, steps = remove_outliers(df, "value", method="iqr")
        assert len(result_df) >= 90
        assert len(steps) <= 1


class TestPreprocessDataEdgeCases:
    def test_auto_detect_all_columns(self):
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        df = pd.DataFrame({"date": dates, "value": y})
        result_df, report = preprocess_data(df)
        assert report.final_rows > 0
        assert isinstance(report, PreprocessReport)

    def test_report_tracks_all_steps(self):
        np.random.seed(42)
        df = pd.DataFrame({
            " My Col ": [1, 2, 3],
            "Another Col": [4, 5, 6],
        })
        result_df, report = preprocess_data(df)
        assert len(report.steps_applied) > 0
        assert report.original_cols == 2

    def test_warning_on_few_rows(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = np.random.randn(5)
        df = pd.DataFrame({"date": dates, "value": y})
        result_df, report = preprocess_data(df, missing_strategy="drop")
        assert any("rows remaining" in w.lower() for w in report.warnings)

    def test_constructed_date_warning(self):
        df = pd.DataFrame({
            "year": [2020, 2021, 2022],
            "month": [1, 2, 3],
            "value": [1, 2, 3],
        })
        result_df, report = preprocess_data(df)
        assert len(report.warnings) > 0
        assert any("date" in w.lower() for w in report.warnings)

    def test_outlier_removal_flag(self):
        np.random.seed(42)
        data = np.concatenate([np.random.randn(100) * 2, [100, -100]])
        dates = pd.date_range("2020-01-01", periods=len(data), freq="D")
        df = pd.DataFrame({"date": dates, "value": data})
        result_df, report = preprocess_data(
            df, remove_outliers_flag=True, outlier_method="iqr"
        )
        assert any("outlier" in s.lower() for s in report.steps_applied)

    def test_forward_fill_strategy(self):
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[10] = np.nan
        df = pd.DataFrame({"date": dates, "value": y})
        result_df, report = preprocess_data(df, missing_strategy="forward_fill")
        assert result_df["value"].isna().sum() == 0

    def test_mean_strategy(self):
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 50 + np.random.randn(n)
        y[10] = np.nan
        y[20] = np.nan
        df = pd.DataFrame({"date": dates, "value": y})
        result_df, report = preprocess_data(df, missing_strategy="mean")
        assert result_df["value"].isna().sum() == 0
