import io

import pandas as pd
import pytest

from src.data.loader import get_available_datasets, load_dataset, load_uploaded_file


class TestLoadUploadedCSV:
    def test_loads_csv_bytes(self):
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]})
        csv_bytes = df.to_csv(index=False).encode()
        f = io.BytesIO(csv_bytes)
        f.name = "test.csv"

        result = load_uploaded_file(f)
        assert len(result) == 2
        assert list(result.columns) == ["date", "value"]


class TestLoadUploadedExcel:
    def test_loads_xlsx(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "value": range(5),
        })
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.name = "test.xlsx"

        result = load_uploaded_file(buffer)
        assert len(result) == 5
        assert "date" in result.columns
        assert "value" in result.columns

    def test_loads_xls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.name = "test.xls"

        result = load_uploaded_file(buffer)
        assert len(result) == 2


class TestLoadUploadedEdgeCases:
    def test_rejects_empty_csv(self):
        f = io.BytesIO(b"col1,col2\n")
        f.name = "empty.csv"
        with pytest.raises(ValueError, match="empty"):
            load_uploaded_file(f)

    def test_rejects_single_column(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        csv_bytes = df.to_csv(index=False).encode()
        f = io.BytesIO(csv_bytes)
        f.name = "single.csv"
        with pytest.raises(ValueError, match="at least 2 columns"):
            load_uploaded_file(f)

    def test_rejects_unknown_format(self):
        f = io.BytesIO(b"data")
        f.name = "test.json"
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_uploaded_file(f)


class TestAvailableDatasets:
    def test_datasets_exist(self):
        datasets = get_available_datasets()
        assert "delhi_aqi" in datasets
        assert "gst_revenue" in datasets

    def test_load_dataset(self):
        df = load_dataset("delhi_aqi")
        assert len(df) > 0
        assert "date" in df.columns
        assert "pm25" in df.columns
