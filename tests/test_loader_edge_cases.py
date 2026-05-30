import io

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    AVAILABLE_DATASETS,
    load_dataset,
    load_uploaded_file,
)


def _xlwt_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("xlwt") is not None


class TestLoadUploadedFileEdgeCases:
    def test_csv_with_extra_whitespace(self):
        csv_data = "date , value\n2020-01-01 , 10\n2020-01-02 , 20\n"
        f = io.BytesIO(csv_data.encode())
        f.name = "test.csv"
        df = load_uploaded_file(f)
        assert len(df) == 2

    def test_csv_with_bom(self):
        csv_data = "\ufeffdate,value\n2020-01-01,10\n2020-01-02,20\n"
        f = io.BytesIO(csv_data.encode("utf-8"))
        f.name = "test.csv"
        df = load_uploaded_file(f)
        assert len(df) == 2

    def test_excel_file(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        buf.name = "test.xlsx"
        buf.size = len(buf.getvalue())
        loaded = load_uploaded_file(buf)
        assert len(loaded) == 5

    def test_file_too_large(self):
        f = io.BytesIO(b"data\n" * 100)
        f.name = "large.csv"
        f.size = 60_000_000  # 60MB
        with pytest.raises(ValueError, match="too large"):
            load_uploaded_file(f)

    def test_file_with_only_one_column(self):
        csv_data = "value\n1\n2\n3\n"
        f = io.BytesIO(csv_data.encode())
        f.name = "one_col.csv"
        with pytest.raises(ValueError, match="at least 2 columns"):
            load_uploaded_file(f)

    def test_unsupported_format(self):
        f = io.BytesIO(b"some data")
        f.name = "test.json"
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_uploaded_file(f)

    @pytest.mark.skipif(
        not _xlwt_available(),
        reason="xlwt engine not installed"
    )
    def test_xls_format(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="xlwt")
        buf.seek(0)
        buf.name = "test.xls"
        buf.size = len(buf.getvalue())
        loaded = load_uploaded_file(buf)
        assert len(loaded) == 5


class TestLoadDatasetEdgeCases:
    def test_all_datasets_are_loadable(self):
        for name in AVAILABLE_DATASETS:
            df = load_dataset(name)
            assert len(df) > 0
            assert len(df.columns) >= 2

    def test_unknown_dataset_name(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_metadata_json_matches_datasets(self):
        import json
        from pathlib import Path

        metadata_path = Path(__file__).parent.parent / "src" / "data" / "datasets" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            for name in AVAILABLE_DATASETS:
                assert name in metadata, f"Dataset '{name}' not in metadata.json"
