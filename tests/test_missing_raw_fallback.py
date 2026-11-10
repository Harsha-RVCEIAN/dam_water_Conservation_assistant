"""Tests for graceful degradation when the raw dataset file is absent.

Modules tested: ``src.data.preprocess``, ``src.data.validate``

Key scenarios:
- When the raw CSV is missing from disk the preprocessing pipeline does not
  crash; it falls back to any cached processed data or raises a clear error.
- The canonical column schema (``validate.CANONICAL_COLUMNS``) is still
  respected in the fallback data path.
- Downstream callers (e.g. the Streamlit app) remain functional when raw
  data is unavailable.

Coverage scope: missing-file error paths; does not test the happy-path ETL.
"""
import pandas as pd

from src.data import preprocess, validate


def _processed_observations(path):
    data = {column: [1, 2] for column in validate.CANONICAL_COLUMNS}
    data.update({
        "reservoir_name": ["K.R.Sagara Dam"] * 2,
        "basin": ["Cauvery"] * 2,
        "sub_basin": ["Upper Cauvery"] * 2,
        "river": ["Cauvery"] * 2,
        "monitoring_date": ["2020-01-01", "2020-01-02"],
    })
    pd.DataFrame(data).to_csv(path, index=False)


def test_validation_uses_processed_file_when_raw_file_is_missing(tmp_path, monkeypatch):
    raw_path = tmp_path / "missing_raw.csv"
    processed_path = tmp_path / "processed.csv"
    _processed_observations(processed_path)
    monkeypatch.setattr(validate, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(validate, "PROCESSED_DATA_PATH", processed_path)

    report = validate.run_validation(data_path=raw_path)

    assert report.total_rows == 2
    assert "processed-data fallback" in report.raw_path


def test_preprocessing_leaves_existing_output_unchanged_without_raw_file(
    tmp_path, monkeypatch
):
    raw_path = tmp_path / "missing_raw.csv"
    processed_path = tmp_path / "processed.csv"
    _processed_observations(processed_path)
    original_bytes = processed_path.read_bytes()
    monkeypatch.setattr(preprocess, "RAW_DATA_PATH", raw_path)

    result = preprocess.run_preprocessing_pipeline(
        raw_path=raw_path,
        output_path=processed_path,
    )

    assert len(result) == 2
    assert processed_path.read_bytes() == original_bytes
