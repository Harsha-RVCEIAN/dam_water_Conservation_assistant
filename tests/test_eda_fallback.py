"""Tests for EDA (Exploratory Data Analysis) fallback behaviour.

Module tested: ``src.data.explore``

Key scenarios:
- When the raw input CSV is absent, the EDA module falls back to the
  preprocessed / cached dataset instead of raising an unhandled error.
- Generated figures are either produced from the fallback data or handled
  gracefully when figure artifacts are unavailable.

Coverage scope: fallback path only; does not test the full EDA chart library.
"""
import pandas as pd

from src.data import explore


def test_eda_uses_processed_data_when_raw_input_is_missing(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    inflow = pd.Series(range(1000, 3400, 100), dtype=float)
    processed_data = pd.DataFrame({
        "monitoring_date": dates,
        "inflow_cusecs": inflow,
        "outflow_cusecs": inflow / 2,
        "withdrawal_cusecs": inflow / 4,
        "evaporation_cusecs": inflow / 10,
        "gross_capacity_tmc": range(20, 44),
        "reservoir_level_ft": range(2440, 2464),
        "cumulative_inflow_tmc": inflow.cumsum() * explore.CUSEC_DAY_TO_TMC,
    })
    raw_path = tmp_path / "missing_raw.csv"
    processed_path = tmp_path / "processed.csv"
    report_path = tmp_path / "eda_summary.md"
    processed_data.to_csv(processed_path, index=False)

    monkeypatch.setattr(explore, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(explore, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(explore, "EDA_REPORT_PATH", report_path)

    result_path = explore.run_eda(raw_path)

    report = result_path.read_text(encoding="utf-8")
    assert result_path == report_path
    assert "read-only processed-data fallback" in report
    assert "24 observations" in report
