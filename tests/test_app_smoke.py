"""Smoke tests for application imports and UI helper functions."""

from pathlib import Path
import pytest


def test_app_imports_cleanly():
    """Verify that src.app and its components import without error."""
    import src.app as app
    assert hasattr(app, "main")
    assert hasattr(app, "load_processed_data")
    assert hasattr(app, "load_forecaster")


def test_processed_dataset_contract():
    """Verify that processed dataset exists and matches expected minimum schema."""
    import pandas as pd
    processed_path = Path("data/processed/krs_dam_processed.csv")
    if not processed_path.exists():
        pytest.skip("Processed dataset not generated yet.")

    df = pd.read_csv(processed_path, nrows=5)
    expected_cols = [
        "calendar_date",
        "inflow_cusecs",
        "outflow_cusecs",
        "withdrawal_cusecs",
        "gross_capacity_tmc",
        "inflow_vol_tmc",
        "net_flow_tmc",
        "target_inflow_next_day_cusecs",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} in processed data"
