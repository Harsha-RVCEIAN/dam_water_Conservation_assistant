"""Tests for rainfall feature timing, anti-leakage safeguards, and rolling window integrity.

Follows Section 12.3 of the Rainfall MLOps Reproducibility Plan.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.rainfall_features import build_rainfall_features, RAINFALL_FEATURE_COLUMNS


@pytest.fixture
def synthetic_dam_and_rainfall():
    """Generate 30 days of synthetic dam dates and rainfall observations."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    dam_dates_series = dates.to_series().dt.strftime("%Y-%m-%d").reset_index(drop=True)
    
    np.random.seed(42)
    rainfall = pd.DataFrame({
        "Monitoring Date": dates.strftime("%Y-%m-%d"),
        "Rainfall_MM": np.random.uniform(0.0, 25.0, size=30),
        "Stations_Reporting": [10] * 30,
        "Station_Coverage": [1.0] * 30
    })
    return dam_dates_series, rainfall


def test_feature_columns_contract(synthetic_dam_and_rainfall):
    """Verify that build_rainfall_features produces the exact expected feature columns."""
    dam_dates, rainfall = synthetic_dam_and_rainfall
    features = build_rainfall_features(dam_dates, rainfall)
    
    for col in RAINFALL_FEATURE_COLUMNS:
        assert col in features.columns, f"Missing feature column: {col}"
    assert len(features) == len(dam_dates)


def test_no_future_rainfall_affects_earlier_features(synthetic_dam_and_rainfall):
    """Modifying future rainfall (day >= t) must produce zero change in features at <= t."""
    dam_dates, rainfall = synthetic_dam_and_rainfall
    
    # Baseline features
    f_base = build_rainfall_features(dam_dates, rainfall)
    
    # Perturb rainfall from day 15 onwards (e.g. inject extreme storm)
    rainfall_perturbed = rainfall.copy()
    rainfall_perturbed.loc[15:, "Rainfall_MM"] += 500.0
    
    f_perturbed = build_rainfall_features(dam_dates, rainfall_perturbed)
    
    # Check that features for days 0 through 15 are byte-for-byte identical
    cols = list(RAINFALL_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        f_base.iloc[:15][cols],
        f_perturbed.iloc[:15][cols]
    )


def test_no_centered_rolling_windows(synthetic_dam_and_rainfall):
    """Verify rolling sums are strictly backward-looking (sum_3d on day t uses t-1, t-2, t-3)."""
    dam_dates, rainfall = synthetic_dam_and_rainfall
    # Set explicit rainfall values
    rainfall = rainfall.copy()
    rainfall["Rainfall_MM"] = 0.0
    rainfall.loc[rainfall["Monitoring Date"] == "2024-01-10", "Rainfall_MM"] = 10.0
    
    features = build_rainfall_features(dam_dates, rainfall)
    
    # Day 2024-01-10: lag_1 is 2024-01-09 (0.0), so sum_3d must NOT see day 10 yet
    row_10 = features[features["calendar_date"] == "2024-01-10"].iloc[0]
    assert row_10["rainfall_mm_lag_1"] == 0.0
    assert row_10["rainfall_mm_sum_3d"] == 0.0
    
    # Day 2024-01-11: lag_1 is 2024-01-10 (10.0) -> sum_3d should reflect 10.0
    row_11 = features[features["calendar_date"] == "2024-01-11"].iloc[0]
    assert row_11["rainfall_mm_lag_1"] == 10.0
    assert row_11["rainfall_mm_sum_3d"] == 10.0


def test_missing_rainfall_not_silently_zero_filled():
    """Missing rainfall values must cause rolling sum to be NaN or flag incomplete, not zero."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    dam_dates = dates.strftime("%Y-%m-%d").tolist()
    
    rainfall = pd.DataFrame({
        "Monitoring Date": dates.strftime("%Y-%m-%d"),
        "Rainfall_MM": [5.0, 5.0, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "Stations_Reporting": [10, 10, 0, 10, 10, 10, 10, 10, 10, 10],
        "Station_Coverage": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    features = build_rainfall_features(dam_dates, rainfall)
    
    # Day 4 (2024-01-04): lag_1 is day 3 (nan). Features must show window incomplete
    row_4 = features[features["calendar_date"] == "2024-01-04"].iloc[0]
    assert row_4["rainfall_lag_1_available"] == 0
    assert row_4["rainfall_window_3d_complete"] == 0


def test_target_never_enters_feature_matrix():
    """Assert that inflow target columns are not part of RAINFALL_FEATURE_COLUMNS."""
    target_names = {"inflow_cfs", "inflow_cfs_target", "next_day_inflow", "target"}
    feature_set = set(RAINFALL_FEATURE_COLUMNS)
    assert len(target_names.intersection(feature_set)) == 0
