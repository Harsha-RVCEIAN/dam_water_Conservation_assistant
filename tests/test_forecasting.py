"""Tests for chronological splitting, feature leakage prevention, baselines, and model inference."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import add_forecasting_lags
from src.models.predict import InflowForecaster
from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
    compute_wape,
    evaluate_predictions,
    load_and_split_data,
)


def test_chronological_split_ordering():
    """Verify strictly that training dates precede validation dates, and validation dates precede test dates."""
    train_df, val_df, test_df = load_and_split_data()

    max_train_date = train_df["calendar_date"].max()
    min_val_date = val_df["calendar_date"].min()
    max_val_date = val_df["calendar_date"].max()
    min_test_date = test_df["calendar_date"].min()

    assert max_train_date < min_val_date, f"Train max date {max_train_date} >= Val min date {min_val_date}"
    assert max_val_date < min_test_date, f"Val max date {max_val_date} >= Test min date {min_test_date}"


def test_validation_partition_not_excluded():
    """Verify that 2023-2024 is strictly included as the validation partition with exact calendar coverage."""
    train_df, val_df, test_df = load_and_split_data()

    assert len(val_df) == 731, f"Expected 731 days for 2023-2024 validation period, got {len(val_df)}"
    assert val_df["calendar_date"].min() == pd.Timestamp("2023-01-01")
    assert val_df["calendar_date"].max() == pd.Timestamp("2024-12-31")
    assert len(train_df) > 10000
    assert len(test_df) > 500


def test_no_feature_leakage_in_lags():
    """Test that feature values at time t do not depend on inflow at time t or future."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    synthetic = pd.DataFrame({
        "calendar_date": dates,
        "inflow_cusecs": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "gross_capacity_tmc": [30.0] * 10,
        "outflow_cusecs": [50.0] * 10,
    })

    lagged = add_forecasting_lags(synthetic, lags=(1, 2), rolling_windows=(3,))

    # At row 2 (index 2, date 2020-01-03, inflow=300):
    # inflow_lag_1 should be 200 (row 1)
    # inflow_lag_2 should be 100 (row 0)
    assert lagged.loc[2, "inflow_lag_1"] == 200.0
    assert lagged.loc[2, "inflow_lag_2"] == 100.0

    # rolling mean of past 3 values excluding today:
    # At index 3 (2020-01-04, inflow=400): past 3 are [100, 200, 300] -> mean = 200.0
    assert lagged.loc[3, "inflow_roll_mean_3"] == pytest.approx(200.0)

    # Target is next-day inflow:
    # At index 2, target should be row 3's inflow (400.0)
    assert lagged.loc[2, "target_inflow_next_day_cusecs"] == 400.0


def test_compute_wape_known_values():
    """Verify Weighted Absolute Percentage Error calculation."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # abs errors: 10 + 10 + 0 = 20; sum(y_true) = 600; WAPE = 20/600 * 100 = 3.333%
    assert compute_wape(y_true, y_pred) == pytest.approx(3.3333, rel=1e-3)


def test_model_artifact_loading_and_prediction():
    """Test inference on synthetic feature inputs."""
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    forecaster = InflowForecaster()
    sample_input = pd.DataFrame([{
        "inflow_lag_1": 2500.0,
        "inflow_lag_2": 2400.0,
        "inflow_lag_3": 2300.0,
        "inflow_lag_7": 2100.0,
        "inflow_lag_14": 1800.0,
        "inflow_roll_mean_7": 2400.0,
        "inflow_roll_std_7": 150.0,
        "inflow_roll_max_7": 2600.0,
        "inflow_roll_mean_14": 2200.0,
        "inflow_roll_mean_30": 2000.0,
        "storage_lag_1": 35.0,
        "storage_lag_7": 34.5,
        "outflow_lag_1": 1500.0,
        "outflow_lag_7": 1200.0,
        "month": 8,
        "day_of_year": 230,
        "is_monsoon": 1,
        "sin_day_of_year": 0.5,
        "cos_day_of_year": -0.866,
    }])

    pred = forecaster.predict(sample_input)
    assert len(pred) == 1
    assert pred[0] >= 0.0, "Predicted flow must be non-negative"
    assert isinstance(pred[0], (float, np.floating))


def test_prediction_with_intervals():
    """Verify empirical prediction intervals produce valid non-negative bounds."""
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    forecaster = InflowForecaster()
    sample_input = pd.DataFrame([{
        "inflow_lag_1": 1500.0,
        "inflow_lag_2": 1400.0,
        "inflow_lag_3": 1300.0,
        "inflow_lag_7": 1200.0,
        "inflow_lag_14": 1100.0,
        "inflow_roll_mean_7": 1400.0,
        "inflow_roll_std_7": 100.0,
        "inflow_roll_max_7": 1600.0,
        "inflow_roll_mean_14": 1300.0,
        "inflow_roll_mean_30": 1200.0,
        "storage_lag_1": 30.0,
        "storage_lag_7": 29.5,
        "outflow_lag_1": 1000.0,
        "outflow_lag_7": 900.0,
        "month": 7,
        "day_of_year": 200,
        "is_monsoon": 1,
        "sin_day_of_year": 0.2,
        "cos_day_of_year": -0.9,
    }])

    point, lower, upper = forecaster.predict_with_interval(sample_input)
    assert lower[0] >= 0.0, "Lower bound must be non-negative"
    assert lower[0] <= point[0], f"Lower bound ({lower[0]}) must be <= point prediction ({point[0]})"
    assert point[0] <= upper[0], f"Point prediction ({point[0]}) must be <= upper bound ({upper[0]})"


def test_domain_boundary_warnings():
    """Verify warning when input flow exceeds historical maximum."""
    forecaster = InflowForecaster()
    extreme_input = pd.DataFrame([{
        "inflow_lag_1": 300000.0,  # Exceeds historical max of ~204,200 cusecs
        "reservoir_level_ft": 2490.0,
    }])
    warnings = forecaster.check_input_domain(extreme_input)
    assert len(warnings) > 0
    assert any("exceeds historical maximum" in w for w in warnings)


def test_prediction_fails_on_missing_features():
    """Verify ValueError when required feature columns are missing."""
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    forecaster = InflowForecaster()
    incomplete_input = pd.DataFrame([{"inflow_lag_1": 1000.0}])
    with pytest.raises(ValueError, match="missing required feature columns"):
        forecaster.predict(incomplete_input)
