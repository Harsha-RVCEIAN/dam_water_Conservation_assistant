"""Model risk, quality gates, and metamorphic safety checks."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.evaluate_gate import (
    MIN_RMSE_IMPROVEMENT_PCT,
    MIN_TEST_R2,
    run_model_quality_gate,
)
from src.models.predict import InflowForecaster
from src.models.train import METADATA_PATH, MODEL_ARTIFACT_PATH


def test_quality_gate_passes_on_production_model():
    """Verify that current production model passes all pre-deployment quality gates."""
    result = run_model_quality_gate()
    assert result.passed is True, f"Quality gate failed: {result.checks_failed}"
    assert len(result.checks_failed) == 0
    assert result.metrics_summary["rmse_improvement_pct"] >= MIN_RMSE_IMPROVEMENT_PCT
    assert result.metrics_summary["test_r2"] >= MIN_TEST_R2


def test_model_beats_persistence_baseline():
    """Verify strictly that Random Forest RMSE is superior to the persistence baseline on held-out test data."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    test_metrics = metadata["metrics"]["test"]

    rf_rmse = test_metrics["selected_model"]["rmse"]
    persistence_rmse = test_metrics["persistence"]["rmse"]

    assert rf_rmse < persistence_rmse, f"Model RMSE ({rf_rmse}) is not better than persistence ({persistence_rmse})"
    improvement_pct = (persistence_rmse - rf_rmse) / persistence_rmse * 100
    assert improvement_pct >= MIN_RMSE_IMPROVEMENT_PCT


def test_non_negativity_invariance_under_zero_flow():
    """Metamorphic test: When all recent inflows are near zero, forecast must remain non-negative and low."""
    forecaster = InflowForecaster()
    zero_flow_input = pd.DataFrame([{
        "inflow_lag_1": 0.0,
        "inflow_lag_2": 0.0,
        "inflow_lag_3": 0.0,
        "inflow_lag_7": 0.0,
        "inflow_lag_14": 0.0,
        "inflow_roll_mean_7": 0.0,
        "inflow_roll_std_7": 0.0,
        "inflow_roll_max_7": 0.0,
        "inflow_roll_mean_14": 0.0,
        "inflow_roll_mean_30": 0.0,
        "storage_lag_1": 15.0,
        "storage_lag_7": 15.0,
        "outflow_lag_1": 100.0,
        "outflow_lag_7": 100.0,
        "month": 2,
        "day_of_year": 45,
        "is_monsoon": 0,
        "sin_day_of_year": 0.7,
        "cos_day_of_year": 0.7,
    }])

    pred = forecaster.predict(zero_flow_input)
    assert pred[0] >= 0.0
    # During dry season with zero recent inflow, predicted inflow shouldn't spike to high monsoon levels
    assert pred[0] < 5000.0, f"Expected low forecast for zero dry season input, got {pred[0]}"


def test_domain_guard_on_flood_surcharge():
    """Verify warning triggers when reservoir level exceeds design Full Reservoir Level (2,482 ft MSL)."""
    forecaster = InflowForecaster()
    surcharge_input = pd.DataFrame([{
        "inflow_lag_1": 15000.0,
        "reservoir_level_ft": 2490.5,  # Exceeds FRL of 2482 ft
    }])
    warnings = forecaster.check_input_domain(surcharge_input)
    assert any("exceeds Full Reservoir Level" in w for w in warnings)


def test_prediction_intervals_order_and_bounds():
    """Verify calibrated empirical prediction intervals are monotonic (lower <= point <= upper)."""
    forecaster = InflowForecaster()
    sample_df = pd.DataFrame([{
        "inflow_lag_1": 3000.0,
        "inflow_lag_2": 2900.0,
        "inflow_lag_3": 2800.0,
        "inflow_lag_7": 2500.0,
        "inflow_lag_14": 2000.0,
        "inflow_roll_mean_7": 2800.0,
        "inflow_roll_std_7": 200.0,
        "inflow_roll_max_7": 3200.0,
        "inflow_roll_mean_14": 2600.0,
        "inflow_roll_mean_30": 2400.0,
        "storage_lag_1": 40.0,
        "storage_lag_7": 39.0,
        "outflow_lag_1": 2000.0,
        "outflow_lag_7": 1800.0,
        "month": 7,
        "day_of_year": 195,
        "is_monsoon": 1,
        "sin_day_of_year": -0.2,
        "cos_day_of_year": -0.98,
    }])

    point, lower, upper = forecaster.predict_with_interval(sample_df)
    assert lower[0] >= 0.0
    assert lower[0] <= point[0]
    assert point[0] <= upper[0]
