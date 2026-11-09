"""Integration and contract tests for FastAPI inference endpoints and schemas."""

import pytest
from pydantic import ValidationError

from src.api.main import (
    app,
    health_check,
    load_production_model,
    predict_inflow,
    predict_inflow_batch,
    predict_latest_observation,
    readiness_check,
    root_endpoint,
    simulate_conservation,
)
from src.api.schemas import (
    BatchInflowForecastRequest,
    ConservationSimulationRequest,
    InflowFeaturesInput,
)


@pytest.fixture(autouse=True, scope="module")
def init_model():
    """Ensure production model is loaded for API tests."""
    load_production_model()


def test_root_endpoint():
    """Verify root discovery endpoint metadata."""
    data = root_endpoint()
    assert data["status"] == "operational"
    assert "/docs" in data["documentation"]
    assert data["version"] == "2.0.0"


def test_health_check_endpoint():
    """Verify liveness probe returns healthy status and confirmed model loaded."""
    data = health_check()
    assert data.status == "healthy"
    assert data.model_loaded is True
    assert "RandomForest" in data.model_type


def test_readiness_check_endpoint():
    """Verify readiness probe confirms operational feature count and split dates."""
    data = readiness_check()
    assert data["ready"] is True
    assert data["features_expected"] == 19
    assert "split" in data


def test_predict_single_instance():
    """Verify valid feature payload returns bounded prediction and intervals."""
    payload = InflowFeaturesInput(
        inflow_lag_1=2500.0,
        inflow_lag_2=2400.0,
        inflow_lag_3=2300.0,
        inflow_lag_7=2100.0,
        inflow_lag_14=1800.0,
        inflow_roll_mean_7=2400.0,
        inflow_roll_std_7=150.0,
        inflow_roll_max_7=2600.0,
        inflow_roll_mean_14=2200.0,
        inflow_roll_mean_30=2000.0,
        storage_lag_1=35.0,
        storage_lag_7=34.5,
        outflow_lag_1=1500.0,
        outflow_lag_7=1200.0,
        month=8,
        day_of_year=230,
        is_monsoon=1,
        sin_day_of_year=0.5,
        cos_day_of_year=-0.866,
    )
    res = predict_inflow(payload)
    assert res.predicted_inflow_cusecs >= 0.0
    assert res.interval_lower_cusecs <= res.predicted_inflow_cusecs
    assert res.predicted_inflow_cusecs <= res.interval_upper_cusecs
    assert res.persistence_baseline_cusecs == 2500.0
    assert "cusecs" in res.unit


def test_predict_rejects_negative_inflows():
    """Verify Pydantic input validation rejects physically impossible negative flows."""
    with pytest.raises(ValidationError):
        InflowFeaturesInput(
            inflow_lag_1=-500.0,  # Negative flow violation (ge=0.0)
            inflow_lag_2=2400.0,
            inflow_lag_3=2300.0,
            inflow_lag_7=2100.0,
            inflow_lag_14=1800.0,
            inflow_roll_mean_7=2400.0,
            inflow_roll_std_7=150.0,
            inflow_roll_max_7=2600.0,
            inflow_roll_mean_14=2200.0,
            inflow_roll_mean_30=2000.0,
            storage_lag_1=35.0,
            storage_lag_7=34.5,
            outflow_lag_1=1500.0,
            outflow_lag_7=1200.0,
            month=8,
            day_of_year=230,
            is_monsoon=1,
            sin_day_of_year=0.5,
            cos_day_of_year=-0.866,
        )


def test_predict_latest_from_archive():
    """Verify automated prediction using latest archive record."""
    data = predict_latest_observation()
    assert "forecast_inflow_cusecs" in data
    assert "latest_observed_date" in data
    assert data["forecast_inflow_cusecs"] >= 0.0


def test_batch_prediction():
    """Verify batch prediction endpoint."""
    instance = InflowFeaturesInput(
        inflow_lag_1=1000.0,
        inflow_lag_2=1000.0,
        inflow_lag_3=1000.0,
        inflow_lag_7=1000.0,
        inflow_lag_14=1000.0,
        inflow_roll_mean_7=1000.0,
        inflow_roll_std_7=50.0,
        inflow_roll_max_7=1100.0,
        inflow_roll_mean_14=1000.0,
        inflow_roll_mean_30=1000.0,
        storage_lag_1=25.0,
        storage_lag_7=25.0,
        outflow_lag_1=500.0,
        outflow_lag_7=500.0,
        month=5,
        day_of_year=130,
        is_monsoon=0,
        sin_day_of_year=0.8,
        cos_day_of_year=-0.6,
    )
    payload = BatchInflowForecastRequest(instances=[instance, instance])
    res = predict_inflow_batch(payload)
    assert res.count == 2
    assert len(res.predictions) == 2


def test_simulate_conservation_endpoint():
    """Verify scenario simulator endpoint."""
    payload = ConservationSimulationRequest(
        current_storage_tmc=30.0,
        assumed_inflow_cusecs=3000.0,
        planned_outflow_cusecs=1000.0,
        baseline_withdrawal_cusecs=2000.0,
        reduction_percentage=20.0,
        simulation_days=7,
        evaporation_cusecs=100.0,
    )
    res = simulate_conservation(payload)
    assert res.water_saved_tmc > 0.0
    assert res.projected_scenario_storage_tmc > res.projected_baseline_storage_tmc
    assert "Hypothetical educational simulation" in res.disclaimer
