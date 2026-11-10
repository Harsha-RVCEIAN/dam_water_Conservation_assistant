"""Unit and integration tests for the KRS Dam Water Conservation FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app, app_state
from src.models.registry import load_production_model


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the loaded model in app_state."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_27_features() -> dict:
    """Sample valid 27-feature dictionary for testing."""
    return {
        "inflow_lag_1": 4500.0,
        "inflow_lag_2": 4200.0,
        "inflow_lag_3": 4100.0,
        "inflow_lag_7": 3800.0,
        "inflow_lag_14": 3500.0,
        "inflow_roll_mean_7": 4150.0,
        "inflow_roll_std_7": 250.0,
        "inflow_roll_max_7": 4500.0,
        "inflow_roll_mean_14": 3900.0,
        "inflow_roll_mean_30": 3700.0,
        "storage_lag_1": 42.5,
        "storage_lag_7": 41.8,
        "outflow_lag_1": 2500.0,
        "outflow_lag_7": 2400.0,
        "month": 7,
        "day_of_year": 200,
        "is_monsoon": 1,
        "sin_day_of_year": -0.25,
        "cos_day_of_year": -0.97,
        "rainfall_mm_lag_1": 15.5,
        "rainfall_mm_sum_3d": 45.0,
        "rainfall_mm_sum_7d": 85.0,
        "rainfall_stations_reporting_lag_1": 8,
        "rainfall_station_coverage_lag_1": 0.8,
        "rainfall_lag_1_available": 1,
        "rainfall_window_3d_complete": 1,
        "rainfall_window_7d_complete": 1,
    }


def test_health_endpoint(client: TestClient):
    """Test GET /health returns 200 and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "krs-dam-inflow-api"


def test_readiness_endpoint(client: TestClient):
    """Test GET /ready confirms model is loaded with 27 features."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 27


def test_predict_nominal_telemetry(client: TestClient, sample_27_features: dict):
    """Test POST /predict with full 27 features produces a nominal forecast."""
    response = client.post("/predict", json=sample_27_features)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_inflow_cusecs"] >= 0.0
    assert data["telemetry_status"] == "NOMINAL"
    assert data["rainfall_available"] is True
    assert len(data["imputed_features"]) == 0
    assert data["confidence_advisory"] is None


def test_predict_degraded_rainfall_telemetry(client: TestClient, sample_27_features: dict):
    """Test POST /predict when all rainfall features are omitted (telemetry outage)."""
    # Create input with only the 19 dam features (rainfall fields omitted)
    dam_only_input = {k: v for k, v in sample_27_features.items() if not k.startswith("rainfall_")}

    response = client.post("/predict", json=dam_only_input)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_inflow_cusecs"] >= 0.0
    assert data["telemetry_status"] == "DEGRADED"
    assert data["rainfall_available"] is False
    assert len(data["imputed_features"]) > 0
    assert "rainfall_mm_lag_1" in data["imputed_features"]
    assert data["confidence_advisory"] is not None
    assert "telemetry was missing" in data["confidence_advisory"].lower()


def test_predict_batch_endpoint(client: TestClient, sample_27_features: dict):
    """Test POST /predict/batch generates multi-day forecasts."""
    batch_payload = {"records": [sample_27_features, sample_27_features]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["predicted_inflow_cusecs"] >= 0.0


def test_predict_latest_endpoint(client: TestClient):
    """Test GET /predict/latest runs against historical merged data."""
    response = client.get("/predict/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_inflow_cusecs"] >= 0.0


def test_simulate_conservation_normal(client: TestClient):
    """Test POST /simulate/conservation with normal reservoir level."""
    payload = {
        "current_storage_tmc": 35.0,
        "forecast_inflow_cusecs": 5000.0,
        "planned_outflow_cusecs": 3000.0,
        "planned_canal_withdrawal_cusecs": 1500.0,
        "horizon_days": 7,
    }
    response = client.post("/simulate/conservation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["current_storage_tmc"] == 35.0
    assert data["water_availability_tier"] in ("Normal", "Surplus")
    assert len(data["action_items"]) > 0


def test_simulate_conservation_deficit(client: TestClient):
    """Test POST /simulate/conservation under critical deficit storage."""
    payload = {
        "current_storage_tmc": 8.0,  # ~16% full
        "forecast_inflow_cusecs": 500.0,
        "planned_outflow_cusecs": 1000.0,
        "planned_canal_withdrawal_cusecs": 1500.0,
        "horizon_days": 7,
    }
    response = client.post("/simulate/conservation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["water_availability_tier"] == "Deficit"
    assert "CRITICAL DEFICIT" in data["conservation_advisory"]
