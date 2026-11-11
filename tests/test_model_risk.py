"""Model risk, input validation, and physical boundary tests for KRS Dam Forecaster."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app, app_state
from src.models.registry import load_production_model

MODEL_PATH = PROJECT_ROOT / "models/rainfall_enhanced/rainfall_tuned_random_forest_v1.joblib"


class MockPipeline:
    """Mock model pipeline for isolated risk testing when real artifact is unavailable."""

    def __init__(self):
        self.named_steps = {
            "imputer": type("MockImputer", (), {"statistics_": np.zeros(27)})()
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            base = row.get("inflow_lag_1", 3500.0)
            if pd.isna(base):
                base = 3500.0
            rain = row.get("rainfall_mm_lag_1", 0.0)
            if pd.notna(rain):
                base += float(rain) * 10.0
            preds.append(max(float(base), 0.0))
        return np.array(preds, dtype=float)


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with loaded production model (real model if present, mock fallback for CI)."""
    orig_model = app_state.get("model")
    orig_info = app_state.get("model_info")

    with TestClient(app) as test_client:
        if app_state.get("model") is None:
            app_state["model"] = MockPipeline()
            app_state["model_info"] = {
                "model_name": "Tuned Random Forest (Rainfall-Enhanced)",
                "source": "test_fixture_mock",
                "version": "1.0 (champion)",
            }
        try:
            yield test_client
        finally:
            if orig_model is None:
                app_state["model"] = None
                app_state["model_info"] = orig_info or {}


@pytest.fixture
def valid_input() -> dict:
    return {
        "inflow_lag_1": 2000.0,
        "inflow_lag_2": 2100.0,
        "inflow_lag_3": 2200.0,
        "inflow_lag_7": 2300.0,
        "inflow_lag_14": 2400.0,
        "inflow_roll_mean_7": 2150.0,
        "inflow_roll_std_7": 100.0,
        "inflow_roll_max_7": 2500.0,
        "inflow_roll_mean_14": 2200.0,
        "inflow_roll_mean_30": 2100.0,
        "storage_lag_1": 30.0,
        "storage_lag_7": 29.5,
        "outflow_lag_1": 1500.0,
        "outflow_lag_7": 1500.0,
        "month": 8,
        "day_of_year": 220,
        "is_monsoon": 1,
        "sin_day_of_year": -0.5,
        "cos_day_of_year": -0.86,
        "rainfall_mm_lag_1": 10.0,
        "rainfall_mm_sum_3d": 25.0,
        "rainfall_mm_sum_7d": 50.0,
        "rainfall_stations_reporting_lag_1": 7,
        "rainfall_station_coverage_lag_1": 0.7,
        "rainfall_lag_1_available": 1,
        "rainfall_window_3d_complete": 1,
        "rainfall_window_7d_complete": 1,
    }


def test_negative_inflow_input_rejected(client: TestClient, valid_input: dict):
    """Ensure Pydantic schema rejects physically impossible negative inflow lags."""
    bad_input = valid_input.copy()
    bad_input["inflow_lag_1"] = -500.0
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_storage_exceeding_capacity_rejected(client: TestClient, valid_input: dict):
    """Ensure reservoir storage above maximum capacity (55 TMC safety cap) is rejected."""
    bad_input = valid_input.copy()
    bad_input["storage_lag_1"] = 60.0  # Above 55.0 cap
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_invalid_month_rejected(client: TestClient, valid_input: dict):
    """Ensure calendar month > 12 is rejected."""
    bad_input = valid_input.copy()
    bad_input["month"] = 13
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_extreme_drought_inflow_is_non_negative(client: TestClient, valid_input: dict):
    """Test model risk under zero-inflow drought conditions: must never forecast negative inflow."""
    zero_inflow_input = valid_input.copy()
    for key in (
        "inflow_lag_1", "inflow_lag_2", "inflow_lag_3", "inflow_lag_7", "inflow_lag_14",
        "inflow_roll_mean_7", "inflow_roll_std_7", "inflow_roll_max_7", "inflow_roll_mean_14",
        "inflow_roll_mean_30", "rainfall_mm_lag_1", "rainfall_mm_sum_3d", "rainfall_mm_sum_7d"
    ):
        zero_inflow_input[key] = 0.0
    zero_inflow_input["is_monsoon"] = 0
    zero_inflow_input["month"] = 3  # March dry season

    response = client.post("/predict", json=zero_inflow_input)
    assert response.status_code == 200
    pred = response.json()["predicted_inflow_cusecs"]
    assert pred >= 0.0, f"Predicted negative inflow: {pred}"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Requires trained model artifact to verify bounded flood hydrology risk",
)
def test_extreme_monsoon_flood_bounded(client: TestClient, valid_input: dict):
    """Test model risk under historic extreme flood inflow."""
    flood_input = valid_input.copy()
    flood_input["inflow_lag_1"] = 120000.0  # 120,000 cfs historic peak
    flood_input["inflow_roll_mean_7"] = 80000.0
    flood_input["inflow_roll_max_7"] = 120000.0
    flood_input["rainfall_mm_lag_1"] = 150.0  # Extreme 150mm cloudburst
    flood_input["rainfall_mm_sum_3d"] = 350.0
    flood_input["rainfall_mm_sum_7d"] = 600.0

    response = client.post("/predict", json=flood_input)
    assert response.status_code == 200
    pred = response.json()["predicted_inflow_cusecs"]
    assert np.isfinite(pred)
    assert pred > 10000.0  # Must reflect large inflow
    assert pred < 300000.0  # Plausibly bounded below impossible runoff levels


def test_imputer_resilience_all_rainfall_none(client: TestClient, valid_input: dict):
    """Verify that when all 8 rainfall features are None, imputer handles it safely."""
    no_rain_input = {k: v for k, v in valid_input.items() if not k.startswith("rainfall_")}
    response = client.post("/predict", json=no_rain_input)
    assert response.status_code == 200
    data = response.json()
    assert np.isfinite(data["predicted_inflow_cusecs"])
    assert data["predicted_inflow_cusecs"] >= 0.0
    assert data["telemetry_status"] == "DEGRADED"
