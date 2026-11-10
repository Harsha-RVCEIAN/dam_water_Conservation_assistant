"""FastAPI inference and decision-support service for KRS Dam Water Conservation.

This production service serves real-time forecasts using the 27-feature Tuned Random Forest
rainfall-enhanced model registered in the MLflow Model Registry, including automated
handling of degraded rainfall telemetry.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache and single-thread inference in service
(PROJECT_ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from fastapi import FastAPI, HTTPException, status
import numpy as np
import pandas as pd

from src.api.schemas import (
    BatchPredictionInput,
    BatchPredictionResponse,
    PredictionInput,
    PredictionResponse,
    WaterConservationInput,
    WaterConservationResponse,
)
from src.models import train_rainfall
from src.models.registry import load_production_model

# In-memory application state
app_state: dict[str, Any] = {
    "model": None,
    "model_info": {},
    "feature_names": list(train_rainfall.LEARNED_FEATURE_COLUMNS),
    "rainfall_features": list(train_rainfall.RAINFALL_FEATURE_COLUMNS),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model from MLflow registry or local fallback on startup."""
    if app_state.get("model") is None:
        try:
            model, info = load_production_model()
            app_state["model"] = model
            app_state["model_info"] = info
            print(f"FastAPI startup: Production model loaded successfully from {info.get('source')}.")
        except Exception as exc:
            print(f"FastAPI startup WARNING: Model could not be loaded at startup: {exc}")
    yield
    print("FastAPI shutdown: Cleaning up resources.")


app = FastAPI(
    title="KRS Dam Water Conservation Assistant API",
    description=(
        "Production ML service for daily inflow forecasting and water conservation guidance "
        "at K.R.Sagara Dam (Cauvery Basin, Karnataka, India). Powered by a 27-feature "
        "Tuned Random Forest model integrating 19 dam lags and 8 upstream catchment rainfall indicators."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _prepare_feature_row(item: PredictionInput) -> tuple[pd.DataFrame, list[str]]:
    """Convert a PredictionInput into a 27-feature dataframe with degraded telemetry tracking."""
    raw_dict = item.model_dump()
    feature_names = app_state["feature_names"]
    rainfall_cols = app_state["rainfall_features"]

    # Identify missing rainfall telemetry
    imputed_features: list[str] = []

    # Check if primary rainfall telemetry was omitted
    if raw_dict.get("rainfall_mm_lag_1") is None:
        imputed_features.append("rainfall_mm_lag_1")
        raw_dict["rainfall_mm_lag_1"] = np.nan
        raw_dict["rainfall_lag_1_available"] = 0

    if raw_dict.get("rainfall_mm_sum_3d") is None:
        imputed_features.append("rainfall_mm_sum_3d")
        raw_dict["rainfall_mm_sum_3d"] = np.nan
        raw_dict["rainfall_window_3d_complete"] = 0

    if raw_dict.get("rainfall_mm_sum_7d") is None:
        imputed_features.append("rainfall_mm_sum_7d")
        raw_dict["rainfall_mm_sum_7d"] = np.nan
        raw_dict["rainfall_window_7d_complete"] = 0

    if raw_dict.get("rainfall_stations_reporting_lag_1") is None:
        imputed_features.append("rainfall_stations_reporting_lag_1")
        raw_dict["rainfall_stations_reporting_lag_1"] = 0

    if raw_dict.get("rainfall_station_coverage_lag_1") is None:
        imputed_features.append("rainfall_station_coverage_lag_1")
        raw_dict["rainfall_station_coverage_lag_1"] = 0.0

    if raw_dict.get("rainfall_lag_1_available") is None:
        raw_dict["rainfall_lag_1_available"] = 0 if "rainfall_mm_lag_1" in imputed_features else 1

    if raw_dict.get("rainfall_window_3d_complete") is None:
        raw_dict["rainfall_window_3d_complete"] = 0 if "rainfall_mm_sum_3d" in imputed_features else 1

    if raw_dict.get("rainfall_window_7d_complete") is None:
        raw_dict["rainfall_window_7d_complete"] = 0 if "rainfall_mm_sum_7d" in imputed_features else 1

    row = pd.DataFrame([[raw_dict[name] for name in feature_names]], columns=feature_names)
    return row, imputed_features


def _predict_single(item: PredictionInput) -> PredictionResponse:
    """Run prediction on a single 27-feature input vector."""
    model = app_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded.",
        )

    row, imputed = _prepare_feature_row(item)
    raw_pred = float(model.predict(row)[0])
    forecast = max(round(raw_pred, 2), 0.0)

    is_degraded = len(imputed) > 0
    advisory = None
    if is_degraded:
        advisory = (
            f"Rainfall gauge telemetry was missing for {len(imputed)} feature(s) ({', '.join(imputed)}). "
            "Forecast was produced using dam hydrological lags with historical median rainfall imputation. "
            "Extreme storm runoff peaks may be underestimated until water enters the reservoir."
        )

    return PredictionResponse(
        predicted_inflow_cusecs=forecast,
        model_name="Tuned Random Forest (Rainfall-Enhanced)",
        model_version="1.0 (champion)",
        telemetry_status="DEGRADED" if is_degraded else "NOMINAL",
        rainfall_available=not is_degraded,
        imputed_features=imputed,
        confidence_advisory=advisory,
    )


@app.get("/health", summary="Service health check")
def health_check() -> dict[str, str]:
    """Return operational health status."""
    return {"status": "ok", "service": "krs-dam-inflow-api", "version": "2.0.0"}


@app.get("/ready", summary="Service readiness check")
def readiness_check() -> dict[str, Any]:
    """Confirm whether model artifacts are loaded and ready to serve inference."""
    is_ready = app_state["model"] is not None
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_loaded": is_ready,
        "feature_count": len(app_state["feature_names"]),
        "model_info": app_state["model_info"],
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict next-day inflow")
def predict_next_day_inflow(payload: PredictionInput) -> PredictionResponse:
    """Generate a next-day inflow forecast using the 27-feature model.

    If rainfall features are omitted, the service flags DEGRADED telemetry and uses median imputation.
    """
    return _predict_single(payload)


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch inflow predictions")
def predict_batch_inflow(payload: BatchPredictionInput) -> BatchPredictionResponse:
    """Generate forecasts for a sequence of days."""
    results = [_predict_single(item) for item in payload.records]
    return BatchPredictionResponse(count=len(results), predictions=results)


@app.get("/predict/latest", response_model=PredictionResponse, summary="Predict using latest historical archive record")
def predict_latest_archive_record() -> PredictionResponse:
    """Generate a forecast using the latest date in the historical merged dataset."""
    merged_path = PROJECT_ROOT / "data/interim/krs_dam_rainfall_merged.csv"
    if not merged_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Merged historical dataset not found.",
        )

    df, _ = train_rainfall._read_and_validate_dataset()
    latest_row = df.iloc[-1]

    input_dict = {name: float(latest_row[name]) for name in app_state["feature_names"]}
    input_dict["month"] = int(latest_row["month"])
    input_dict["day_of_year"] = int(latest_row["day_of_year"])
    input_dict["is_monsoon"] = int(latest_row["is_monsoon"])

    payload = PredictionInput(**input_dict)
    return _predict_single(payload)


@app.post("/simulate/conservation", response_model=WaterConservationResponse, summary="Simulate 7-day reservoir balance & advisory")
def simulate_water_conservation(payload: WaterConservationInput) -> WaterConservationResponse:
    """Simulate reservoir storage trajectory and generate conservation actions.

    Converts cusecs (cfs) to TMC: 1 cusec for 1 day = 0.0000864 TMC.
    """
    CUSECS_TO_TMC_PER_DAY = 0.0000864

    inflow_tmc_total = payload.forecast_inflow_cusecs * CUSECS_TO_TMC_PER_DAY * payload.horizon_days
    outflow_tmc_total = (payload.planned_outflow_cusecs + payload.planned_canal_withdrawal_cusecs) * CUSECS_TO_TMC_PER_DAY * payload.horizon_days
    net_change_tmc = inflow_tmc_total - outflow_tmc_total

    projected_storage_tmc = max(min(payload.current_storage_tmc + net_change_tmc, 49.452), 0.0)
    percent_full = (projected_storage_tmc / 49.452) * 100.0

    if percent_full >= 85.0:
        tier = "Surplus"
        advisory = "Reservoir is near Full Reservoir Level (FRL). Prepare flood discharge protocol if heavy catchment rain is forecast."
        actions = ["Monitor upstream Harangi and Hemavathi discharges", "Inspect spillway gates", "Optimize hydropower releases"]
    elif percent_full >= 50.0:
        tier = "Normal"
        advisory = "Storage is in healthy operational range. Maintain standard irrigation and drinking water allocations."
        actions = ["Follow scheduled canal releases", "Track weekly monsoon runoff trends"]
    elif percent_full >= 25.0:
        tier = "Stress"
        advisory = "Storage is below seasonal targets. Enforce managed conservation on secondary canal networks."
        actions = ["Prioritize Bengaluru & Mysuru municipal drinking water supply", "Limit rotational canal irrigation", "Audit canal transmission losses"]
    else:
        tier = "Deficit"
        advisory = "CRITICAL DEFICIT: Reservoir approaching dead storage (4.4 TMC). Restrict non-essential withdrawals immediately."
        actions = ["Cease agricultural canal discharges", "Reserve 100% remaining live storage for emergency drinking water", "Convene Cauvery Basin regulatory emergency committee"]

    return WaterConservationResponse(
        current_storage_tmc=round(payload.current_storage_tmc, 3),
        projected_storage_tmc=round(projected_storage_tmc, 3),
        net_storage_change_tmc=round(net_change_tmc, 3),
        water_availability_tier=tier,
        conservation_advisory=advisory,
        action_items=actions,
    )
