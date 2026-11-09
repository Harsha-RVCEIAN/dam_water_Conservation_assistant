"""FastAPI production inference application for K.R.Sagara Dam Inflow Forecaster.

Provides high-performance REST endpoints for daily inflow predictions,
empirical 80% uncertainty bounds, domain boundary checks, and hypothetical
water-conservation scenario simulations.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchInflowForecastRequest,
    BatchInflowForecastResponse,
    ConservationSimulationRequest,
    ConservationSimulationResponse,
    HealthResponse,
    InflowFeaturesInput,
    InflowForecastResponse,
)
from src.models.predict import InflowForecaster
from src.models.water_budget import simulate_conservation_scenario

# Global model instance
forecaster: InflowForecaster | None = None


def load_production_model() -> None:
    """Initialize InflowForecaster on service startup."""
    global forecaster
    try:
        forecaster = InflowForecaster()
        print("Model loaded successfully into FastAPI service.")
    except Exception as e:
        print(f"Warning: Could not load model artifact on startup: {e}")
        forecaster = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and manage InflowForecaster lifecycle."""
    load_production_model()
    yield


app = FastAPI(
    title="K.R.Sagara Dam Water Conservation Assistant - Inference API",
    description=(
        "Production REST API for hydrological inflow forecasting and hypothetical "
        "water-budget decision support (Cauvery Basin, Karnataka, India)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_forecaster() -> InflowForecaster:
    """Dependency helper to ensure model is operational."""
    if forecaster is None or forecaster.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded or unavailable.",
        )
    return forecaster


@app.get("/", tags=["General"])
def root_endpoint() -> Dict[str, str]:
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": "KRS Dam Inflow Forecasting Service",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Liveness probe to confirm service is responding."""
    is_loaded = forecaster is not None and forecaster.pipeline is not None
    m_type = forecaster.metadata.get("model_type", "RandomForestRegressor") if forecaster else "Unknown"
    return HealthResponse(
        status="healthy",
        service="krs_dam_inflow_api",
        model_loaded=is_loaded,
        model_type=m_type,
    )


@app.get("/ready", tags=["Monitoring"])
def readiness_check() -> Dict[str, Any]:
    """Readiness probe to verify model artifact and metadata are fully operational."""
    f = get_forecaster()
    return {
        "ready": True,
        "model_type": f.metadata.get("model_type", "RandomForestRegressor"),
        "features_expected": len(f.metadata.get("features", [])),
        "split": f.metadata.get("split", {}),
    }


@app.post("/predict", response_model=InflowForecastResponse, tags=["Inference"])
def predict_inflow(payload: InflowFeaturesInput) -> InflowForecastResponse:
    """Generate next-day inflow prediction from input features with 80% prediction intervals."""
    f = get_forecaster()

    # Convert Pydantic payload to DataFrame
    df = pd.DataFrame([payload.model_dump()])

    # Domain safety audit
    warnings = f.check_input_domain(df)

    # Compute point and interval predictions
    point, lower, upper = f.predict_with_interval(df)
    point_val = float(point[0])
    lower_val = float(lower[0])
    upper_val = float(upper[0])

    persistence_val = float(payload.inflow_lag_1)
    diff_val = round(point_val - persistence_val, 2)

    return InflowForecastResponse(
        forecast_horizon="1 Day Ahead (t+1)",
        predicted_inflow_cusecs=round(point_val, 2),
        interval_lower_cusecs=round(lower_val, 2),
        interval_upper_cusecs=round(upper_val, 2),
        confidence_level="80% Empirical Prediction Interval",
        unit="cusecs (mean daily rate). 1 cusec-day = 0.0000864 TMC",
        persistence_baseline_cusecs=round(persistence_val, 2),
        difference_from_baseline=diff_val,
        model_type=f.metadata.get("model_type", "RandomForestRegressor"),
        domain_warnings=warnings,
    )


@app.get("/predict/latest", tags=["Inference"])
def predict_latest_observation() -> Dict[str, Any]:
    """Predict next-day inflow following the latest observation in processed dataset."""
    f = get_forecaster()
    try:
        return f.predict_latest_from_processed()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not compute forecast from latest processed data: {e}",
        )


@app.post("/predict/batch", response_model=BatchInflowForecastResponse, tags=["Inference"])
def predict_inflow_batch(payload: BatchInflowForecastRequest) -> BatchInflowForecastResponse:
    """Generate inflow predictions for a batch of input feature records."""
    f = get_forecaster()

    rows = [inst.model_dump() for inst in payload.instances]
    df = pd.DataFrame(rows)

    points, lowers, uppers = f.predict_with_interval(df)

    results = []
    for i, inst in enumerate(payload.instances):
        single_df = df.iloc[[i]]
        warnings = f.check_input_domain(single_df)
        p_val = float(points[i])
        l_val = float(lowers[i])
        u_val = float(uppers[i])
        base_val = float(inst.inflow_lag_1)

        results.append(
            InflowForecastResponse(
                forecast_horizon="1 Day Ahead (t+1)",
                predicted_inflow_cusecs=round(p_val, 2),
                interval_lower_cusecs=round(l_val, 2),
                interval_upper_cusecs=round(u_val, 2),
                confidence_level="80% Empirical Prediction Interval",
                persistence_baseline_cusecs=round(base_val, 2),
                difference_from_baseline=round(p_val - base_val, 2),
                domain_warnings=warnings,
            )
        )

    return BatchInflowForecastResponse(count=len(results), predictions=results)


@app.post(
    "/simulate/conservation",
    response_model=ConservationSimulationResponse,
    tags=["Decision Support"],
)
def simulate_conservation(payload: ConservationSimulationRequest) -> ConservationSimulationResponse:
    """Simulate hypothetical water-conservation scenario over a designated day horizon."""
    try:
        res = simulate_conservation_scenario(
            current_storage_tmc=payload.current_storage_tmc,
            assumed_inflow_cusecs=payload.assumed_inflow_cusecs,
            planned_outflow_cusecs=payload.planned_outflow_cusecs,
            baseline_withdrawal_cusecs=payload.baseline_withdrawal_cusecs,
            reduction_percentage=payload.reduction_percentage,
            simulation_days=payload.simulation_days,
            evaporation_cusecs=payload.evaporation_cusecs,
        )
        return ConservationSimulationResponse(
            scenario_name=res.scenario_name,
            simulation_days=res.simulation_days,
            current_storage_tmc=res.current_storage_tmc,
            water_saved_tmc=res.water_saved_tmc,
            savings_as_pct_capacity=res.savings_as_pct_capacity,
            projected_baseline_storage_tmc=res.projected_baseline_storage_tmc,
            projected_scenario_storage_tmc=res.projected_scenario_storage_tmc,
            disclaimer=res.disclaimer,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error executing scenario simulation: {e}",
        )
