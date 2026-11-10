"""Health and Readiness Probes for KRS Dam Inflow Serving."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health & Readiness"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process Liveness Probe",
    description="Reports that the FastAPI application process is up and responding. Does not indicate model readiness.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="krs-dam-inflow-serving",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        503: {
            "description": "Model is not loaded or not ready for inference.",
            "model": ReadyResponse,
        }
    },
    summary="Model Readiness Probe",
    description="Checks whether the registered MLflow model (Version 1) has been successfully loaded into memory.",
)
async def readiness_check(request: Request):
    model_holder = getattr(request.app.state, "model_holder", None)
    now_iso = datetime.now(timezone.utc).isoformat()

    if model_holder is not None and model_holder.is_ready():
        return ReadyResponse(
            ready=True,
            model_name=model_holder.model_name,
            model_version=model_holder.version,
            status_detail="Model loaded from MLflow Model Registry and ready for inference.",
            timestamp=now_iso,
        )

    error_msg = model_holder.load_error if model_holder else "Model holder not initialized."
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "ready": False,
            "model_name": "KRS_Dam_Inflow_Forecaster",
            "model_version": 1,
            "status_detail": f"Model unavailable: {error_msg}",
            "timestamp": now_iso,
        },
    )

