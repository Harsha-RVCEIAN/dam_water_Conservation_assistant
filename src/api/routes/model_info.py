"""Model Information Endpoint for KRS Dam Inflow Serving."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import ADVISORY_DISCLAIMER, ModelInfoResponse

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Provenance",
    description="Returns safe model metadata, quality status, feature specifications, and advisory disclaimers.",
)
async def get_model_info(request: Request) -> ModelInfoResponse:
    model_holder = getattr(request.app.state, "model_holder", None)
    if model_holder is None or not model_holder.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or still loading.",
        )

    meta = model_holder.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "KRS_Dam_Inflow_Forecaster"),
        model_version=meta.get("version", 1),
        alias=meta.get("alias", "candidate"),
        target_variable=meta.get("target", "target_inflow_next_day_cusecs"),
        target_unit=meta.get("target_unit", "cusecs"),
        quality_status=meta.get("quality_status", "CONDITIONAL"),
        deployment_approved=meta.get("deployment_approved", False),
        intended_use=meta.get("intended_use", "Historical inflow forecasting"),
        feature_count=meta.get("feature_count", 19),
        feature_columns=meta.get("feature_columns", []),
        advisory_disclaimer=ADVISORY_DISCLAIMER,
    )

