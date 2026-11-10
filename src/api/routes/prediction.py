"""Inference Route for KRS Dam Next-Day Inflow Forecasting."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import (
    ADVISORY_DISCLAIMER,
    InflowPredictionRequest,
    InflowPredictionResponse,
)

router = APIRouter(tags=["Forecasting"])


@router.post(
    "/predict",
    response_model=InflowPredictionResponse,
    summary="Predict Next-Day Reservoir Inflow",
    description=(
        "Computes next-day inflow forecast in cusecs using the registered "
        "KRS_Dam_Inflow_Forecaster (Version 1). Applies non-negative clipping "
        "and includes quality and advisory provenance."
    ),
)
async def predict_inflow(
    payload: InflowPredictionRequest,
    request: Request,
) -> InflowPredictionResponse:
    model_holder = getattr(request.app.state, "model_holder", None)
    if model_holder is None or not model_holder.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not ready to serve predictions.",
        )

    try:
        # Convert validated Pydantic model to properly-ordered DataFrame
        feature_df = payload.to_feature_dataframe()
        pred_cusecs = model_holder.predict(feature_df)

        meta = model_holder.metadata
        return InflowPredictionResponse(
            predicted_inflow_cusecs=pred_cusecs,
            target_variable=meta.get("target", "target_inflow_next_day_cusecs"),
            target_unit=meta.get("target_unit", "cusecs"),
            model_name=meta.get("model_name", "KRS_Dam_Inflow_Forecaster"),
            model_version=meta.get("version", 1),
            model_alias=meta.get("alias", "candidate"),
            quality_status=meta.get("quality_status", "CONDITIONAL"),
            deployment_approved=meta.get("deployment_approved", False),
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            disclaimer=ADVISORY_DISCLAIMER,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature validation error during inference: {ve}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {type(exc).__name__}",
        )

