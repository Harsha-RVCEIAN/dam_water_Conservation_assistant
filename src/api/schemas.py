"""Pydantic Request and Response Schemas for KRS Dam Inflow Serving API.

Defines strict input validation, feature ordering, and explicit output formats
derived directly from the trained RandomForestRegressor pipeline contracts.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN

ADVISORY_DISCLAIMER = (
    "Advisory forecast only. This prediction is generated from historical hydrological "
    "reservoir observations for water-conservation decision simulation. It must NOT be used "
    "as an automated spillway gate-control or official dam emergency command system."
)


def _check_finite(val: Any, field_name: str) -> float:
    """Ensure numeric value is finite and not NaN or infinite."""
    if val is None:
        raise ValueError(f"Field '{field_name}' cannot be null.")
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"Field '{field_name}' must be a valid number.")

    if math.isnan(f_val):
        raise ValueError(f"Field '{field_name}' cannot be NaN.")
    if math.isinf(f_val):
        raise ValueError(f"Field '{field_name}' cannot be infinite.")
    return f_val


class InflowPredictionRequest(BaseModel):
    """Input payload containing the 19 backward-looking features required for next-day inflow forecasting."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "inflow_lag_1": 1500.0,
                "inflow_lag_2": 1450.0,
                "inflow_lag_3": 1400.0,
                "inflow_lag_7": 1300.0,
                "inflow_lag_14": 1200.0,
                "inflow_roll_mean_7": 1420.0,
                "inflow_roll_std_7": 75.0,
                "inflow_roll_max_7": 1550.0,
                "inflow_roll_mean_14": 1350.0,
                "inflow_roll_mean_30": 1300.0,
                "storage_lag_1": 35.5,
                "storage_lag_7": 36.0,
                "outflow_lag_1": 2200.0,
                "outflow_lag_7": 2150.0,
                "month": 7,
                "day_of_year": 195,
                "is_monsoon": 1,
                "sin_day_of_year": -0.15,
                "cos_day_of_year": -0.98,
            }
        },
    )

    inflow_lag_1: float = Field(..., description="Inflow observation 1 day prior (cusecs).")
    inflow_lag_2: float = Field(..., description="Inflow observation 2 days prior (cusecs).")
    inflow_lag_3: float = Field(..., description="Inflow observation 3 days prior (cusecs).")
    inflow_lag_7: float = Field(..., description="Inflow observation 7 days prior (cusecs).")
    inflow_lag_14: float = Field(..., description="Inflow observation 14 days prior (cusecs).")

    inflow_roll_mean_7: float = Field(..., description="7-day rolling mean of historical inflow (cusecs).")
    inflow_roll_std_7: float = Field(..., description="7-day rolling standard deviation of historical inflow (cusecs).")
    inflow_roll_max_7: float = Field(..., description="7-day rolling maximum of historical inflow (cusecs).")
    inflow_roll_mean_14: float = Field(..., description="14-day rolling mean of historical inflow (cusecs).")
    inflow_roll_mean_30: float = Field(..., description="30-day rolling mean of historical inflow (cusecs).")

    storage_lag_1: float = Field(..., description="Reservoir gross storage 1 day prior (TMC).")
    storage_lag_7: float = Field(..., description="Reservoir gross storage 7 days prior (TMC).")

    outflow_lag_1: float = Field(..., description="Total dam outflow 1 day prior (cusecs).")
    outflow_lag_7: float = Field(..., description="Total dam outflow 7 days prior (cusecs).")

    month: int = Field(..., ge=1, le=12, description="Calendar month (1 to 12).")
    day_of_year: int = Field(..., ge=1, le=366, description="Day of year (1 to 366).")
    is_monsoon: int = Field(..., ge=0, le=1, description="Binary monsoon indicator (1 for June-Sept, 0 otherwise).")

    sin_day_of_year: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Sine cyclical day-of-year encoding. Auto-derived if omitted."
    )
    cos_day_of_year: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Cosine cyclical day-of-year encoding. Auto-derived if omitted."
    )

    @field_validator(
        "inflow_lag_1", "inflow_lag_2", "inflow_lag_3", "inflow_lag_7", "inflow_lag_14",
        "inflow_roll_mean_7", "inflow_roll_std_7", "inflow_roll_max_7",
        "inflow_roll_mean_14", "inflow_roll_mean_30",
        "storage_lag_1", "storage_lag_7",
        "outflow_lag_1", "outflow_lag_7",
        mode="before"
    )
    @classmethod
    def validate_finite_floats(cls, v: Any, info) -> float:
        return _check_finite(v, info.field_name)

    @field_validator("sin_day_of_year", "cos_day_of_year", mode="before")
    @classmethod
    def validate_optional_cyclical(cls, v: Any, info) -> Optional[float]:
        if v is None:
            return None
        return _check_finite(v, info.field_name)

    @model_validator(mode="after")
    def derive_cyclical_features_if_missing(self) -> InflowPredictionRequest:
        """Derive sine and cosine day-of-year cyclical components if not supplied."""
        doy = self.day_of_year
        if self.sin_day_of_year is None:
            self.sin_day_of_year = float(np.sin(2.0 * np.pi * doy / 365.25))
        if self.cos_day_of_year is None:
            self.cos_day_of_year = float(np.cos(2.0 * np.pi * doy / 365.25))
        return self

    def to_feature_dataframe(self) -> pd.DataFrame:
        """Convert validated request into a single-row DataFrame ordered by FEATURE_COLUMNS."""
        row_dict = {
            "inflow_lag_1": float(self.inflow_lag_1),
            "inflow_lag_2": float(self.inflow_lag_2),
            "inflow_lag_3": float(self.inflow_lag_3),
            "inflow_lag_7": float(self.inflow_lag_7),
            "inflow_lag_14": float(self.inflow_lag_14),
            "inflow_roll_mean_7": float(self.inflow_roll_mean_7),
            "inflow_roll_std_7": float(self.inflow_roll_std_7),
            "inflow_roll_max_7": float(self.inflow_roll_max_7),
            "inflow_roll_mean_14": float(self.inflow_roll_mean_14),
            "inflow_roll_mean_30": float(self.inflow_roll_mean_30),
            "storage_lag_1": float(self.storage_lag_1),
            "storage_lag_7": float(self.storage_lag_7),
            "outflow_lag_1": float(self.outflow_lag_1),
            "outflow_lag_7": float(self.outflow_lag_7),
            "month": float(self.month),
            "day_of_year": float(self.day_of_year),
            "is_monsoon": float(self.is_monsoon),
            "sin_day_of_year": float(self.sin_day_of_year),
            "cos_day_of_year": float(self.cos_day_of_year),
        }
        df = pd.DataFrame([row_dict])
        return df[FEATURE_COLUMNS]


class InflowPredictionResponse(BaseModel):
    """Response payload containing next-day inflow prediction and model provenance metadata."""

    predicted_inflow_cusecs: float = Field(..., description="Forecasted next-day reservoir inflow (cusecs).")
    target_variable: str = Field(TARGET_COLUMN, description="Target variable name.")
    target_unit: str = Field("cusecs", description="Target variable physical unit.")
    model_name: str = Field("KRS_Dam_Inflow_Forecaster", description="Registered model name in MLflow.")
    model_version: int = Field(1, description="Registered model version number.")
    model_alias: Optional[str] = Field("candidate", description="Assigned model version alias.")
    quality_status: str = Field("CONDITIONAL", description="Quality acceptance status from Step 1 evaluation.")
    deployment_approved: bool = Field(False, description="Whether model is approved for automated production dispatch.")
    prediction_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of inference execution.")
    disclaimer: str = Field(ADVISORY_DISCLAIMER, description="Advisory disclaimer.")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field("healthy", description="API process status.")
    service: str = Field("krs-dam-inflow-serving", description="Service identifier.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ReadyResponse(BaseModel):
    """Readiness probe response indicating model availability."""

    ready: bool = Field(..., description="Whether the MLflow registered model is loaded and ready.")
    model_name: str = Field("KRS_Dam_Inflow_Forecaster", description="Model name.")
    model_version: int = Field(1, description="Model version.")
    status_detail: str = Field(..., description="Readiness description.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ModelInfoResponse(BaseModel):
    """Comprehensive safe metadata describing the registered serving model."""

    model_name: str = Field(..., description="Registered model name.")
    model_version: int = Field(..., description="Registered model version.")
    alias: Optional[str] = Field(None, description="Registered model alias.")
    target_variable: str = Field(..., description="Prediction target name.")
    target_unit: str = Field(..., description="Target unit of measurement.")
    quality_status: str = Field(..., description="Quality acceptance decision.")
    deployment_approved: bool = Field(..., description="Production deployment approval flag.")
    intended_use: str = Field(..., description="Intended operational scope.")
    feature_count: int = Field(..., description="Expected number of input features.")
    feature_columns: List[str] = Field(..., description="Ordered list of required feature names.")
    advisory_disclaimer: str = Field(..., description="Domain safety disclaimer.")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""

    detail: str = Field(..., description="Human-readable error description.")
    error_type: str = Field(..., description="Error category classification.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")

