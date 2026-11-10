"""Pydantic request and response schemas for the KRS Dam Water Conservation API.

These schemas define the input and output contracts for the 27-feature Tuned Random Forest
rainfall-enhanced inflow forecaster, including telemetry degradation warnings.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class DamFeatures(BaseModel):
    """19 Dam hydrological lags and calendar features (mandatory)."""
    inflow_lag_1: float = Field(..., description="Observed reservoir inflow at t-1 in cusecs (cfs)", ge=0.0)
    inflow_lag_2: float = Field(..., description="Observed reservoir inflow at t-2 in cusecs", ge=0.0)
    inflow_lag_3: float = Field(..., description="Observed reservoir inflow at t-3 in cusecs", ge=0.0)
    inflow_lag_7: float = Field(..., description="Observed reservoir inflow at t-7 in cusecs", ge=0.0)
    inflow_lag_14: float = Field(..., description="Observed reservoir inflow at t-14 in cusecs", ge=0.0)
    inflow_roll_mean_7: float = Field(..., description="Trailing 7-day rolling mean inflow in cusecs", ge=0.0)
    inflow_roll_std_7: float = Field(..., description="Trailing 7-day rolling standard deviation in cusecs", ge=0.0)
    inflow_roll_max_7: float = Field(..., description="Trailing 7-day peak inflow in cusecs", ge=0.0)
    inflow_roll_mean_14: float = Field(..., description="Trailing 14-day rolling mean inflow in cusecs", ge=0.0)
    inflow_roll_mean_30: float = Field(..., description="Trailing 30-day rolling mean inflow in cusecs", ge=0.0)
    storage_lag_1: float = Field(..., description="Observed reservoir storage at t-1 in TMC", ge=0.0, le=55.0)
    storage_lag_7: float = Field(..., description="Observed reservoir storage at t-7 in TMC", ge=0.0, le=55.0)
    outflow_lag_1: float = Field(..., description="River outflow at t-1 in cusecs", ge=0.0)
    outflow_lag_7: float = Field(..., description="River outflow at t-7 in cusecs", ge=0.0)
    month: int = Field(..., description="Calendar month (1-12)", ge=1, le=12)
    day_of_year: int = Field(..., description="Day of year (1-366)", ge=1, le=366)
    is_monsoon: int = Field(..., description="Binary monsoon indicator (1 for Jun-Sep, else 0)", ge=0, le=1)
    sin_day_of_year: float = Field(..., description="Sine of day of year seasonal cycle", ge=-1.0, le=1.0)
    cos_day_of_year: float = Field(..., description="Cosine of day of year seasonal cycle", ge=-1.0, le=1.0)


class RainfallFeatures(BaseModel):
    """8 Catchment rainfall features from 10 upstream gauge stations.

    If live rain gauge telemetry is unavailable, these fields can be omitted or set to None.
    The service will automatically trigger median imputation and return an explicit degraded
    telemetry advisory to dam operators.
    """
    rainfall_mm_lag_1: Optional[float] = Field(None, description="Catchment-mean daily rainfall at t-1 in mm", ge=0.0)
    rainfall_mm_sum_3d: Optional[float] = Field(None, description="Trailing 3-day rainfall sum [t-3, t-1] in mm", ge=0.0)
    rainfall_mm_sum_7d: Optional[float] = Field(None, description="Trailing 7-day rainfall sum [t-7, t-1] in mm", ge=0.0)
    rainfall_stations_reporting_lag_1: Optional[int] = Field(None, description="Count of reporting stations at t-1 (0-10)", ge=0, le=10)
    rainfall_station_coverage_lag_1: Optional[float] = Field(None, description="Station coverage ratio (0.0 to 1.0)", ge=0.0, le=1.0)
    rainfall_lag_1_available: Optional[int] = Field(None, description="1 if t-1 rainfall observed, 0 if missing", ge=0, le=1)
    rainfall_window_3d_complete: Optional[int] = Field(None, description="1 if all 3 prior days recorded, else 0", ge=0, le=1)
    rainfall_window_7d_complete: Optional[int] = Field(None, description="1 if all 7 prior days recorded, else 0", ge=0, le=1)


class PredictionInput(DamFeatures, RainfallFeatures):
    """Unified 27-feature prediction input schema."""
    pass


class PredictionResponse(BaseModel):
    """Inference output schema with transparency flags for degraded telemetry."""
    predicted_inflow_cusecs: float = Field(..., description="Next-day forecasted inflow in cusecs (cfs)")
    model_name: str = Field(..., description="Name of the champion model serving this forecast")
    model_version: str = Field(..., description="Version of the model registered in MLflow")
    telemetry_status: str = Field(..., description="'NOMINAL' if all rainfall telemetry was provided; 'DEGRADED' if imputed")
    rainfall_available: bool = Field(..., description="Whether valid upstream rainfall gauge data was present")
    imputed_features: list[str] = Field(default_factory=list, description="List of feature names that were missing and imputed")
    confidence_advisory: Optional[str] = Field(None, description="Operational advisory if telemetry was degraded")


class BatchPredictionInput(BaseModel):
    """Batch input containing multiple day feature rows."""
    records: list[PredictionInput] = Field(..., min_length=1, max_length=365)


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""
    count: int
    predictions: list[PredictionResponse]


class WaterConservationInput(BaseModel):
    """Inputs to simulate water balance and conservation recommendations."""
    current_storage_tmc: float = Field(..., description="Current reservoir storage in TMC", ge=0.0, le=49.452)
    forecast_inflow_cusecs: float = Field(..., description="Predicted next-day inflow in cusecs", ge=0.0)
    planned_outflow_cusecs: float = Field(..., description="Planned river discharge in cusecs", ge=0.0)
    planned_canal_withdrawal_cusecs: float = Field(..., description="Planned canal withdrawal in cusecs", ge=0.0)
    horizon_days: int = Field(7, description="Simulation horizon (1-14 days)", ge=1, le=14)


class WaterConservationResponse(BaseModel):
    """Conservation guidance and projected reservoir trajectory."""
    current_storage_tmc: float
    projected_storage_tmc: float
    net_storage_change_tmc: float
    water_availability_tier: str  # "Surplus", "Normal", "Stress", "Deficit"
    conservation_advisory: str
    action_items: list[str]
