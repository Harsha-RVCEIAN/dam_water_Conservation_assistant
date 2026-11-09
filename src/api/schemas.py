"""Pydantic request and response schemas for FastAPI inference service.

Defines strict input validation, physical domain boundary constraints,
and structured output models with confidence intervals and warnings.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class InflowFeaturesInput(BaseModel):
    """Input payload containing all 19 strictly backward-looking features."""
    inflow_lag_1: float = Field(..., description="Observed river inflow at t-1 in cusecs", ge=0.0, json_schema_extra={"example": 2500.0})
    inflow_lag_2: float = Field(..., description="Observed river inflow at t-2 in cusecs", ge=0.0, json_schema_extra={"example": 2400.0})
    inflow_lag_3: float = Field(..., description="Observed river inflow at t-3 in cusecs", ge=0.0, json_schema_extra={"example": 2300.0})
    inflow_lag_7: float = Field(..., description="Observed river inflow at t-7 in cusecs", ge=0.0, json_schema_extra={"example": 2100.0})
    inflow_lag_14: float = Field(..., description="Observed river inflow at t-14 in cusecs", ge=0.0, json_schema_extra={"example": 1800.0})
    inflow_roll_mean_7: float = Field(..., description="Trailing 7-day mean inflow up to t-1", ge=0.0, json_schema_extra={"example": 2400.0})
    inflow_roll_std_7: float = Field(..., description="Trailing 7-day inflow std dev up to t-1", ge=0.0, json_schema_extra={"example": 150.0})
    inflow_roll_max_7: float = Field(..., description="Trailing 7-day peak inflow up to t-1", ge=0.0, json_schema_extra={"example": 2600.0})
    inflow_roll_mean_14: float = Field(..., description="Trailing 14-day mean inflow up to t-1", ge=0.0, json_schema_extra={"example": 2200.0})
    inflow_roll_mean_30: float = Field(..., description="Trailing 30-day mean inflow up to t-1", ge=0.0, json_schema_extra={"example": 2000.0})
    storage_lag_1: float = Field(..., description="Gross reservoir storage at t-1 in TMC", ge=0.0, le=55.0, json_schema_extra={"example": 35.0})
    storage_lag_7: float = Field(..., description="Gross reservoir storage at t-7 in TMC", ge=0.0, le=55.0, json_schema_extra={"example": 34.5})
    outflow_lag_1: float = Field(..., description="River outflow release at t-1 in cusecs", ge=0.0, json_schema_extra={"example": 1500.0})
    outflow_lag_7: float = Field(..., description="River outflow release at t-7 in cusecs", ge=0.0, json_schema_extra={"example": 1200.0})
    month: int = Field(..., description="Calendar month (1-12)", ge=1, le=12, json_schema_extra={"example": 8})
    day_of_year: int = Field(..., description="Day of calendar year (1-366)", ge=1, le=366, json_schema_extra={"example": 230})
    is_monsoon: int = Field(..., description="Binary monsoon indicator (1 for Jun-Sep, else 0)", ge=0, le=1, json_schema_extra={"example": 1})
    sin_day_of_year: float = Field(..., description="Cyclic harmonic sin(2pi*doy/365.25)", ge=-1.0, le=1.0, json_schema_extra={"example": 0.5})
    cos_day_of_year: float = Field(..., description="Cyclic harmonic cos(2pi*doy/365.25)", ge=-1.0, le=1.0, json_schema_extra={"example": -0.866})


class InflowForecastResponse(BaseModel):
    """Structured response for daily next-day inflow prediction."""
    forecast_horizon: str = Field(default="1 Day Ahead (t+1)")
    predicted_inflow_cusecs: float = Field(..., description="Predicted next-day river inflow in cusecs")
    interval_lower_cusecs: float = Field(..., description="Empirical 10th percentile lower bound (80% interval)")
    interval_upper_cusecs: float = Field(..., description="Empirical 90th percentile upper bound (80% interval)")
    confidence_level: str = Field(default="80% Empirical Prediction Interval")
    unit: str = Field(default="cusecs (mean daily rate). 1 cusec-day = 0.0000864 TMC")
    persistence_baseline_cusecs: Optional[float] = Field(None, description="Latest observed inflow rate (baseline)")
    difference_from_baseline: Optional[float] = Field(None, description="Forecast minus persistence baseline in cusecs")
    model_type: str = Field(default="RandomForestRegressor")
    domain_warnings: List[str] = Field(default_factory=list, description="Hydrological domain safety warnings")


class BatchInflowForecastRequest(BaseModel):
    """Batch input for multiple forecasting queries."""
    instances: List[InflowFeaturesInput] = Field(..., min_length=1, max_length=100)


class BatchInflowForecastResponse(BaseModel):
    """Batch response container."""
    count: int
    predictions: List[InflowForecastResponse]


class ConservationSimulationRequest(BaseModel):
    """Request payload for hypothetical water-conservation simulation."""
    current_storage_tmc: float = Field(..., ge=0.0, le=49.452, description="Current reservoir storage in TMC", json_schema_extra={"example": 35.0})
    assumed_inflow_cusecs: float = Field(..., ge=0.0, description="Assumed mean inflow rate in cusecs", json_schema_extra={"example": 3000.0})
    planned_outflow_cusecs: float = Field(..., ge=0.0, description="Planned river release rate in cusecs", json_schema_extra={"example": 1000.0})
    baseline_withdrawal_cusecs: float = Field(..., ge=0.0, description="Baseline canal withdrawal rate in cusecs", json_schema_extra={"example": 2000.0})
    reduction_percentage: float = Field(..., ge=0.0, le=100.0, description="Canal rationing reduction % (0-100%)", json_schema_extra={"example": 15.0})
    simulation_days: int = Field(default=7, ge=1, le=90, description="Simulation period in days", json_schema_extra={"example": 7})
    evaporation_cusecs: float = Field(default=150.0, ge=0.0, description="Estimated reservoir evaporation in cusecs", json_schema_extra={"example": 150.0})


class ConservationSimulationResponse(BaseModel):
    """Simulation output containing projected storages and conserved volume."""
    scenario_name: str
    simulation_days: int
    current_storage_tmc: float
    water_saved_tmc: float = Field(..., description="Total water conserved across horizon in TMC")
    savings_as_pct_capacity: float = Field(..., description="Savings as percentage of design gross capacity")
    projected_baseline_storage_tmc: float = Field(..., description="Projected storage under baseline withdrawal in TMC")
    projected_scenario_storage_tmc: float = Field(..., description="Projected storage under conservation in TMC")
    disclaimer: str


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str
    service: str
    model_loaded: bool
    model_type: str
    version: str = "2.0.0"
