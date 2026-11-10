"""Unit Tests for FastAPI Serving Pydantic Request and Response Schemas.

Module tested: ``src.api.schemas``

Key scenarios:
- Valid payloads parse successfully and preserve numeric precision.
- Missing required features, invalid data types, and null values raise ValidationError.
- NaN, positive infinity, and negative infinity are strictly rejected.
- Unexpected extra fields are rejected per extra='forbid' policy.
- Bounded calendar fields (month: 1-12, day_of_year: 1-366, is_monsoon: 0-1) are validated.
- Cyclical sine/cosine day-of-year features are correctly auto-derived when omitted.
- to_feature_dataframe() produces a single-row DataFrame ordered identically to FEATURE_COLUMNS.
- Response models (InflowPredictionResponse, HealthResponse, ReadyResponse, ModelInfoResponse)
  validate and serialize expected fields and types.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ADVISORY_DISCLAIMER,
    HealthResponse,
    InflowPredictionRequest,
    InflowPredictionResponse,
    ModelInfoResponse,
    ReadyResponse,
)
from src.models.train import FEATURE_COLUMNS


@pytest.fixture
def valid_payload_dict() -> dict:
    """Return a dictionary representing a valid complete prediction request."""
    return {
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


def test_valid_request_parses_successfully(valid_payload_dict):
    """Verify that a valid feature payload is correctly parsed into InflowPredictionRequest."""
    req = InflowPredictionRequest(**valid_payload_dict)
    assert req.inflow_lag_1 == 1500.0
    assert req.month == 7
    assert req.day_of_year == 195
    assert req.is_monsoon == 1
    assert req.sin_day_of_year == -0.15
    assert req.cos_day_of_year == -0.98


def test_to_feature_dataframe_matches_feature_columns_and_order(valid_payload_dict):
    """Verify that to_feature_dataframe produces exact columns and column ordering."""
    req = InflowPredictionRequest(**valid_payload_dict)
    df = req.to_feature_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == list(FEATURE_COLUMNS)
    assert df["inflow_lag_1"].iloc[0] == 1500.0
    assert df["storage_lag_1"].iloc[0] == 35.5


def test_cyclical_features_auto_derived_when_omitted(valid_payload_dict):
    """Verify that sin_day_of_year and cos_day_of_year are derived from day_of_year if not supplied."""
    payload = valid_payload_dict.copy()
    del payload["sin_day_of_year"]
    del payload["cos_day_of_year"]

    req = InflowPredictionRequest(**payload)
    doy = payload["day_of_year"]
    expected_sin = float(np.sin(2.0 * np.pi * doy / 365.25))
    expected_cos = float(np.cos(2.0 * np.pi * doy / 365.25))

    assert math.isclose(req.sin_day_of_year, expected_sin, abs_tol=1e-5)
    assert math.isclose(req.cos_day_of_year, expected_cos, abs_tol=1e-5)


def test_missing_required_feature_raises_validation_error(valid_payload_dict):
    """Verify that omitting a required feature raises a ValidationError."""
    payload = valid_payload_dict.copy()
    del payload["inflow_lag_1"]

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "inflow_lag_1" in str(exc_info.value)


def test_null_feature_raises_validation_error(valid_payload_dict):
    """Verify that passing None for a feature raises a ValidationError."""
    payload = valid_payload_dict.copy()
    payload["storage_lag_1"] = None

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "storage_lag_1" in str(exc_info.value)


def test_nan_feature_raises_validation_error(valid_payload_dict):
    """Verify that passing NaN for a feature raises a ValidationError."""
    payload = valid_payload_dict.copy()
    payload["inflow_roll_mean_7"] = float("nan")

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "cannot be NaN" in str(exc_info.value)


def test_positive_infinity_raises_validation_error(valid_payload_dict):
    """Verify that positive infinity is rejected."""
    payload = valid_payload_dict.copy()
    payload["outflow_lag_1"] = float("inf")

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "cannot be infinite" in str(exc_info.value)


def test_negative_infinity_raises_validation_error(valid_payload_dict):
    """Verify that negative infinity is rejected."""
    payload = valid_payload_dict.copy()
    payload["inflow_lag_2"] = float("-inf")

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "cannot be infinite" in str(exc_info.value)


def test_extra_fields_strictly_forbidden(valid_payload_dict):
    """Verify that extra unexpected fields trigger a ValidationError per extra='forbid'."""
    payload = valid_payload_dict.copy()
    payload["unexpected_live_telemetry_reading"] = 123.45

    with pytest.raises(ValidationError) as exc_info:
        InflowPredictionRequest(**payload)
    assert "unexpected_live_telemetry_reading" in str(exc_info.value)
    assert "extra_forbidden" in str(exc_info.value)


def test_month_out_of_bounds_raises_validation_error(valid_payload_dict):
    """Verify that month must be between 1 and 12."""
    payload = valid_payload_dict.copy()
    payload["month"] = 13

    with pytest.raises(ValidationError):
        InflowPredictionRequest(**payload)


def test_day_of_year_out_of_bounds_raises_validation_error(valid_payload_dict):
    """Verify that day_of_year must be between 1 and 366."""
    payload = valid_payload_dict.copy()
    payload["day_of_year"] = 400

    with pytest.raises(ValidationError):
        InflowPredictionRequest(**payload)


def test_is_monsoon_out_of_bounds_raises_validation_error(valid_payload_dict):
    """Verify that is_monsoon must be 0 or 1."""
    payload = valid_payload_dict.copy()
    payload["is_monsoon"] = 2

    with pytest.raises(ValidationError):
        InflowPredictionRequest(**payload)


def test_inflow_prediction_response_schema():
    """Verify that InflowPredictionResponse correctly models inference output."""
    resp = InflowPredictionResponse(
        predicted_inflow_cusecs=1542.35,
        prediction_timestamp="2026-11-10T11:00:00Z",
    )
    assert resp.predicted_inflow_cusecs == 1542.35
    assert resp.target_variable == "target_inflow_next_day_cusecs"
    assert resp.target_unit == "cusecs"
    assert resp.model_name == "KRS_Dam_Inflow_Forecaster"
    assert resp.model_version == 1
    assert resp.quality_status == "CONDITIONAL"
    assert resp.deployment_approved is False
    assert resp.disclaimer == ADVISORY_DISCLAIMER


def test_health_and_ready_responses():
    """Verify that HealthResponse and ReadyResponse model diagnostic states."""
    h = HealthResponse(status="healthy", timestamp="2026-11-10T11:00:00Z")
    assert h.status == "healthy"
    assert h.service == "krs-dam-inflow-serving"

    r = ReadyResponse(
        ready=True,
        status_detail="Model loaded",
        timestamp="2026-11-10T11:00:00Z",
    )
    assert r.ready is True
    assert r.model_name == "KRS_Dam_Inflow_Forecaster"
    assert r.model_version == 1


def test_model_info_response():
    """Verify that ModelInfoResponse correctly contains safe metadata without secrets."""
    info = ModelInfoResponse(
        model_name="KRS_Dam_Inflow_Forecaster",
        model_version=1,
        alias="candidate",
        target_variable="target_inflow_next_day_cusecs",
        target_unit="cusecs",
        quality_status="CONDITIONAL",
        deployment_approved=False,
        intended_use="Historical inflow forecasting",
        feature_count=19,
        feature_columns=list(FEATURE_COLUMNS),
        advisory_disclaimer=ADVISORY_DISCLAIMER,
    )
    assert info.model_name == "KRS_Dam_Inflow_Forecaster"
    assert info.quality_status == "CONDITIONAL"
    assert info.feature_count == 19

