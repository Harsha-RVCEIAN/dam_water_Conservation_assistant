"""Unit tests for the Streamlit-to-FastAPI client module (src.services.api_client)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.services.api_client import (
    DEFAULT_API_BASE_URL,
    ORDERED_FEATURE_COLUMNS,
    check_backend_health,
    extract_feature_payload,
    fetch_model_info,
    get_api_base_url,
    get_inflow_forecast,
    predict_inflow_via_api,
)


def test_get_api_base_url_default(monkeypatch):
    """Verify default base URL when API_BASE_URL env var is unset."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert get_api_base_url() == DEFAULT_API_BASE_URL


def test_get_api_base_url_override(monkeypatch):
    """Verify custom base URL from environment variable."""
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    assert get_api_base_url() == "http://api:8000"


def test_check_backend_health_unreachable():
    """Verify health check returns connected=False when server is not running."""
    res = check_backend_health("http://127.0.0.1:59999", timeout=0.5)
    assert res["connected"] is False
    assert res["ready"] is False
    assert "Unreachable" in res["status_detail"]


def test_check_backend_health_mock_success():
    """Verify health check returns correct metadata when backend responds 200."""
    mock_payload = json.dumps({
        "ready": True,
        "model_name": "KRS_Dam_Inflow_Forecaster",
        "model_version": 1,
        "status_detail": "Model loaded successfully.",
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = mock_payload
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = check_backend_health("http://mock-api:8000")
        assert res["connected"] is True
        assert res["ready"] is True
        assert res["model_name"] == "KRS_Dam_Inflow_Forecaster"
        assert res["model_version"] == 1


def test_predict_inflow_via_api_mock_success():
    """Verify predict_inflow_via_api parses response correctly."""
    mock_payload = json.dumps({
        "predicted_inflow_cusecs": 2708.38,
        "target_variable": "target_inflow_next_day_cusecs",
        "target_unit": "cusecs",
        "model_name": "KRS_Dam_Inflow_Forecaster",
        "model_version": 1,
        "quality_status": "CONDITIONAL",
        "deployment_approved": False,
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = mock_payload
    mock_resp.__enter__.return_value = mock_resp

    sample_features = {"inflow_lag_1": 1500.0, "month": 7}
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = predict_inflow_via_api(sample_features, "http://mock-api:8000")
        assert res["success"] is True
        assert res["predicted_inflow_cusecs"] == 2708.38
        assert res["source"] == "fastapi_backend"


def test_predict_inflow_via_api_connection_error():
    """Verify graceful error reporting when connection fails."""
    sample_features = {"inflow_lag_1": 1500.0}
    res = predict_inflow_via_api(sample_features, "http://127.0.0.1:59999", timeout=0.5)
    assert res["success"] is False
    assert res["predicted_inflow_cusecs"] is None
    assert "Connection error" in res["error"]


def test_extract_feature_payload():
    """Verify extraction of ordered 19 features from a pandas Series or dict."""
    row_data = {col: float(i * 10) for i, col in enumerate(ORDERED_FEATURE_COLUMNS)}
    payload = extract_feature_payload(row_data)
    assert len(payload) == 19
    assert payload["inflow_lag_1"] == 0.0

    # Missing column should raise KeyError
    incomplete = {"inflow_lag_1": 100.0}
    with pytest.raises(KeyError, match="Missing required feature"):
        extract_feature_payload(incomplete)


def test_get_inflow_forecast_fallback_when_api_offline():
    """Verify that get_inflow_forecast falls back to local forecaster ONLY for connectivity failures."""
    mock_forecaster = MagicMock()
    mock_forecaster.predict_latest_from_processed.return_value = {
        "forecast_origin_date": "2024-11-20",
        "forecast_target_date": "2024-11-21",
        "forecast_inflow_cusecs": 1234.5,
        "interval_lower_cusecs": 500.0,
        "interval_upper_cusecs": 2500.0,
        "persistence_baseline_cusecs": 1200.0,
        "is_stale": False,
        "data_age_days": 1,
        "freshness_status": "CURRENT",
        "freshness_note": "Archive is fresh",
        "domain_warnings": [],
    }

    dummy_df = pd.DataFrame([{col: 10.0 for col in ORDERED_FEATURE_COLUMNS}])
    dummy_df["calendar_date"] = "2024-11-20"
    dummy_df["inflow_cusecs"] = 1200.0

    # Point to non-running server
    res = get_inflow_forecast(dummy_df, forecaster=mock_forecaster, base_url="http://127.0.0.1:59998")
    assert res is not None
    assert res["forecast_inflow_cusecs"] == 1234.5
    assert "Fallback" in res["inference_source"]
    assert res["backend_connected"] is False


def test_get_inflow_forecast_api_success():
    """Verify that get_inflow_forecast uses API prediction when backend is reachable."""
    mock_forecaster = MagicMock()
    mock_forecaster.metadata = {
        "uncertainty_intervals_cusecs": {
            "residual_quantile_10": -800.0,
            "residual_quantile_90": 1200.0,
        }
    }
    mock_forecaster.predict_latest_from_processed.return_value = {
        "forecast_origin_date": "2024-11-20",
        "forecast_target_date": "2024-11-21",
        "forecast_inflow_cusecs": 1000.0,
        "interval_lower_cusecs": 200.0,
        "interval_upper_cusecs": 2200.0,
        "persistence_baseline_cusecs": 1200.0,
        "is_stale": False,
        "data_age_days": 1,
        "freshness_status": "CURRENT",
        "freshness_note": "Archive is fresh",
        "domain_warnings": [],
    }

    dummy_df = pd.DataFrame([{col: 10.0 for col in ORDERED_FEATURE_COLUMNS}])
    dummy_df["calendar_date"] = "2024-11-20"
    dummy_df["inflow_cusecs"] = 1200.0

    mock_health = {"connected": True, "ready": True, "status_detail": "Ready"}
    mock_predict = {
        "success": True,
        "predicted_inflow_cusecs": 3500.0,
        "response": {
            "model_name": "KRS_Dam_Inflow_Forecaster",
            "model_version": 1,
            "quality_status": "CONDITIONAL",
            "deployment_approved": False,
        },
    }

    with patch("src.services.api_client.check_backend_health", return_value=mock_health), \
         patch("src.services.api_client.predict_inflow_via_api", return_value=mock_predict):
        res = get_inflow_forecast(dummy_df, forecaster=mock_forecaster, base_url="http://mock-api:8000")
        assert res is not None
        assert res["forecast_inflow_cusecs"] == 3500.0
        assert res["interval_lower_cusecs"] == 2700.0  # 3500 - 800
        assert res["interval_upper_cusecs"] == 4700.0  # 3500 + 1200
        assert "FastAPI" in res["inference_source"]
        assert res["backend_connected"] is True


def test_get_inflow_forecast_no_fallback_on_api_error_response():
    """Verify that an API-side error (bad response, not connectivity) does NOT silently fall back."""
    mock_forecaster = MagicMock()
    mock_forecaster.predict_latest_from_processed.return_value = {
        "forecast_inflow_cusecs": 999.0,
        "interval_lower_cusecs": 100.0,
        "interval_upper_cusecs": 2000.0,
        "persistence_baseline_cusecs": 900.0,
        "is_stale": False,
        "data_age_days": 1,
        "freshness_status": "CURRENT",
        "freshness_note": "Fresh",
        "domain_warnings": [],
        "forecast_origin_date": "2024-11-20",
        "forecast_target_date": "2024-11-21",
    }

    dummy_df = pd.DataFrame([{col: 10.0 for col in ORDERED_FEATURE_COLUMNS}])
    dummy_df["calendar_date"] = "2024-11-20"
    dummy_df["inflow_cusecs"] = 1200.0

    # Backend is reachable and "ready" but returns an error response (e.g. 422 schema error)
    mock_health = {"connected": True, "ready": True, "status_detail": "Ready"}
    mock_predict_fail = {
        "success": False,
        "predicted_inflow_cusecs": None,
        "response": None,
        "error": "HTTP 422: Unprocessable Content - schema mismatch",
        "source": "error",
    }

    with patch("src.services.api_client.check_backend_health", return_value=mock_health), \
         patch("src.services.api_client.predict_inflow_via_api", return_value=mock_predict_fail):
        res = get_inflow_forecast(dummy_df, forecaster=mock_forecaster, base_url="http://mock-api:8000")
        # Must NOT return a fallback result — the API was reachable but returned a logic error
        assert res is None, (
            "get_inflow_forecast must NOT silently fall back to local model when "
            "the backend is reachable but returned an error response."
        )


def test_get_inflow_forecast_fallback_eligible_on_connection_error():
    """Verify fallback IS used when backend is not connected (network-level failure)."""
    mock_forecaster = MagicMock()
    mock_forecaster.predict_latest_from_processed.return_value = {
        "forecast_inflow_cusecs": 1234.0,
        "interval_lower_cusecs": 400.0,
        "interval_upper_cusecs": 2500.0,
        "persistence_baseline_cusecs": 1100.0,
        "is_stale": False,
        "data_age_days": 1,
        "freshness_status": "CURRENT",
        "freshness_note": "Fresh",
        "domain_warnings": [],
        "forecast_origin_date": "2024-11-20",
        "forecast_target_date": "2024-11-21",
    }

    dummy_df = pd.DataFrame([{col: 10.0 for col in ORDERED_FEATURE_COLUMNS}])
    dummy_df["calendar_date"] = "2024-11-20"
    dummy_df["inflow_cusecs"] = 1100.0

    # Backend is completely unreachable
    mock_health = {"connected": False, "ready": False, "status_detail": "Unreachable: Connection refused"}

    with patch("src.services.api_client.check_backend_health", return_value=mock_health):
        res = get_inflow_forecast(dummy_df, forecaster=mock_forecaster, base_url="http://dead-host:8000")
        assert res is not None
        assert res["forecast_inflow_cusecs"] == 1234.0
        assert "Fallback" in res["inference_source"]
        assert res["backend_connected"] is False
        assert res["deployment_approved"] is False
        assert res["quality_status"] == "CONDITIONAL"
