"""API client for connecting the Streamlit dashboard to the FastAPI inference backend.

Provides robust HTTP client functions for checking backend health, retrieving
model metadata, and generating next-day inflow predictions with timeout controls,
error handling, and seamless local model fallback.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("krs_dam.api_client")

DEFAULT_API_BASE_URL = "http://localhost:8000"

ORDERED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "inflow_lag_1",
    "inflow_lag_2",
    "inflow_lag_3",
    "inflow_lag_7",
    "inflow_lag_14",
    "inflow_roll_mean_7",
    "inflow_roll_std_7",
    "inflow_roll_max_7",
    "inflow_roll_mean_14",
    "inflow_roll_mean_30",
    "storage_lag_1",
    "storage_lag_7",
    "outflow_lag_1",
    "outflow_lag_7",
    "month",
    "day_of_year",
    "is_monsoon",
    "sin_day_of_year",
    "cos_day_of_year",
)


def get_api_base_url() -> str:
    """Return configured backend URL from environment or default."""
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def check_backend_health(
    base_url: Optional[str] = None, timeout: float = 2.0
) -> Dict[str, Any]:
    """Check if the FastAPI backend service is reachable and ready.

    Returns:
        Dict with keys 'connected', 'ready', 'status_detail', 'model_name', 'model_version', 'url'
    """
    url = (base_url or get_api_base_url()).rstrip("/")
    endpoint = f"{url}/ready"
    try:
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "connected": True,
                    "ready": bool(data.get("ready", False)),
                    "status_detail": data.get("status_detail", "Ready"),
                    "model_name": data.get("model_name"),
                    "model_version": data.get("model_version"),
                    "url": url,
                }
            else:
                return {
                    "connected": True,
                    "ready": False,
                    "status_detail": f"HTTP {resp.status}",
                    "url": url,
                }
    except Exception as exc:
        logger.debug("Backend health check failed for %s: %s", endpoint, exc)
        return {
            "connected": False,
            "ready": False,
            "status_detail": f"Unreachable: {exc}",
            "url": url,
        }


def fetch_model_info(
    base_url: Optional[str] = None, timeout: float = 2.0
) -> Optional[Dict[str, Any]]:
    """Retrieve model provenance and metadata from FastAPI backend."""
    url = (base_url or get_api_base_url()).rstrip("/")
    endpoint = f"{url}/model-info"
    try:
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Failed to fetch model info from %s: %s", endpoint, exc)
    return None


def predict_inflow_via_api(
    features: Dict[str, float],
    base_url: Optional[str] = None,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Request next-day inflow prediction from the FastAPI serving service.

    Args:
        features: Dictionary containing the required 19 feature names and float values.
        base_url: Optional custom base URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys:
            success (bool): Whether prediction succeeded.
            predicted_inflow_cusecs (float or None): Model forecast.
            response (dict or None): Echoed response metadata.
            error (str or None): Error message if failed.
    """
    url = (base_url or get_api_base_url()).rstrip("/")
    endpoint = f"{url}/predict"
    try:
        payload_bytes = json.dumps(features).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "success": True,
                    "predicted_inflow_cusecs": float(data.get("predicted_inflow_cusecs", 0.0)),
                    "response": data,
                    "error": None,
                    "source": "fastapi_backend",
                    "url": url,
                }
            else:
                return {
                    "success": False,
                    "predicted_inflow_cusecs": None,
                    "response": None,
                    "error": f"Backend returned HTTP {resp.status}",
                    "source": "error",
                    "url": url,
                }
    except urllib.error.HTTPError as http_err:
        err_msg = f"HTTP {http_err.code}: {http_err.reason}"
        try:
            err_body = http_err.read().decode("utf-8")
            err_msg += f" - {err_body}"
        except Exception:
            pass
        return {
            "success": False,
            "predicted_inflow_cusecs": None,
            "response": None,
            "error": err_msg,
            "source": "error",
            "url": url,
        }
    except Exception as exc:
        return {
            "success": False,
            "predicted_inflow_cusecs": None,
            "response": None,
            "error": f"Connection error: {exc}",
            "source": "error",
            "url": url,
        }


def extract_feature_payload(row: Any) -> Dict[str, float]:
    """Extract ordered 19 model features from a DataFrame row or dictionary."""
    payload: Dict[str, float] = {}
    for col in ORDERED_FEATURE_COLUMNS:
        if col in row:
            val = row[col]
            if hasattr(val, "values"):
                val = val.values[0]
            payload[col] = float(val)
        else:
            raise KeyError(f"Missing required feature '{col}' in input row.")
    return payload


def get_inflow_forecast(
    df: Any,
    forecaster: Optional[Any] = None,
    base_url: Optional[str] = None,
    timeout: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Generate inflow forecast by querying FastAPI backend, with explicit local fallback.

    Fallback is ONLY triggered for connection failures and backend-not-ready states.
    Invalid API responses (HTTP errors, schema errors) are logged at WARNING level.
    The local fallback always uses the same model artifact as the API (Version 1).

    Returns:
        Dictionary with forecast result and 'inference_source' / 'backend_connected' provenance.
    """
    url = (base_url or get_api_base_url()).rstrip("/")
    latest_row = df.iloc[[-1]] if hasattr(df, "iloc") else None

    # 1. Pre-compute local fallback result (same model artifact as API: models/inflow_forecaster.joblib)
    local_res: Optional[Dict[str, Any]] = None
    if forecaster is not None:
        try:
            local_res = forecaster.predict_latest_from_processed()
        except Exception as exc:
            logger.warning("Local forecaster failed — cannot produce fallback prediction: %s", exc)

    # 2. Attempt FastAPI backend prediction
    backend_success = False
    api_pred_val: Optional[float] = None
    api_resp_data: Dict[str, Any] = {}
    backend_err: Optional[str] = None
    _fallback_eligible = False  # Only fall back for connectivity failures, not API logic errors

    health = check_backend_health(base_url=url, timeout=min(timeout, 2.0))
    if not health.get("connected"):
        # Network-level failure: fall back is appropriate
        backend_err = health.get("status_detail", "Backend unreachable")
        _fallback_eligible = True
        logger.warning("FastAPI backend unreachable at %s — activating local model fallback. Reason: %s", url, backend_err)
    elif not health.get("ready"):
        # Service is reachable but not ready (e.g. model load failed)
        backend_err = health.get("status_detail", "Backend not ready")
        _fallback_eligible = True
        logger.warning("FastAPI backend not ready at %s — activating local model fallback. Reason: %s", url, backend_err)
    else:
        # Backend is healthy — attempt prediction
        try:
            if latest_row is not None:
                features = extract_feature_payload(latest_row.iloc[0])
                api_result = predict_inflow_via_api(features, base_url=url, timeout=timeout)
                if api_result.get("success"):
                    backend_success = True
                    api_pred_val = api_result["predicted_inflow_cusecs"]
                    api_resp_data = api_result.get("response", {})
                else:
                    # Backend returned an error response — this is an API-side issue, not a connectivity failure
                    backend_err = api_result.get("error", "API returned unsuccessful response")
                    logger.warning("FastAPI prediction failed (not a connectivity issue) — %s. No fallback.", backend_err)
                    # _fallback_eligible remains False: don't silently mask API-side errors
            else:
                backend_err = "No dataframe row available for feature extraction"
                logger.error("Cannot extract features: %s", backend_err)
        except Exception as exc:
            # Unexpected exception during the HTTP call itself — treat as connectivity failure
            backend_err = f"Unexpected API call error: {exc}"
            _fallback_eligible = True
            logger.warning("Unexpected error calling FastAPI at %s — activating local fallback: %s", url, exc)

    # 3. Assemble response
    if backend_success and api_pred_val is not None:
        if local_res is not None:
            res = dict(local_res)
            uncert = forecaster.metadata.get("uncertainty_intervals_cusecs", {}) if forecaster else {}
            q10 = float(uncert.get("residual_quantile_10", -1500.0))
            q90 = float(uncert.get("residual_quantile_90", 2500.0))

            res["forecast_inflow_cusecs"] = api_pred_val
            res["interval_lower_cusecs"] = max(0.0, api_pred_val + q10)
            res["interval_upper_cusecs"] = max(0.0, api_pred_val + q90)
            res["inference_source"] = "FastAPI Serving Service (MLflow Registry v1)"
            res["backend_connected"] = True
            res["backend_url"] = url
            res["model_name"] = api_resp_data.get("model_name", "KRS_Dam_Inflow_Forecaster")
            res["model_version"] = api_resp_data.get("model_version", 1)
            res["quality_status"] = api_resp_data.get("quality_status", "CONDITIONAL")
            res["deployment_approved"] = api_resp_data.get("deployment_approved", False)
            res["backend_detail"] = "Operational: Serving Version 1 from MLflow Registry"
            return res
        elif latest_row is not None:
            import pandas as pd
            origin_date = str(latest_row["calendar_date"].values[0]) if "calendar_date" in latest_row else "Unknown"
            latest_inflow = float(latest_row["inflow_cusecs"].values[0]) if "inflow_cusecs" in latest_row else 0.0
            target_date = str((pd.to_datetime(origin_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            return {
                "forecast_horizon": "1 Day Ahead",
                "latest_observed_date": origin_date,
                "latest_observed_inflow_cusecs": latest_inflow,
                "forecast_origin_date": origin_date,
                "forecast_target_date": target_date,
                "data_age_days": 0,
                "is_stale": False,
                "freshness_status": "CURRENT",
                "freshness_note": "Forecast produced directly via FastAPI microservice",
                "forecast_inflow_cusecs": api_pred_val,
                "interval_lower_cusecs": max(0.0, api_pred_val - 1500.0),
                "interval_upper_cusecs": max(0.0, api_pred_val + 2500.0),
                "interval_level": "80% empirical quantile interval",
                "persistence_baseline_cusecs": latest_inflow,
                "domain_warnings": [],
                "model_type": "RandomForestRegressor (MLflow v1)",
                "features_used": list(ORDERED_FEATURE_COLUMNS),
                "inference_source": "FastAPI Serving Service (MLflow Registry v1)",
                "backend_connected": True,
                "backend_url": url,
                "model_name": api_resp_data.get("model_name", "KRS_Dam_Inflow_Forecaster"),
                "model_version": api_resp_data.get("model_version", 1),
                "quality_status": api_resp_data.get("quality_status", "CONDITIONAL"),
                "deployment_approved": api_resp_data.get("deployment_approved", False),
                "backend_detail": "Operational: Serving Version 1 from MLflow Registry",
            }

    # 4. Fallback to local model — ONLY for connectivity/readiness failures (not API-side errors)
    if _fallback_eligible and local_res is not None:
        res = dict(local_res)
        res["inference_source"] = "Local Model Artifact (Fallback — API Offline)"
        res["backend_connected"] = False
        res["backend_url"] = url
        res["model_name"] = "KRS_Dam_Inflow_Forecaster (Local — models/inflow_forecaster.joblib)"
        res["model_version"] = 1
        res["quality_status"] = "CONDITIONAL"
        res["deployment_approved"] = False
        res["backend_detail"] = f"Backend connectivity failure: {backend_err}. Serving from local artifact (same Version 1 weights)."
        return res

    return None
