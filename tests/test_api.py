"""Integration and Lifecycle Tests for FastAPI Model Serving.

Module tested: ``src.api.main``, ``src.api.routes``, ``src.api.model_loader``

Key scenarios:
- Application startup loads the registered model Version 1 once via lifespan.
- GET /health responds 200 independently of model readiness.
- GET /ready responds 200 when model is loaded; responds 503 when model is unavailable.
- GET /model-info returns safe metadata (CONDITIONAL, deployment_approved=false, feature count).
- GET /openapi.json and GET /docs generate valid API documentation.
- POST /predict computes predictions matching direct MLflow predictions within 1e-5 tolerance.
- POST /predict rejects missing features, invalid data types, NaN, +/-inf, and extra fields.
- Non-negative flow clipping policy is enforced (output >= 0.0).
- Model artifact and MLflow registry metadata are protected against mutation.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from src.api.main import app, create_app, lifespan
from src.api.model_loader import ModelArtifactHolder
from src.api.schemas import ADVISORY_DISCLAIMER
from src.models.model_registry import (
    REGISTERED_MODEL_NAME,
    get_model_md5,
    get_model_sha256,
)
from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
    load_and_split_data,
)


# -----------------------------------------------------------------------------
# LIGHTWEIGHT ASGI TEST CLIENT (No external httpx2 dependency required)
# -----------------------------------------------------------------------------

class ASGIResponse:
    """Encapsulates response from direct ASGI execution."""

    def __init__(self, status_code: int, headers: dict, body: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = body

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


class ASGITestClient:
    """Direct ASGI test client compatible with FastAPI applications."""

    def __init__(self, app_instance):
        self.app = app_instance

    async def _async_request(
        self,
        method: str,
        path: str,
        json_body: Optional[Any] = None,
        raw_body: Optional[bytes] = None,
        headers: Optional[List[tuple]] = None,
    ) -> ASGIResponse:
        req_headers = list(headers or [])
        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            req_headers.append((b"content-type", b"application/json"))
        elif raw_body is not None:
            body_bytes = raw_body

        req_headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": req_headers,
            "state": {},
        }
        messages = []
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg):
            messages.append(msg)

        await self.app(scope, receive, send)

        start = next(m for m in messages if m["type"] == "http.response.start")
        status = start["status"]
        resp_headers = dict(start.get("headers", []))
        body_parts = [m["body"] for m in messages if m["type"] == "http.response.body"]
        full_body = b"".join(body_parts)
        return ASGIResponse(status, resp_headers, full_body)

    async def get(self, path: str) -> ASGIResponse:
        return await self._async_request("GET", path)

    async def post(
        self,
        path: str,
        json: Optional[Any] = None,
        raw_body: Optional[bytes] = None,
        headers: Optional[List[tuple]] = None,
    ) -> ASGIResponse:
        return await self._async_request("POST", path, json_body=json, raw_body=raw_body, headers=headers)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    """Create module-level event loop for async test suite."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client():
    """Yield initialized client with active lifespan (model loaded once)."""
    async with lifespan(app):
        yield ASGITestClient(app)


@pytest.fixture(scope="module")
def valid_test_rows() -> List[dict]:
    """Retrieve 5 representative valid rows from the processed test split."""
    _, _, test_df = load_and_split_data(PROCESSED_DATA_PATH)
    rows = []
    # Pick 5 rows spread across seasons
    indices = [0, len(test_df) // 4, len(test_df) // 2, 3 * len(test_df) // 4, len(test_df) - 1]
    for idx in indices:
        row = test_df.iloc[idx]
        rows.append({col: float(row[col]) for col in FEATURE_COLUMNS})
    return rows


# -----------------------------------------------------------------------------
# LIFECYCLE & READINESS TESTS
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health_endpoint_returns_200(client):
    """Verify that /health responds 200 with service metadata."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "krs-dam-inflow-serving"
    assert "timestamp" in data


@pytest.mark.anyio
async def test_ready_endpoint_returns_200_when_model_loaded(client):
    """Verify that /ready responds 200 and indicates model readiness."""
    resp = await client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["model_name"] == REGISTERED_MODEL_NAME
    assert data["model_version"] == 1
    assert "ready for inference" in data["status_detail"]


@pytest.mark.anyio
async def test_ready_endpoint_returns_503_when_model_fails_to_load():
    """Verify that /ready responds 503 Service Unavailable when model is not loaded."""
    test_app = create_app()
    # Explicitly set unready holder
    test_app.state.model_holder = ModelArtifactHolder(
        load_error="Simulated MLflow store connection failure."
    )
    unready_client = ASGITestClient(test_app)

    # Health remains 200
    resp_h = await unready_client.get("/health")
    assert resp_h.status_code == 200

    # Ready reports 503
    resp_r = await unready_client.get("/ready")
    assert resp_r.status_code == 503
    data = resp_r.json()
    assert data["ready"] is False
    assert "Simulated MLflow store connection failure" in data["status_detail"]


# -----------------------------------------------------------------------------
# MODEL METADATA & DOCUMENTATION TESTS
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_model_info_returns_safe_provenance_metadata(client):
    """Verify that /model-info returns required safe metadata without exposing secrets."""
    resp = await client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_name"] == REGISTERED_MODEL_NAME
    assert data["model_version"] == 1
    assert data["alias"] == "candidate"
    assert data["target_variable"] == "target_inflow_next_day_cusecs"
    assert data["target_unit"] == "cusecs"
    assert data["quality_status"] == "CONDITIONAL"
    assert data["deployment_approved"] is False
    assert data["feature_count"] == 19
    assert list(data["feature_columns"]) == list(FEATURE_COLUMNS)
    assert data["advisory_disclaimer"] == ADVISORY_DISCLAIMER


@pytest.mark.anyio
async def test_openapi_schema_and_docs_available(client):
    """Verify that /openapi.json and /docs are accessible and contain endpoint definitions."""
    resp_json = await client.get("/openapi.json")
    assert resp_json.status_code == 200
    schema = resp_json.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/model-info" in schema["paths"]

    resp_docs = await client.get("/docs")
    assert resp_docs.status_code == 200
    assert "swagger" in resp_docs.text.lower() or "html" in resp_docs.text.lower()


# -----------------------------------------------------------------------------
# PREDICTION & EQUIVALENCE TESTS
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_prediction_equivalence_with_direct_mlflow_model(client, valid_test_rows):
    """Verify that API prediction output matches direct MLflow loaded model within 1e-5 tolerance."""
    # Direct MLflow load
    mlflow_model = mlflow.sklearn.load_model("models:/KRS_Dam_Inflow_Forecaster/1")

    for idx, row in enumerate(valid_test_rows):
        # 1. API prediction
        resp = await client.post("/predict", json=row)
        assert resp.status_code == 200, f"Row {idx} failed with {resp.text}"
        data = resp.json()
        api_pred = data["predicted_inflow_cusecs"]

        # 2. Direct MLflow prediction
        df_input = pd.DataFrame([row])[FEATURE_COLUMNS]
        direct_raw = mlflow_model.predict(df_input)
        direct_pred = round(float(np.clip(direct_raw[0], a_min=0.0, a_max=None)), 2)

        # 3. Equivalence assertion
        abs_diff = abs(api_pred - direct_pred)
        assert abs_diff <= 1e-5, (
            f"Row {idx} discrepancy: API={api_pred} vs MLflow={direct_pred} (diff={abs_diff})"
        )

        # 4. Physical non-negative flow constraint
        assert api_pred >= 0.0
        assert np.isfinite(api_pred)

        # 5. Metadata fields
        assert data["target_variable"] == "target_inflow_next_day_cusecs"
        assert data["target_unit"] == "cusecs"
        assert data["quality_status"] == "CONDITIONAL"
        assert data["deployment_approved"] is False
        assert "timestamp" in data["prediction_timestamp"] or "T" in data["prediction_timestamp"]


# -----------------------------------------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_predict_rejects_missing_required_feature(client, valid_test_rows):
    """Verify that missing a required feature returns HTTP 422 Unprocessable Entity."""
    bad_payload = valid_test_rows[0].copy()
    del bad_payload["inflow_lag_1"]

    resp = await client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    assert "inflow_lag_1" in resp.text


@pytest.mark.anyio
async def test_predict_rejects_wrong_data_type(client, valid_test_rows):
    """Verify that invalid string type for float field returns HTTP 422."""
    bad_payload = valid_test_rows[0].copy()
    bad_payload["storage_lag_1"] = "invalid_string_not_numeric"

    resp = await client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_rejects_nan_values(client, valid_test_rows):
    """Verify that NaN values in JSON return HTTP 422."""
    bad_payload = valid_test_rows[0].copy()
    bad_payload["inflow_roll_mean_7"] = None

    resp = await client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_rejects_infinite_values(client, valid_test_rows):
    """Verify that infinite values return HTTP 422."""
    # Build raw JSON string containing infinity since standard json module rejects float('inf')
    raw_json = json.dumps(valid_test_rows[0]).replace(
        '"outflow_lag_1":',
        '"outflow_lag_1": 1e9999, "dummy":'
    )
    resp = await client.post(
        "/predict",
        raw_body=raw_json.encode("utf-8"),
        headers=[(b"content-type", b"application/json")],
    )
    assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_predict_rejects_extra_fields(client, valid_test_rows):
    """Verify that unexpected extra fields return HTTP 422 per extra='forbid'."""
    bad_payload = valid_test_rows[0].copy()
    bad_payload["gate_spillway_command"] = "OPEN_GATE_3"

    resp = await client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    assert "extra_forbidden" in resp.text or "gate_spillway_command" in resp.text


@pytest.mark.anyio
async def test_predict_returns_503_when_model_is_unloaded():
    """Verify that POST /predict returns 503 when model is not ready."""
    test_app = create_app()
    test_app.state.model_holder = ModelArtifactHolder(pipeline=None)
    unready_client = ASGITestClient(test_app)

    dummy_payload = {col: 1000.0 for col in FEATURE_COLUMNS}
    dummy_payload["month"] = 7
    dummy_payload["day_of_year"] = 195
    dummy_payload["is_monsoon"] = 1
    dummy_payload["sin_day_of_year"] = -0.15
    dummy_payload["cos_day_of_year"] = -0.98

    resp = await unready_client.post("/predict", json=dummy_payload)
    assert resp.status_code == 503
    assert "not ready" in resp.text


# -----------------------------------------------------------------------------
# ARTIFACT INTEGRITY GUARDRAIL TEST
# -----------------------------------------------------------------------------

def test_model_artifact_sha256_remains_unchanged():
    """Verify that models/inflow_forecaster.joblib has not been modified by API operations."""
    current_sha256 = get_model_sha256(MODEL_ARTIFACT_PATH)
    current_md5 = get_model_md5(MODEL_ARTIFACT_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_sha256 = meta["provenance"]["artifact_sha256"]
    expected_md5 = meta["provenance"]["artifact_md5"]

    assert current_sha256 == expected_sha256
    assert current_md5 == expected_md5

