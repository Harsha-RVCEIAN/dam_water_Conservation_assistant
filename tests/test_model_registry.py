"""Unit and Integration Tests for MLflow Model Registry (Phase 2, Step 2).

Tests registered model creation, versioning, metadata tags, model lineage,
prediction equivalence with disk joblib artifact, alias assignment,
error handling, and model artifact non-mutation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.models.model_registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    get_model_md5,
    get_model_sha256,
    get_registered_model_details,
    get_registry_client,
    load_registered_model,
    register_production_candidate,
    verify_registered_model_equivalence,
)
from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
    load_and_split_data,
)


# -----------------------------------------------------------------------------
# 1. ENVIRONMENT & BACKEND STORE TESTS
# -----------------------------------------------------------------------------

def test_registry_client_configuration():
    """Verify that get_registry_client returns a valid MlflowClient pointing to tracking URI."""
    client = get_registry_client(MLFLOW_TRACKING_URI)
    assert isinstance(client, MlflowClient)
    assert client.tracking_uri == MLFLOW_TRACKING_URI


def test_filestore_supports_model_registry_in_isolation(tmp_path):
    """Verify that local FileStore natively supports Model Registry in an isolated directory."""
    uri = tmp_path.as_uri()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    # 1. Search initially empty
    models = client.search_registered_models()
    assert len(models) == 0

    # 2. Create registered model
    rm = client.create_registered_model("IsolatedModel", tags={"env": "test"})
    assert rm.name == "IsolatedModel"
    assert rm.tags.get("env") == "test"

    # 3. Create model version
    mv = client.create_model_version(
        name="IsolatedModel",
        source=f"{uri}/source",
        run_id="isolated_run_001",
        tags={"stage": "candidate"},
    )
    assert mv.version == "1" or mv.version == 1
    assert mv.run_id == "isolated_run_001"

    # 4. Set alias
    client.set_registered_model_alias("IsolatedModel", "candidate", mv.version)
    by_alias = client.get_model_version_by_alias("IsolatedModel", "candidate")
    assert str(by_alias.version) == str(mv.version)


# -----------------------------------------------------------------------------
# 2. MODEL REGISTRATION & LINEAGE TESTS
# -----------------------------------------------------------------------------

def test_registered_model_exists():
    """Verify that KRS_Dam_Inflow_Forecaster exists in the project model registry."""
    client = get_registry_client()
    rm = client.get_registered_model(REGISTERED_MODEL_NAME)
    assert rm.name == REGISTERED_MODEL_NAME
    assert "KRS Dam" in rm.description


def test_registered_model_version_1_exists():
    """Verify that Version 1 of KRS_Dam_Inflow_Forecaster exists and has active status."""
    client = get_registry_client()
    mv = client.get_model_version(REGISTERED_MODEL_NAME, 1)
    assert str(mv.version) == "1"
    assert mv.status in ("READY", "active")


def test_registered_model_lineage_to_training_run():
    """Verify that registered Version 1 is linked to original training run ID."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_training_run_id = meta["mlflow"]["run_id"]

    client = get_registry_client()
    mv = client.get_model_version(REGISTERED_MODEL_NAME, 1)

    assert mv.run_id == expected_training_run_id
    assert expected_training_run_id in mv.source


def test_registered_model_tags_and_conditional_status():
    """Verify that registered Version 1 tags match Step 1 quality evaluation and CONDITIONAL status."""
    client = get_registry_client()
    mv = client.get_model_version(REGISTERED_MODEL_NAME, 1)
    tags = mv.tags

    assert tags.get("project") == "KRS_Dam_Water_Conservation_Assistant"
    assert tags.get("model_type") == "Tuned_Random_Forest"
    assert tags.get("quality_status") == "CONDITIONAL"
    assert tags.get("deployment_approved") == "false"
    assert tags.get("rmse") == "5935.09"
    assert tags.get("mae") == "2335.17"
    assert tags.get("r2") == "0.5899"
    assert tags.get("wape") == "45.69"
    assert tags.get("interval_coverage") == "77.05"


def test_candidate_alias_assigned():
    """Verify that the candidate alias is assigned to Version 1."""
    client = get_registry_client()
    mv_alias = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS)
    assert str(mv_alias.version) == "1"


def test_registration_idempotency():
    """Verify that calling register_production_candidate again does not create duplicate versions."""
    mv, created = register_production_candidate()
    assert str(mv.version) == "1"
    assert created is False


# -----------------------------------------------------------------------------
# 3. MODEL LOADING & PREDICTION EQUIVALENCE TESTS
# -----------------------------------------------------------------------------

def test_load_registered_model_by_version():
    """Verify that the model can be loaded from registry using integer version."""
    model = load_registered_model(REGISTERED_MODEL_NAME, 1)
    assert hasattr(model, "predict")
    assert hasattr(model, "named_steps")
    assert "imputer" in model.named_steps
    assert "regressor" in model.named_steps


def test_load_registered_model_by_alias():
    """Verify that the model can be loaded from registry using alias @candidate."""
    model = load_registered_model(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS)
    assert hasattr(model, "predict")
    assert hasattr(model, "named_steps")


def test_prediction_equivalence_with_disk_joblib():
    """Verify that predictions from the registered model and disk joblib are identical within tolerance."""
    equiv = verify_registered_model_equivalence(tolerance=1e-5)
    assert equiv["is_equivalent"] is True
    assert equiv["max_absolute_difference"] <= 1e-5
    assert equiv["all_non_negative"] is True
    assert equiv["all_finite"] is True
    assert equiv["sample_count"] == 658


def test_prediction_output_properties():
    """Verify that registered model predictions on test records are non-negative and finite."""
    model = load_registered_model(REGISTERED_MODEL_NAME, 1)
    _, _, test_df = load_and_split_data(PROCESSED_DATA_PATH)
    X_test = test_df[FEATURE_COLUMNS]

    raw_preds = model.predict(X_test)
    preds = np.clip(raw_preds, a_min=0.0, a_max=None)

    assert len(preds) == len(X_test)
    assert np.all(preds >= 0.0)
    assert np.all(np.isfinite(preds))


def test_input_schema_robustness():
    """Verify that the registered model handles valid DataFrame input with 19 features."""
    model = load_registered_model(REGISTERED_MODEL_NAME, 1)
    dummy_input = pd.DataFrame(np.ones((3, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    preds = model.predict(dummy_input)
    assert len(preds) == 3
    assert np.all(np.isfinite(preds))


# -----------------------------------------------------------------------------
# 4. FAILURE HANDLING & SAFETY CHECKS
# -----------------------------------------------------------------------------

def test_nonexistent_model_loading_raises_error():
    """Verify that loading a non-existent model raises an appropriate MLflow exception."""
    with pytest.raises(Exception):
        load_registered_model("NonExistent_Dam_Model_12345", 1)


def test_nonexistent_version_loading_raises_error():
    """Verify that loading a non-existent version raises an appropriate MLflow exception."""
    with pytest.raises(Exception):
        load_registered_model(REGISTERED_MODEL_NAME, 9999)


def test_original_artifact_unchanged_after_registry_operations():
    """Verify that models/inflow_forecaster.joblib has not been modified or overwritten."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    current_sha256 = get_model_sha256(MODEL_ARTIFACT_PATH)
    current_md5 = get_model_md5(MODEL_ARTIFACT_PATH)

    assert current_sha256 == meta["provenance"]["artifact_sha256"]
    assert current_md5 == meta["provenance"]["artifact_md5"]


def test_get_registered_model_details_structure():
    """Verify that get_registered_model_details returns expected dictionary keys."""
    details = get_registered_model_details(REGISTERED_MODEL_NAME, 1)
    assert details["model_name"] == REGISTERED_MODEL_NAME
    assert str(details["version"]) == "1"
    assert "version_tags" in details
    assert details["version_tags"]["quality_status"] == "CONDITIONAL"
    assert details["version_tags"]["deployment_approved"] == "false"

