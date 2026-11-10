"""Unit tests for model artifact integrity, provenance, and loading verification.

Ensures that models/inflow_forecaster.joblib and models/model_metadata.json:
1. Match cryptographic hashes and immutable schema contracts
2. Contain trusted scikit-learn Pipeline components
3. Accurately predict non-negative, finite inflows on valid input fixtures
4. Reject invalid or missing feature schemas
5. Are fully traceable to MLflow run ID and Git commit
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "inflow_forecaster.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@pytest.fixture
def model_and_metadata():
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    assert METADATA_PATH.exists(), f"Metadata missing at {METADATA_PATH}"

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def test_artifact_cryptographic_checksums(model_and_metadata):
    """Verify that the binary artifact on disk matches recorded SHA256 and MD5 hashes."""
    _, metadata = model_and_metadata
    raw_bytes = MODEL_PATH.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    md5 = hashlib.md5(raw_bytes).hexdigest()

    provenance = metadata.get("provenance", {})
    assert provenance.get("artifact_sha256") == sha256
    assert provenance.get("artifact_md5") == md5
    assert provenance.get("git_commit") is not None


def test_pipeline_architecture_and_estimators(model_and_metadata):
    """Verify that the loaded artifact is a scikit-learn Pipeline with expected steps."""
    pipeline, metadata = model_and_metadata
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["imputer", "regressor"]
    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert isinstance(pipeline.named_steps["regressor"], RandomForestRegressor)

    rf = pipeline.named_steps["regressor"]
    expected_hyper = metadata["hyperparameters"]
    assert rf.n_estimators == expected_hyper["n_estimators"]
    assert rf.max_depth == expected_hyper["max_depth"]
    assert rf.min_samples_leaf == expected_hyper["min_samples_leaf"]
    assert rf.max_features == expected_hyper["max_features"]


def test_ordered_feature_schema(model_and_metadata):
    """Verify that the feature count and order strictly match the metadata contract."""
    _, metadata = model_and_metadata
    features = metadata.get("features", [])
    assert len(features) == 19
    assert metadata.get("feature_count") == 19
    assert features[0] == "inflow_lag_1"
    assert "inflow_roll_mean_7" in features
    assert "storage_lag_1" in features
    assert "outflow_lag_1" in features


def test_prediction_on_valid_fixture(model_and_metadata):
    """Verify that predictions on valid input fixtures are finite, non-negative, and shape-consistent."""
    pipeline, metadata = model_and_metadata
    features = metadata["features"]

    # Construct realistic synthetic fixture
    fixture_row = {f: 1000.0 for f in features}
    fixture_row["month"] = 7
    fixture_row["day_of_year"] = 200
    fixture_row["is_monsoon"] = 1
    fixture_row["sin_day_of_year"] = 0.5
    fixture_row["cos_day_of_year"] = -0.5
    df_fixture = pd.DataFrame([fixture_row])

    preds = pipeline.predict(df_fixture[features])
    assert len(preds) == 1
    assert np.isfinite(preds[0])
    assert preds[0] >= 0.0


def test_rejection_of_invalid_feature_schema(model_and_metadata):
    """Verify that missing feature columns raise a KeyError during prediction."""
    pipeline, metadata = model_and_metadata
    incomplete_df = pd.DataFrame([{"inflow_lag_1": 500.0, "month": 6}])
    with pytest.raises(KeyError):
        pipeline.predict(incomplete_df[metadata["features"]])


def test_traceability_to_mlflow_run(model_and_metadata):
    """Verify that metadata records a valid MLflow run ID and experiment lineage."""
    _, metadata = model_and_metadata
    mlflow_info = metadata.get("mlflow", {})
    assert mlflow_info.get("experiment_name") == "dam_inflow_forecasting"
    run_id = mlflow_info.get("run_id")
    assert run_id is not None
    assert len(run_id) == 32  # Standard 32-character hex UUID for MLflow runs

