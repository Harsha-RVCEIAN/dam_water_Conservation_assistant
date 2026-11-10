"""Provision MLflow FileStore and Model Registry for CI and Clean Checkouts.

Populates the local MLflow FileStore from authoritative tracked repository
artifacts (models/inflow_forecaster.joblib and models/model_metadata.json).
Enables headless CI runners and fresh developer clones to execute model-registry
and FastAPI serving tests without retraining or copying developer-local mlruns/.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn

from src.models.model_registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    get_registry_client,
    register_production_candidate,
)
from src.models.train import (
    METADATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACT_PATH,
)

logger = logging.getLogger("krs_dam.ci_provisioner")

EXPERIMENT_ID_DEFAULT = "844396270743384489"


def ensure_ci_registry_provisioned(
    tracking_uri: Optional[str] = None,
    joblib_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = METADATA_PATH,
) -> bool:
    """Ensure the registered model Version 1 is available in the MLflow tracking store.

    If the registered model version 1 already exists and is loadable, this function
    is a no-op and returns False.

    If missing (e.g. on a clean GitHub Actions runner), it provisions the required
    run and registers the model from tracked repository artifacts.

    Returns:
        bool: True if provisioning was performed, False if already present.
    """
    effective_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

    # 1. Check if model version 1 already exists
    try:
        client = get_registry_client(effective_uri)
        mv = client.get_model_version(REGISTERED_MODEL_NAME, 1)
        if mv and str(mv.version) == "1":
            # Verify the underlying run artifact exists
            mlflow.set_tracking_uri(effective_uri)
            mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/1")
            return False
    except Exception:
        # Needs provisioning
        pass

    logger.info("Provisioning MLflow tracking store at %s for CI...", effective_uri)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    if not joblib_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {joblib_path}")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    exp_id = EXPERIMENT_ID_DEFAULT
    run_id = meta.get("mlflow", {}).get("run_id", "157ee4e92e284c82a63f385d6c4c9469")
    exp_name = meta.get("mlflow", {}).get("experiment_name", "dam_inflow_forecasting")

    # Resolve local disk directory from file URI
    clean_uri = effective_uri
    if clean_uri.startswith("file:///"):
        store_dir = Path(clean_uri[7:]).resolve()
    elif clean_uri.startswith("file:/"):
        store_dir = Path(clean_uri[5:]).resolve()
    elif clean_uri.startswith("file:"):
        store_dir = Path(clean_uri[5:]).resolve()
    else:
        store_dir = Path(clean_uri).resolve()

    store_uri_clean = store_dir.as_uri()

    # 2. Create experiment and run directory structure
    exp_dir = store_dir / exp_id
    run_dir = exp_dir / run_id
    artifacts_dir = run_dir / "artifacts" / "model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 3. Write experiment meta.yaml
    exp_meta = (
        f"artifact_location: {store_uri_clean}/{exp_id}\n"
        "creation_time: 1792605264068\n"
        f"experiment_id: '{exp_id}'\n"
        "last_update_time: 1792605264068\n"
        "lifecycle_stage: active\n"
        f"name: {exp_name}\n"
    )
    (exp_dir / "meta.yaml").write_text(exp_meta, encoding="utf-8")

    # 4. Write run meta.yaml
    run_meta = (
        f"artifact_uri: {store_uri_clean}/{exp_id}/{run_id}/artifacts\n"
        "end_time: 1793901868181\n"
        "entry_point_name: ''\n"
        f"experiment_id: '{exp_id}'\n"
        "lifecycle_stage: active\n"
        f"run_id: {run_id}\n"
        "run_name: tuned_random_forest_chronological\n"
        f"run_uuid: {run_id}\n"
        "start_time: 1793901851129\n"
        "status: 3\n"
        "tags: []\n"
        "user_id: ci_runner\n"
    )
    (run_dir / "meta.yaml").write_text(run_meta, encoding="utf-8")

    # 5. Populate model artifact from tracked joblib pipeline
    model_pipeline = joblib.load(joblib_path)
    mlflow.sklearn.save_model(model_pipeline, path=str(artifacts_dir))

    # 6. Register model version in registry
    register_production_candidate(
        model_name=REGISTERED_MODEL_NAME,
        tracking_uri=effective_uri,
        training_run_id=run_id,
        joblib_path=joblib_path,
        metadata_path=metadata_path,
        alias=CANDIDATE_ALIAS,
    )

    logger.info("Successfully provisioned %s v1 in %s", REGISTERED_MODEL_NAME, effective_uri)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provisioned = ensure_ci_registry_provisioned()
    print(f"MLflow store status: {'Provisioned' if provisioned else 'Already present'}")

