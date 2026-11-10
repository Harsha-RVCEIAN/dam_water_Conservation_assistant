"""MLflow Model Registry integration for the KRS Dam Water Conservation Assistant.

This module registers the validated 27-feature Tuned Random Forest forecasting model
with MLflow Model Registry using an SQLite tracking backend (sqlite:///mlflow.db),
tags it with the full 27-feature signature, and promotes it to Production / 'champion'.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence matplotlib and loky multiprocessing overhead in sandboxed/service environments
(PROJECT_ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.models import train_rainfall

# SQLite tracking database ensures persistent experiment tracking and full Model Registry support
DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
EXPERIMENT_NAME = "dam_inflow_rainfall_enhanced_registry"
REGISTERED_MODEL_NAME = "krs_dam_rainfall_inflow_forecaster"
MODEL_PATH = PROJECT_ROOT / "models/rainfall_enhanced/rainfall_tuned_random_forest_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models/rainfall_enhanced/rainfall_tuned_random_forest_v1.json"
REPORT_PATH = PROJECT_ROOT / "reports/quality_gate_report.json"


def configure_mlflow() -> MlflowClient:
    """Set tracking URI to SQLite backend and return an MlflowClient."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def register_production_model() -> dict[str, Any]:
    """Register the validated 27-feature Tuned Random Forest model in the Model Registry."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Quality gate report not found: {REPORT_PATH}. Run evaluate_gate.py first."
        )

    client = configure_mlflow()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    gate_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    if gate_report.get("status") != "PASSED":
        raise ValueError(
            f"Cannot register model: Quality gate status is '{gate_report.get('status')}'. "
            "Model must pass all quality gates before promotion."
        )

    # Load model and sample data for signature
    model = joblib.load(MODEL_PATH)
    feature_names = list(train_rainfall.LEARNED_FEATURE_COLUMNS)
    sample_input = pd.DataFrame([{f: 0.0 for f in feature_names}])
    sample_output = np.array([1000.0])
    signature = infer_signature(sample_input, sample_output)

    # Set or create experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "KRS Dam Water Conservation Assistant",
                "phase": "Phase 2 - Productionization & Deployment",
                "framework": "scikit-learn Pipeline (SimpleImputer + RandomForestRegressor)",
                "features": "27 (19 dam + 8 catchment rainfall)",
            },
        )
    else:
        experiment_id = experiment.experiment_id

    # Create run and log model
    with mlflow.start_run(experiment_id=experiment_id, run_name="register_tuned_rf_v1") as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("model_name", "Tuned Random Forest")
        mlflow.log_param("pipeline", "SimpleImputer(median) + RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_leaf", 4)
        mlflow.log_param("max_features", 0.7)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("dam_feature_count", 19)
        mlflow.log_param("rainfall_feature_count", 8)

        # Log metrics from quality gate report
        metrics = gate_report["metrics"]
        mlflow.log_metric("rmse_cfs", metrics["rmse_cfs"])
        mlflow.log_metric("r2_score", metrics["r2_score"])
        mlflow.log_metric("mae_cfs", metrics["mae_cfs"])
        mlflow.log_metric("conventional_wape_pct", metrics["conventional_wape_pct"])
        mlflow.log_metric("harshad_ref_wape_pct", metrics["harshad_reference_wape_pct"])
        mlflow.log_metric("rmse_improvement_pct", metrics["rmse_improvement_pct"])

        # Log tags
        mlflow.set_tag("quality_gate_status", gate_report["status"])
        mlflow.set_tag("evaluated_date_range", gate_report["test_date_range"])
        mlflow.set_tag("pipeline_architecture", "rainfall_enhanced_v2")

        # Log sklearn model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            serialization_format="pickle",
        )

        # Log quality report and metadata as artifacts
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="quality_gate")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="metadata")

    # Transition the latest version to Production and set 'champion' alias
    latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
    if not latest_versions:
        raise RuntimeError(f"No versions found for registered model '{REGISTERED_MODEL_NAME}'.")

    latest_version = latest_versions[-1].version
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=latest_version,
    )

    print(f"Model successfully registered in MLflow Model Registry:")
    print(f"  Registered Name : {REGISTERED_MODEL_NAME}")
    print(f"  Version         : {latest_version}")
    print(f"  Stage           : Production")
    print(f"  Alias           : champion")
    print(f"  Run ID          : {run_id}")
    print(f"  Tracking Backend: {TRACKING_URI}")

    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": latest_version,
        "stage": "Production",
        "alias": "champion",
        "run_id": run_id,
        "tracking_uri": TRACKING_URI,
    }


def load_production_model() -> tuple[Any, dict[str, Any]]:
    """Load the champion model pipeline from verified artifact with registry metadata."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        # Use single thread in production serving to eliminate multiprocessing/loky overhead
        if hasattr(model, "named_steps") and "regressor" in model.named_steps:
            model.named_steps["regressor"].n_jobs = 1
        return model, {
            "source": "production_champion_artifact",
            "registered_model_name": REGISTERED_MODEL_NAME,
            "stage": "Production",
            "alias": "champion",
            "metadata": metadata,
        }

    # Fallback to MLflow Model Registry URI
    configure_mlflow()
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    if hasattr(model, "named_steps") and "regressor" in model.named_steps:
        model.named_steps["regressor"].n_jobs = 1
    return model, {"source": "mlflow_registry", "uri": model_uri, "metadata": metadata}


def main() -> None:
    print("=" * 70)
    print("KRS Dam Water Conservation Assistant: MLflow Model Registry Promotion")
    print("=" * 70)
    res = register_production_model()
    print("=" * 70)


if __name__ == "__main__":
    main()
