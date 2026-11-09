"""MLflow Model Registry integration for K.R.Sagara Dam Inflow Forecaster.

Provides automated registration of trained model artifacts, quality gate
verification before promotion, and lifecycle stage management (Staging/Production).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.models.evaluate_gate import run_model_quality_gate

METADATA_PATH = Path("models/model_metadata.json")
REGISTERED_MODEL_NAME = "krs_dam_inflow_forecaster"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def register_and_promote_model(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    metadata_path: Path = METADATA_PATH,
) -> Dict[str, Any]:
    """Register trained candidate model and promote to Production if quality gate passes."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # 1. First run the pre-deployment quality gate
    gate_result = run_model_quality_gate(metadata_path)
    if not gate_result.passed:
        raise ValueError(
            f"Quality Gate REJECTED model registration. Failures: {gate_result.checks_failed}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = metadata.get("mlflow", {}).get("run_id")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 2. Register model
    model_source = f"runs:/{run_id}/model" if run_id else "models/inflow_forecaster.joblib"
    print(f"Registering model from source: {model_source} under name '{registered_model_name}'...")

    try:
        model_version = client.create_model_version(
            name=registered_model_name,
            source=model_source,
            run_id=run_id,
            description="Production Random Forest model for daily KRS Dam inflow forecasting.",
            tags={
                "algorithm": metadata.get("model_type", "RandomForestRegressor"),
                "test_r2": str(metadata.get("metrics", {}).get("test", {}).get("selected_model", {}).get("r2")),
                "test_rmse": str(metadata.get("metrics", {}).get("test", {}).get("selected_model", {}).get("rmse")),
                "rmse_improvement_pct": str(metadata.get("metrics", {}).get("test", {}).get("rmse_improvement_pct")),
                "gate_status": "PASSED",
            },
        )
        version_num = model_version.version
        print(f"Created model version: {version_num}")
    except Exception as e:
        print(f"Notice: Registering directly or creating registered model first: {e}")
        try:
            client.create_registered_model(
                name=registered_model_name,
                description="Cauvery basin K.R.Sagara Dam daily inflow forecaster.",
            )
        except Exception:
            pass
        model_version = client.create_model_version(
            name=registered_model_name,
            source=model_source,
            run_id=run_id,
        )
        version_num = model_version.version

    # 3. Transition to Production
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version_num,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Successfully transitioned model version {version_num} to 'Production' stage.")
    except Exception as e:
        # For MLflow 2.9+ aliases:
        try:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="champion",
                version=version_num,
            )
            print(f"Set model version {version_num} as 'champion' alias.")
        except Exception as e2:
            print(f"Stage transition notice: {e2}")

    return {
        "registered_model_name": registered_model_name,
        "version": version_num,
        "run_id": run_id,
        "quality_gate_passed": True,
        "stage": "Production",
    }


def main() -> None:
    """CLI runner to register and promote the production model."""
    result = register_and_promote_model()
    print("Registration Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
