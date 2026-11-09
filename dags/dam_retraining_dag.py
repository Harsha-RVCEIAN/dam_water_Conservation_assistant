"""Apache Airflow DAG: K.R.Sagara Dam Hydrology Monthly Retraining Pipeline.

Orchestrates monthly automated ingestion, validation, preprocessing, model retraining,
quality gating, and MLflow Model Registry promotion without duplicating business logic.
All tasks invoke modular, validated functions directly from the 'src' package.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path so Airflow workers find src modules cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional Airflow imports with safe fallback for local non-Airflow test environments
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def airflow_task_validate() -> str:
    """Task 1: Execute raw schema validation and anomaly audit."""
    from src.data.validate import run_validation
    report = run_validation()
    return f"Validation passed: {report.total_rows} rows checked. Anomalies: {len(report.flagged_anomalies)}"


def airflow_task_preprocess() -> str:
    """Task 2: Execute leakage-free feature engineering and lag generation."""
    from src.data.preprocess import run_preprocessing_pipeline
    df = run_preprocessing_pipeline()
    return f"Preprocessing completed: {len(df)} rows with {len(df.columns)} features."


def airflow_task_train() -> str:
    """Task 3: Execute chronological model training and MLflow experiment tracking."""
    from src.models.train import train_and_evaluate
    metadata = train_and_evaluate()
    run_id = metadata.get("mlflow", {}).get("run_id")
    return f"Training completed. MLflow run_id: {run_id}"


def airflow_task_quality_gate() -> str:
    """Task 4: Run pre-deployment quality and risk gate against baseline thresholds."""
    from src.models.evaluate_gate import run_model_quality_gate
    gate_res = run_model_quality_gate()
    if not gate_res.passed:
        raise ValueError(f"Quality gate REJECTED model candidate: {gate_res.checks_failed}")
    return f"Quality gate PASSED with {len(gate_res.checks_passed)} verified checks."


def airflow_task_register_model() -> str:
    """Task 5: Register model into MLflow Model Registry and promote to Production."""
    from src.models.registry import register_and_promote_model
    reg_res = register_and_promote_model()
    return f"Model registered as '{reg_res['registered_model_name']}' version {reg_res['version']} (Stage: {reg_res['stage']})."


default_args = {
    "owner": "krs_dam_mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id="krs_dam_inflow_retraining_pipeline",
        default_args=default_args,
        description="Automated monthly validation, feature preprocessing, model retraining, and registry promotion",
        schedule_interval="@monthly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "hydrology", "inflow_forecasting", "cauvery_basin"],
    )

    with dag:
        task_validate = PythonOperator(
            task_id="1_validate_hydrology_data",
            python_callable=airflow_task_validate,
        )

        task_preprocess = PythonOperator(
            task_id="2_preprocess_lags_and_volumes",
            python_callable=airflow_task_preprocess,
        )

        task_train = PythonOperator(
            task_id="3_train_chronological_models",
            python_callable=airflow_task_train,
        )

        task_gate = PythonOperator(
            task_id="4_verify_model_quality_gate",
            python_callable=airflow_task_quality_gate,
        )

        task_register = PythonOperator(
            task_id="5_promote_to_model_registry",
            python_callable=airflow_task_register_model,
        )

        # Sequential dependency flow with quality gate before registry promotion
        task_validate >> task_preprocess >> task_train >> task_gate >> task_register
else:
    dag = None
