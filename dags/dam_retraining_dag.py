"""Apache Airflow Retraining DAG for KRS Dam Rainfall-Enhanced Forecaster.

This DAG orchestrates the automated retraining, evaluation, and promotion pipeline
using Harshad's actual validated pipeline modules and the Phase 2 quality gate.

Pipeline Task Sequence:
  validate_source_data
          │
          ▼
   merge_dam_rainfall
          │
          ▼
  train_rainfall_models
          │
          ▼
  evaluate_quality_gate
          │
          ▼
  promote_model_registry
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path so Airflow workers can import pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import airflow; provide fallback Mock classes for Windows environments
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

    class DAG:  # type: ignore
        """Mock DAG class allowing syntax and structure validation on Windows without Airflow installed."""
        def __init__(self, dag_id: str, default_args: dict, schedule: str, catchup: bool, **kwargs):
            self.dag_id = dag_id
            self.default_args = default_args
            self.schedule = schedule
            self.catchup = catchup
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class PythonOperator:  # type: ignore
        """Mock PythonOperator allowing operator chaining validation."""
        def __init__(self, task_id: str, python_callable, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self.upstream_list = []
            self.downstream_list = []

        def __rshift__(self, other):
            self.downstream_list.append(other)
            other.upstream_list.append(self)
            return other


# Callable task definitions linking directly to actual pipeline functions

def task_validate_source_data() -> dict[str, Any]:
    """Validate freshness and schema of processed dam hydrology and daily rainfall datasets."""
    from src.models import train_rainfall
    df, dataset_hash = train_rainfall._read_and_validate_dataset()
    print(f"Airflow: Validated source datasets with hash {dataset_hash[:12]} ({len(df)} rows).")
    return {"status": "VALIDATED", "row_count": len(df), "dataset_hash": dataset_hash}


def task_merge_dam_rainfall() -> dict[str, str]:
    """Merge dam hydrological features with daily catchment rainfall."""
    from src.data import merge_dam_rainfall
    merge_dam_rainfall.main()
    print("Airflow: Successfully merged dam and rainfall series.")
    return {"status": "MERGED"}


def task_train_rainfall_models() -> dict[str, int]:
    """Retrain all 7 candidate models and serialize winning Tuned Random Forest."""
    from src.models import train_rainfall
    metrics_df = train_rainfall.train_and_evaluate()
    print(f"Airflow: Training completed for {len(metrics_df)} model-split pairs.")
    return {"status": "TRAINED", "rows_evaluated": len(metrics_df)}


def task_evaluate_quality_gate() -> dict[str, Any]:
    """Execute automated model quality gate checking RMSE improvement, R^2, and WAPE."""
    from src.models import evaluate_gate
    result = evaluate_gate.evaluate_model_quality_gate()
    if result.status != "PASSED":
        raise ValueError(f"Airflow Pipeline Stopped: Quality gate check failed with status '{result.status}'.")
    print(f"Airflow: Quality gate {result.status} with RMSE imp {result.metrics['rmse_improvement_pct']}%.")
    return {"status": result.status, "metrics": result.metrics}


def task_promote_model_registry() -> dict[str, str]:
    """Promote validated model to Production in MLflow Model Registry."""
    from src.models import registry
    promotion_info = registry.register_production_model()
    print(f"Airflow: Model promoted to Production with alias '{promotion_info['alias']}'.")
    return promotion_info


# Default execution arguments
default_args = {
    "owner": "Hadi",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG structure
with DAG(
    dag_id="krs_dam_retraining_pipeline",
    default_args=default_args,
    description="Automated weekly retraining and quality gate pipeline for KRS Dam Forecaster",
    schedule="0 2 * * 1",  # Run weekly every Monday at 02:00 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hydrology", "krs-dam", "mlops", "phase2"],
) as dag:

    t1_validate = PythonOperator(
        task_id="validate_source_data",
        python_callable=task_validate_source_data,
    )

    t2_merge = PythonOperator(
        task_id="merge_dam_rainfall",
        python_callable=task_merge_dam_rainfall,
    )

    t3_train = PythonOperator(
        task_id="train_rainfall_models",
        python_callable=task_train_rainfall_models,
    )

    t4_gate = PythonOperator(
        task_id="evaluate_quality_gate",
        python_callable=task_evaluate_quality_gate,
    )

    t5_promote = PythonOperator(
        task_id="promote_model_registry",
        python_callable=task_promote_model_registry,
    )

    # Establish strictly sequential dependency chain
    t1_validate >> t2_merge >> t3_train >> t4_gate >> t5_promote
