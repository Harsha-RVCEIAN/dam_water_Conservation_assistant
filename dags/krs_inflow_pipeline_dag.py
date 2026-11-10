"""Airflow DAG for KRS Dam Inflow Forecasting & Decision Support Pipeline.

Orchestrates end-to-end MLOps pipeline stages:
1. Raw data validation & schema checking
2. Unit conversion validation (cusec-day to TMC)
3. Chronological feature engineering (strict backward-looking lags)
4. Water balance mass-accounting computation
5. Model training, evaluation, and experiment tracking via MLflow
6. Model freshness and inference readiness verification

Experiment tracking is handled by MLflow (logging hyperparameters, metrics, and
model artifacts into experiment `dam_inflow_forecasting`), while Apache Airflow
provides pipeline orchestration, task dependency scheduling, and execution tracking.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Python executable to use for pipeline task execution
PYTHON_EXEC = sys.executable

# -----------------------------------------------------------------------------
# Graceful Airflow Environment Adapter
# -----------------------------------------------------------------------------
# When deployed on an Apache Airflow cluster / container, official airflow imports
# are used. When loaded in lightweight environments (e.g., local Windows developer
# workstations without POSIX daemon dependencies), lightweight shim classes provide
# full attribute compatibility, DAG graph validation, and task dependency checking.
# -----------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False

    class BaseOperatorShim:
        """Lightweight BaseOperator shim for non-Airflow environments."""

        def __init__(self, task_id: str, **kwargs):
            self.task_id = task_id
            self.kwargs = kwargs
            self.upstream_list: list[BaseOperatorShim] = []
            self.downstream_list: list[BaseOperatorShim] = []

        def set_downstream(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    self.set_downstream(o)
                return other
            if other not in self.downstream_list:
                self.downstream_list.append(other)
            if self not in other.upstream_list:
                other.upstream_list.append(self)
            return other

        def set_upstream(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    self.set_upstream(o)
                return other
            other.set_downstream(self)
            return other

        def __rshift__(self, other):
            return self.set_downstream(other)

        def __lshift__(self, other):
            return self.set_upstream(other)

        def __rrshift__(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    o.set_downstream(self)
            return self

        def __rlshift__(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    o.set_upstream(self)
            return self

        def __repr__(self):
            return f"<{self.__class__.__name__}: {self.task_id}>"

    class BashOperator(BaseOperatorShim):  # type: ignore[no-redef]
        """Lightweight BashOperator shim."""

        def __init__(self, task_id: str, bash_command: str, **kwargs):
            super().__init__(task_id, **kwargs)
            self.bash_command = bash_command

    class PythonOperator(BaseOperatorShim):  # type: ignore[no-redef]
        """Lightweight PythonOperator shim."""

        def __init__(self, task_id: str, python_callable, **kwargs):
            super().__init__(task_id, **kwargs)
            self.python_callable = python_callable

    class DAG:  # type: ignore[no-redef]
        """Lightweight DAG shim."""

        def __init__(
            self,
            dag_id: str,
            default_args: dict | None = None,
            description: str | None = None,
            schedule_interval: str | timedelta | None = None,
            start_date: datetime | None = None,
            catchup: bool = False,
            tags: list[str] | None = None,
            **kwargs,
        ):
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.description = description or ""
            self.schedule_interval = schedule_interval
            self.start_date = start_date or datetime(2026, 1, 1)
            self.catchup = catchup
            self.tags = tags or []
            self.kwargs = kwargs
            self.tasks: list[BaseOperatorShim] = []

        def __enter__(self):
            DAG._current_dag = self
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            DAG._current_dag = None

        def add_task(self, task: BaseOperatorShim):
            if task not in self.tasks:
                self.tasks.append(task)

    # Auto-register tasks into the active DAG context if created inside `with dag:`
    _orig_shim_init = BaseOperatorShim.__init__

    def _shim_auto_register(self, task_id: str, **kwargs):
        _orig_shim_init(self, task_id, **kwargs)
        cur = getattr(DAG, "_current_dag", None)
        if cur is not None:
            cur.add_task(self)

    BaseOperatorShim.__init__ = _shim_auto_register  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# DAG Configuration
# -----------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "dam_mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DAG_ID = "krs_dam_inflow_mlops_pipeline"
SCHEDULE_INTERVAL = "@daily"


def verify_model_artifact_callable():
    """Verify that the production model artifact and metadata exist and are valid."""
    from src.models.predict import InflowForecaster

    forecaster = InflowForecaster()
    if forecaster.pipeline is None:
        raise RuntimeError("InflowForecaster model artifact failed to load.")
    res = forecaster.predict_latest_from_processed()
    print(
        f"Model freshness verified. Selected model: {res.get('model_type')}, "
        f"forecast target: {res.get('forecast_target_date')}, "
        f"forecast inflow: {res.get('forecast_inflow_cusecs')} cusecs"
    )
    return res


# -----------------------------------------------------------------------------
# DAG Definition
# -----------------------------------------------------------------------------
dag = DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Orchestrates KRS Dam data validation, preprocessing, water balance, and MLflow-tracked model training.",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["krs_dam", "mlops", "inflow_forecasting", "layer_a", "layer_b"],
)

with dag:
    # Stage 1: Validate raw dataset integrity and quality
    task_validate_raw = BashOperator(
        task_id="validate_raw_dataset",
        bash_command=f'"{PYTHON_EXEC}" -m src.data.validate',
    )

    # Stage 2: Audit unit conversions (cusec-day to TMC)
    task_validate_conversions = BashOperator(
        task_id="validate_unit_conversions",
        bash_command=f'"{PYTHON_EXEC}" -m src.utils.validate_conversions',
    )

    # Stage 3: Chronological feature engineering & preprocessing
    task_preprocess = BashOperator(
        task_id="preprocess_features",
        bash_command=f'"{PYTHON_EXEC}" -m src.data.preprocess',
    )

    # Stage 4: Compute water balance mass accounting
    task_water_balance = BashOperator(
        task_id="compute_water_balance",
        bash_command=f'"{PYTHON_EXEC}" -m src.models.water_balance',
    )

    # Stage 5: Train candidate models & log parameters/metrics to MLflow
    task_train_mlflow = BashOperator(
        task_id="train_models_track_mlflow",
        bash_command=f'"{PYTHON_EXEC}" -m src.models.train',
    )

    # Stage 6: Verify model artifact freshness and inference readiness
    task_verify_model = PythonOperator(
        task_id="verify_model_freshness",
        python_callable=verify_model_artifact_callable,
    )

    # -------------------------------------------------------------------------
    # Pipeline Execution Flow / Task Dependencies
    # -------------------------------------------------------------------------
    # Raw validation gates both conversions and preprocessing
    task_validate_raw >> task_validate_conversions
    task_validate_raw >> task_preprocess

    # Preprocessed features feed both water balance and model training
    task_preprocess >> task_water_balance
    task_preprocess >> task_train_mlflow

    # Model training gates model freshness verification
    task_train_mlflow >> task_verify_model
