"""Unit tests for Apache Airflow DAG orchestration.

Validates DAG structure, task graph dependencies, execution commands,
and integration between Airflow orchestration and MLflow experiment tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dags.krs_inflow_pipeline_dag as dag_module


def test_dag_metadata_and_schedule():
    """Verify DAG ID, tags, default arguments, and schedule interval."""
    dag = dag_module.dag
    assert dag.dag_id == "krs_dam_inflow_mlops_pipeline"
    assert dag.schedule_interval == "@daily"
    assert "krs_dam" in dag.tags
    assert "mlops" in dag.tags
    assert "inflow_forecasting" in dag.tags
    assert dag.default_args.get("owner") == "dam_mlops_team"
    assert dag.default_args.get("retries") == 1


def test_dag_tasks_declared():
    """Verify all 6 pipeline stages are declared in the DAG."""
    dag = dag_module.dag
    task_ids = {t.task_id for t in dag.tasks}
    expected_tasks = {
        "validate_raw_dataset",
        "validate_unit_conversions",
        "preprocess_features",
        "compute_water_balance",
        "train_models_track_mlflow",
        "verify_model_freshness",
    }
    assert expected_tasks.issubset(task_ids), f"Missing tasks: {expected_tasks - task_ids}"


def test_dag_task_dependencies():
    """Verify task execution order and upstream/downstream dependencies."""
    dag = dag_module.dag
    task_map = {t.task_id: t for t in dag.tasks}

    val_raw = task_map["validate_raw_dataset"]
    val_conv = task_map["validate_unit_conversions"]
    prep = task_map["preprocess_features"]
    wb = task_map["compute_water_balance"]
    train = task_map["train_models_track_mlflow"]
    verify = task_map["verify_model_freshness"]

    # Raw validation gates conversions and preprocessing
    assert val_conv in val_raw.downstream_list or val_raw in val_conv.upstream_list
    assert prep in val_raw.downstream_list or val_raw in prep.upstream_list

    # Preprocessing gates water balance and model training
    assert wb in prep.downstream_list or prep in wb.upstream_list
    assert train in prep.downstream_list or prep in train.upstream_list

    # Training gates verification
    assert verify in train.downstream_list or train in verify.upstream_list


def test_dag_bash_commands_target_correct_modules():
    """Verify bash commands call the expected Python modules."""
    dag = dag_module.dag
    task_map = {t.task_id: t for t in dag.tasks}

    assert "src.data.validate" in task_map["validate_raw_dataset"].bash_command
    assert "src.utils.validate_conversions" in task_map["validate_unit_conversions"].bash_command
    assert "src.data.preprocess" in task_map["preprocess_features"].bash_command
    assert "src.models.water_balance" in task_map["compute_water_balance"].bash_command
    assert "src.models.train" in task_map["train_models_track_mlflow"].bash_command


def test_verify_model_artifact_callable():
    """Verify the Python callable successfully loads and checks production model freshness."""
    result = dag_module.verify_model_artifact_callable()
    assert isinstance(result, dict)
    assert "model_type" in result
    assert "forecast_inflow_cusecs" in result
    assert "forecast_target_date" in result
    assert result["forecast_inflow_cusecs"] >= 0.0

