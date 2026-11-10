"""Tests to validate Airflow DAG structure and task callables without Airflow installed."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dags.dam_retraining_dag import (
    dag,
    task_validate_source_data,
    task_merge_dam_rainfall,
    task_train_rainfall_models,
    task_evaluate_quality_gate,
    task_promote_model_registry,
)


def test_dag_structure_and_metadata():
    """Verify DAG ID, schedule, and task count."""
    assert dag.dag_id == "krs_dam_retraining_pipeline"
    assert dag.schedule == "0 2 * * 1"
    assert dag.catchup is False


def test_dag_task_sequence():
    """Verify sequential operator dependency chain."""
    from dags.dam_retraining_dag import t1_validate, t2_merge, t3_train, t4_gate, t5_promote
    assert t2_merge in t1_validate.downstream_list
    assert t3_train in t2_merge.downstream_list
    assert t4_gate in t3_train.downstream_list
    assert t5_promote in t4_gate.downstream_list


def test_task_validate_source_callable():
    """Verify task_validate_source_data runs and finds source datasets."""
    res = task_validate_source_data()
    assert res["status"] == "VALIDATED"
    assert res["row_count"] > 10000


def test_task_quality_gate_callable():
    """Verify task_evaluate_quality_gate runs and passes quality gates."""
    res = task_evaluate_quality_gate()
    assert res["status"] == "PASSED"
    assert res["metrics"]["rmse_improvement_pct"] >= 20.0
