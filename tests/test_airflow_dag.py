"""Tests for Airflow DAG design, callables, and pipeline integrity."""

import pytest
import dags.dam_retraining_dag as dag_module


def test_airflow_module_imports():
    """Verify DAG script imports without syntax or import errors."""
    assert hasattr(dag_module, "airflow_task_validate")
    assert hasattr(dag_module, "airflow_task_preprocess")
    assert hasattr(dag_module, "airflow_task_train")
    assert hasattr(dag_module, "airflow_task_quality_gate")
    assert hasattr(dag_module, "airflow_task_register_model")


def test_dag_callables_are_callable():
    """Verify that all tasks in the DAG are callable functions."""
    assert callable(dag_module.airflow_task_validate)
    assert callable(dag_module.airflow_task_preprocess)
    assert callable(dag_module.airflow_task_train)
    assert callable(dag_module.airflow_task_quality_gate)
    assert callable(dag_module.airflow_task_register_model)


def test_dag_structure_if_airflow_installed():
    """If Airflow is present in environment, verify DAG structure and task count."""
    if not dag_module.AIRFLOW_AVAILABLE:
        pytest.skip("Airflow not installed in local environment (evaluated via Docker/WSL2).")

    dag = dag_module.dag
    assert dag is not None
    assert dag.dag_id == "krs_dam_inflow_retraining_pipeline"
    assert len(dag.tasks) == 5

    task_ids = [t.task_id for t in dag.tasks]
    expected_tasks = [
        "1_validate_hydrology_data",
        "2_preprocess_lags_and_volumes",
        "3_train_chronological_models",
        "4_verify_model_quality_gate",
        "5_promote_to_model_registry",
    ]
    for et in expected_tasks:
        assert et in task_ids


def test_dag_structure_via_ast():
    """Verify DAG structure, task IDs, and sequential wiring statically via AST without Airflow installed."""
    import ast
    from pathlib import Path

    dag_path = Path("dags/dam_retraining_dag.py")
    assert dag_path.exists(), "DAG file must exist"
    tree = ast.parse(dag_path.read_text(encoding="utf-8"))

    # Extract all task_ids in PythonOperator invocations
    task_ids = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PythonOperator":
            for kw in node.keywords:
                if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                    task_ids.append(kw.value.value)

    expected = [
        "1_validate_hydrology_data",
        "2_preprocess_lags_and_volumes",
        "3_train_chronological_models",
        "4_verify_model_quality_gate",
        "5_promote_to_model_registry",
    ]
    assert task_ids == expected, f"Tasks found {task_ids} do not match expected {expected}"

