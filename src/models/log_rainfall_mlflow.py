"""Log the completed rainfall experiment to a separate local MLflow store.

The store is ``mlruns_rainfall_enhanced/`` at the project root, separate from
the original trainer's ``mlruns/`` URI. This script reads saved results and
artifacts only; it does not retrain models or modify model files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "dam_inflow_rainfall_enhanced"
TRACKING_PATH = PROJECT_ROOT / "mlruns_rainfall_enhanced"
TRACKING_URI = TRACKING_PATH.as_uri()
MODEL_DIR = PROJECT_ROOT / "models/rainfall_enhanced"
METRICS_PATH = PROJECT_ROOT / "reports/rainfall_enhanced_model_metrics.csv"
DATASET_PATH = PROJECT_ROOT / "data/interim/krs_dam_rainfall_merged.csv"
TRAINING_SCRIPT = PROJECT_ROOT / "src/models/train_rainfall.py"
FEATURE_SCRIPT = PROJECT_ROOT / "src/data/rainfall_features.py"
BASELINE_TRAINING_SCRIPT = PROJECT_ROOT / "src/models/train.py"

RUNS = {
    "Rainfall_Persistence": {
        "model_name": "Persistence",
        "artifact_filename": "rainfall_persistence_v1.joblib",
        "metadata_filename": "rainfall_persistence_v1.json",
    },
    "Rainfall_7Day_Moving_Average": {
        "model_name": "7-Day Moving Average",
        "artifact_filename": "rainfall_7day_moving_average_v1.joblib",
        "metadata_filename": "rainfall_7day_moving_average_v1.json",
    },
    "Rainfall_Seasonal_Naive": {
        "model_name": "Seasonal Naive",
        "artifact_filename": "rainfall_seasonal_naive_v1.joblib",
        "metadata_filename": "rainfall_seasonal_naive_v1.json",
    },
    "Rainfall_Ridge_Regression": {
        "model_name": "Ridge Regression",
        "artifact_filename": "rainfall_ridge_regression_v1.joblib",
        "metadata_filename": "rainfall_ridge_regression_v1.json",
    },
    "Rainfall_Random_Forest": {
        "model_name": "Random Forest",
        "artifact_filename": "rainfall_random_forest_v1.joblib",
        "metadata_filename": "rainfall_random_forest_v1.json",
    },
    "Rainfall_Tuned_Random_Forest": {
        "model_name": "Tuned Random Forest",
        "artifact_filename": "rainfall_tuned_random_forest_v1.joblib",
        "metadata_filename": "rainfall_tuned_random_forest_v1.json",
    },
    "Rainfall_HistGradientBoosting": {
        "model_name": "HistGradientBoosting",
        "artifact_filename": "rainfall_histgradientboosting_v1.joblib",
        "metadata_filename": "rainfall_histgradientboosting_v1.json",
    },
}

SOURCE_TAG = "source_experiment"
SOURCE_TAG_VALUE = "completed_rainfall_experiment_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_saved_results() -> list[dict[str, Any]]:
    required_paths = [
        METRICS_PATH,
        DATASET_PATH,
        TRAINING_SCRIPT,
        FEATURE_SCRIPT,
        BASELINE_TRAINING_SCRIPT,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required logging inputs are missing: {missing}")

    metrics = pd.read_csv(METRICS_PATH)
    required_columns = {
        "model_name",
        "model_type",
        "algorithm",
        "split",
        "sample_count",
        "mae",
        "rmse",
        "r2",
        "feature_count",
        "rainfall_features_used",
    }
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"Metrics report is missing columns: {sorted(missing_columns)}")
    if len(metrics) != 14:
        raise ValueError(f"Expected 14 saved metric rows; found {len(metrics)}.")

    dataset_hash = _sha256(DATASET_PATH)
    code_hashes = {
        "rainfall_training_script_sha256": _sha256(TRAINING_SCRIPT),
        "rainfall_feature_script_sha256": _sha256(FEATURE_SCRIPT),
        "baseline_training_script_sha256": _sha256(BASELINE_TRAINING_SCRIPT),
    }
    revision = _git_revision()
    entries = []

    for run_name, spec in RUNS.items():
        artifact_path = MODEL_DIR / spec["artifact_filename"]
        metadata_path = MODEL_DIR / spec["metadata_filename"]
        if not artifact_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Saved artifact or metadata missing for {run_name}: "
                f"{artifact_path}, {metadata_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != spec["model_name"]:
            raise ValueError(f"Model metadata name mismatch for {run_name}.")
        if metadata.get("dataset_sha256") != dataset_hash:
            raise ValueError(
                f"Dataset hash differs from saved metadata for {run_name}; "
                "refusing to log mismatched results."
            )
        if metadata.get("target") != "target_inflow_next_day_cusecs":
            raise ValueError(f"Unexpected target in metadata for {run_name}.")

        model_metrics = metrics.loc[metrics["model_name"].eq(spec["model_name"])]
        if set(model_metrics["split"]) != {"validation", "test"} or len(model_metrics) != 2:
            raise ValueError(f"Expected validation and test metrics for {run_name}.")

        for split in ("validation", "test"):
            row = model_metrics.loc[model_metrics["split"].eq(split)].iloc[0]
            saved = metadata["metrics"].get(split)
            if saved is None:
                raise ValueError(f"{split} metrics absent from metadata for {run_name}.")
            for key in ("sample_count", "mae", "rmse", "r2"):
                if not _values_match(row[key], saved[key]):
                    raise ValueError(
                        f"{run_name} {split} {key} differs between metrics CSV "
                        "and metadata."
                    )

        expected_model_type = "learned" if metadata["model_type"] == "learned" else "rule-based"
        if not model_metrics["model_type"].eq(expected_model_type).all():
            raise ValueError(f"Model type differs between files for {run_name}.")
        if not model_metrics["feature_count"].eq(metadata["feature_count"]).all():
            raise ValueError(f"Feature count differs between files for {run_name}.")
        rainfall_used = bool(metadata["rainfall_features_used"])
        if not model_metrics["rainfall_features_used"].map(_as_bool).all() == rainfall_used:
            raise ValueError(f"Rainfall usage differs between files for {run_name}.")

        entries.append(
            {
                "run_name": run_name,
                "spec": spec,
                "artifact_path": artifact_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "metrics": model_metrics.set_index("split").to_dict(orient="index"),
                "code_hashes": code_hashes,
                "git_revision": revision,
            }
        )

    return entries


def _values_match(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-8
    except (TypeError, ValueError):
        return str(left) == str(right)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _parameter_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _run_parameters(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry["metadata"]
    params: dict[str, Any] = {
        "model_name": entry["run_name"],
        "algorithm": metadata["algorithm"],
        "model_type": metadata["model_type"],
        "input_feature_names": json.dumps(metadata["input_feature_names"]),
        "feature_count": int(metadata["feature_count"]),
        "target": metadata["target"],
        "dataset_path": metadata["dataset_path"],
        "dataset_sha256": metadata["dataset_sha256"],
        "rainfall_features_used": bool(metadata["rainfall_features_used"]),
        "training_timestamp_utc": metadata["training_timestamp_utc"],
        "source_training_script": "src/models/train_rainfall.py",
        "logging_provenance": "existing completed experiment; no retraining",
    }
    if entry["git_revision"]:
        params["logging_git_revision"] = entry["git_revision"]

    for split_name in ("train", "validation", "test"):
        requested = metadata["requested_date_ranges"][split_name]
        actual = metadata["actual_date_ranges"][split_name]
        params[f"{split_name}_requested_start_date"] = requested["start"]
        params[f"{split_name}_requested_end_date"] = requested["end"]
        params[f"{split_name}_actual_start_date"] = actual["start"]
        params[f"{split_name}_actual_end_date"] = actual["end"]
        params[f"{split_name}_sample_count"] = int(metadata["sample_counts"][split_name])

    for key, value in metadata.get("hyperparameters", {}).items():
        params[f"hyperparameter_{key}"] = _parameter_value(value)
    return params


def _run_metrics(entry: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for split_name in ("validation", "test"):
        saved = entry["metrics"][split_name]
        for metric in ("mae", "rmse", "r2"):
            result[f"{split_name}_{metric}"] = float(saved[metric])
        result[f"{split_name}_sample_count"] = float(saved["sample_count"])
    return result


def _matching_existing_run(
    client: MlflowClient,
    run: Any,
    entry: dict[str, Any],
) -> bool:
    data = run.data
    expected_metrics = _run_metrics(entry)
    expected_params = _run_parameters(entry)
    if data.tags.get(SOURCE_TAG) != SOURCE_TAG_VALUE:
        return False
    if data.params.get("dataset_sha256") != entry["metadata"]["dataset_sha256"]:
        return False
    if any(
        key not in data.metrics or not _values_match(data.metrics[key], value)
        for key, value in expected_metrics.items()
    ):
        return False
    if any(
        key not in data.params or str(data.params[key]) != str(value)
        for key, value in expected_params.items()
    ):
        return False

    artifacts = client.list_artifacts(run.info.run_id)
    top_level = {item.path for item in artifacts}
    return {"model", "metadata", "evaluation", "code"}.issubset(top_level)


def _existing_runs_by_name(
    client: MlflowClient,
    experiment_id: str,
) -> dict[str, Any]:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    )
    matching = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName")
        if name in RUNS:
            if name in matching:
                raise RuntimeError(
                    f"Multiple MLflow runs already use requested name {name!r}; "
                    "inspect them manually before logging."
                )
            matching[name] = run
    return matching


def _log_entry(
    client: MlflowClient,
    experiment_id: str,
    entry: dict[str, Any],
) -> str:
    metadata = entry["metadata"]
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=entry["run_name"],
    ) as active_run:
        mlflow.set_tags(
            {
                SOURCE_TAG: SOURCE_TAG_VALUE,
                "model_type": metadata["model_type"],
                "rainfall_features_used": str(
                    bool(metadata["rainfall_features_used"])
                ).lower(),
                "target": metadata["target"],
                "dataset_sha256": metadata["dataset_sha256"],
                "result_status": "existing_completed_experiment",
            }
        )
        mlflow.log_params(_run_parameters(entry))
        mlflow.log_metrics(_run_metrics(entry))
        mlflow.log_artifact(str(entry["artifact_path"]), artifact_path="model")
        mlflow.log_artifact(str(entry["metadata_path"]), artifact_path="metadata")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
        for script in (
            TRAINING_SCRIPT,
            FEATURE_SCRIPT,
            BASELINE_TRAINING_SCRIPT,
            Path(__file__).resolve(),
        ):
            mlflow.log_artifact(str(script), artifact_path="code")
        for key, value in entry["code_hashes"].items():
            mlflow.set_tag(key, value)
        if entry["git_revision"]:
            mlflow.set_tag("logging_git_revision", entry["git_revision"])
        return active_run.info.run_id


def log_completed_experiment() -> dict[str, Any]:
    """Log the seven existing model results without retraining or overwriting runs."""
    entries = _load_saved_results()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "purpose": "completed rainfall-enhanced model comparison",
                "tracking_location": "project-local separate store",
                "source": "existing saved artifacts and metrics; no retraining",
            },
        )
    else:
        experiment_id = experiment.experiment_id

    existing = _existing_runs_by_name(client, experiment_id)
    for entry in entries:
        run = existing.get(entry["run_name"])
        if run is not None and not _matching_existing_run(client, run, entry):
            raise RuntimeError(
                f"Run {entry['run_name']!r} already exists but does not match the "
                "saved experiment results. No runs were created; inspect it first."
            )

    result_runs = {}
    for entry in entries:
        run = existing.get(entry["run_name"])
        if run is not None:
            result_runs[entry["run_name"]] = {
                "run_id": run.info.run_id,
                "status": "already_logged_and_verified",
            }
        else:
            result_runs[entry["run_name"]] = {
                "run_id": _log_entry(client, experiment_id, entry),
                "status": "logged",
            }

    _validate_logged_runs(client, experiment_id, entries, result_runs)
    return {
        "tracking_uri": TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "runs": result_runs,
    }


def _validate_logged_runs(
    client: MlflowClient,
    experiment_id: str,
    entries: list[dict[str, Any]],
    result_runs: dict[str, dict[str, str]],
) -> None:
    experiment = client.get_experiment(experiment_id)
    if experiment is None or experiment.name != EXPERIMENT_NAME:
        raise AssertionError("The intended MLflow experiment was not found.")

    for entry in entries:
        run_id = result_runs[entry["run_name"]]["run_id"]
        run = client.get_run(run_id)
        if run.data.tags.get("mlflow.runName") != entry["run_name"]:
            raise AssertionError(f"MLflow run name mismatch for {entry['run_name']}.")
        if not _matching_existing_run(client, run, entry):
            raise AssertionError(f"MLflow run validation failed for {entry['run_name']}.")
        expected_metrics = _run_metrics(entry)
        if any(key not in run.data.metrics for key in expected_metrics):
            raise AssertionError(f"Metrics are missing for {entry['run_name']}.")
        for artifact_dir, expected_filename in (
            ("model", entry["artifact_path"].name),
            ("metadata", entry["metadata_path"].name),
            ("evaluation", METRICS_PATH.name),
            ("code", Path(__file__).name),
        ):
            names = {
                item.path.rsplit("/", 1)[-1]
                for item in client.list_artifacts(run_id, artifact_dir)
            }
            if expected_filename not in names:
                raise AssertionError(
                    f"Logged artifact {artifact_dir}/{expected_filename} missing "
                    f"for {entry['run_name']}."
                )


def main() -> None:
    result = log_completed_experiment()
    print(f"Tracking URI: {result['tracking_uri']}")
    print(
        f"Experiment: {result['experiment_name']} "
        f"(ID {result['experiment_id']})"
    )
    for run_name, details in result["runs"].items():
        print(
            f"{run_name}: {details['run_id']} "
            f"({details['status']})"
        )


if __name__ == "__main__":
    main()
