"""Model training and evaluation pipeline with MLflow experiment tracking.

Implements chronological train/validation/test splits, computes baseline persistence
and moving average forecasts, trains a RandomForest regressor, logs metrics and
parameters to MLflow, and serializes production model artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Prevent Loky warning on Windows subprocess
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

PROCESSED_DATA_PATH = Path("data/processed/krs_dam_processed.csv")
MODEL_OUTPUT_DIR = Path("models")
MODEL_ARTIFACT_PATH = MODEL_OUTPUT_DIR / "inflow_forecaster.joblib"
METADATA_PATH = MODEL_OUTPUT_DIR / "model_metadata.json"
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "dam_inflow_forecasting"

FEATURE_COLUMNS = [
    "inflow_lag_1",
    "inflow_lag_2",
    "inflow_lag_3",
    "inflow_lag_7",
    "inflow_lag_14",
    "inflow_roll_mean_7",
    "inflow_roll_std_7",
    "inflow_roll_mean_14",
    "inflow_roll_mean_30",
    "storage_lag_1",
    "storage_lag_7",
    "outflow_lag_1",
    "outflow_lag_7",
    "month",
    "day_of_year",
    "is_monsoon",
]
TARGET_COLUMN = "target_inflow_next_day_cusecs"

# Chronological split boundaries
VAL_START_DATE = "2023-01-01"
TEST_START_DATE = "2025-01-01"


def load_and_split_data(
    processed_path: Path = PROCESSED_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data and perform chronological train/val/test split."""
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path.resolve()}")

    df = pd.read_csv(processed_path)
    # Ensure rows with missing target or primary lag are dropped for training/evaluation
    valid = df.dropna(subset=[TARGET_COLUMN, "inflow_lag_1"]).sort_values("calendar_date").reset_index(drop=True)

    train_df = valid[valid["calendar_date"] < VAL_START_DATE].copy()
    val_df = valid[
        (valid["calendar_date"] >= VAL_START_DATE) & (valid["calendar_date"] < TEST_START_DATE)
    ].copy()
    test_df = valid[valid["calendar_date"] >= TEST_START_DATE].copy()

    return train_df, val_df, test_df


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def train_and_evaluate(
    processed_path: Path = PROCESSED_DATA_PATH,
) -> Dict[str, Any]:
    """Run full chronological model training, evaluation, and MLflow logging."""
    train_df, val_df, test_df = load_and_split_data(processed_path)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print(f"Dataset split chronologically:")
    print(f"  Train: {len(X_train):,} samples ({train_df['calendar_date'].min()} to {train_df['calendar_date'].max()})")
    print(f"  Val:   {len(X_val):,} samples ({val_df['calendar_date'].min()} to {val_df['calendar_date'].max()})")
    print(f"  Test:  {len(X_test):,} samples ({test_df['calendar_date'].min()} to {test_df['calendar_date'].max()})")

    # 1. Baseline: Persistence (forecast equals latest observed inflow_lag_1)
    persistence_pred = test_df["inflow_lag_1"].to_numpy()
    persistence_metrics = evaluate_predictions(y_test, persistence_pred)

    # 2. Baseline: 7-day Moving Average
    roll7_pred = test_df["inflow_roll_mean_7"].fillna(test_df["inflow_lag_1"]).to_numpy()
    roll7_metrics = evaluate_predictions(y_test, roll7_pred)

    print("\nBaseline Results on Held-out Test Set:")
    print(f"  Persistence Baseline -> MAE: {persistence_metrics['mae']}, RMSE: {persistence_metrics['rmse']}, R2: {persistence_metrics['r2']}")
    print(f"  7-Day Moving Avg    -> MAE: {roll7_metrics['mae']}, RMSE: {roll7_metrics['rmse']}, R2: {roll7_metrics['r2']}")

    # 3. Machine Learning Model: Pipeline with median imputation and RandomForest
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(**model_params)),
        ]
    )

    print("\nTraining RandomForestRegressor pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions
    val_pred = pipeline.predict(X_val)
    val_metrics = evaluate_predictions(y_val, val_pred)

    test_pred = pipeline.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred)

    print("\nRandomForest Results on Held-out Test Set:")
    print(f"  Test MAE:  {test_metrics['mae']} cusecs")
    print(f"  Test RMSE: {test_metrics['rmse']} cusecs")
    print(f"  Test R2:   {test_metrics['r2']}")

    rmse_improvement = (persistence_metrics["rmse"] - test_metrics["rmse"]) / persistence_metrics["rmse"] * 100
    print(f"  RMSE Improvement over Persistence Baseline: {rmse_improvement:.2f}%")

    # 4. MLflow Logging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest_chronological") as run:
        # Parameters
        mlflow.log_params(model_params)
        mlflow.log_param("val_start_date", VAL_START_DATE)
        mlflow.log_param("test_start_date", TEST_START_DATE)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("num_features", len(FEATURE_COLUMNS))

        # Metrics
        mlflow.log_metric("baseline_persistence_mae", persistence_metrics["mae"])
        mlflow.log_metric("baseline_persistence_rmse", persistence_metrics["rmse"])
        mlflow.log_metric("baseline_persistence_r2", persistence_metrics["r2"])
        mlflow.log_metric("baseline_roll7_mae", roll7_metrics["mae"])
        mlflow.log_metric("baseline_roll7_rmse", roll7_metrics["rmse"])

        mlflow.log_metric("val_mae", val_metrics["mae"])
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("val_r2", val_metrics["r2"])

        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_r2", test_metrics["r2"])
        mlflow.log_metric("rmse_improvement_pct", rmse_improvement)

        # Log scikit-learn pipeline model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id
        print(f"\nMLflow run logged successfully with run_id: {run_id}")

    # 5. Serialize model artifact and metadata
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)
    print(f"Model artifact saved to: {MODEL_ARTIFACT_PATH}")

    metadata = {
        "model_type": "RandomForestRegressor",
        "pipeline_steps": ["SimpleImputer(median)", "RandomForestRegressor"],
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "split": {
            "train_range": [str(train_df["calendar_date"].min()), str(train_df["calendar_date"].max())],
            "val_range": [str(val_df["calendar_date"].min()), str(val_df["calendar_date"].max())],
            "test_range": [str(test_df["calendar_date"].min()), str(test_df["calendar_date"].max())],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "metrics": {
            "baseline_persistence": persistence_metrics,
            "baseline_roll7": roll7_metrics,
            "random_forest_val": val_metrics,
            "random_forest_test": test_metrics,
            "rmse_improvement_pct": round(rmse_improvement, 2),
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model metadata saved to: {METADATA_PATH}")

    return metadata


def main() -> None:
    """CLI entrypoint for train module."""
    train_and_evaluate()


if __name__ == "__main__":
    main()
