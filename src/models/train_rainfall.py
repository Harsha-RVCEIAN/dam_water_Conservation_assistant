"""Train and evaluate rainfall-enhanced forecasting experiments separately."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.rainfall_features import (
    RAINFALL_DATA_PATH,
    RAINFALL_FEATURE_COLUMNS,
    build_rainfall_features,
)
from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DATA_PATH = PROJECT_ROOT / "data/interim/krs_dam_rainfall_merged.csv"
PROCESSED_DAM_PATH = PROJECT_ROOT / "data/processed/krs_dam_processed.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models/rainfall_enhanced"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "reports/rainfall_enhanced_model_metrics.csv"

TRAIN_START = "1990-01-01"
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

TARGET_DATE_RANGES = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}

BASELINE_SPECS = {
    "persistence": {
        "name": "Persistence",
        "filename": "rainfall_persistence_v1.joblib",
        "method": "persistence_t_minus_1",
        "features": ["inflow_lag_1"],
    },
    "moving_average": {
        "name": "7-Day Moving Average",
        "filename": "rainfall_7day_moving_average_v1.joblib",
        "method": "inflow_roll_mean_7_fallback_inflow_lag_1",
        "features": ["inflow_roll_mean_7", "inflow_lag_1"],
    },
    "seasonal_naive": {
        "name": "Seasonal Naive",
        "filename": "rainfall_seasonal_naive_v1.joblib",
        "method": "day_of_year_training_target_median_fallback_inflow_lag_1",
        "features": ["day_of_year", "inflow_lag_1"],
    },
}

LEARNED_SPECS = {
    "ridge": {
        "name": "Ridge Regression",
        "filename": "rainfall_ridge_regression_v1.joblib",
        "estimator": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=100.0, random_state=42)),
            ]
        ),
        "hyperparameters": {
            "imputer_strategy": "median",
            "scaler": "StandardScaler",
            "alpha": 100.0,
            "random_state": 42,
        },
    },
    "random_forest": {
        "name": "Random Forest",
        "filename": "rainfall_random_forest_v1.joblib",
        "estimator": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hyperparameters": {
            "imputer_strategy": "median",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "tuned_random_forest": {
        "name": "Tuned Random Forest",
        "filename": "rainfall_tuned_random_forest_v1.joblib",
        "estimator": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=4,
                        max_features=0.7,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hyperparameters": {
            "imputer_strategy": "median",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 4,
            "max_features": 0.7,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "histgradientboosting": {
        "name": "HistGradientBoosting",
        "filename": "rainfall_histgradientboosting_v1.joblib",
        "estimator": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_iter=100,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hyperparameters": {
            "imputer_strategy": "median",
            "max_iter": 100,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
    },
}

LEARNED_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *RAINFALL_FEATURE_COLUMNS]
if TARGET_COLUMN in LEARNED_FEATURE_COLUMNS:
    raise ValueError("The target must not be included in the model features.")


def _read_and_validate_dataset(
    merged_path: Path = MERGED_DATA_PATH,
    rainfall_path: Path = RAINFALL_DATA_PATH,
    dam_path: Path = PROCESSED_DAM_PATH,
) -> tuple[pd.DataFrame, str]:
    if not merged_path.exists():
        raise FileNotFoundError(f"Combined dam-rainfall dataset not found: {merged_path}")
    if not rainfall_path.exists():
        raise FileNotFoundError(f"Daily rainfall dataset not found: {rainfall_path}")
    if not dam_path.exists():
        raise FileNotFoundError(f"Processed dam dataset not found: {dam_path}")

    merged = pd.read_csv(merged_path)
    required = {
        "calendar_date",
        "inflow_cusecs",
        "inflow_lag_1",
        "inflow_roll_mean_7",
        "day_of_year",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Combined dataset is missing required columns: {sorted(missing)}")

    merged["calendar_date"] = pd.to_datetime(
        merged["calendar_date"], errors="coerce"
    ).dt.normalize()
    if merged["calendar_date"].isna().any():
        raise ValueError("Combined dataset contains invalid calendar dates.")
    if merged["calendar_date"].duplicated().any():
        raise ValueError("Combined dataset contains duplicate calendar dates.")
    merged = merged.sort_values("calendar_date").reset_index(drop=True)

    rainfall = pd.read_csv(
        rainfall_path,
        usecols=["Monitoring Date", "Rainfall_MM", "Stations_Reporting"],
        dtype={"Monitoring Date": "string"},
    )
    dam_dates = pd.read_csv(dam_path, usecols=["calendar_date"])["calendar_date"]
    features = build_rainfall_features(dam_dates, rainfall)
    if features["calendar_date"].duplicated().any():
        raise ValueError("Rainfall feature builder returned duplicate dates.")
    if not features["calendar_date"].isin(merged["calendar_date"]).all():
        raise ValueError("Rainfall feature dates do not align with the merged dataset.")

    features = features.rename(columns={"calendar_date": "_feature_date"})
    merged = merged.merge(
        features,
        left_on="calendar_date",
        right_on="_feature_date",
        how="left",
        validate="one_to_one",
        sort=False,
    ).drop(columns="_feature_date")
    if len(merged) != len(dam_dates):
        raise AssertionError("Joining rainfall features changed the merged row count.")
    missing_features = set(LEARNED_FEATURE_COLUMNS) - set(merged.columns)
    if missing_features:
        raise ValueError(f"Engineered model features are missing: {sorted(missing_features)}")
    if len(LEARNED_FEATURE_COLUMNS) != 27:
        raise AssertionError(
            f"Expected 27 learned-model features; found {len(LEARNED_FEATURE_COLUMNS)}."
        )

    next_date_inflow = merged.set_index("calendar_date")["inflow_cusecs"]
    expected_target = merged["calendar_date"].add(pd.Timedelta(days=1)).map(
        next_date_inflow
    )
    target_available = merged[TARGET_COLUMN].notna()
    if not np.allclose(
        merged.loc[target_available, TARGET_COLUMN].to_numpy(dtype=float),
        expected_target.loc[target_available].to_numpy(dtype=float),
        equal_nan=False,
    ):
        raise ValueError(
            "Target alignment check failed: target must equal next-calendar-day inflow."
        )

    valid = merged.dropna(subset=[TARGET_COLUMN, "inflow_lag_1"]).copy()
    if valid.empty:
        raise ValueError("No rows have both a valid target and persistence input.")
    valid_dates = valid["calendar_date"]
    for split_name, (start, end) in TARGET_DATE_RANGES.items():
        selected = valid.loc[valid_dates.between(start, end)]
        if selected.empty:
            raise ValueError(f"No eligible observations in the {split_name} period.")
        if selected["calendar_date"].min() < pd.Timestamp(start):
            raise AssertionError(f"{split_name} split begins before its configured range.")
        if selected["calendar_date"].max() > pd.Timestamp(end):
            raise AssertionError(f"{split_name} split ends after its configured range.")
    if valid["calendar_date"].max() > pd.Timestamp(TEST_END):
        valid = valid.loc[valid["calendar_date"] <= TEST_END].copy()

    dataset_hash = hashlib.sha256(merged_path.read_bytes()).hexdigest()
    return valid.reset_index(drop=True), dataset_hash


def _split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: df.loc[df["calendar_date"].between(start, end)].copy()
        for name, (start, end) in TARGET_DATE_RANGES.items()
    }


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Cannot calculate metrics for empty or misaligned predictions.")
    return {
        "sample_count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _baseline_predictions(
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    train_df = splits["train"]
    seasonal_medians = train_df.groupby("day_of_year")[TARGET_COLUMN].median()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for key in BASELINE_SPECS:
        predictions[key] = {}
        for split_name in ("validation", "test"):
            part = splits[split_name]
            if key == "persistence":
                pred = part["inflow_lag_1"].to_numpy(dtype=float)
            elif key == "moving_average":
                pred = (
                    part["inflow_roll_mean_7"]
                    .fillna(part["inflow_lag_1"])
                    .to_numpy(dtype=float)
                )
            else:
                pred = (
                    part["day_of_year"]
                    .map(seasonal_medians)
                    .fillna(part["inflow_lag_1"])
                    .to_numpy(dtype=float)
                )
            predictions[key][split_name] = np.clip(pred, a_min=0.0, a_max=None)
    return predictions


def _output_paths() -> list[Path]:
    paths = [METRICS_OUTPUT_PATH]
    for spec in (*BASELINE_SPECS.values(), *LEARNED_SPECS.values()):
        artifact = MODEL_OUTPUT_DIR / spec["filename"]
        paths.extend([artifact, artifact.with_suffix(".json")])
    return paths


def _assert_output_paths_available() -> None:
    conflicts = [str(path) for path in _output_paths() if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs: "
            + ", ".join(conflicts)
        )


def train_and_evaluate() -> pd.DataFrame:
    """Train seven separate methods and save versioned artifacts and metrics."""
    _assert_output_paths_available()
    df, dataset_hash = _read_and_validate_dataset()
    splits = _split_by_date(df)
    feature_missing_counts = (
        df[list(RAINFALL_FEATURE_COLUMNS)].isna().sum().to_dict()
    )
    if df["calendar_date"].max() > pd.Timestamp(TEST_END):
        raise AssertionError("Rows after 2025-12-31 entered the experiment cohort.")

    print("Eligible target rows by split:")
    for split_name, part in splits.items():
        print(
            f"  {split_name}: {len(part)} "
            f"({part['calendar_date'].min().date()} to "
            f"{part['calendar_date'].max().date()})"
        )
    print("Rainfall feature missing-value counts:", feature_missing_counts)

    training_timestamp = datetime.now(timezone.utc).isoformat()
    common_metadata: dict[str, Any] = {
        "target": TARGET_COLUMN,
        "dataset_path": str(MERGED_DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_hash,
        "requested_date_ranges": {
            name: {"start": start, "end": end}
            for name, (start, end) in TARGET_DATE_RANGES.items()
        },
        "actual_date_ranges": {
            name: {
                "start": str(part["calendar_date"].min().date()),
                "end": str(part["calendar_date"].max().date()),
            }
            for name, part in splits.items()
        },
        "sample_counts": {name: int(len(part)) for name, part in splits.items()},
        "rainfall_source_end_date": "2025-12-31",
        "rainfall_missing_value_counts": {
            column: int(count) for column, count in feature_missing_counts.items()
        },
        "training_timestamp_utc": training_timestamp,
    }

    metric_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    metadata_items: dict[str, dict[str, Any]] = {}

    baseline_predictions = _baseline_predictions(splits)
    for key, spec in BASELINE_SPECS.items():
        split_metrics = {}
        for split_name in ("validation", "test"):
            part = splits[split_name]
            metrics = _metrics(
                part[TARGET_COLUMN],
                baseline_predictions[key][split_name],
            )
            split_metrics[split_name] = metrics
            metric_rows.append(
                _metric_row(spec, split_name, metrics, rainfall_used=False)
            )
        artifact_payload: dict[str, Any] = {
            "model_type": "rule-based",
            "method": spec["method"],
            "feature_names": spec["features"],
            "rainfall_features_used": False,
        }
        if key == "seasonal_naive":
            artifact_payload["day_of_year_target_medians"] = (
                splits["train"].groupby("day_of_year")[TARGET_COLUMN].median()
            )
        artifacts[key] = artifact_payload
        metadata_items[key] = {
            **common_metadata,
            "model_name": spec["name"],
            "algorithm": spec["method"],
            "model_type": "rule-based",
            "input_feature_names": spec["features"],
            "feature_count": len(spec["features"]),
            "rainfall_features_used": False,
            "training_sample_count": int(len(splits["train"])),
            "hyperparameters": {"forecast_rule": spec["method"]},
            "missing_data_handling": (
                "No rainfall inputs or rainfall imputation. "
                "Uses the original rule; rolling mean and seasonal median fall back "
                "to inflow_lag_1 as in the original implementation."
            ),
            "metrics": split_metrics,
            "artifact_path": str(MODEL_OUTPUT_DIR / spec["filename"]),
        }

    X_train = splits["train"][LEARNED_FEATURE_COLUMNS]
    y_train = splits["train"][TARGET_COLUMN]
    for key, spec in LEARNED_SPECS.items():
        estimator = spec["estimator"]
        estimator.fit(X_train, y_train)
        split_metrics = {}
        for split_name in ("validation", "test"):
            part = splits[split_name]
            predictions = np.clip(
                estimator.predict(part[LEARNED_FEATURE_COLUMNS]),
                a_min=0.0,
                a_max=None,
            )
            metrics = _metrics(part[TARGET_COLUMN], predictions)
            split_metrics[split_name] = metrics
            metric_rows.append(
                _metric_row(spec, split_name, metrics, rainfall_used=True)
            )
        artifacts[key] = estimator
        metadata_items[key] = {
            **common_metadata,
            "model_name": spec["name"],
            "algorithm": spec["name"],
            "model_type": "learned",
            "input_feature_names": LEARNED_FEATURE_COLUMNS,
            "feature_count": len(LEARNED_FEATURE_COLUMNS),
            "rainfall_features_used": True,
            "training_sample_count": int(len(splits["train"])),
            "hyperparameters": spec["hyperparameters"],
            "missing_data_handling": {
                "strategy": "SimpleImputer(strategy='median')",
                "fit_scope": "training split only, inside sklearn Pipeline",
                "rainfall_observations_filled_in_source": False,
                "missingness_indicators_preserved": True,
                "training_rainfall_missing_counts": {
                    column: int(count)
                    for column, count in splits["train"][
                        list(RAINFALL_FEATURE_COLUMNS)
                    ].isna().sum().items()
                },
            },
            "metrics": split_metrics,
            "artifact_path": str(MODEL_OUTPUT_DIR / spec["filename"]),
        }

    metrics_df = pd.DataFrame(metric_rows)
    if len(metrics_df) != 14:
        raise AssertionError(f"Expected 14 split metric rows, got {len(metrics_df)}.")

    _assert_output_paths_available()
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_specs = {**BASELINE_SPECS, **LEARNED_SPECS}
    for key, artifact in artifacts.items():
        spec = all_specs[key]
        artifact_path = MODEL_OUTPUT_DIR / spec["filename"]
        temporary_path = artifact_path.with_name(artifact_path.name + ".tmp")
        joblib.dump(artifact, temporary_path)
        temporary_path.replace(artifact_path)

        metadata_path = artifact_path.with_suffix(".json")
        temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_metadata.write_text(
            json.dumps(metadata_items[key], indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)

    temporary_metrics = METRICS_OUTPUT_PATH.with_name(
        METRICS_OUTPUT_PATH.name + ".tmp"
    )
    metrics_df.to_csv(temporary_metrics, index=False)
    temporary_metrics.replace(METRICS_OUTPUT_PATH)

    print("\nRainfall-enhanced experiment completed:")
    print(metrics_df.to_string(index=False))
    print(f"\nModel artifacts and metadata: {MODEL_OUTPUT_DIR}")
    print(f"Consolidated metrics: {METRICS_OUTPUT_PATH}")
    return metrics_df


def _metric_row(
    spec: dict[str, Any],
    split_name: str,
    metrics: dict[str, float | int],
    *,
    rainfall_used: bool,
) -> dict[str, Any]:
    return {
        "model_name": spec["name"],
        "model_type": "learned" if rainfall_used else "rule-based",
        "algorithm": spec.get("method", spec["name"]),
        "split": split_name,
        **metrics,
        "feature_count": len(LEARNED_FEATURE_COLUMNS)
        if rainfall_used
        else len(spec["features"]),
        "rainfall_features_used": rainfall_used,
    }


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
