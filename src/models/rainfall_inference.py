"""Inference helpers for the separately trained rainfall forecasting experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.rainfall_features import (
    RAINFALL_DATA_PATH,
    RAINFALL_FEATURE_COLUMNS,
    build_rainfall_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DATA_PATH = PROJECT_ROOT / "data/interim/krs_dam_rainfall_merged.csv"
MODEL_DIR = PROJECT_ROOT / "models/rainfall_enhanced"

MODEL_ARTIFACT_FILES = {
    "Persistence": "rainfall_persistence_v1.joblib",
    "7-Day Moving Average": "rainfall_7day_moving_average_v1.joblib",
    "Seasonal Naive": "rainfall_seasonal_naive_v1.joblib",
    "Ridge Regression": "rainfall_ridge_regression_v1.joblib",
    "Random Forest": "rainfall_random_forest_v1.joblib",
    "Tuned Random Forest": "rainfall_tuned_random_forest_v1.joblib",
    "HistGradientBoosting": "rainfall_histgradientboosting_v1.joblib",
}
MODEL_METADATA_FILES = {
    name: Path(filename).with_suffix(".json").name
    for name, filename in MODEL_ARTIFACT_FILES.items()
}


class RainfallInferenceError(ValueError):
    """Raised when the saved experiment cannot produce a safe forecast."""


def load_forecast_data(
    merged_path: Path = MERGED_DATA_PATH,
    rainfall_path: Path = RAINFALL_DATA_PATH,
    model_dir: Path = MODEL_DIR,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Load dam rows and attach the same cutoff-safe rainfall features used in training."""
    metadata_path = model_dir / MODEL_METADATA_FILES["Tuned Random Forest"]
    if not metadata_path.is_file():
        raise RainfallInferenceError(f"Model metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise RainfallInferenceError(
            f"Could not read model metadata {metadata_path}: {exc}"
        ) from exc

    feature_names = metadata.get("input_feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise RainfallInferenceError(
            f"Model metadata has no valid input feature list: {metadata_path}"
        )
    dam_features = tuple(
        name for name in feature_names if name not in RAINFALL_FEATURE_COLUMNS
    )
    if (
        len(feature_names) != 27
        or len(dam_features) != 19
        or not set(RAINFALL_FEATURE_COLUMNS).issubset(feature_names)
    ):
        raise RainfallInferenceError(
            "Saved learned-model metadata does not describe the expected "
            "19 dam/calendar and eight rainfall features."
        )

    for path, label in ((merged_path, "Combined dataset"), (rainfall_path, "Rainfall data")):
        if not path.is_file():
            raise RainfallInferenceError(f"{label} not found: {path}")
    try:
        dam = pd.read_csv(merged_path, parse_dates=["calendar_date"])
        rainfall = pd.read_csv(
            rainfall_path,
            usecols=["Monitoring Date", "Rainfall_MM", "Stations_Reporting"],
            dtype={"Monitoring Date": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError) as exc:
        raise RainfallInferenceError(f"Could not read prediction inputs: {exc}") from exc

    required = {"calendar_date", "inflow_cusecs", *dam_features}
    missing = required - set(dam.columns)
    if missing:
        raise RainfallInferenceError(
            f"Combined dataset is missing forecast columns: {sorted(missing)}"
        )
    if dam["calendar_date"].isna().any() or dam["calendar_date"].duplicated().any():
        raise RainfallInferenceError(
            "Combined dataset dates must be valid and unique for forecasting."
        )
    dam = dam.sort_values("calendar_date").reset_index(drop=True)
    try:
        rainfall_features = build_rainfall_features(dam["calendar_date"], rainfall)
    except (ValueError, KeyError) as exc:
        raise RainfallInferenceError(
            f"Could not construct aligned rainfall features: {exc}"
        ) from exc
    data = dam.merge(
        rainfall_features,
        on="calendar_date",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(data) != len(dam):
        raise RainfallInferenceError(
            "Attaching rainfall features changed the dam dataset row count."
        )

    learned_metadata: list[dict[str, Any]] = []
    for model_name in (
        "Ridge Regression",
        "Random Forest",
        "Tuned Random Forest",
        "HistGradientBoosting",
    ):
        path = model_dir / MODEL_METADATA_FILES[model_name]
        if not path.is_file():
            raise RainfallInferenceError(f"Model metadata not found: {path}")
        try:
            learned_metadata.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise RainfallInferenceError(
                f"Could not read model metadata {path}: {exc}"
            ) from exc
    if any(item.get("input_feature_names") != feature_names for item in learned_metadata):
        raise RainfallInferenceError(
            "Learned-model feature lists do not agree; refusing to build predictions."
        )
    return data, tuple(feature_names), dam_features


def load_forecast_models(
    model_dir: Path = MODEL_DIR,
) -> dict[str, Any]:
    """Load the seven existing experiment artifacts without training."""
    artifacts: dict[str, Any] = {}
    for model_name, filename in MODEL_ARTIFACT_FILES.items():
        path = model_dir / filename
        if not path.is_file():
            raise RainfallInferenceError(f"Model artifact not found: {path}")
        try:
            artifacts[model_name] = joblib.load(path)
        except (OSError, EOFError, ValueError, ImportError) as exc:
            raise RainfallInferenceError(
                f"Could not load model artifact {path}: {exc}"
            ) from exc
    return artifacts


def feature_values_for_row(
    row: pd.Series,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Extract the exact saved-model inputs without filling missing rainfall."""
    missing = set(feature_names) - set(row.index)
    if missing:
        raise RainfallInferenceError(
            f"Forecast row is missing model inputs: {sorted(missing)}"
        )
    values: dict[str, float] = {}
    for name in feature_names:
        value = row[name]
        if pd.isna(value):
            values[name] = float("nan")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise RainfallInferenceError(
                f"Forecast input '{name}' is not numeric: {value!r}"
            ) from exc
        if not np.isfinite(numeric_value):
            raise RainfallInferenceError(
                f"Forecast input '{name}' is not finite."
            )
        values[name] = numeric_value
    return values


def predict_all_models(
    feature_values: dict[str, float],
    models: dict[str, Any],
) -> pd.DataFrame:
    """Apply the saved learned estimators and the original three baseline rules."""
    required_models = set(MODEL_ARTIFACT_FILES)
    if set(models) != required_models:
        raise RainfallInferenceError(
            "Forecast requires all seven saved model artifacts."
        )
    required_values = {
        "inflow_lag_1",
        "inflow_roll_mean_7",
        "day_of_year",
        *RAINFALL_FEATURE_COLUMNS,
    }
    missing_values = required_values - set(feature_values)
    if missing_values:
        raise RainfallInferenceError(
            f"Forecast inputs are missing features: {sorted(missing_values)}"
        )
    row = pd.DataFrame([feature_values])
    predictions: list[dict[str, str | float]] = []

    persistence = max(float(row.at[0, "inflow_lag_1"]), 0.0)
    rolling_mean = row.at[0, "inflow_roll_mean_7"]
    moving_average = (
        float(row.at[0, "inflow_lag_1"])
        if pd.isna(rolling_mean)
        else float(rolling_mean)
    )
    seasonal_payload = models["Seasonal Naive"]
    if not isinstance(seasonal_payload, dict):
        raise RainfallInferenceError("Seasonal Naive artifact has an invalid format.")
    medians = seasonal_payload.get("day_of_year_target_medians")
    if medians is None or not hasattr(medians, "get"):
        raise RainfallInferenceError(
            "Seasonal Naive artifact is missing its training seasonal medians."
        )
    seasonal = medians.get(
        int(row.at[0, "day_of_year"]), row.at[0, "inflow_lag_1"]
    )
    rules = {
        "Persistence": persistence,
        "7-Day Moving Average": max(moving_average, 0.0),
        "Seasonal Naive": max(float(seasonal), 0.0),
    }
    for model_name, prediction in rules.items():
        predictions.append({"model_name": model_name, "forecast_cusecs": prediction})

    for model_name in (
        "Ridge Regression",
        "Random Forest",
        "Tuned Random Forest",
        "HistGradientBoosting",
    ):
        estimator = models[model_name]
        if not hasattr(estimator, "predict"):
            raise RainfallInferenceError(
                f"Learned-model artifact has no predict method: {model_name}"
            )
        try:
            prediction = float(estimator.predict(row)[0])
        except (ValueError, TypeError, KeyError) as exc:
            raise RainfallInferenceError(
                f"Could not generate a forecast with {model_name}: {exc}"
            ) from exc
        if not np.isfinite(prediction):
            raise RainfallInferenceError(
                f"{model_name} returned a non-finite forecast."
            )
        predictions.append(
            {"model_name": model_name, "forecast_cusecs": max(prediction, 0.0)}
        )

    result = pd.DataFrame(predictions)
    if len(result) != len(MODEL_ARTIFACT_FILES):
        raise RainfallInferenceError("Did not produce all seven model forecasts.")
    return result
