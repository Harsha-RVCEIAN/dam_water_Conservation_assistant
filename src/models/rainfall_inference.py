"""Inference helpers for the separately trained rainfall forecasting experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
LEARNED_MODEL_NAMES = (
    "Ridge Regression",
    "Random Forest",
    "Tuned Random Forest",
    "HistGradientBoosting",
)


class RainfallInferenceError(ValueError):
    """Raised when the saved experiment cannot produce a safe forecast."""


@dataclass(frozen=True)
class NextDayForecast:
    issue_date: pd.Timestamp
    target_date: pd.Timestamp
    predicted_inflow_cusecs: float
    actual_inflow_cusecs: float | None
    signed_error_cusecs: float | None
    feature_values: dict[str, float]
    missing_rainfall_features: tuple[str, ...]



def select_best_learned_model(metrics: pd.DataFrame) -> str:
    """Choose the top learned model by mean rank across saved test metrics."""
    required = {"model_name", "model_type", "split", "mae", "rmse", "r2"}
    missing = required - set(metrics.columns)
    if missing:
        raise RainfallInferenceError(
            f"Metrics are missing model-selection columns: {sorted(missing)}"
        )
    test = metrics.loc[
        metrics["split"].astype(str).str.lower().eq("test")
        & metrics["model_name"].isin(LEARNED_MODEL_NAMES)
        & metrics["model_type"].astype(str).str.lower().eq("learned")
    ].copy()
    if set(test["model_name"]) != set(LEARNED_MODEL_NAMES):
        raise RainfallInferenceError(
            "Saved test metrics must include exactly the four learned rainfall models."
        )
    for column in ("mae", "rmse", "r2"):
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if test[column].isna().any() or not np.isfinite(test[column]).all():
            raise RainfallInferenceError(
                f"Saved test metric '{column}' contains missing or invalid values."
            )
    if test.duplicated("model_name").any():
        raise RainfallInferenceError(
            "Saved test metrics contain duplicate learned-model rows."
        )
    ranks = pd.DataFrame(
        {
            "model_name": test["model_name"],
            "mae": test["mae"].rank(ascending=True, method="min"),
            "rmse": test["rmse"].rank(ascending=True, method="min"),
            "r2": test["r2"].rank(ascending=False, method="min"),
        }
    )
    ranks["mean_rank"] = ranks[["mae", "rmse", "r2"]].mean(axis=1)
    ranked = ranks.sort_values(
        ["mean_rank", "mae", "rmse", "model_name"],
        ascending=[True, True, True, True],
    )
    return str(ranked.iloc[0]["model_name"])


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


def eligible_prediction_issue_dates(
    model_data: pd.DataFrame,
    dam_history: pd.DataFrame,
    dam_features: tuple[str, ...],
) -> tuple[pd.Timestamp, ...]:
    """Return observed issue dates with all required dam/calendar inputs present."""
    required_dam_columns = {"calendar_date", "inflow_cusecs"}
    missing_dam_columns = required_dam_columns - set(dam_history.columns)
    if missing_dam_columns:
        raise RainfallInferenceError(
            f"Dam history is missing required columns: {sorted(missing_dam_columns)}"
        )
    required_model_columns = {"calendar_date", *dam_features}
    missing_model_columns = required_model_columns - set(model_data.columns)
    if missing_model_columns:
        raise RainfallInferenceError(
            f"Forecast data is missing required columns: {sorted(missing_model_columns)}"
        )
    dam_dates = pd.to_datetime(dam_history["calendar_date"], errors="coerce").dt.normalize()
    model_dates = pd.to_datetime(model_data["calendar_date"], errors="coerce").dt.normalize()
    if dam_dates.isna().any() or dam_dates.duplicated().any():
        raise RainfallInferenceError("Dam history dates must be valid and unique.")
    if model_dates.isna().any() or model_dates.duplicated().any():
        raise RainfallInferenceError("Forecast data dates must be valid and unique.")
    observed_dates = set(dam_dates)
    numeric_features = model_data.loc[:, list(dam_features)].apply(
        pd.to_numeric, errors="coerce"
    )
    complete_features = np.isfinite(numeric_features.to_numpy(dtype=float)).all(axis=1)
    eligible = [
        pd.Timestamp(issue_date)
        for issue_date, is_complete in zip(model_dates, complete_features)
        if issue_date in observed_dates and is_complete
    ]
    if not eligible:
        raise RainfallInferenceError(
            "No issue dates have both an observed dam row and all required "
            "dam/calendar lag features."
        )
    return tuple(eligible)


def predict_for_issue_date(
    issue_date: str | pd.Timestamp,
    *,
    model_data: pd.DataFrame,
    dam_history: pd.DataFrame,
    feature_names: tuple[str, ...],
    dam_features: tuple[str, ...],
    models: dict[str, Any],
    model_name: str = "Tuned Random Forest",
) -> NextDayForecast:
    """Predict next-calendar-day inflow from issue-date inputs only.

    Historical target observations are joined only after prediction to report an
    actual and error; they are never passed to the model.
    """
    try:
        selected_date = pd.Timestamp(issue_date).normalize()
    except (TypeError, ValueError) as exc:
        raise RainfallInferenceError(
            f"Invalid prediction issue date: {issue_date!r}."
        ) from exc
    if pd.isna(selected_date):
        raise RainfallInferenceError(f"Invalid prediction issue date: {issue_date!r}.")

    eligible_dates = eligible_prediction_issue_dates(
        model_data,
        dam_history,
        dam_features,
    )
    if selected_date not in eligible_dates:
        all_dam_dates = pd.to_datetime(
            dam_history["calendar_date"], errors="coerce"
        ).dt.normalize()
        if selected_date not in set(all_dam_dates):
            raise RainfallInferenceError(
                f"Issue date {selected_date.date()} is outside the available dam "
                "observation dates."
            )
        raise RainfallInferenceError(
            f"Issue date {selected_date.date()} does not have all required "
            "dam/calendar lag features; choose an eligible observed date."
        )

    matching = model_data.loc[
        pd.to_datetime(model_data["calendar_date"], errors="coerce")
        .dt.normalize()
        .eq(selected_date)
    ]
    if len(matching) != 1:
        raise RainfallInferenceError(
            f"Issue date {selected_date.date()} does not map to exactly one feature row."
        )
    row = matching.iloc[0]
    feature_values = feature_values_for_row(row, feature_names)
    missing_rainfall = tuple(
        name for name in RAINFALL_FEATURE_COLUMNS if not np.isfinite(feature_values[name])
    )
    estimator = models.get(model_name)
    if estimator is None or not hasattr(estimator, "predict"):
        raise RainfallInferenceError(
            f"Saved forecasting model is unavailable: {model_name}."
        )
    if missing_rainfall:
        steps = getattr(estimator, "named_steps", None)
        imputer = steps.get("imputer") if steps is not None else None
        statistics = getattr(imputer, "statistics_", None)
        if statistics is None:
            raise RainfallInferenceError(
                f"{model_name} has missing rainfall inputs, but no fitted imputer "
                "is available to handle them safely."
            )
        statistics = np.asarray(statistics, dtype=float)
        missing_indices = [feature_names.index(name) for name in missing_rainfall]
        if any(
            index >= len(statistics) or not np.isfinite(statistics[index])
            for index in missing_indices
        ):
            raise RainfallInferenceError(
                f"{model_name}'s fitted imputer has no valid training value for "
                "one or more missing rainfall features."
            )
    input_frame = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names,
    )
    try:
        prediction = float(estimator.predict(input_frame)[0])
    except (ValueError, TypeError, KeyError) as exc:
        raise RainfallInferenceError(
            f"Could not generate a forecast with {model_name}: {exc}"
        ) from exc
    if not np.isfinite(prediction):
        raise RainfallInferenceError(
            f"{model_name} returned a non-finite forecast."
        )
    prediction = max(prediction, 0.0)

    target_date = selected_date + pd.Timedelta(days=1)
    normalized_dam_dates = pd.to_datetime(
        dam_history["calendar_date"], errors="coerce"
    ).dt.normalize()
    actual_rows = dam_history.loc[normalized_dam_dates.eq(target_date), "inflow_cusecs"]
    actual: float | None = None
    if (
        target_date <= pd.Timestamp.now().normalize()
        and len(actual_rows) == 1
        and pd.notna(actual_rows.iloc[0])
    ):
        actual_value = pd.to_numeric(actual_rows.iloc[0], errors="coerce")
        if pd.notna(actual_value) and np.isfinite(actual_value):
            actual = float(actual_value)

    return NextDayForecast(
        issue_date=selected_date,
        target_date=target_date,
        predicted_inflow_cusecs=prediction,
        actual_inflow_cusecs=actual,
        signed_error_cusecs=(
            prediction - actual if actual is not None else None
        ),
        feature_values=feature_values,
        missing_rainfall_features=missing_rainfall,
    )


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
