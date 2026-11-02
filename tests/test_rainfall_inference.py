import numpy as np
import pandas as pd

from src.data.rainfall_features import RAINFALL_FEATURE_COLUMNS
from src.models.rainfall_inference import (
    MODEL_ARTIFACT_FILES,
    feature_values_for_row,
    load_forecast_data,
    load_forecast_models,
    predict_all_models,
)


class ConstantEstimator:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class RainfallInputEstimator:
    def predict(self, frame):
        return frame["rainfall_mm_lag_1"].to_numpy(dtype=float)


def _sample_inputs():
    return {
        "inflow_lag_1": 10.0,
        "inflow_roll_mean_7": 20.0,
        "day_of_year": 123.0,
        **{feature: float("nan") for feature in RAINFALL_FEATURE_COLUMNS},
    }


def _sample_models():
    models = {
        name: ConstantEstimator(50.0)
        for name in MODEL_ARTIFACT_FILES
        if name not in {"Persistence", "7-Day Moving Average", "Seasonal Naive"}
    }
    models.update(
        {
            "Persistence": {"model_type": "rule-based"},
            "7-Day Moving Average": {"model_type": "rule-based"},
            "Seasonal Naive": {
                "model_type": "rule-based",
                "day_of_year_target_medians": pd.Series({123: 30.0}),
            },
        }
    )
    return models


def test_baseline_rules_keep_their_original_formulas_and_ignore_rainfall():
    first = _sample_inputs()
    second = {**first, "rainfall_mm_lag_1": 77.0, "rainfall_mm_sum_7d": 123.0}

    first_predictions = predict_all_models(first, _sample_models()).set_index(
        "model_name"
    )
    second_predictions = predict_all_models(second, _sample_models()).set_index(
        "model_name"
    )

    assert first_predictions.at["Persistence", "forecast_cusecs"] == 10.0
    assert first_predictions.at["7-Day Moving Average", "forecast_cusecs"] == 20.0
    assert first_predictions.at["Seasonal Naive", "forecast_cusecs"] == 30.0
    for model_name in ("Persistence", "7-Day Moving Average", "Seasonal Naive"):
        assert (
            first_predictions.at[model_name, "forecast_cusecs"]
            == second_predictions.at[model_name, "forecast_cusecs"]
        )


def test_seasonal_naive_falls_back_to_persistence_when_date_has_no_median():
    inputs = {**_sample_inputs(), "day_of_year": 124.0}

    predictions = predict_all_models(inputs, _sample_models()).set_index("model_name")

    assert predictions.at["Seasonal Naive", "forecast_cusecs"] == 10.0


def test_rainfall_scenario_inputs_reach_learned_models():
    first = {**_sample_inputs(), "rainfall_mm_lag_1": 10.0}
    second = {**_sample_inputs(), "rainfall_mm_lag_1": 20.0}
    models = _sample_models()
    for model_name in (
        "Ridge Regression",
        "Random Forest",
        "Tuned Random Forest",
        "HistGradientBoosting",
    ):
        models[model_name] = RainfallInputEstimator()

    first_predictions = predict_all_models(first, models).set_index("model_name")
    second_predictions = predict_all_models(second, models).set_index("model_name")

    for model_name in (
        "Ridge Regression",
        "Random Forest",
        "Tuned Random Forest",
        "HistGradientBoosting",
    ):
        assert first_predictions.at[model_name, "forecast_cusecs"] == 10.0
        assert second_predictions.at[model_name, "forecast_cusecs"] == 20.0


def test_feature_values_preserve_missing_rainfall():
    row = pd.Series(
        {
            "inflow_lag_1": 10.0,
            "rainfall_mm_lag_1": np.nan,
            "rainfall_lag_1_available": 0,
        }
    )

    values = feature_values_for_row(
        row,
        ("inflow_lag_1", "rainfall_mm_lag_1", "rainfall_lag_1_available"),
    )

    assert values["inflow_lag_1"] == 10.0
    assert np.isnan(values["rainfall_mm_lag_1"])
    assert values["rainfall_lag_1_available"] == 0


def test_saved_experiment_artifacts_make_a_latest_date_forecast():
    data, feature_names, dam_features = load_forecast_data()
    latest = data.iloc[-1]
    feature_values = feature_values_for_row(latest, feature_names)
    models = load_forecast_models()

    forecasts = predict_all_models(feature_values, models)

    assert len(feature_names) == 27
    assert len(dam_features) == 19
    assert len(forecasts) == 7
    assert set(forecasts["model_name"]) == set(MODEL_ARTIFACT_FILES)
    assert np.isfinite(forecasts["forecast_cusecs"]).all()
    if pd.Timestamp(latest["calendar_date"]) > pd.Timestamp("2025-12-31"):
        assert np.isnan(feature_values["rainfall_mm_lag_1"])
        assert feature_values["rainfall_lag_1_available"] == 0
