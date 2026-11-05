import numpy as np
import pandas as pd
import pytest

from src.data.rainfall_features import RAINFALL_FEATURE_COLUMNS
from src.models.rainfall_inference import (
    MODEL_ARTIFACT_FILES,
    RainfallInferenceError,
    eligible_prediction_issue_dates,
    feature_values_for_row,
    load_forecast_data,
    load_forecast_models,
    predict_all_models,
    predict_for_issue_date,
    select_best_learned_model,
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


def test_best_learned_model_is_selected_from_saved_test_metrics():
    metrics = pd.read_csv("reports/rainfall_enhanced_model_metrics.csv")

    assert select_best_learned_model(metrics) == "Tuned Random Forest"


def _saved_forecast_inputs():
    model_data, feature_names, dam_features = load_forecast_data()
    dam_history = pd.read_csv(
        "data/processed/krs_dam_processed.csv",
        parse_dates=["calendar_date"],
    )
    return (
        model_data,
        feature_names,
        dam_features,
        dam_history,
        load_forecast_models(),
    )


def test_historical_issue_date_predicts_following_day_and_reports_actual():
    model_data, features, dam_features, dam_history, models = _saved_forecast_inputs()

    result = predict_for_issue_date(
        "2025-12-20",
        model_data=model_data,
        dam_history=dam_history,
        feature_names=features,
        dam_features=dam_features,
        models=models,
    )
    actual = dam_history.loc[
        dam_history["calendar_date"].eq(pd.Timestamp("2025-12-21")),
        "inflow_cusecs",
    ].iloc[0]

    assert result.issue_date == pd.Timestamp("2025-12-20")
    assert result.target_date == pd.Timestamp("2025-12-21")
    assert np.isfinite(result.predicted_inflow_cusecs)
    assert result.actual_inflow_cusecs == pytest.approx(actual)
    assert result.signed_error_cusecs == pytest.approx(
        result.predicted_inflow_cusecs - actual
    )
    assert not result.missing_rainfall_features


def test_2026_issue_date_uses_saved_fitted_imputer_for_missing_rainfall():
    model_data, features, dam_features, dam_history, models = _saved_forecast_inputs()

    result = predict_for_issue_date(
        "2026-10-21",
        model_data=model_data,
        dam_history=dam_history,
        feature_names=features,
        dam_features=dam_features,
        models=models,
    )

    assert result.target_date == pd.Timestamp("2026-10-22")
    assert np.isfinite(result.predicted_inflow_cusecs)
    assert result.missing_rainfall_features
    assert all(
        np.isnan(result.feature_values[name])
        for name in result.missing_rainfall_features
    )
    assert result.actual_inflow_cusecs is None


def test_missing_rainfall_is_rejected_if_saved_model_has_no_fitted_imputer():
    model_data, features, dam_features, dam_history, _ = _saved_forecast_inputs()

    with pytest.raises(RainfallInferenceError, match="no fitted imputer"):
        predict_for_issue_date(
            "2026-10-21",
            model_data=model_data,
            dam_history=dam_history,
            feature_names=features,
            dam_features=dam_features,
            models={"Tuned Random Forest": ConstantEstimator(100)},
        )


def test_prediction_rejects_dates_outside_archive_and_missing_required_lags():
    model_data, features, dam_features, dam_history, models = _saved_forecast_inputs()

    for issue_date in ("2027-01-01", "1990-01-01"):
        with pytest.raises(RainfallInferenceError):
            predict_for_issue_date(
                issue_date,
                model_data=model_data,
                dam_history=dam_history,
                feature_names=features,
                dam_features=dam_features,
                models=models,
            )

    eligible = eligible_prediction_issue_dates(
        model_data, dam_history, dam_features
    )
    assert eligible[0] > pd.Timestamp("1990-01-01")
    assert eligible[-1] == pd.Timestamp("2026-10-21")


def test_historical_prediction_does_not_read_future_feature_rows():
    model_data, features, dam_features, dam_history, models = _saved_forecast_inputs()
    issue_date = pd.Timestamp("2025-12-20")
    baseline = predict_for_issue_date(
        issue_date,
        model_data=model_data,
        dam_history=dam_history,
        feature_names=features,
        dam_features=dam_features,
        models=models,
    )
    altered = model_data.copy().astype(
        {name: "float64" for name in RAINFALL_FEATURE_COLUMNS}
    )
    future_rows = pd.to_datetime(altered["calendar_date"]).gt(issue_date)
    altered.loc[future_rows, list(dam_features)] = 1_000_000.0
    altered.loc[future_rows, list(RAINFALL_FEATURE_COLUMNS)] = 1_000_000.0

    after_future_change = predict_for_issue_date(
        issue_date,
        model_data=altered,
        dam_history=dam_history,
        feature_names=features,
        dam_features=dam_features,
        models=models,
    )

    assert after_future_change.predicted_inflow_cusecs == pytest.approx(
        baseline.predicted_inflow_cusecs
    )


def test_prediction_rejects_issue_date_without_observed_dam_row():
    model_data, features, dam_features, dam_history, models = _saved_forecast_inputs()
    dam_history = dam_history.loc[
        ~dam_history["calendar_date"].eq(pd.Timestamp("2025-12-20"))
    ]

    with pytest.raises(RainfallInferenceError, match="outside the available dam"):
        predict_for_issue_date(
            "2025-12-20",
            model_data=model_data,
            dam_history=dam_history,
            feature_names=features,
            dam_features=dam_features,
            models=models,
        )
