import pandas as pd
import pytest

from src.rainfall_dashboard import (
    DashboardDataError,
    EXPECTED_MODELS,
    load_metrics,
    load_model_metadata,
    prepare_comparison,
    validate_metrics_metadata,
)


METRICS_PATH = "reports/rainfall_enhanced_model_metrics.csv"


def test_load_metrics_validates_and_returns_all_models():
    metrics = load_metrics()

    assert set(metrics["model_name"]) == set(EXPECTED_MODELS)
    assert set(metrics["split"]) == {"validation", "test"}
    assert len(metrics) == 14
    assert metrics.groupby(["model_name", "split"]).size().eq(1).all()


def test_load_metrics_reports_missing_file(tmp_path):
    with pytest.raises(DashboardDataError, match="Metrics report not found"):
        load_metrics(tmp_path / "missing.csv")


def test_load_metrics_rejects_missing_values(tmp_path):
    metrics = pd.read_csv(METRICS_PATH)
    metrics.loc[0, "mae"] = None
    path = tmp_path / "missing-value.csv"
    metrics.to_csv(path, index=False)

    with pytest.raises(DashboardDataError, match="missing or invalid"):
        load_metrics(path)


def test_load_metrics_rejects_duplicate_rows(tmp_path):
    metrics = pd.read_csv(METRICS_PATH)
    metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicate.csv"
    metrics.to_csv(path, index=False)

    with pytest.raises(DashboardDataError, match="duplicate model/split"):
        load_metrics(path)


def test_metadata_is_loaded_and_has_common_periods():
    metadata = load_model_metadata()

    assert set(metadata) == set(EXPECTED_MODELS)
    assert metadata["Tuned Random Forest"]["actual_date_ranges"]["test"] == {
        "start": "2025-01-01",
        "end": "2025-12-31",
    }


def test_ranking_is_calculated_from_metrics():
    metrics = load_metrics()

    comparison, best_model = prepare_comparison(metrics)

    assert best_model == "Tuned Random Forest"
    assert comparison.index[0] == best_model
    assert ("Test ranking", "average_test_rank") in comparison.columns
    assert comparison.loc[best_model, ("Test", "sample_count")] == 365


def test_r2_is_not_interpreted_as_percentage_and_features_are_27():
    metadata = load_model_metadata()
    metrics = load_metrics()

    assert metadata["HistGradientBoosting"]["feature_count"] == 27
    assert metrics["r2"].max() < 1.0
    assert not metrics["rainfall_features_used"].all()


def test_metadata_and_metrics_must_agree():
    metrics = load_metrics()
    metadata = load_model_metadata()
    validate_metrics_metadata(metrics, metadata)

    metadata["Persistence"]["metrics"]["test"]["mae"] = -1
    with pytest.raises(DashboardDataError, match="differs between metrics and metadata"):
        validate_metrics_metadata(metrics, metadata)


def test_missing_metadata_is_reported(tmp_path):
    with pytest.raises(DashboardDataError, match="Model metadata file not found"):
        load_model_metadata(tmp_path)
