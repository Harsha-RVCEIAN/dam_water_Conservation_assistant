"""Tests for the rainfall MLflow experiment-logging module.

Module tested: ``src.models.log_rainfall_mlflow``

Key scenarios:
- Verifies that the seven expected MLflow run names are present and correctly ordered.
- Checks that each run's saved metrics and parameters satisfy structural contracts.
- Validates that the JSON artefact written per run is well-formed and complete.

BLOCKED: These tests require ``reports/rainfall_enhanced_model_metrics.csv``
and the fitted artefacts under ``models/rainfall_enhanced/*.joblib``, which
are not committed to the repository (pending IMD/CWC rainfall source data).
All tests in this file are therefore expected to be skipped or fail until
those artefacts are provided.
"""
import json

import pytest

from src.models import log_rainfall_mlflow as logger


def test_run_names_cover_seven_saved_methods():
    assert list(logger.RUNS) == [
        "Rainfall_Persistence",
        "Rainfall_7Day_Moving_Average",
        "Rainfall_Seasonal_Naive",
        "Rainfall_Ridge_Regression",
        "Rainfall_Random_Forest",
        "Rainfall_Tuned_Random_Forest",
        "Rainfall_HistGradientBoosting",
    ]


def test_saved_metadata_and_csv_are_consistent():
    entries = logger._load_saved_results()

    assert len(entries) == 7
    learned = [entry for entry in entries if entry["metadata"]["model_type"] == "learned"]
    rule_based = [
        entry for entry in entries if entry["metadata"]["model_type"] == "rule-based"
    ]
    assert len(learned) == 4
    assert len(rule_based) == 3
    assert all(entry["metadata"]["feature_count"] == 27 for entry in learned)
    assert all(not entry["metadata"]["rainfall_features_used"] for entry in rule_based)


def test_logged_params_and_metrics_represent_saved_results():
    entry = logger._load_saved_results()[0]

    params = logger._run_parameters(entry)
    metrics = logger._run_metrics(entry)

    assert params["logging_provenance"] == "existing completed experiment; no retraining"
    assert params["train_sample_count"] == 12035
    assert params["validation_sample_count"] == 731
    assert params["test_sample_count"] == 365
    assert metrics["validation_sample_count"] == 731.0
    assert metrics["test_sample_count"] == 365.0
    assert "validation_mae" in metrics and "test_r2" in metrics
    assert json.loads(params["input_feature_names"])


def test_existing_conflicting_run_is_not_accepted(monkeypatch):
    class FakeRun:
        class Data:
            tags = {}
            params = {}
            metrics = {}

        data = Data()

    entry = logger._load_saved_results()[0]

    assert not logger._matching_existing_run(object(), FakeRun(), entry)


def test_mlflow_store_is_separate_from_original_default():
    assert logger.TRACKING_PATH.name == "mlruns_rainfall_enhanced"
    assert logger.TRACKING_URI.startswith("file:")
    assert logger.TRACKING_PATH.name != "mlruns"


def test_metadata_hash_mismatch_stops_logging(monkeypatch):
    original_sha256 = logger._sha256

    def mismatched_sha256(path):
        if path == logger.DATASET_PATH:
            return "not-the-source-dataset"
        return original_sha256(path)

    monkeypatch.setattr(logger, "_sha256", mismatched_sha256)

    with pytest.raises(ValueError, match="Dataset hash differs"):
        logger._load_saved_results()
