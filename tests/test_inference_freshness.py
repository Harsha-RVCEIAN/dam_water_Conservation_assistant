"""Tests for dynamic inference freshness and integrity checks."""

from pathlib import Path
import pandas as pd
import pytest

from src.models.predict import InflowForecaster


def test_forecast_freshness_evaluation():
    forecaster = InflowForecaster()
    # Case 1: Observation is 15 days old
    res_stale = forecaster.evaluate_freshness(
        latest_observation_date="2026-10-21",
        current_date="2026-11-05",
    )
    assert res_stale["data_age_days"] == 15
    assert res_stale["is_stale"] is True
    assert res_stale["data_age_status"] == "STALE"
    assert res_stale["data_source_mode"] == "HISTORICAL_ARCHIVE"
    assert res_stale["telemetry_connection_status"] == "NOT_CONFIGURED"
    assert res_stale["freshness_status"] == "HISTORICAL_STALE"
    assert res_stale["target_date"] == "2026-10-22"
    assert "historical evaluation" in res_stale["freshness_note"]

    # Case 2: Fresh observation (0 days old)
    res_fresh = forecaster.evaluate_freshness(
        latest_observation_date="2026-11-05",
        current_date="2026-11-05",
    )
    assert res_fresh["data_age_days"] == 0
    assert res_fresh["is_stale"] is False
    assert res_fresh["data_age_status"] == "FRESH"
    assert res_fresh["data_source_mode"] == "HISTORICAL_ARCHIVE"
    assert res_fresh["telemetry_connection_status"] == "NOT_CONFIGURED"
    assert res_fresh["freshness_status"] == "CURRENT_OPERATIONAL"
    assert res_fresh["target_date"] == "2026-11-06"


def test_live_telemetry_requires_explicit_connector():
    """Verify that historical archive data never claims to be live telemetry solely from recent timestamp."""
    forecaster = InflowForecaster()
    # Even if timestamp is today, source mode defaults to HISTORICAL_ARCHIVE
    res = forecaster.evaluate_freshness(
        latest_observation_date="2026-11-05",
        current_date="2026-11-05",
        data_source_mode="HISTORICAL_ARCHIVE",
    )
    assert res["data_source_mode"] == "HISTORICAL_ARCHIVE"
    assert res["telemetry_connection_status"] == "NOT_CONFIGURED"
    assert "Live telemetry: NOT_CONFIGURED" in res["freshness_note"]


def test_inference_target_derived_from_origin():
    forecaster = InflowForecaster()
    res = forecaster.predict_latest_from_processed(current_date="2026-11-05")
    assert res["forecast_origin_date"] == "2026-10-21"
    assert res["forecast_target_date"] == "2026-10-22"
    assert res["is_stale"] is True


def test_missing_features_raises_value_error():
    forecaster = InflowForecaster()
    incomplete_df = pd.DataFrame([{"inflow_lag_1": 1000.0}])  # Missing other required features
    with pytest.raises(ValueError, match="missing required feature columns"):
        forecaster.predict(incomplete_df)


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        InflowForecaster(model_path=tmp_path / "non_existent.joblib")


def test_missing_metadata_raises_file_not_found(tmp_path):
    fake_model = tmp_path / "model.joblib"
    fake_model.touch()
    with pytest.raises(FileNotFoundError, match="Model metadata not found"):
        InflowForecaster(model_path=fake_model, metadata_path=tmp_path / "missing_meta.json")

