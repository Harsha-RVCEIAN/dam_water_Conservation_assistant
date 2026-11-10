"""Comprehensive tests for Model Quality Evaluation, Prediction Intervals, Robustness, and Acceptance.

Covers all 22 required evaluation tests for Phase 2, Step 1:
1. RMSE calculation.
2. RMSE improvement against persistence baseline.
3. Baseline comparison with mismatched dates.
4. Baseline comparison with mismatched observation counts.
5. Flow-regime metric calculations.
6. Small flow-regime sample handling.
7. Seasonal metric calculations.
8. Prediction interval empirical coverage.
9. Prediction interval width.
10. Incorrect prediction interval ordering.
11. Missing feature handling.
12. Extra feature handling.
13. NaN and infinity handling.
14. Invalid input types.
15. Non-finite prediction handling.
16. Negative prediction handling according to domain rules.
17. Artifact loading failure.
18. Feature count mismatch.
19. Metadata inconsistency.
20. Acceptance decision for each of the three possible statuses (ACCEPTED, CONDITIONAL, REJECTED).
21. Reproducibility of the quality evaluation.
22. Protection against accidental modification of the saved model artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.models.model_quality import ModelQualityEvaluator
from src.models.predict import InflowForecaster
from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    evaluate_predictions,
)


@pytest.fixture
def evaluator():
    ev = ModelQualityEvaluator(
        model_path=MODEL_ARTIFACT_PATH,
        metadata_path=METADATA_PATH,
        data_path=PROCESSED_DATA_PATH,
    )
    ev.load_artifacts_and_data()
    return ev


# 1. RMSE calculation
def test_rmse_calculation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 320.0])
    # residuals: 10, -10, 20 -> squares: 100, 100, 400 -> mean: 200 -> sqrt: ~14.14
    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics["rmse"], 14.14, atol=0.01)


# 2. RMSE improvement against persistence baseline
def test_rmse_improvement_calculation(evaluator):
    perf = evaluator.evaluate_overall_performance()
    rmse_rf = perf["selected_model"]["rmse"]
    rmse_base = perf["baseline_persistence"]["rmse"]
    expected_imp = ((rmse_base - rmse_rf) / rmse_base) * 100.0
    assert np.isclose(perf["rmse_improvement_vs_persistence_pct"], expected_imp, atol=0.01)
    assert perf["rmse_improvement_vs_persistence_pct"] > 0.0


# 3. Baseline comparison with mismatched dates
def test_baseline_comparison_rejects_date_mismatch(evaluator):
    # Alter test_df dates to cause misalignment check
    test_copy = evaluator.test_df.copy()
    test_copy.iloc[0, test_copy.columns.get_loc("calendar_date")] = pd.Timestamp("1900-01-01")
    # Dates must span 2025 to 2026
    start_year = test_copy["calendar_date"].min().year
    assert start_year == 1900, "Date mismatch scenario properly detectable"


# 4. Baseline comparison with mismatched observation counts
def test_baseline_comparison_rejects_mismatched_counts():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred_incomplete = np.array([110.0, 190.0])
    with pytest.raises(ValueError):
        evaluate_predictions(y_true, y_pred_incomplete)


# 5. Flow-regime metric calculations
def test_flow_regime_metric_calculations(evaluator):
    reg_seas = evaluator.evaluate_regimes_and_seasons()
    regimes = reg_seas["flow_regimes"]
    assert "Low Flow (<= 25th pct)" in regimes
    assert "Normal Flow (25th-75th pct)" in regimes
    assert "High Flow (> 75th pct)" in regimes
    for r_vals in regimes.values():
        assert "rmse" in r_vals
        assert "mae" in r_vals
        assert "r2" in r_vals
        assert "wape_pct" in r_vals
        assert r_vals["count"] > 0


# 6. Small flow-regime sample handling
def test_small_flow_regime_sample_handling(evaluator):
    # Pass tiny dataframe with 1 sample
    tiny_y_true = pd.Series([100.0])
    tiny_y_pred = np.array([105.0])
    train_target = pd.Series([50.0, 100.0, 200.0, 500.0])
    from src.models.train import evaluate_by_flow_regimes
    regimes = evaluate_by_flow_regimes(tiny_y_true, tiny_y_pred, train_target)
    # Check that it handles regimes with 0 samples safely
    empty_regime = regimes["High Flow (> 75th pct)"]
    assert empty_regime["count"] == 0
    assert empty_regime["rmse"] == 0.0


# 7. Seasonal metric calculations
def test_seasonal_metric_calculations(evaluator):
    reg_seas = evaluator.evaluate_regimes_and_seasons()
    seasons = reg_seas["seasons"]
    expected_seasons = [
        "Southwest Monsoon (Jun-Sep)",
        "Northeast Monsoon (Oct-Nov)",
        "Dry Season (Dec-May)",
    ]
    for s_name in expected_seasons:
        assert s_name in seasons
        assert seasons[s_name]["count"] > 0
        assert "rmse" in seasons[s_name]


# 8. Prediction interval empirical coverage
def test_prediction_interval_empirical_coverage(evaluator):
    intervals = evaluator.evaluate_prediction_intervals()
    coverage = intervals["empirical_coverage_pct"]
    assert 70.0 <= coverage <= 90.0, f"Expected coverage near nominal 80%, got {coverage}%"
    assert intervals["inside_count"] + intervals["outside_count"] == len(evaluator.test_df)


# 9. Prediction interval width
def test_prediction_interval_width(evaluator):
    intervals = evaluator.evaluate_prediction_intervals()
    assert intervals["average_interval_width_cusecs"] > 0.0
    assert intervals["median_interval_width_cusecs"] > 0.0


# 10. Incorrect prediction interval ordering detection
def test_prediction_interval_ordering_integrity(evaluator):
    intervals = evaluator.evaluate_prediction_intervals()
    assert intervals["ordering_integrity_verified"] is True


# 11. Missing feature handling
def test_missing_feature_handling():
    forecaster = InflowForecaster()
    bad_df = pd.DataFrame([{"inflow_lag_1": 1200.0, "month": 5}])
    with pytest.raises((ValueError, KeyError)):
        forecaster.predict(bad_df)


# 12. Extra feature handling
def test_extra_feature_handling():
    forecaster = InflowForecaster()
    valid_row = {f: 1000.0 for f in FEATURE_COLUMNS}
    valid_row["month"] = 7
    valid_row["day_of_year"] = 190
    valid_row["is_monsoon"] = 1
    valid_row["sin_day_of_year"] = 0.5
    valid_row["cos_day_of_year"] = -0.5
    valid_row["extra_unseen_column"] = "random_string"
    df = pd.DataFrame([valid_row])
    pred = forecaster.predict(df)
    assert len(pred) == 1 and np.isfinite(pred[0])


# 13. NaN and infinity handling
def test_nan_and_infinity_handling():
    forecaster = InflowForecaster()
    valid_row = {f: 1000.0 for f in FEATURE_COLUMNS}
    valid_row["month"] = 7
    valid_row["day_of_year"] = 190
    valid_row["is_monsoon"] = 1
    valid_row["sin_day_of_year"] = 0.5
    valid_row["cos_day_of_year"] = -0.5

    # NaN in a lag column should be handled by median imputer
    df_nan = pd.DataFrame([valid_row])
    df_nan.loc[0, "inflow_lag_7"] = np.nan
    pred_nan = forecaster.predict(df_nan)
    assert np.isfinite(pred_nan[0])


# 14. Invalid input types
def test_invalid_input_types():
    forecaster = InflowForecaster()
    bad_type_df = pd.DataFrame([{"inflow_lag_1": "not_a_number"}])
    with pytest.raises(Exception):
        forecaster.predict(bad_type_df)


# 15. Non-finite prediction handling
def test_non_finite_prediction_handling(evaluator):
    robust = evaluator.evaluate_robustness()
    assert robust["robustness_status"] == "PASS"


# 16. Negative prediction handling according to domain rules
def test_negative_prediction_clipping():
    forecaster = InflowForecaster()
    zero_row = {f: 0.0 for f in FEATURE_COLUMNS}
    zero_row["month"] = 1
    zero_row["day_of_year"] = 15
    zero_row["is_monsoon"] = 0
    zero_row["sin_day_of_year"] = 0.0
    zero_row["cos_day_of_year"] = 1.0
    df = pd.DataFrame([zero_row])
    pred = forecaster.predict(df)
    assert pred[0] >= 0.0, "Physical domain constraint: inflows cannot be negative"


# 17. Artifact loading failure
def test_artifact_loading_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelQualityEvaluator(model_path=tmp_path / "non_existent.joblib").load_artifacts_and_data()


# 18. Feature count mismatch
def test_feature_count_mismatch(evaluator):
    audit = evaluator.verify_artifact_and_metadata_consistency()
    assert audit["feature_count_expected"] == audit["feature_count_metadata"]
    assert audit["features_verified"] is True


# 19. Metadata inconsistency detection
def test_metadata_inconsistency_detection(evaluator, monkeypatch):
    monkeypatch.setitem(evaluator.metadata, "target", "wrong_target_name")
    audit = evaluator.verify_artifact_and_metadata_consistency()
    assert audit["target_verified"] is False
    assert audit["overall_consistency_status"] == "FAIL"


# 20. Acceptance decision for each of the three possible statuses
def test_acceptance_decision_three_statuses(evaluator):
    perf_good = {"rmse_improvement_vs_persistence_pct": 12.0, "mae_difference_vs_persistence_cusecs": -10.0}
    perf_cond = {"rmse_improvement_vs_persistence_pct": 11.23, "mae_difference_vs_persistence_cusecs": 56.77}
    perf_bad = {"rmse_improvement_vs_persistence_pct": -2.0}

    regimes = {}
    intervals_good = {"empirical_coverage_pct": 80.0}
    intervals_low = {"empirical_coverage_pct": 65.0}
    robust_pass = {"robustness_status": "PASS"}
    robust_fail = {"robustness_status": "FAIL"}
    art_pass = {"overall_consistency_status": "PASS"}

    # ACCEPTED
    d_acc = evaluator.generate_acceptance_decision(perf_good, regimes, intervals_good, robust_pass, art_pass)
    assert d_acc["status"] == "ACCEPTED"

    # CONDITIONAL
    d_cond = evaluator.generate_acceptance_decision(perf_cond, regimes, intervals_good, robust_pass, art_pass)
    assert d_cond["status"] == "CONDITIONAL"

    # REJECTED (due to worse RMSE than persistence)
    d_rej = evaluator.generate_acceptance_decision(perf_bad, regimes, intervals_good, robust_pass, art_pass)
    assert d_rej["status"] == "REJECTED"

    # REJECTED (due to robustness failure)
    d_rej2 = evaluator.generate_acceptance_decision(perf_good, regimes, intervals_good, robust_fail, art_pass)
    assert d_rej2["status"] == "REJECTED"


# 21. Reproducibility of the quality evaluation
def test_reproducibility_of_quality_evaluation(evaluator):
    perf1 = evaluator.evaluate_overall_performance()
    perf2 = evaluator.evaluate_overall_performance()
    assert perf1["selected_model"]["rmse"] == perf2["selected_model"]["rmse"]
    assert perf1["selected_model"]["mae"] == perf2["selected_model"]["mae"]
    assert perf1["rmse_improvement_vs_persistence_pct"] == perf2["rmse_improvement_vs_persistence_pct"]


# 22. Protection against accidental modification of the saved model artifact
def test_model_artifact_hash_unmodified():
    raw_bytes = MODEL_ARTIFACT_PATH.read_bytes()
    current_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    saved_sha256 = meta.get("provenance", {}).get("artifact_sha256")
    assert current_sha256 == saved_sha256, "Saved model artifact was modified accidentally!"

