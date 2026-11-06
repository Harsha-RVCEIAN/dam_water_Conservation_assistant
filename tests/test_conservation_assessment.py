"""Tests for Layer B reservoir condition assessment, flow anomaly review, and shortage risk."""

import pandas as pd
import pytest

from src.models.conservation_assessment import (
    assess_reservoir_condition,
    detect_abnormal_flow_patterns,
    assess_water_shortage_risk,
)


def test_assess_reservoir_condition_declining_and_watch():
    # 7 daily records with declining storage
    dates = pd.date_range("2024-06-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0],  # 16 TMC -> in WATCH band (<= 20)
        "inflow_cusecs": [2000.0] * 7,
        "outflow_cusecs": [1000.0] * 7,
        "withdrawal_cusecs": [2500.0] * 7,
    })

    res = assess_reservoir_condition(df, as_of_date="2024-06-07")
    assert res.condition == "WATCH"
    assert res.storage_trend == "DECLINING"
    assert res.storage_value_tmc == 16.0
    assert res.data_freshness == "HISTORICAL_ARCHIVE"  # 2024 date is historical


def test_assess_reservoir_condition_insufficient_data():
    # Only 2 rows when 5 required
    df = pd.DataFrame({
        "calendar_date": pd.date_range("2024-01-01", periods=2, freq="D"),
        "live_above_cill_tmc": [25.0, 24.0],
    })
    res = assess_reservoir_condition(df)
    assert res.condition == "INSUFFICIENT_DATA"


def test_detect_abnormal_flow_patterns_mad():
    # Normal flow is 1000 cusecs, then one extreme spike to 15000 cusecs
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    inflow = [1000.0] * 19 + [15000.0]
    df = pd.DataFrame({
        "calendar_date": dates,
        "inflow_cusecs": inflow,
        "outflow_cusecs": [500.0] * 20,
        "withdrawal_cusecs": [500.0] * 20,
    })
    flags = detect_abnormal_flow_patterns(df)
    inflow_flags = [f for f in flags if f.metric == "inflow_cusecs"]
    assert len(inflow_flags) >= 1
    assert inflow_flags[-1].flag_type == "UNUSUAL_FLOW"
    assert inflow_flags[-1].observed_value == 15000.0


def test_assess_water_shortage_risk_high_when_low_storage_and_imbalance():
    # Setup condition assessment with low storage (HIGH_RISK) and withdrawal > inflow
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0],  # 9 TMC -> <= 10 TMC is HIGH_RISK
        "inflow_cusecs": [1000.0] * 7,
        "outflow_cusecs": [500.0] * 7,
        "withdrawal_cusecs": [2500.0] * 7,  # 2.5x inflow
    })
    cond = assess_reservoir_condition(df)
    risk = assess_water_shortage_risk(cond, inflow_forecast_cfs=800.0)

    assert risk.risk_level == "HIGH"
    assert len(risk.risk_reasons) >= 3
    assert risk.is_forecast_based is True
    assert "contingency scenarios" in risk.suggested_review_action

