"""Tests for Layer B reservoir condition assessment, flow anomaly review, and shortage risk."""

import numpy as np
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
    assert res.storage_measurement_date == "2024-06-07"
    assert res.observation_date == "2024-06-07"
    assert res.data_freshness == "HISTORICAL_ARCHIVE"


def test_assess_reservoir_condition_insufficient_data():
    # Only 2 rows when 5 required
    df = pd.DataFrame({
        "calendar_date": pd.date_range("2024-01-01", periods=2, freq="D"),
        "live_above_cill_tmc": [25.0, 24.0],
    })
    res = assess_reservoir_condition(df)
    assert res.condition == "INSUFFICIENT_DATA"


def test_assess_reservoir_condition_enforces_five_valid_storage_readings():
    # 7 rows, but only 3 valid storage values (below configured min_valid_observations of 5)
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [25.0, np.nan, np.nan, 24.0, np.nan, np.nan, 23.0],
        "inflow_cusecs": [1000.0] * 7,
        "outflow_cusecs": [500.0] * 7,
        "withdrawal_cusecs": [500.0] * 7,
    })
    res = assess_reservoir_condition(df)
    assert res.condition == "INSUFFICIENT_DATA"
    assert "minimum 5 required" in res.limitations[-1]


def test_storage_measurement_date_distinguishes_missing_latest_storage():
    # 7 rows, 5 valid storage values, but the very latest row (day 7) is missing storage
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [25.0, 24.5, 24.0, 23.5, 23.0, 22.5, np.nan],
        "inflow_cusecs": [1000.0] * 7,
        "outflow_cusecs": [500.0] * 7,
        "withdrawal_cusecs": [500.0] * 7,
    })
    res = assess_reservoir_condition(df)
    assert res.condition == "NORMAL"  # 22.5 TMC is between 20 and 35
    assert res.observation_date == "2024-01-07"
    assert res.storage_measurement_date == "2024-01-06"
    assert any("Latest storage value is from" in l for l in res.limitations)


def test_assess_reservoir_condition_surplus():
    # Storage >= 35 TMC triggers SURPLUS condition
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0],
        "inflow_cusecs": [5000.0] * 7,
        "outflow_cusecs": [2000.0] * 7,
        "withdrawal_cusecs": [1000.0] * 7,
    })
    res = assess_reservoir_condition(df)
    assert res.condition == "SURPLUS"
    assert res.storage_trend == "RISING"


def test_detect_abnormal_flow_patterns_mad_uncontaminated():
    # Normal historical flow is 1000 cusecs for 19 days, then one extreme spike to 15000 cusecs
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
    # Historical baseline median is strictly from shifted prior window (1000.0)
    assert inflow_flags[-1].baseline_median == 1000.0


def test_assess_water_shortage_risk_consecutive_decline():
    # 7 rows with consecutive declining storage
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({
        "calendar_date": dates,
        "live_above_cill_tmc": [12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0],  # 6 consecutive declines
        "inflow_cusecs": [1000.0] * 7,
        "outflow_cusecs": [500.0] * 7,
        "withdrawal_cusecs": [2500.0] * 7,
    })
    cond = assess_reservoir_condition(df)
    risk = assess_water_shortage_risk(cond, inflow_forecast_cfs=800.0)

    assert risk.risk_level == "HIGH"
    assert any("consecutive observations" in r for r in risk.risk_reasons)
    assert risk.supporting_metrics["consecutive_storage_decline_days"] >= 3
