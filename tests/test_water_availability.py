from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.models.water_availability import (
    MAX_LIVE_STORAGE_TMC,
    WaterAvailabilityError,
    assess_water_alert,
    build_conservation_recommendations,
    calculate_storage_status,
    derive_inflow_references,
    summarize_rainfall,
)


def _storage_history(latest_storage=20.0):
    dates = pd.date_range("2026-10-01", "2026-10-21", freq="D")
    values = np.linspace(25.0, latest_storage, len(dates))
    return pd.DataFrame(
        {"calendar_date": dates, "live_above_cill_tmc": values}
    )


def _rainfall(days=31):
    dates = pd.date_range("2025-12-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Monitoring Date": dates.strftime("%Y-%m-%d"),
            "Rainfall_MM": np.zeros(days),
            "Stations_Reporting": np.full(days, 8),
        }
    )


def test_storage_uses_live_water_and_derived_max_live_capacity():
    history = _storage_history()

    status = calculate_storage_status(history, as_of=date(2026, 10, 21))

    assert status.live_storage_tmc == pytest.approx(20.0)
    assert status.max_live_storage_tmc == pytest.approx(MAX_LIVE_STORAGE_TMC)
    assert status.storage_pct == pytest.approx(20 / MAX_LIVE_STORAGE_TMC * 100)
    assert status.remaining_live_storage_tmc == pytest.approx(
        MAX_LIVE_STORAGE_TMC - 20
    )
    assert status.observation_age_days == 0
    assert status.change_7d_tmc is not None


def test_missing_latest_storage_is_not_replaced_with_zero():
    history = _storage_history()
    history.loc[history.index[-1], "live_above_cill_tmc"] = np.nan

    with pytest.raises(WaterAvailabilityError, match="missing or non-finite"):
        calculate_storage_status(history, as_of=date(2026, 10, 21))


def test_storage_above_capacity_is_flagged_not_clamped():
    history = _storage_history(latest_storage=MAX_LIVE_STORAGE_TMC + 1)

    status = calculate_storage_status(history, as_of=date(2026, 10, 21))
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=None,
        low_inflow_reference_cusecs=None,
    )

    assert status.is_outside_capacity
    assert status.storage_pct > 100
    assert alert.level == "Insufficient data"


def test_zero_rainfall_is_measured_and_summarized_as_zero():
    summary = summarize_rainfall(_rainfall())

    assert summary["latest_mm"] == 0.0
    assert summary["latest_stations_reporting"] == 8
    assert summary["weekly"]["sum_mm"] == 0.0
    assert summary["weekly"]["days_available"] == 7
    assert summary["monthly"]["sum_mm"] == 0.0
    assert summary["monthly"]["days_available"] == 31


def test_missing_rainfall_stays_missing_in_rolling_summaries():
    rainfall = _rainfall()
    rainfall.loc[28, "Rainfall_MM"] = np.nan

    summary = summarize_rainfall(rainfall)

    assert summary["weekly"]["sum_mm"] is None
    assert summary["weekly"]["days_available"] == 6
    assert summary["monthly"]["sum_mm"] is None
    assert summary["monthly"]["days_available"] == 30


@pytest.mark.parametrize(
    ("storage", "expected"),
    [
        (5.0, "Critical Warning"),
        (20.0, "Low Storage Warning"),
        (30.0, "Low Storage Warning"),
        (40.0, "Moderate Warning"),
        (60.0, "Normal"),
    ],
)
def test_storage_alert_levels_use_configured_provisional_thresholds(
    storage, expected
):
    history = _storage_history(latest_storage=storage)
    status = calculate_storage_status(
        history,
        as_of=date(2026, 10, 21),
        max_live_storage_tmc=100.0,
    )

    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=None,
        low_inflow_reference_cusecs=None,
        thresholds={"moderate_pct": 50, "low_pct": 30, "critical_pct": 15},
    )

    assert alert.level == expected


def test_declining_storage_and_low_relative_inflow_trigger_moderate_warning():
    dates = pd.date_range("2026-10-14", "2026-10-21", freq="D")
    history = pd.DataFrame(
        {
            "calendar_date": dates,
            "live_above_cill_tmc": np.arange(30.0, 22.0, -1.0),
        }
    )
    status = calculate_storage_status(
        history, as_of=date(2026, 10, 21), max_live_storage_tmc=40.0
    )

    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=100,
        low_inflow_reference_cusecs=200,
        thresholds={"moderate_pct": 50, "low_pct": 30, "critical_pct": 15},
    )

    assert alert.level == "Moderate Warning"
    assert any("decreased" in reason for reason in alert.reasons)


def test_training_inflow_references_use_historical_quartiles():
    history = pd.DataFrame(
        {
            "calendar_date": pd.date_range("2020-01-01", periods=4),
            "inflow_cusecs": [100.0, 200.0, 300.0, 400.0],
        }
    )

    references = derive_inflow_references(history, train_end="2020-12-31")

    assert references["low"] == pytest.approx(175)
    assert references["median"] == pytest.approx(250)
    assert references["high"] == pytest.approx(325)


def test_alert_rejects_misordered_thresholds():
    status = calculate_storage_status(
        _storage_history(), as_of=date(2026, 10, 21)
    )

    with pytest.raises(WaterAvailabilityError, match="satisfy"):
        assess_water_alert(
            status,
            predicted_inflow_cusecs=None,
            low_inflow_reference_cusecs=None,
            thresholds={"moderate_pct": 20, "low_pct": 30, "critical_pct": 15},
        )


def test_recommendations_are_condition_triggered_and_explain_evidence():
    history = _storage_history(latest_storage=5.0)
    status = calculate_storage_status(
        history, as_of=date(2026, 10, 21), max_live_storage_tmc=45.073
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=50,
        low_inflow_reference_cusecs=200,
        thresholds={"moderate_pct": 50, "low_pct": 30, "critical_pct": 15},
    )

    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=50,
        prediction_date=pd.Timestamp("2026-10-22"),
        low_inflow_reference_cusecs=200,
        high_inflow_reference_cusecs=800,
        latest_outflow_cusecs=500,
        latest_outflow_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_date=pd.Timestamp("2026-10-20"),
        latest_rainfall_mm=0.0,
        reference_date=date(2026, 10, 21),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )

    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.priority == "High"
    assert recommendation.category == "Combined water conservation recommendation"
    assert "approved rules" in recommendation.action
    assert "5.000 TMC" in recommendation.supporting_values
    assert "Predicted next-day inflow 50 cusecs for 2026-10-22" in recommendation.supporting_values
    assert "observed outflow 500 cusecs on 2026-10-21" in recommendation.supporting_values
    assert "rainfall 0.00 mm" in recommendation.supporting_values.lower()
    assert all(item.reason and item.action and item.supporting_values for item in recommendations)


def test_stale_dam_data_and_rainfall_create_data_quality_recommendations():
    status = calculate_storage_status(
        _storage_history(), as_of=date(2026, 11, 3)
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=None,
        low_inflow_reference_cusecs=None,
    )

    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=None,
        low_inflow_reference_cusecs=None,
        latest_rainfall_date=pd.Timestamp("2025-12-31"),
        reference_date=date(2026, 11, 3),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )

    assert any(item.priority == "Data freshness" for item in recommendations)
    assert not recommendations[0].action.startswith("Refresh reservoir observations")
    assert any(
        "current operational decisions" in item.action
        for item in recommendations
        if item.priority == "Data freshness"
    )
    assert any(
        item.title == "Refresh rainfall observations"
        and "was not treated as current" in item.reason
        for item in recommendations
    )


def test_fresh_low_rainfall_and_inflow_trigger_preparedness_recommendation():
    history = pd.DataFrame(
        {
            "calendar_date": pd.date_range("2026-10-01", "2026-10-21"),
            "live_above_cill_tmc": np.full(21, 30.0),
        }
    )
    status = calculate_storage_status(
        history, as_of=date(2026, 10, 21), max_live_storage_tmc=40.0
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=50,
        low_inflow_reference_cusecs=200,
    )

    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=50,
        prediction_date=pd.Timestamp("2026-10-22"),
        low_inflow_reference_cusecs=200,
        latest_rainfall_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_mm=0.0,
        low_rainfall_reference_mm=1.0,
        reference_date=date(2026, 10, 21),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )

    recommendation = recommendations[0]
    assert recommendation.title == "Use water efficiently and monitor future inflow"
    assert "historical 25th-percentile" in recommendation.reason
    assert "0.00 mm" in recommendation.supporting_values
    assert "2026-10-22" in recommendation.supporting_values


@pytest.mark.parametrize(
    (
        "storage",
        "change_7d",
        "prediction",
        "outflow",
        "expected_title",
    ),
    [
        (20, -1, 200, 500, "Review water allocation and monitor reservoir availability."),
        (20, -1, 900, 500, "Review water allocation and monitor reservoir availability."),
        (80, 1, 200, 500, "Use water efficiently and monitor future inflow"),
        (45, -1, 200, 500, "Review water allocation and monitor reservoir availability."),
        (80, 1, 900, 500, "Monitor storage capacity after the predicted inflow"),
        (40, 1, 500, 700, "Review water allocation and monitor reservoir availability."),
    ],
)
def test_prediction_storage_combinations_select_explainable_recommendations(
    storage,
    change_7d,
    prediction,
    outflow,
    expected_title,
):
    dates = pd.date_range("2026-10-14", periods=8)
    history = pd.DataFrame(
        {
            "calendar_date": dates,
            "live_above_cill_tmc": np.linspace(
                storage - change_7d, storage, len(dates)
            ),
        }
    )
    status = calculate_storage_status(
        history, as_of=date(2026, 10, 21), max_live_storage_tmc=100
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=prediction,
        low_inflow_reference_cusecs=300,
        thresholds={"moderate_pct": 50, "low_pct": 30, "critical_pct": 15},
    )
    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=prediction,
        prediction_date=pd.Timestamp("2026-10-22"),
        low_inflow_reference_cusecs=300,
        high_inflow_reference_cusecs=800,
        latest_outflow_cusecs=outflow,
        latest_outflow_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_mm=0,
        reference_date=date(2026, 10, 21),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )

    recommendation = recommendations[0]
    assert recommendation.title == expected_title
    assert recommendation.action
    assert recommendation.reason
    assert "storage" in recommendation.supporting_values.lower()
    assert "Predicted next-day inflow" in recommendation.supporting_values
    assert "Latest observed outflow" in recommendation.supporting_values
    assert "30-day storage change" in recommendation.supporting_values
    assert "2026-10-22" in recommendation.supporting_values


def test_watch_storage_rising_trend_and_normal_prediction_produce_specific_actions():
    dates = pd.date_range("2026-10-14", periods=8)
    history = pd.DataFrame(
        {
            "calendar_date": dates,
            "live_above_cill_tmc": np.linspace(19.106, 19.772, len(dates)),
        }
    )
    status = calculate_storage_status(
        history, as_of=date(2026, 11, 3), max_live_storage_tmc=45.073
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=3264,
        low_inflow_reference_cusecs=373,
        thresholds={"moderate_pct": 50, "low_pct": 30, "critical_pct": 15},
    )

    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=3264,
        prediction_date=pd.Timestamp("2026-10-22"),
        low_inflow_reference_cusecs=373,
        high_inflow_reference_cusecs=5034,
        latest_inflow_cusecs=3452,
        latest_inflow_date=pd.Timestamp("2026-10-21"),
        latest_outflow_cusecs=1366,
        latest_outflow_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_date=pd.Timestamp("2025-12-31"),
        latest_rainfall_mm=0.0,
        reference_date=date(2026, 11, 3),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )
    action_recommendation = recommendations[0]
    actions = action_recommendation.action.lower()

    assert action_recommendation.title == (
        "Review water allocation and monitor reservoir availability."
    )
    assert "43.9%" in action_recommendation.reason
    assert "50.0%" in action_recommendation.reason
    assert "positive" in action_recommendation.reason
    assert "not unusually low or high" in action_recommendation.reason
    assert "review planned water allocations" in actions
    assert "essential needs" in actions
    assert "positive" in actions
    assert "actual measurement" in actions
    assert "refresh forecast inputs" not in actions
    assert any(
        item.priority == "Data freshness"
        and "current operational decisions" in item.action.lower()
        for item in recommendations
    )
    assert any(
        item.title == "Forecast target date has passed"
        and "historical rather than a current" in item.reason
        for item in recommendations
    )


def test_fresh_heavy_rain_and_high_inflow_trigger_monitoring_not_causality():
    status = calculate_storage_status(
        _storage_history(latest_storage=35),
        as_of=date(2026, 10, 21),
        max_live_storage_tmc=45.073,
    )
    alert = assess_water_alert(
        status,
        predicted_inflow_cusecs=900,
        low_inflow_reference_cusecs=200,
    )
    recommendations = build_conservation_recommendations(
        status,
        alert,
        predicted_inflow_cusecs=900,
        prediction_date=pd.Timestamp("2026-10-22"),
        low_inflow_reference_cusecs=200,
        high_inflow_reference_cusecs=800,
        latest_inflow_cusecs=700,
        latest_inflow_date=pd.Timestamp("2026-10-21"),
        latest_outflow_cusecs=500,
        latest_outflow_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_date=pd.Timestamp("2026-10-21"),
        latest_rainfall_mm=100,
        low_rainfall_reference_mm=1,
        high_rainfall_reference_mm=50,
        reference_date=date(2026, 10, 21),
        max_observation_age_days=3,
        max_rainfall_age_days=7,
    )

    recommendation = recommendations[0]
    assert recommendation.title == (
        "Monitor measured reservoir levels after high rainfall and inflow"
    )
    assert "does not establish" in recommendation.reason
    assert "approved reservoir and" in recommendation.action
