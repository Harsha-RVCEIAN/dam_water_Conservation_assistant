"""Tests for Layer B rule-based conservation recommender."""

from src.models.conservation_assessment import (
    FlowAnomalyFlag,
    ReservoirConditionAssessment,
    ShortageRiskAssessment,
)
from src.models.conservation_recommender import generate_conservation_recommendations


def test_recommender_triggers_withdrawal_evaluation_and_storage_decline():
    cond = ReservoirConditionAssessment(
        condition="WATCH",
        observation_date="2024-06-07",
        storage_measurement_date="2024-06-07",
        data_age_days=0,
        data_freshness="FRESH",
        storage_value_tmc=18.5,
        storage_basis="live_above_cill_tmc",
        storage_trend="DECLINING",
        inflow_trend="DECLINING",
        river_outflow_trend="STEADY",
        canal_withdrawal_trend="STEADY",
        evidence={
            "mean_inflow_cusecs": 1200.0,
            "mean_withdrawal_cusecs": 2200.0,
            "recent_window_days": 7,
        },
    )
    risk = ShortageRiskAssessment(
        risk_level="MEDIUM",
        risk_reasons=["Storage declining", "Withdrawal > inflow"],
        supporting_metrics={"score": 2},
        suggested_review_action="Evaluate reduction scenarios",
    )
    anom = [
        FlowAnomalyFlag(
            metric="withdrawal_cusecs",
            date="2024-06-07",
            observed_value=3500.0,
            baseline_median=2200.0,
            threshold_mad=600.0,
            flag_type="UNUSUAL_FLOW",
            reason="Unusual surge in canal withdrawal",
        )
    ]

    recs = generate_conservation_recommendations(
        condition=cond,
        risk=risk,
        anomalies=anom,
        water_balance_status="NORMAL",
    )

    rec_ids = [r.recommendation_id for r in recs]
    assert "REC_STORAGE_DECLINE" in rec_ids
    assert "REC_WITHDRAWAL_EVALUATION" in rec_ids
    assert "REC_UNUSUAL_WITHDRAWAL" in rec_ids

    # Assert no operational gate commands or physical leakage claims
    for r in recs:
        assert "not an official" in r.limitations[0].lower() or "advisory" in r.limitations[0].lower()
        assert "leakage" not in r.title.lower()


def test_recommender_stale_or_historical_suppresses_operational_advice():
    cond = ReservoirConditionAssessment(
        condition="WATCH",
        observation_date="2024-06-07",
        storage_measurement_date="2024-06-07",
        data_age_days=100,
        data_freshness="HISTORICAL_ARCHIVE",
        storage_value_tmc=18.5,
        storage_basis="live_above_cill_tmc",
        storage_trend="DECLINING",
        inflow_trend="DECLINING",
        river_outflow_trend="STEADY",
        canal_withdrawal_trend="STEADY",
        evidence={
            "mean_inflow_cusecs": 1200.0,
            "mean_withdrawal_cusecs": 2200.0,
            "recent_window_days": 7,
        },
    )
    risk = ShortageRiskAssessment(risk_level="HISTORICAL_ONLY")
    recs = generate_conservation_recommendations(condition=cond, risk=risk)
    assert len(recs) == 1
    assert recs[0].recommendation_id == "REC_DATA_FRESHNESS"


def test_recommender_water_balance_discrepancy_rules():
    cond = ReservoirConditionAssessment(
        condition="NORMAL",
        observation_date="2024-06-07",
        storage_measurement_date="2024-06-07",
        data_age_days=0,
        data_freshness="FRESH",
        storage_value_tmc=25.0,
        storage_basis="live_above_cill_tmc",
        storage_trend="STEADY",
        inflow_trend="STEADY",
        river_outflow_trend="STEADY",
        canal_withdrawal_trend="STEADY",
        evidence={"recent_window_days": 7},
    )
    risk = ShortageRiskAssessment(risk_level="LOW")

    # 1. Complete status with discrepancy >= 2.0 triggers REC_WATER_BALANCE_REVIEW
    recs_complete = generate_conservation_recommendations(
        condition=cond,
        risk=risk,
        water_balance_status="COMPLETE",
        water_balance_discrepancy_tmc=2.5,
    )
    rec_ids = [r.recommendation_id for r in recs_complete]
    assert "REC_WATER_BALANCE_REVIEW" in rec_ids

    # 2. Incomplete status must NOT trigger REC_WATER_BALANCE_REVIEW even with large discrepancy
    recs_incomplete = generate_conservation_recommendations(
        condition=cond,
        risk=risk,
        water_balance_status="INCOMPLETE",
        water_balance_discrepancy_tmc=5.0,
    )
    rec_ids_incomp = [r.recommendation_id for r in recs_incomplete]
    assert "REC_WATER_BALANCE_REVIEW" not in rec_ids_incomp

    # 3. UNVERIFIED_EVAPORATION must NOT trigger REC_WATER_BALANCE_REVIEW
    recs_unverified = generate_conservation_recommendations(
        condition=cond,
        risk=risk,
        water_balance_status="UNVERIFIED_EVAPORATION",
        water_balance_discrepancy_tmc=3.0,
    )
    rec_ids_unver = [r.recommendation_id for r in recs_unverified]
    assert "REC_WATER_BALANCE_REVIEW" not in rec_ids_unver


def test_recommender_invalid_anomalies_and_deduplication():
    import numpy as np

    cond = ReservoirConditionAssessment(
        condition="NORMAL",
        observation_date="2024-06-07",
        storage_measurement_date="2024-06-07",
        data_age_days=0,
        data_freshness="FRESH",
        storage_value_tmc=25.0,
        storage_basis="live_above_cill_tmc",
        storage_trend="STEADY",
        inflow_trend="STEADY",
        river_outflow_trend="STEADY",
        canal_withdrawal_trend="STEADY",
        evidence={"recent_window_days": 7},
    )
    risk = ShortageRiskAssessment(risk_level="LOW")

    # Invalid anomaly flags (negative, NaN, non-finite)
    invalid_anoms = [
        FlowAnomalyFlag(
            metric="withdrawal_cusecs",
            date="2024-06-07",
            observed_value=-100.0,
            baseline_median=2000.0,
            threshold_mad=500.0,
            flag_type="UNUSUAL_FLOW",
            reason="Negative value",
        ),
        FlowAnomalyFlag(
            metric="withdrawal_cusecs",
            date="2024-06-07",
            observed_value=np.nan,
            baseline_median=2000.0,
            threshold_mad=500.0,
            flag_type="UNUSUAL_FLOW",
            reason="NaN value",
        ),
    ]

    recs = generate_conservation_recommendations(
        condition=cond,
        risk=risk,
        anomalies=invalid_anoms,
    )
    # Invalid anomalies must be excluded from producing anomaly recommendations
    assert "REC_UNUSUAL_WITHDRAWAL" not in [r.recommendation_id for r in recs]

    # Check deduplication and priority ordering
    prio_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    prios = [prio_order.get(r.priority, 3) for r in recs]
    assert prios == sorted(prios)
    rec_ids = [r.recommendation_id for r in recs]
    assert len(rec_ids) == len(set(rec_ids))



