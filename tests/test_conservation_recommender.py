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
    assert "REC_DATA_FRESHNESS" in rec_ids

    # Assert no operational gate commands or physical leakage claims
    for r in recs:
        assert "not an official" in r.limitations[0].lower() or "advisory" in r.limitations[0].lower()
        assert "leakage" not in r.title.lower()


def test_recommender_suppresses_operational_advice_on_insufficient_data():
    cond = ReservoirConditionAssessment(
        condition="INSUFFICIENT_DATA",
        observation_date=None,
        data_age_days=None,
        data_freshness="UNKNOWN",
        storage_value_tmc=None,
        storage_basis="live_above_cill_tmc",
        storage_trend="UNKNOWN",
        inflow_trend="UNKNOWN",
        river_outflow_trend="UNKNOWN",
        canal_withdrawal_trend="UNKNOWN",
    )
    risk = ShortageRiskAssessment(risk_level="INSUFFICIENT_DATA")

    recs = generate_conservation_recommendations(condition=cond, risk=risk)
    assert len(recs) == 1
    assert recs[0].recommendation_id == "REC_DATA_FRESHNESS"

