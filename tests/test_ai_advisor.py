"""Tests for Google Gemini AI Conservation Advisor and deterministic fallback."""

from unittest.mock import MagicMock, patch
import pytest

from src.models.conservation_assessment import (
    FlowAnomalyFlag,
    ReservoirConditionAssessment,
    ShortageRiskAssessment,
)
from src.models.conservation_recommender import ConservationRecommendation
from src.models.water_budget import ConservationScenarioResult
from src.services.ai_advisor import (
    _build_context_summary,
    _generate_deterministic_briefing,
    generate_ai_conservation_briefing,
    get_gemini_api_key,
    is_gemini_configured,
)


@pytest.fixture
def sample_condition():
    return ReservoirConditionAssessment(
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


@pytest.fixture
def sample_risk():
    return ShortageRiskAssessment(
        risk_level="HISTORICAL_ONLY",
        risk_reasons=["Observations are historical archive data"],
        supporting_metrics={"score": 2},
        suggested_review_action="Obtain updated reservoir observations",
    )


@pytest.fixture
def sample_recommendations():
    return [
        ConservationRecommendation(
            recommendation_id="REC_DATA_FRESHNESS",
            title="Verify Historical Observations",
            priority="MEDIUM",
            trigger="Available observations are historical archive",
            evidence={},
            explanation="Data is historical.",
            suggested_action="Obtain latest observations before live actions.",
        )
    ]


def test_deterministic_fallback_when_api_key_blank(
    sample_condition, sample_risk, sample_recommendations
):
    with patch("src.services.ai_advisor.get_gemini_api_key", return_value=""):
        res = generate_ai_conservation_briefing(
            condition=sample_condition,
            risk=sample_risk,
            recommendations=sample_recommendations,
        )
        assert res["is_ai_generated"] is False
        assert res["status"] == "deterministic_fallback"
        assert "Executive Hydrological Briefing" in res["briefing_markdown"]
        assert "leakage" in res["briefing_markdown"].lower() or "not proof of" in res["briefing_markdown"].lower()
        assert "historical" in res["briefing_markdown"].lower()


def test_context_summary_preserves_safety_disclaimers(
    sample_condition, sample_risk, sample_recommendations
):
    summary = _build_context_summary(
        condition=sample_condition,
        risk=sample_risk,
        recommendations=sample_recommendations,
        diagnostic_residual_tmc=0.15,
    )
    assert "DO NOT prove physical dam leakage" in summary
    assert "academic decision-support prototype" in summary
    assert "NOT live dam SCADA telemetry" in summary
    assert "Evaporation is excluded" in summary


def test_mocked_gemini_api_success(
    sample_condition, sample_risk, sample_recommendations
):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "### AI Executive Briefing\nReservoir storage is in WATCH condition."}]
                }
            }
        ]
    }

    with patch("src.services.ai_advisor.get_gemini_api_key", return_value="fake-key-123"):
        with patch("requests.post", return_value=fake_response):
            res = generate_ai_conservation_briefing(
                condition=sample_condition,
                risk=sample_risk,
                recommendations=sample_recommendations,
            )
            assert res["is_ai_generated"] is True
            assert res["status"] == "success"
            assert "AI Executive Briefing" in res["briefing_markdown"]


def test_gemini_api_network_failure_falls_back(
    sample_condition, sample_risk, sample_recommendations
):
    with patch("src.services.ai_advisor.get_gemini_api_key", return_value="fake-key-123"):
        with patch("requests.post", side_effect=Exception("Connection timeout")):
            res = generate_ai_conservation_briefing(
                condition=sample_condition,
                risk=sample_risk,
                recommendations=sample_recommendations,
            )
            assert res["is_ai_generated"] is False
            assert res["status"] == "deterministic_fallback"

