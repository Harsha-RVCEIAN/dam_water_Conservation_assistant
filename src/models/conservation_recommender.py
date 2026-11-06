"""Deterministic rule-based conservation recommendation engine.

Layer B academic decision-support prototype.
Generates explainable, condition-triggered review suggestions without gate-control,
official allocation, or leakage claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.models.conservation_assessment import (
    FlowAnomalyFlag,
    ReservoirConditionAssessment,
    ShortageRiskAssessment,
)


@dataclass
class ConservationRecommendation:
    recommendation_id: str
    title: str
    priority: str  # HIGH, MEDIUM, LOW
    trigger: str
    evidence: Dict[str, Any]
    explanation: str
    suggested_action: str
    estimated_benefit: Optional[float] = None
    confidence: str = "PROVISIONAL"
    limitations: List[str] = field(default_factory=list)


def generate_conservation_recommendations(
    condition: ReservoirConditionAssessment,
    risk: ShortageRiskAssessment,
    anomalies: Optional[List[FlowAnomalyFlag]] = None,
    water_balance_status: Optional[str] = None,
    water_balance_discrepancy_tmc: Optional[float] = None,
) -> List[ConservationRecommendation]:
    """Generate prioritized, deduplicated conservation review recommendations."""
    recommendations: List[ConservationRecommendation] = []
    if anomalies is None:
        anomalies = []

    standard_limitation = (
        "Advisory recommendation for operational review only. "
        "Does not constitute an official water-allocation directive or gate-operation command."
    )

    # Rule 1: Stale or Incomplete Data
    if condition.condition == "INSUFFICIENT_DATA" or condition.data_freshness in ["STALE", "HISTORICAL_ARCHIVE"]:
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_DATA_FRESHNESS",
                title="Verify and Refresh Observation Telemetry",
                priority="HIGH" if condition.condition == "INSUFFICIENT_DATA" else "MEDIUM",
                trigger="Observation data is stale or incomplete",
                evidence={
                    "observation_date": condition.observation_date,
                    "data_age_days": condition.data_age_days,
                    "freshness": condition.data_freshness,
                },
                explanation=(
                    f"Telemetry records end at {condition.observation_date} ({condition.data_age_days} days old). "
                    "Operational decisions should not rely on unverified historical conditions."
                ),
                suggested_action="Refresh reservoir monitoring records before initiating revised conservation measures.",
                confidence="HIGH",
                limitations=[standard_limitation],
            )
        )
        # Suppress aggressive operational recommendations if data is completely insufficient
        if condition.condition == "INSUFFICIENT_DATA":
            return recommendations

    # Rule 2: Persistent Storage Decline
    if condition.storage_trend == "DECLINING":
        priority = "HIGH" if condition.condition == "HIGH_RISK" else "MEDIUM"
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_STORAGE_DECLINE",
                title="Review Consecutive Storage Depletion Trend",
                priority=priority,
                trigger="Storage declined across the recent observation window",
                evidence={
                    "storage_tmc": condition.storage_value_tmc,
                    "storage_trend": condition.storage_trend,
                    "condition": condition.condition,
                },
                explanation=(
                    f"Live storage ({condition.storage_value_tmc:.2f} TMC) is exhibiting a declining trend "
                    f"in the {condition.evidence.get('recent_window_days', 7)}-day evaluation interval."
                ),
                suggested_action="Review storage trajectory and verify incoming tributary flows across monitoring stations.",
                confidence="MEDIUM",
                limitations=[standard_limitation, "Do not prescribe downstream release cuts without multi-agency review."],
            )
        )

    # Rule 3: Low Inflow / High Withdrawal Imbalance
    mean_inflow = condition.evidence.get("mean_inflow_cusecs")
    mean_with = condition.evidence.get("mean_withdrawal_cusecs")
    if mean_inflow is not None and mean_with is not None and mean_inflow > 0:
        ratio = mean_with / mean_inflow
        if ratio >= 1.2:
            priority = "HIGH" if ratio >= 1.8 or risk.risk_level == "HIGH" else "MEDIUM"
            recommendations.append(
                ConservationRecommendation(
                    recommendation_id="REC_WITHDRAWAL_EVALUATION",
                    title="Evaluate Canal Withdrawal Reduction Scenarios",
                    priority=priority,
                    trigger="Canal withdrawal exceeds inflow across recent window",
                    evidence={
                        "mean_inflow_cusecs": round(mean_inflow, 1),
                        "mean_withdrawal_cusecs": round(mean_with, 1),
                        "withdrawal_to_inflow_ratio": round(ratio, 2),
                    },
                    explanation=(
                        f"Mean canal withdrawal ({mean_with:.0f} cusecs) is {ratio:.2f}x higher than mean inflow "
                        f"({mean_inflow:.0f} cusecs). Continuing this pattern accelerates storage depletion."
                    ),
                    suggested_action="Run hypothetical 5% to 20% canal rationing scenarios to model conserved volume.",
                    confidence="MEDIUM",
                    limitations=[
                        standard_limitation,
                        "Voluntary reduction is not automatically approved and must honor established irrigation schedules.",
                    ],
                )
            )

    # Rule 4: Unusual Flow Flags in Withdrawal or Outflow
    withdrawal_anomalies = [a for a in anomalies if "withdrawal" in a.metric and a.flag_type == "UNUSUAL_FLOW"]
    if withdrawal_anomalies:
        latest_anom = withdrawal_anomalies[-1]
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_UNUSUAL_WITHDRAWAL",
                title="Review Unusual Canal Withdrawal Measurement",
                priority="MEDIUM",
                trigger="Canal withdrawal flagged by robust MAD anomaly filter",
                evidence={
                    "metric": latest_anom.metric,
                    "date": latest_anom.date,
                    "observed_cusecs": latest_anom.observed_value,
                    "baseline_median_cusecs": latest_anom.baseline_median,
                },
                explanation=(
                    f"Observed withdrawal of {latest_anom.observed_value:.1f} cusecs on {latest_anom.date} "
                    f"deviates significantly from recent baseline ({latest_anom.baseline_median:.1f} cusecs)."
                ),
                suggested_action="Inspect canal gauge logs and check whether rotational irrigation scheduling caused the spike.",
                confidence="PROVISIONAL",
                limitations=[standard_limitation, "Statistical flag is not proof of canal wastage or operational error."],
            )
        )

    outflow_anomalies = [a for a in anomalies if "outflow" in a.metric and a.flag_type == "UNUSUAL_FLOW"]
    if outflow_anomalies:
        latest_out = outflow_anomalies[-1]
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_UNUSUAL_OUTFLOW",
                title="Review Unusual River Outflow Observation",
                priority="LOW",
                trigger="River outflow flagged by robust MAD anomaly filter",
                evidence={
                    "metric": latest_out.metric,
                    "date": latest_out.date,
                    "observed_cusecs": latest_out.observed_value,
                    "baseline_median_cusecs": latest_out.baseline_median,
                },
                explanation=(
                    f"River outflow ({latest_out.observed_value:.1f} cusecs) on {latest_out.date} "
                    f"exceeds standard rolling deviation."
                ),
                suggested_action="Verify downstream river flow logs against spillway/sluice gate operation records.",
                confidence="PROVISIONAL",
                limitations=[standard_limitation, "Do not instruct gate operations; review historical records only."],
            )
        )

    # Rule 5: Water-Balance Discrepancy
    if water_balance_status in ["MODERATE_DISCREPANCY", "HIGH_DISCREPANCY"] or (
        water_balance_discrepancy_tmc is not None and abs(water_balance_discrepancy_tmc) >= 2.0
    ):
        disc_val = water_balance_discrepancy_tmc if water_balance_discrepancy_tmc is not None else 0.0
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_WATER_BALANCE_REVIEW",
                title="Review Water-Balance Accounting Discrepancy",
                priority="LOW",
                trigger="Daily mass balance residual exceeds advisory screening threshold",
                evidence={
                    "status": water_balance_status,
                    "discrepancy_tmc": round(disc_val, 3),
                },
                explanation=(
                    f"Water-balance diagnostic shows a discrepancy of {disc_val:+.3f} TMC. "
                    "This indicates unmodeled terms such as localized rainfall or evaporation uncertainty."
                ),
                suggested_action="Audit stage-discharge ratings, evaporation estimates, and measurement timestamps.",
                confidence="PROVISIONAL",
                limitations=[
                    standard_limitation,
                    "Water-balance discrepancies are mass-accounting residuals; they DO NOT confirm physical leakage.",
                ],
            )
        )

    # Sort by priority order: HIGH, MEDIUM, LOW
    prio_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda r: prio_map.get(r.priority, 3))
    return recommendations

