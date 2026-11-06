"""Deterministic rule-based conservation recommendation engine.

Layer B academic decision-support prototype.
Generates explainable, condition-triggered review suggestions without gate-control,
official allocation, or leakage claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
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
    """
    Generate deterministic, prioritized conservation-review recommendations.

    Recommendations are advisory only. Historical/stale data suppresses
    condition-dependent operational suggestions, and only a COMPLETE water
    balance may trigger the complete-balance discrepancy rule.
    """
    recommendations: List[ConservationRecommendation] = []
    anomalies = anomalies or []

    standard_limitation = (
        "Advisory recommendation for operational review only. "
        "Does not constitute an official water-allocation directive or "
        "gate-operation command."
    )

    def finite_nonnegative(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(np.isfinite(number) and number >= 0)

    def finite_number(value: Any) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def safe_float(value: Any) -> Optional[float]:
        return float(value) if finite_number(value) else None

    # Data freshness: historical/stale observations must not be presented as
    # current operational evidence. Return after the freshness warning.
    freshness = str(condition.data_freshness).upper()
    insufficient = str(condition.condition).upper() == "INSUFFICIENT_DATA"

    if insufficient or freshness in {"STALE", "HISTORICAL_ARCHIVE"}:
        if insufficient:
            title = "Refresh Reservoir Monitoring Data"
            priority = "HIGH"
            trigger = "Insufficient valid reservoir observations"
            explanation = (
                "The available observations are insufficient for a reliable "
                "reservoir-condition or shortage-risk assessment."
            )
        else:
            title = "Verify and Refresh Historical Observations"
            priority = "MEDIUM"
            trigger = "Available observations are stale or historical"
            explanation = (
                f"Available records end at {condition.observation_date} "
                f"({condition.data_age_days} days old). They describe historical "
                "conditions and should not be treated as current operating status."
            )

        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_DATA_FRESHNESS",
                title=title,
                priority=priority,
                trigger=trigger,
                evidence={
                    "observation_date": condition.observation_date,
                    "data_age_days": condition.data_age_days,
                    "freshness": condition.data_freshness,
                    "condition": condition.condition,
                },
                explanation=explanation,
                suggested_action=(
                    "Obtain and validate the latest reservoir, inflow, outflow, "
                    "and withdrawal observations before making current conservation assessments."
                ),
                confidence="HIGH",
                limitations=[
                    standard_limitation,
                    "Historical records cannot establish present reservoir conditions.",
                ],
            )
        )
        return recommendations

    # Rule 2: Storage decline. Only use finite, nonnegative storage values.
    storage_value = safe_float(condition.storage_value_tmc)
    if (
        str(condition.storage_trend).upper() == "DECLINING"
        and storage_value is not None
        and storage_value >= 0
    ):
        priority = (
            "HIGH"
            if str(condition.condition).upper() == "HIGH_RISK"
            else "MEDIUM"
        )
        recent_window = condition.evidence.get("recent_window_days", 7)
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_STORAGE_DECLINE",
                title="Review Recent Storage Decline",
                priority=priority,
                trigger="Storage trend is declining over the evaluated observation window",
                evidence={
                    "storage_tmc": round(storage_value, 3),
                    "storage_trend": condition.storage_trend,
                    "condition": condition.condition,
                    "recent_window_days": recent_window,
                },
                explanation=(
                    f"Recorded storage is {storage_value:.2f} TMC and the assessed "
                    f"trend is declining over the configured recent window "
                    f"({recent_window})."
                ),
                suggested_action=(
                    "Review the storage time series and validate inflow and "
                    "withdrawal measurements across the same dates."
                ),
                confidence="MEDIUM",
                limitations=[
                    standard_limitation,
                    "A declining trend alone does not establish the cause of storage loss.",
                    "Do not prescribe downstream release cuts without multi-agency review.",
                ],
            )
        )

    # Rule 3: Inflow/withdrawal comparison. Invalid values must not be used.
    mean_inflow = safe_float(condition.evidence.get("mean_inflow_cusecs"))
    mean_withdrawal = safe_float(
        condition.evidence.get("mean_withdrawal_cusecs")
    )

    if (
        mean_inflow is not None
        and mean_withdrawal is not None
        and mean_inflow > 0
        and mean_withdrawal >= 0
    ):
        ratio = mean_withdrawal / mean_inflow
        if ratio >= 1.2:
            high_risk = str(risk.risk_level).upper() == "HIGH"
            priority = "HIGH" if ratio >= 1.8 or high_risk else "MEDIUM"
            recommendations.append(
                ConservationRecommendation(
                    recommendation_id="REC_WITHDRAWAL_EVALUATION",
                    title="Evaluate Canal Withdrawal Scenarios",
                    priority=priority,
                    trigger=(
                        "Mean canal withdrawal is elevated relative to mean inflow "
                        "over the assessed observation window"
                    ),
                    evidence={
                        "mean_inflow_cusecs": round(mean_inflow, 1),
                        "mean_withdrawal_cusecs": round(mean_withdrawal, 1),
                        "withdrawal_to_inflow_ratio": round(ratio, 2),
                    },
                    explanation=(
                        f"Mean withdrawal ({mean_withdrawal:.1f} cusecs) is "
                        f"{ratio:.2f} times mean inflow ({mean_inflow:.1f} cusecs). "
                        "If sustained, this imbalance can contribute to storage decline."
                    ),
                    suggested_action=(
                        "Compare hypothetical 5%, 10%, and 20% withdrawal-reduction "
                        "scenarios and quantify their estimated storage benefit."
                    ),
                    confidence="MEDIUM",
                    limitations=[
                        standard_limitation,
                        "The ratio is a screening indicator, not proof of unsustainable demand.",
                        "Any schedule or allocation change requires the responsible authority's review.",
                    ],
                )
            )

    # Rule 4: Latest unusual withdrawal/outflow observations.
    def latest_valid_anomaly(metric_keyword: str) -> Optional[FlowAnomalyFlag]:
        matches = [
            anomaly
            for anomaly in anomalies
            if metric_keyword in str(anomaly.metric).lower()
            and str(anomaly.flag_type).upper() == "UNUSUAL_FLOW"
            and finite_nonnegative(anomaly.observed_value)
            and finite_nonnegative(anomaly.baseline_median)
        ]
        if not matches:
            return None
        # Input is expected to be date ordered; sorting makes this explicit.
        return sorted(matches, key=lambda item: str(item.date))[-1]

    latest_withdrawal = latest_valid_anomaly("withdrawal")
    if latest_withdrawal is not None:
        observed = float(latest_withdrawal.observed_value)
        baseline = float(latest_withdrawal.baseline_median)
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_UNUSUAL_WITHDRAWAL",
                title="Review Unusual Canal Withdrawal Reading",
                priority="MEDIUM",
                trigger="Canal withdrawal observation was flagged as statistically unusual",
                evidence={
                    "metric": latest_withdrawal.metric,
                    "date": latest_withdrawal.date,
                    "observed_cusecs": observed,
                    "baseline_median_cusecs": baseline,
                },
                explanation=(
                    f"Withdrawal was {observed:.1f} cusecs on "
                    f"{latest_withdrawal.date}, compared with a rolling baseline "
                    f"median of {baseline:.1f} cusecs."
                ),
                suggested_action=(
                    "Check the corresponding canal-gauge record and determine "
                    "whether the observation reflects a real operational change "
                    "or a measurement/data-quality issue."
                ),
                confidence="PROVISIONAL",
                limitations=[
                    standard_limitation,
                    "A statistical anomaly is not proof of canal wastage or operational error.",
                ],
            )
        )

    latest_outflow = latest_valid_anomaly("outflow")
    if latest_outflow is not None:
        observed = float(latest_outflow.observed_value)
        baseline = float(latest_outflow.baseline_median)
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_UNUSUAL_OUTFLOW",
                title="Review Unusual River Outflow Reading",
                priority="LOW",
                trigger="River outflow observation was flagged as statistically unusual",
                evidence={
                    "metric": latest_outflow.metric,
                    "date": latest_outflow.date,
                    "observed_cusecs": observed,
                    "baseline_median_cusecs": baseline,
                },
                explanation=(
                    f"River outflow was {observed:.1f} cusecs on "
                    f"{latest_outflow.date}, compared with a rolling baseline "
                    f"median of {baseline:.1f} cusecs."
                ),
                suggested_action=(
                    "Cross-check the historical outflow reading against gauge "
                    "logs and available spillway/sluice operation records."
                ),
                confidence="PROVISIONAL",
                limitations=[
                    standard_limitation,
                    "This is a record-review prompt, not an instruction to operate gates.",
                ],
            )
        )

    # Rule 5: Complete water-balance discrepancy only.
    # Diagnostic residuals excluding evaporation must not be passed here as
    # complete-balance discrepancies.
    discrepancy = safe_float(water_balance_discrepancy_tmc)
    status = str(water_balance_status or "").upper()

    if (
        status == "COMPLETE"
        and discrepancy is not None
        and abs(discrepancy) >= 2.0
    ):
        recommendations.append(
            ConservationRecommendation(
                recommendation_id="REC_WATER_BALANCE_REVIEW",
                title="Review Complete Water-Balance Discrepancy",
                priority="LOW",
                trigger=(
                    "Complete interval water-balance discrepancy exceeds "
                    "the provisional 2.0 TMC screening threshold"
                ),
                evidence={
                    "status": status,
                    "discrepancy_tmc": round(discrepancy, 3),
                },
                explanation=(
                    f"The complete water-balance calculation has a residual of "
                    f"{discrepancy:+.3f} TMC. Review possible unmeasured or "
                    "uncertain terms before drawing conclusions."
                ),
                suggested_action=(
                    "Audit the interval dates, flow measurement basis, storage "
                    "readings, evaporation convention, and gauge calibration."
                ),
                confidence="PROVISIONAL",
                limitations=[
                    standard_limitation,
                    "A mass-accounting residual does not confirm physical leakage.",
                    "Do not use this result alone to make release or allocation decisions.",
                ],
            )
        )

    # Stable priority ordering; Python's sort preserves rule order for ties.
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(
        key=lambda recommendation: priority_order.get(
            recommendation.priority, 3
        )
    )

    # Deduplicate by recommendation ID while preserving priority order.
    deduplicated: List[ConservationRecommendation] = []
    seen_ids = set()
    for recommendation in recommendations:
        if recommendation.recommendation_id not in seen_ids:
            deduplicated.append(recommendation)
            seen_ids.add(recommendation.recommendation_id)

    return deduplicated
