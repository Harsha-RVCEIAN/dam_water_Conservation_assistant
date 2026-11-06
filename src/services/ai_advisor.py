"""Explainable AI Conservation Advisor for K.R.Sagara Dam using Google Gemini.

Phase 1 Advisory Layer:
- Translates deterministic Layer A & Layer B model outputs into natural language.
- Strictly adheres to domain boundaries: never overrides deterministic calculations,
  never fabricates observations, never claims leakage, and never issues gate commands.
- Gracefully falls back to deterministic briefing when GEMINI_API_KEY is not configured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from src.models.conservation_assessment import (
    FlowAnomalyFlag,
    ReservoirConditionAssessment,
    ShortageRiskAssessment,
)
from src.models.conservation_recommender import ConservationRecommendation
from src.models.water_budget import ConservationScenarioResult


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"


def get_gemini_api_key() -> str:
    """Read Gemini API key from local .env file or environment."""
    # Prioritize project-level .env file
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass

    # Fallback to system environment variable
    return os.environ.get("GEMINI_API_KEY", "").strip()


def is_gemini_configured() -> bool:
    """Return True if a non-empty Gemini API key is available."""
    return bool(get_gemini_api_key())


def _build_context_summary(
    condition: ReservoirConditionAssessment,
    risk: ShortageRiskAssessment,
    recommendations: List[ConservationRecommendation],
    anomalies: Optional[List[FlowAnomalyFlag]] = None,
    water_balance_status: Optional[str] = None,
    water_balance_discrepancy_tmc: Optional[float] = None,
    diagnostic_residual_tmc: Optional[float] = None,
    scenario_results: Optional[List[ConservationScenarioResult]] = None,
    forecast_inflow_cusecs: Optional[float] = None,
    forecast_target_date: Optional[str] = None,
) -> str:
    """Assemble deterministic facts and evidence into structured context for the AI."""
    anomalies = anomalies or []
    scenario_results = scenario_results or []

    lines = [
        "### DETERMINISTIC RESERVOIR EVIDENCE SUMMARY",
        f"- Observation Record Anchor Date: {condition.observation_date}",
        f"- Storage Measurement Date: {condition.storage_measurement_date or 'Unavailable'}",
        f"- Data Age / Elapsed Days: {condition.data_age_days} days (Freshness: {condition.data_freshness})",
        f"- Assessed Reservoir Condition: {condition.condition} (Live Storage: {condition.storage_value_tmc if condition.storage_value_tmc is not None else 'N/A'} TMC, Basis: {condition.storage_basis})",
        f"- Observed Trajectories: Storage {condition.storage_trend}, Inflow {condition.inflow_trend}, River Outflow {condition.river_outflow_trend}, Canal Withdrawal {condition.canal_withdrawal_trend}",
        f"- Assessed Potential Shortage Risk Level: {risk.risk_level}",
    ]

    if risk.risk_reasons:
        lines.append("  * Risk Drivers: " + "; ".join(risk.risk_reasons))

    if forecast_inflow_cusecs is not None:
        lines.append(f"- 1-Day Inflow Forecast: {forecast_inflow_cusecs:.1f} cusecs for target date {forecast_target_date}")

    lines.append(f"- Water Balance Status: {water_balance_status or 'INCOMPLETE'}")
    if water_balance_status == "COMPLETE" and water_balance_discrepancy_tmc is not None:
        lines.append(f"  * Complete Balance Discrepancy: {water_balance_discrepancy_tmc:+.4f} TMC")
    elif diagnostic_residual_tmc is not None:
        lines.append(f"  * Diagnostic Residual (Excluding Evaporation): {diagnostic_residual_tmc:+.4f} TMC (Evaporation unverified)")

    if anomalies:
        lines.append(f"- Statistical Flow Anomaly Review Flags ({len(anomalies)} trailing flags):")
        for a in anomalies[:4]:
            lines.append(f"  * {a.date} | {a.metric}: observed {a.observed_value:.1f} cusecs vs baseline median {a.baseline_median:.1f} cusecs ({a.reason})")
    else:
        lines.append("- Statistical Flow Anomalies: None flagged in recent observation window")

    if recommendations:
        lines.append("- Rule-Based Advisory Recommendations:")
        for r in recommendations:
            lines.append(f"  * [{r.priority}] {r.title}: {r.trigger}. Action: {r.suggested_action}")

    if scenario_results:
        lines.append("- Hypothetical Canal Rationing Scenarios (7-Day Horizon):")
        for sc in scenario_results:
            lines.append(f"  * {sc.scenario_name}: Conserves {sc.water_saved_tmc:.4f} TMC ({sc.savings_as_pct_capacity:.2f}% capacity). Projected storage: {sc.projected_scenario_storage_tmc:.2f} TMC (Bounds: {'OUT_OF_BOUNDS' if sc.scenario_exceeds_bounds else 'NORMAL'}).")

    lines.append(
        "\nIMPORTANT OPERATIONAL DISCLAIMERS:\n"
        "1. This is an academic decision-support prototype operating on historical archive data, NOT live dam SCADA telemetry.\n"
        "2. Water-balance discrepancy and flow anomalies DO NOT prove physical dam leakage.\n"
        "3. Evaporation is excluded due to unverified measurement convention (not zero actual evaporation).\n"
        "4. Recommendations are advisory prompts for engineering review, NOT official water-allocation or gate-operation commands."
    )
    return "\n".join(lines)


def _generate_deterministic_briefing(
    condition: ReservoirConditionAssessment,
    risk: ShortageRiskAssessment,
    recommendations: List[ConservationRecommendation],
    anomalies: Optional[List[FlowAnomalyFlag]] = None,
    water_balance_status: Optional[str] = None,
    water_balance_discrepancy_tmc: Optional[float] = None,
    diagnostic_residual_tmc: Optional[float] = None,
    scenario_results: Optional[List[ConservationScenarioResult]] = None,
) -> Dict[str, Any]:
    """Generate structured natural-language briefing using deterministic templates."""
    anomalies = anomalies or []
    scenario_results = scenario_results or []

    # 1. Executive overview
    if condition.condition == "INSUFFICIENT_DATA":
        status_text = "The reservoir monitoring archive contains insufficient continuous observations to establish a validated storage baseline."
    elif condition.data_freshness in {"HISTORICAL_ARCHIVE", "STALE"}:
        status_text = (
            f"The reservoir is in provisional **{condition.condition}** condition with live storage recorded at "
            f"**{condition.storage_value_tmc:.2f} TMC** as of historical date **{condition.observation_date}** "
            f"({condition.data_age_days} days elapsed). Because these records belong to a historical research archive, "
            f"they describe historical state rather than active live reservoir telemetry."
        )
    else:
        status_text = (
            f"The reservoir is currently in **{condition.condition}** condition with live storage at "
            f"**{condition.storage_value_tmc:.2f} TMC**."
        )

    # 2. Shortage risk narrative
    risk_text = f"**Assessed Potential Shortage Risk: {risk.risk_level}**\n"
    if risk.risk_level == "HISTORICAL_ONLY":
        risk_text += "Current operational risk cannot be determined because the available observations are historical archive data."
    elif risk.risk_reasons:
        risk_text += "Key contributing factors: " + "; ".join(risk.risk_reasons) + "."
    else:
        risk_text += "Storage trajectories and inflow ratios remain within normal baseline parameters."

    # 3. Water Balance & Anomaly narrative
    wb_text = ""
    if water_balance_status == "COMPLETE" and water_balance_discrepancy_tmc is not None:
        wb_text = f"A complete interval water balance yielded a discrepancy of {water_balance_discrepancy_tmc:+.3f} TMC."
    elif diagnostic_residual_tmc is not None:
        wb_text = (
            f"Diagnostic interval flow residual (excluding unverified evaporation) is {diagnostic_residual_tmc:+.3f} TMC. "
            "Evaporation measurement conventions remain unverified. Notice: this mass-accounting residual is not proof of physical dam leakage."
        )
    else:
        wb_text = "Water-balance interval calculation is incomplete due to missing interval measurements or date alignment gaps."

    # 4. Conservation actions
    actions_list = [f"**{r.title}** ({r.priority} Priority): {r.suggested_action}" for r in recommendations]

    markdown_output = f"""### 📋 Executive Hydrological Briefing (Deterministic Advisory Engine)

> **Notice:** *Google Gemini API key not configured in `.env`. Displaying explainable deterministic synthesis based on validated physical calculations.*

#### 1. Reservoir State & Freshness Summary
{status_text}

#### 2. Shortage Risk Analysis
{risk_text}

#### 3. Mass Balance & Flow Anomaly Review
{wb_text}
{f"Found {len(anomalies)} statistical flow flag(s) in the trailing 7-day window requiring engineering gauge cross-check." if anomalies else "No statistical flow anomalies detected in the trailing window."}

#### 4. Advisory Conservation Actions
""" + ("\n".join(f"- {act}" for act in actions_list) if actions_list else "- Maintain regular gauge monitoring and observation verification.")

    if scenario_results:
        top_sc = scenario_results[-1]
        markdown_output += f"""

#### 5. Hypothetical Canal Rationing Insight
Simulating a **{top_sc.reduction_percentage:.0f}%** canal withdrawal reduction over 7 days yields approximately **{top_sc.water_saved_tmc:.4f} TMC** in conserved storage ({top_sc.savings_as_pct_capacity:.2f}% of live design capacity). All simulations are hypothetical decision-support tools and do not constitute official allocation directives.
"""

    markdown_output += """

---
*🛡️ **Engineering Disclaimers:** Water-balance discrepancy or statistical flow flags are mass-accounting and gauge review indicators; they are not proof of physical dam leakage. Evaporation conventions are unverified. Records describe historical observations, not live telemetry.*
"""

    return {
        "status": "deterministic_fallback",
        "model_used": "Deterministic Rule-Based Synthesizer",
        "briefing_markdown": markdown_output,
        "briefing": markdown_output,
        "is_ai_generated": False,
    }


def generate_ai_conservation_briefing(
    condition: ReservoirConditionAssessment,
    risk: ShortageRiskAssessment,
    recommendations: List[ConservationRecommendation],
    anomalies: Optional[List[FlowAnomalyFlag]] = None,
    water_balance_status: Optional[str] = None,
    water_balance_discrepancy_tmc: Optional[float] = None,
    diagnostic_residual_tmc: Optional[float] = None,
    scenario_results: Optional[List[ConservationScenarioResult]] = None,
    forecast_inflow_cusecs: Optional[float] = None,
    forecast_target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an explainable executive briefing using Google Gemini (or deterministic fallback)."""
    api_key = get_gemini_api_key()

    # If no API key is provided, use deterministic briefing
    if not api_key:
        return _generate_deterministic_briefing(
            condition=condition,
            risk=risk,
            recommendations=recommendations,
            anomalies=anomalies,
            water_balance_status=water_balance_status,
            water_balance_discrepancy_tmc=water_balance_discrepancy_tmc,
            diagnostic_residual_tmc=diagnostic_residual_tmc,
            scenario_results=scenario_results,
        )

    context_summary = _build_context_summary(
        condition=condition,
        risk=risk,
        recommendations=recommendations,
        anomalies=anomalies,
        water_balance_status=water_balance_status,
        water_balance_discrepancy_tmc=water_balance_discrepancy_tmc,
        diagnostic_residual_tmc=diagnostic_residual_tmc,
        scenario_results=scenario_results,
        forecast_inflow_cusecs=forecast_inflow_cusecs,
        forecast_target_date=forecast_target_date,
    )

    system_instruction = (
        "You are an explainable AI Water Conservation Advisor for the K.R.Sagara Dam decision-support prototype "
        "(Cauvery Basin, Karnataka, India). Your goal is to explain the deterministic engineering calculations in clear, "
        "accurate, professional language for reservoir managers and irrigation engineers.\n\n"
        "STRICT SAFETY & FACTUALITY BOUNDARIES:\n"
        "1. Never invent or extrapolate reservoir numbers. Use ONLY the supplied figures.\n"
        "2. Never override the deterministic risk level or reservoir condition.\n"
        "3. Water-balance discrepancy or flow anomalies DO NOT prove physical dam leakage. State they are accounting residuals or unmodeled tributary flow.\n"
        "4. Never issue official water-allocation decrees or dam-gate operational commands. All recommendations are advisory options for review.\n"
        "5. Clearly emphasize that records are from a historical archive and NOT live dam telemetry.\n"
        "6. Emphasize that evaporation is excluded because measurement conventions are unverified, NOT because evaporation is zero.\n"
        "7. Format your response cleanly using Markdown headings, bullets, and bold highlights."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"Based strictly on the following deterministic data, provide a comprehensive, structured briefing:\n\n"
        f"{context_summary}\n\n"
        "Structure your briefing into:\n"
        "1. Executive Reservoir Status & Freshness\n"
        "2. Shortage Risk & Trend Assessment\n"
        "3. Water Balance & Statistical Flow Review (with leakage disclaimer)\n"
        "4. Explainable Conservation Actions & Scenario Benefits\n"
        "5. Engineering Limitations & Monitoring Next Steps"
    )

    try:
        url = f"{GEMINI_API_URL}?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": 1024,
            },
        }
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    ai_text = parts[0]["text"]
                    return {
                        "status": "success",
                        "model_used": "Google Gemini (gemini-3.5-flash-lite)",
                        "briefing_markdown": ai_text,
                        "briefing": ai_text,
                        "source": "gemini",
                        "is_ai_generated": True,
                    }

        # If API returned non-200 or unexpected structure, fall back
        return _generate_deterministic_briefing(
            condition=condition,
            risk=risk,
            recommendations=recommendations,
            anomalies=anomalies,
            water_balance_status=water_balance_status,
            water_balance_discrepancy_tmc=water_balance_discrepancy_tmc,
            diagnostic_residual_tmc=diagnostic_residual_tmc,
            scenario_results=scenario_results,
        )
    except Exception:
        # Fallback on network error or timeout
        return _generate_deterministic_briefing(
            condition=condition,
            risk=risk,
            recommendations=recommendations,
            anomalies=anomalies,
            water_balance_status=water_balance_status,
            water_balance_discrepancy_tmc=water_balance_discrepancy_tmc,
            diagnostic_residual_tmc=diagnostic_residual_tmc,
            scenario_results=scenario_results,
        )

