"""Reservoir condition assessment, flow anomaly review, and shortage risk diagnostics.

Layer B deterministic decision-support logic.
Academic prototype only; not a live dam-control or official allocation system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.models.storage_analysis import extract_storage_summary, compute_interval_storage_change


def _load_config() -> Dict[str, Any]:
    config_path = Path("config/conservation_rules.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "assessment": {
            "trend_window_observations": 7,
            "min_valid_observations": 5,
            "storage_provisional_thresholds": {
                "basis": "live_above_cill_tmc",
                "high_tmc": 35.0,
                "watch_tmc": 20.0,
                "low_tmc": 10.0,
            },
            "inflow_provisional_low_cusecs": 1500.0,
        },
        "anomaly_detection": {
            "window_observations": 14,
            "mad_multiplier": 3.0,
            "min_samples": 7,
        },
        "risk_assessment": {
            "consecutive_storage_decline_days": 3,
            "withdrawal_to_inflow_ratio_high": 1.5,
        },
    }


@dataclass
class ReservoirConditionAssessment:
    condition: str  # SURPLUS, NORMAL, WATCH, HIGH_RISK, INSUFFICIENT_DATA
    observation_date: Optional[str]
    storage_measurement_date: Optional[str]
    data_age_days: Optional[int]
    data_freshness: str  # FRESH, STALE, HISTORICAL_ARCHIVE, UNKNOWN
    storage_value_tmc: Optional[float]
    storage_basis: str
    storage_trend: str  # RISING, STEADY, DECLINING, UNKNOWN
    inflow_trend: str  # RISING, STEADY, DECLINING, UNKNOWN
    river_outflow_trend: str
    canal_withdrawal_trend: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


@dataclass
class FlowAnomalyFlag:
    metric: str
    date: str
    observed_value: float
    baseline_median: float
    threshold_mad: float
    flag_type: str  # UNUSUAL_FLOW, DATA_QUALITY
    reason: str


@dataclass
class ShortageRiskAssessment:
    risk_level: str  # LOW, MEDIUM, HIGH, INSUFFICIENT_DATA
    risk_reasons: List[str] = field(default_factory=list)
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    suggested_review_action: str = ""
    data_quality_status: str = "VALID"
    is_forecast_based: bool = False
    forecast_target_date: Optional[str] = None
    limitations: List[str] = field(default_factory=list)


def _calculate_trend(series: pd.Series, tolerance_pct: float = 0.05) -> str:
    """Calculate direction of trend over consecutive observations."""
    clean = series.dropna()
    if len(clean) < 2:
        return "UNKNOWN"
    start_val = clean.iloc[0]
    end_val = clean.iloc[-1]
    if start_val == 0.0:
        diff = end_val - start_val
        return "RISING" if diff > 0 else ("DECLINING" if diff < 0 else "STEADY")
    rel_change = (end_val - start_val) / abs(start_val)
    if rel_change > tolerance_pct:
        return "RISING"
    elif rel_change < -tolerance_pct:
        return "DECLINING"
    return "STEADY"


def assess_reservoir_condition(
    df: pd.DataFrame,
    as_of_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ReservoirConditionAssessment:
    """Evaluate current condition of reservoir using historical observations.

    Enforces interval verification, date alignment, and minimum valid observation counts.
    """
    if config is None:
        config = _load_config()
    cfg_assess = config.get("assessment", {})
    trend_window = cfg_assess.get("trend_window_observations", 7)
    min_obs = cfg_assess.get("min_valid_observations", 5)
    storage_cfg = cfg_assess.get("storage_provisional_thresholds", {})
    storage_basis = storage_cfg.get("basis", "live_above_cill_tmc")
    high_tmc = storage_cfg.get("high_tmc", 35.0)
    watch_tmc = storage_cfg.get("watch_tmc", 20.0)
    low_tmc = storage_cfg.get("low_tmc", 10.0)

    limitations = [
        "Academic prototype: thresholds are provisional and not official KRS operating bands.",
        "Capacity percentage is omitted unless an official verified live-capacity denominator is configured.",
    ]

    if df.empty or "calendar_date" not in df.columns:
        return ReservoirConditionAssessment(
            condition="INSUFFICIENT_DATA",
            observation_date=None,
            storage_measurement_date=None,
            data_age_days=None,
            data_freshness="UNKNOWN",
            storage_value_tmc=None,
            storage_basis=storage_basis,
            storage_trend="UNKNOWN",
            inflow_trend="UNKNOWN",
            river_outflow_trend="UNKNOWN",
            canal_withdrawal_trend="UNKNOWN",
            evidence={},
            limitations=limitations + ["DataFrame is empty or missing 'calendar_date' column."],
        )

    df_sorted = df.copy()
    df_sorted["calendar_date"] = pd.to_datetime(df_sorted["calendar_date"]).dt.normalize()
    df_sorted = df_sorted.sort_values("calendar_date").reset_index(drop=True)

    if as_of_date is not None:
        target_ts = pd.to_datetime(as_of_date).normalize()
        df_sorted = df_sorted[df_sorted["calendar_date"] <= target_ts]
        if df_sorted.empty:
            return ReservoirConditionAssessment(
                condition="INSUFFICIENT_DATA",
                observation_date=as_of_date,
                storage_measurement_date=None,
                data_age_days=None,
                data_freshness="UNKNOWN",
                storage_value_tmc=None,
                storage_basis=storage_basis,
                storage_trend="UNKNOWN",
                inflow_trend="UNKNOWN",
                river_outflow_trend="UNKNOWN",
                canal_withdrawal_trend="UNKNOWN",
                evidence={},
                limitations=limitations + [f"No observations found on or before {as_of_date}."],
            )

    latest_row = df_sorted.iloc[-1]
    obs_date = latest_row["calendar_date"].strftime("%Y-%m-%d")
    today = pd.Timestamp.now().normalize()
    data_age_days = (today - latest_row["calendar_date"]).days

    if data_age_days > 30:
        freshness = "HISTORICAL_ARCHIVE"
        limitations.append(f"Observation date {obs_date} is {data_age_days} days old (historical archive).")
    elif data_age_days > 1:
        freshness = "STALE"
        limitations.append(f"Observation date {obs_date} is {data_age_days} days old (stale).")
    else:
        freshness = "FRESH"

    # Get recent window slice
    recent_slice = df_sorted.iloc[-trend_window:]
    valid_storage_series = (
        recent_slice[storage_basis]
        if storage_basis in recent_slice.columns
        else pd.Series(dtype=float, index=recent_slice.index)
    )

    valid_count = int(valid_storage_series.notna().sum())
    if len(recent_slice) < min_obs or valid_count < min_obs:
        return ReservoirConditionAssessment(
            condition="INSUFFICIENT_DATA",
            observation_date=obs_date,
            storage_measurement_date=None,
            data_age_days=data_age_days,
            data_freshness=freshness,
            storage_value_tmc=None,
            storage_basis=storage_basis,
            storage_trend="UNKNOWN",
            inflow_trend="UNKNOWN",
            river_outflow_trend="UNKNOWN",
            canal_withdrawal_trend="UNKNOWN",
            evidence={
                "observation_count": len(recent_slice),
                "valid_storage_count": valid_count,
                "required_min_observations": min_obs,
            },
            limitations=limitations + [
                f"Recent window has only {valid_count} valid storage readings; minimum {min_obs} required."
            ],
        )

    # Determine storage measurement date and value
    valid_storage_rows = recent_slice[recent_slice[storage_basis].notna()]
    latest_storage_row = valid_storage_rows.iloc[-1]
    latest_storage = float(latest_storage_row[storage_basis])
    storage_meas_date = latest_storage_row["calendar_date"].strftime("%Y-%m-%d")

    if storage_meas_date != obs_date:
        limitations.append(
            f"Latest storage value is from {storage_meas_date}, while latest dataset record is {obs_date}."
        )

    # Count consecutive storage decline days
    storage_diffs = valid_storage_rows[storage_basis].diff().dropna().tolist()
    consecutive_declines = 0
    for diff in reversed(storage_diffs):
        if diff < 0:
            consecutive_declines += 1
        else:
            break

    # Trends
    s_trend = _calculate_trend(valid_storage_series)
    inflow_col = "inflow_cusecs" if "inflow_cusecs" in recent_slice.columns else "inflow_cfs"
    outflow_col = "outflow_cusecs" if "outflow_cusecs" in recent_slice.columns else "outflow_cfs"
    withdrawal_col = "withdrawal_cusecs" if "withdrawal_cusecs" in recent_slice.columns else "withdrawal_cfs"

    inflow_trend = _calculate_trend(recent_slice[inflow_col]) if inflow_col in recent_slice.columns else "UNKNOWN"
    outflow_trend = _calculate_trend(recent_slice[outflow_col]) if outflow_col in recent_slice.columns else "UNKNOWN"
    withdrawal_trend = _calculate_trend(recent_slice[withdrawal_col]) if withdrawal_col in recent_slice.columns else "UNKNOWN"

    # Condition determination using high_tmc, watch_tmc, and low_tmc
    if latest_storage is None:
        condition = "INSUFFICIENT_DATA"
    elif latest_storage <= low_tmc:
        condition = "HIGH_RISK"
    elif latest_storage <= watch_tmc:
        condition = "WATCH"
    elif latest_storage >= high_tmc:
        condition = "SURPLUS"
    else:
        condition = "NORMAL"

    evidence = {
        "latest_storage_tmc": latest_storage,
        "storage_measurement_date": storage_meas_date,
        "recent_window_days": len(recent_slice),
        "valid_storage_count": valid_count,
        "consecutive_storage_decline_days": consecutive_declines,
        "mean_inflow_cusecs": float(recent_slice[inflow_col].mean()) if inflow_col in recent_slice.columns else None,
        "mean_withdrawal_cusecs": float(recent_slice[withdrawal_col].mean()) if withdrawal_col in recent_slice.columns else None,
    }

    return ReservoirConditionAssessment(
        condition=condition,
        observation_date=obs_date,
        storage_measurement_date=storage_meas_date,
        data_age_days=data_age_days,
        data_freshness=freshness,
        storage_value_tmc=latest_storage,
        storage_basis=storage_basis,
        storage_trend=s_trend,
        inflow_trend=inflow_trend,
        river_outflow_trend=outflow_trend,
        canal_withdrawal_trend=withdrawal_trend,
        evidence=evidence,
        limitations=limitations,
    )


def detect_abnormal_flow_patterns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[FlowAnomalyFlag]:
    """Detect unusual flow patterns using robust rolling Median Absolute Deviation (MAD).

    Prevents self-contamination by using shifted historical baseline (shift(1)) and
    enforces minimum sample size against valid historical observations.
    """
    if config is None:
        config = _load_config()
    cfg_anom = config.get("anomaly_detection", {})
    window_obs = cfg_anom.get("window_observations", cfg_anom.get("window_days", 14))
    mad_multiplier = cfg_anom.get("mad_multiplier", 3.0)
    min_samples = cfg_anom.get("min_samples", 7)

    if columns is None:
        columns = [c for c in ["inflow_cusecs", "outflow_cusecs", "withdrawal_cusecs"] if c in df.columns]

    flags: List[FlowAnomalyFlag] = []
    if df.empty or "calendar_date" not in df.columns:
        return flags

    df_work = df.copy().sort_values("calendar_date").reset_index(drop=True)

    for col in columns:
        series = df_work[col]
        # Shift series by 1 so the current observation does not contribute to its own baseline
        hist_series = series.shift(1)
        rolling_median = hist_series.rolling(window=window_obs, min_periods=min_samples).median()
        rolling_dev = (hist_series - rolling_median).abs()
        rolling_mad = rolling_dev.rolling(window=window_obs, min_periods=min_samples).median()

        # Check recent window observations (last 7 rows)
        tail_idx = df_work.index[-7:]
        for idx in tail_idx:
            val = series.iloc[idx]
            if pd.isna(val):
                flags.append(
                    FlowAnomalyFlag(
                        metric=col,
                        date=str(df_work["calendar_date"].iloc[idx])[:10],
                        observed_value=np.nan,
                        baseline_median=np.nan,
                        threshold_mad=np.nan,
                        flag_type="DATA_QUALITY",
                        reason=f"Missing observation in {col}.",
                    )
                )
                continue

            # Ensure minimum sample count in the preceding historical window
            start_window_idx = max(0, idx - window_obs)
            valid_hist_count = hist_series.iloc[start_window_idx:idx].notna().sum()
            if valid_hist_count < min_samples:
                continue

            med = rolling_median.iloc[idx]
            mad = rolling_mad.iloc[idx]
            if pd.isna(med):
                continue

            scale = mad * 1.4826 if not pd.isna(mad) and mad > 0 else 0.1 * abs(med)
            if scale == 0.0:
                scale = 50.0  # Minimal flow floor
            threshold = mad_multiplier * scale
            deviation = abs(val - med)

            if deviation > threshold:
                flags.append(
                    FlowAnomalyFlag(
                        metric=col,
                        date=str(df_work["calendar_date"].iloc[idx])[:10],
                        observed_value=float(val),
                        baseline_median=float(med),
                        threshold_mad=float(threshold),
                        flag_type="UNUSUAL_FLOW",
                        reason=f"{col} value ({val:.1f}) deviates by {deviation:.1f} from historical baseline median ({med:.1f}).",
                    )
                )
    return flags


def assess_water_shortage_risk(
    condition: ReservoirConditionAssessment,
    inflow_forecast_cfs: Optional[float] = None,
    forecast_target_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ShortageRiskAssessment:
    """Assess potential water-shortage risk from evidence without unsupported extrapolation.

    Integrates consecutive_storage_decline_days from configuration.
    """
    if config is None:
        config = _load_config()
    cfg_risk = config.get("risk_assessment", {})
    ratio_high_thresh = cfg_risk.get("withdrawal_to_inflow_ratio_high", 1.5)
    decline_days_thresh = cfg_risk.get("consecutive_storage_decline_days", 3)

    limitations = [
        "Advisory screening: does not constitute an official drought or shortage declaration.",
        "One-day forecast informs next-day transition only; does not forecast multi-week water availability.",
    ]

    if condition.condition == "INSUFFICIENT_DATA":
        return ShortageRiskAssessment(
            risk_level="INSUFFICIENT_DATA",
            risk_reasons=["Baseline reservoir condition cannot be established due to insufficient data."],
            supporting_metrics={},
            suggested_review_action="Verify and update daily dam telemetry before performing shortage risk screening.",
            data_quality_status="INCOMPLETE",
            limitations=limitations,
        )

    risk_reasons: List[str] = []
    score = 0

    # Evidence 1: Storage Level
    if condition.condition == "HIGH_RISK":
        score += 3
        risk_reasons.append(f"Storage ({condition.storage_value_tmc:.2f} TMC) is at or below provisional low threshold.")
    elif condition.condition == "WATCH":
        score += 1
        risk_reasons.append(f"Storage ({condition.storage_value_tmc:.2f} TMC) is in provisional watch band.")

    # Evidence 2: Storage Trend & Consecutive Decline Days
    consec_declines = condition.evidence.get("consecutive_storage_decline_days", 0)
    if consec_declines >= decline_days_thresh:
        score += 1
        risk_reasons.append(
            f"Live storage has declined for {consec_declines} consecutive observations (>= {decline_days_thresh} days configured)."
        )
    elif condition.storage_trend == "DECLINING":
        score += 1
        risk_reasons.append("Live storage is exhibiting a declining trajectory across recent observations.")

    # Evidence 3: Withdrawal vs Inflow Balance
    mean_inflow = condition.evidence.get("mean_inflow_cusecs")
    mean_with = condition.evidence.get("mean_withdrawal_cusecs")
    if mean_inflow is not None and mean_with is not None and mean_inflow > 0:
        ratio = mean_with / mean_inflow
        if ratio > ratio_high_thresh:
            score += 1
            risk_reasons.append(f"Canal withdrawal exceeds inflow by {ratio:.2f}x in the recent observation window.")

    # Evidence 4: One-day forecast
    is_forecast_based = False
    if inflow_forecast_cfs is not None:
        is_forecast_based = True
        low_inflow_ref = config.get("assessment", {}).get("inflow_provisional_low_cusecs", 1500.0)
        if inflow_forecast_cfs < low_inflow_ref:
            score += 1
            risk_reasons.append(
                f"Next-day forecast inflow ({inflow_forecast_cfs:.1f} cfs) is below reference ({low_inflow_ref:.0f} cfs)."
            )

    # Risk level classification
    if score >= 4:
        risk_level = "HIGH"
        action = "Initiate immediate operational review of planned canal withdrawals and prepare contingency scenarios."
    elif score >= 2:
        risk_level = "MEDIUM"
        action = "Monitor daily storage changes closely and evaluate voluntary withdrawal-reduction scenarios."
    else:
        risk_level = "LOW"
        action = "Maintain regular monitoring schedule."

    if condition.data_freshness in ["STALE", "HISTORICAL_ARCHIVE"]:
        limitations.append(
            f"Assessment reflects historical state as of {condition.observation_date}, not current real-time condition."
        )

    return ShortageRiskAssessment(
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        supporting_metrics={
            "score": score,
            "observation_date": condition.observation_date,
            "storage_measurement_date": condition.storage_measurement_date,
            "storage_tmc": condition.storage_value_tmc,
            "forecast_inflow_cfs": inflow_forecast_cfs,
            "consecutive_storage_decline_days": consec_declines,
        },
        suggested_review_action=action,
        data_quality_status=condition.data_freshness,
        is_forecast_based=is_forecast_based,
        forecast_target_date=forecast_target_date,
        limitations=limitations,
    )
