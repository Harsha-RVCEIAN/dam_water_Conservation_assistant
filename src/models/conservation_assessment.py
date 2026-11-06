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
    condition: str  # HIGH_STORAGE, NORMAL, WATCH, HIGH_RISK, INSUFFICIENT_DATA
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
    valid_storage_rows = valid_storage_rows.sort_values("calendar_date")

    consecutive_declines = 0

    for i in range(len(valid_storage_rows) - 1, 0, -1):

        current = valid_storage_rows.iloc[i]
        previous = valid_storage_rows.iloc[i - 1]

        date_gap = (
            current["calendar_date"] - previous["calendar_date"]
        ).days

        storage_change = (
            current[storage_basis] - previous[storage_basis]
        )

        if date_gap != 1 or storage_change >= 0:
            break

        consecutive_declines += 1

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
        condition = "HIGH_STORAGE"
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
    """
    Detect unusual flow patterns using rolling Median Absolute Deviation.

    Features:
    - Uses only preceding observations for the baseline.
    - Excludes negative and non-finite historical flow readings.
    - Enforces a minimum valid historical sample count.
    - Handles missing and invalid current readings as data-quality flags.
    - Does not interpret statistical anomalies as physical leakage.
    """
    if config is None:
        config = _load_config()
    cfg_anom = config.get("anomaly_detection", {})
    window_obs = int(
        cfg_anom.get(
            "window_observations",
            cfg_anom.get("window_days", 14),
        )
    )
    mad_multiplier = float(cfg_anom.get("mad_multiplier", 3.0))
    min_samples = int(cfg_anom.get("min_samples", 7))
    if window_obs < 1:
        raise ValueError("Anomaly window must be at least 1.")
    if min_samples < 1 or min_samples > window_obs:
        raise ValueError(
            "Minimum samples must be between 1 and the rolling window size."
        )
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("MAD multiplier must be finite and positive.")
    if df.empty or "calendar_date" not in df.columns:
        return []
    if columns is None:
        columns = [
            col
            for col in [
                "inflow_cusecs",
                "outflow_cusecs",
                "withdrawal_cusecs",
            ]
            if col in df.columns
        ]
    flags: List[FlowAnomalyFlag] = []
    df_work = df.copy()
    df_work["calendar_date"] = pd.to_datetime(
        df_work["calendar_date"],
        errors="coerce",
    )
    df_work = (
        df_work.dropna(subset=["calendar_date"])
        .sort_values("calendar_date")
        .reset_index(drop=True)
    )
    if df_work.empty:
        return flags
    # Inspect only the most recent seven observations.
    tail_start = max(0, len(df_work) - 7)
    for col in columns:
        if col not in df_work.columns:
            continue
        # Convert invalid strings and values to NaN.
        series = pd.to_numeric(
            df_work[col],
            errors="coerce",
        ).astype(float)
        # Negative and infinite readings are invalid.
        invalid_mask = (
            ~np.isfinite(series)
            | (series < 0)
        )
        # Valid historical flow observations only.
        valid_series = series.where(~invalid_mask)
        # Shift baseline to exclude current observation.
        hist_series = valid_series.shift(1)
        # Rolling historical median.
        rolling_median = hist_series.rolling(
            window=window_obs,
            min_periods=min_samples,
        ).median()
        # Median absolute deviation.
        absolute_deviation = (
            hist_series - rolling_median
        ).abs()
        rolling_mad = absolute_deviation.rolling(
            window=window_obs,
            min_periods=min_samples,
        ).median()
        for idx in range(tail_start, len(df_work)):
            val = series.iloc[idx]
            observation_date = (
                df_work["calendar_date"].iloc[idx].strftime("%Y-%m-%d")
            )
            # Invalid current observation
            if invalid_mask.iloc[idx]:
                if pd.isna(val):
                    reason = f"Missing or non-numeric observation in {col}."
                elif not np.isfinite(val):
                    reason = f"Non-finite observation in {col}."
                else:
                    reason = f"Negative flow observation in {col}."
                flags.append(
                    FlowAnomalyFlag(
                        metric=col,
                        date=observation_date,
                        observed_value=(
                            float(val) if pd.notna(val) and np.isfinite(val)
                            else np.nan
                        ),
                        baseline_median=np.nan,
                        threshold_mad=np.nan,
                        flag_type="DATA_QUALITY",
                        reason=reason,
                    )
                )
                continue
            # Count valid preceding historical observations.
            start_window_idx = max(0, idx - window_obs)
            historical_values = valid_series.iloc[
                start_window_idx:idx
            ]
            valid_hist_count = historical_values.notna().sum()
            if valid_hist_count < min_samples:
                continue
            median_value = rolling_median.iloc[idx]
            mad_value = rolling_mad.iloc[idx]
            if pd.isna(median_value):
                continue
            # Robust scale estimate.
            if pd.notna(mad_value) and mad_value > 0:
                scale = 1.4826 * mad_value
            else:
                # Fallback when historical values have zero MAD.
                scale = 0.1 * abs(median_value)
            # Minimum threshold for constant or zero-flow baselines.
            if scale == 0:
                scale = 50.0
            threshold = mad_multiplier * scale
            deviation = abs(val - median_value)
            if deviation > threshold:
                flags.append(
                    FlowAnomalyFlag(
                        metric=col,
                        date=observation_date,
                        observed_value=float(val),
                        baseline_median=float(median_value),
                        threshold_mad=float(threshold),
                        flag_type="UNUSUAL_FLOW",
                        reason=(
                            f"{col} value ({val:.1f} cfs) deviates by "
                            f"{deviation:.1f} cfs from the historical "
                            f"median ({median_value:.1f} cfs)."
                        ),
                    )
                )
    return flags


    def assess_water_shortage_risk(
        condition: ReservoirConditionAssessment,
        inflow_forecast_cfs: Optional[float] = None,
        forecast_target_date: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ShortageRiskAssessment:
        """
        Assess potential water-shortage risk using available reservoir evidence.

        Uses:
        - Provisional storage condition
        - Consecutive calendar-day storage declines
        - Recent withdrawal-to-inflow ratio
        - Optional one-day inflow forecast

        This is an advisory screening tool, not an official shortage declaration.
        """

        if config is None:
            config = _load_config()

        cfg_risk = config.get("risk_assessment", {})
        cfg_assessment = config.get("assessment", {})

        ratio_high_thresh = float(
            cfg_risk.get("withdrawal_to_inflow_ratio_high", 1.5)
        )
        decline_days_thresh = int(
            cfg_risk.get("consecutive_storage_decline_days", 3)
        )
        low_inflow_ref = float(
            cfg_assessment.get("inflow_provisional_low_cusecs", 1500.0)
        )

        limitations = [
            "Advisory screening only; does not constitute an official drought or water-shortage declaration.",
            "A one-day inflow forecast informs only the next-day transition and does not predict multi-week water availability.",
            "Storage thresholds are provisional and must not be interpreted as official reservoir operating levels.",
        ]

        # Validate configuration
        if (
            not np.isfinite(ratio_high_thresh)
            or ratio_high_thresh <= 0
            or decline_days_thresh < 1
            or not np.isfinite(low_inflow_ref)
            or low_inflow_ref < 0
        ):
            raise ValueError("Invalid shortage-risk configuration values.")

        # Insufficient historical data
        if condition.condition == "INSUFFICIENT_DATA":
            return ShortageRiskAssessment(
                risk_level="INSUFFICIENT_DATA",
                risk_reasons=[
                    "Baseline reservoir condition cannot be established due to insufficient data."
                ],
                supporting_metrics={},
                suggested_review_action=(
                    "Verify and update reservoir observations before performing shortage-risk screening."
                ),
                data_quality_status="INCOMPLETE",
                limitations=limitations,
            )

        # Do not present stale data as current risk.
        stale_statuses = {"STALE", "HISTORICAL_ARCHIVE"}

        if condition.data_freshness in stale_statuses:
            limitations.append(
                f"Assessment reflects historical observations as of "
                f"{condition.observation_date}, not the current reservoir condition."
            )

            return ShortageRiskAssessment(
                risk_level="HISTORICAL_ONLY",
                risk_reasons=[
                    "Current shortage risk cannot be reliably assessed because the latest observations are stale."
                ],
                supporting_metrics={
                    "observation_date": condition.observation_date,
                    "storage_measurement_date": condition.storage_measurement_date,
                    "storage_tmc": condition.storage_value_tmc,
                    "data_freshness": condition.data_freshness,
                },
                suggested_review_action=(
                    "Obtain updated reservoir storage and flow observations before making a current risk assessment."
                ),
                data_quality_status=condition.data_freshness,
                limitations=limitations,
            )

        risk_reasons: List[str] = []
        score = 0

        # Evidence 1: Storage level
        if condition.condition == "HIGH_RISK":
            score += 3

            if condition.storage_value_tmc is not None:
                risk_reasons.append(
                    f"Storage ({condition.storage_value_tmc:.2f} TMC) "
                    "is at or below the provisional low threshold."
                )
            else:
                risk_reasons.append(
                    "Reservoir condition indicates a provisional high-risk storage level."
                )

        elif condition.condition == "WATCH":
            score += 1

            if condition.storage_value_tmc is not None:
                risk_reasons.append(
                    f"Storage ({condition.storage_value_tmc:.2f} TMC) "
                    "is within the provisional watch band."
                )
            else:
                risk_reasons.append(
                    "Reservoir condition indicates a provisional storage watch level."
                )

        # Evidence 2: Consecutive calendar-day storage declines
        consec_declines = condition.evidence.get(
            "consecutive_storage_decline_days", 0
        )

        try:
            consec_declines = int(consec_declines)
        except (TypeError, ValueError):
            consec_declines = 0

        if consec_declines >= decline_days_thresh:
            score += 1

            risk_reasons.append(
                f"Live storage has declined for {consec_declines} "
                f"consecutive calendar days "
                f"(configured threshold: {decline_days_thresh} days)."
            )

        elif condition.storage_trend == "DECLINING":
            score += 1

            risk_reasons.append(
                "Live storage is exhibiting a declining trajectory across recent observations."
            )

        # Evidence 3: Withdrawal-to-inflow ratio
        mean_inflow = condition.evidence.get("mean_inflow_cusecs")
        mean_with = condition.evidence.get("mean_withdrawal_cusecs")

        if mean_inflow is not None and mean_with is not None:
            try:
                mean_inflow = float(mean_inflow)
                mean_with = float(mean_with)

                if (
                    np.isfinite(mean_inflow)
                    and np.isfinite(mean_with)
                    and mean_inflow > 0
                    and mean_with >= 0
                ):
                    ratio = mean_with / mean_inflow

                    if ratio > ratio_high_thresh:
                        score += 1

                        risk_reasons.append(
                            f"Recent canal withdrawal is {ratio:.2f}x "
                            "the mean inflow rate."
                        )

            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Evidence 4: Optional one-day inflow forecast
        is_forecast_based = False
        valid_forecast = None

        if inflow_forecast_cfs is not None:
            try:
                candidate_forecast = float(inflow_forecast_cfs)

                if (
                    np.isfinite(candidate_forecast)
                    and candidate_forecast >= 0
                ):
                    valid_forecast = candidate_forecast
                    is_forecast_based = True

            except (TypeError, ValueError):
                valid_forecast = None

        if valid_forecast is not None:
            if valid_forecast < low_inflow_ref:
                score += 1

                risk_reasons.append(
                    f"Next-day forecast inflow ({valid_forecast:.1f} cfs) "
                    f"is below the provisional reference "
                    f"({low_inflow_ref:.0f} cfs)."
                )

        # Risk classification
        if score >= 4:
            risk_level = "HIGH"
            action = (
                "Review recent storage, inflow and withdrawal trends. "
                "Evaluate conservation scenarios with the relevant reservoir authorities."
            )

        elif score >= 2:
            risk_level = "MEDIUM"
            action = (
                "Monitor updated storage and flow observations and "
                "evaluate hypothetical withdrawal-reduction scenarios."
            )

        else:
            risk_level = "LOW"
            action = (
                "Maintain regular monitoring and verify incoming reservoir observations."
            )

        return ShortageRiskAssessment(
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            supporting_metrics={
                "score": score,
                "observation_date": condition.observation_date,
                "storage_measurement_date": condition.storage_measurement_date,
                "storage_tmc": condition.storage_value_tmc,
                "mean_inflow_cusecs": mean_inflow,
                "mean_withdrawal_cusecs": mean_with,
                "withdrawal_to_inflow_threshold": ratio_high_thresh,
                "forecast_inflow_cfs": valid_forecast,
                "consecutive_storage_decline_days": consec_declines,
            },
            suggested_review_action=action,
            data_quality_status=condition.data_freshness,
            is_forecast_based=is_forecast_based,
            forecast_target_date=(
                forecast_target_date if is_forecast_based else None
            ),
            limitations=limitations,
        )