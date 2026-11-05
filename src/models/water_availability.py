"""Data-grounded storage, rainfall, alert, and conservation guidance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.models.water_budget import (
    KRS_DEAD_STORAGE_TMC,
    KRS_DESIGN_GROSS_CAPACITY_TMC,
)

MAX_LIVE_STORAGE_TMC = (
    KRS_DESIGN_GROSS_CAPACITY_TMC - KRS_DEAD_STORAGE_TMC
)
DEFAULT_STORAGE_THRESHOLDS = {
    "moderate_pct": 50.0,
    "low_pct": 30.0,
    "critical_pct": 15.0,
}


class WaterAvailabilityError(ValueError):
    """Raised when reservoir or rainfall values cannot support a safe calculation."""


@dataclass(frozen=True)
class StorageStatus:
    observation_date: pd.Timestamp
    observation_age_days: int
    live_storage_tmc: float
    max_live_storage_tmc: float
    storage_pct: float
    remaining_live_storage_tmc: float
    change_1d_tmc: float | None
    change_7d_tmc: float | None
    change_30d_tmc: float | None
    is_outside_capacity: bool


@dataclass(frozen=True)
class AlertResult:
    level: str
    reasons: tuple[str, ...]
    triggered_values: tuple[str, ...]
    provisional_thresholds: dict[str, float]


@dataclass(frozen=True)
class Recommendation:
    title: str
    reason: str
    supporting_values: str
    priority: str
    category: str = "Storage-based recommendation"
    action: str = "Follow approved procedures."


def _finite_value(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise WaterAvailabilityError(f"{label} is not numeric.") from exc
    if not np.isfinite(numeric):
        raise WaterAvailabilityError(f"{label} is missing or non-finite.")
    return numeric


def _storage_at_date(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
) -> float | None:
    rows = history.loc[history["calendar_date"].eq(target_date), "live_above_cill_tmc"]
    if len(rows) != 1 or pd.isna(rows.iloc[0]):
        return None
    value = float(rows.iloc[0])
    return value if np.isfinite(value) else None


def calculate_storage_status(
    history: pd.DataFrame,
    *,
    as_of: date | pd.Timestamp | None = None,
    max_live_storage_tmc: float = MAX_LIVE_STORAGE_TMC,
) -> StorageStatus:
    """Calculate live-storage status without substituting missing observations."""
    required = {"calendar_date", "live_above_cill_tmc"}
    missing = required - set(history.columns)
    if missing:
        raise WaterAvailabilityError(
            f"Storage history is missing columns: {sorted(missing)}."
        )
    if history.empty:
        raise WaterAvailabilityError("Storage history has no observations.")
    dates = pd.to_datetime(history["calendar_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise WaterAvailabilityError("Storage dates must be valid and unique.")
    ordered = history.assign(calendar_date=dates).sort_values("calendar_date")
    latest = ordered.iloc[-1]
    observation_date = pd.Timestamp(latest["calendar_date"])
    live_storage = _finite_value(
        latest["live_above_cill_tmc"], "Latest live storage above cill"
    )
    capacity = _finite_value(max_live_storage_tmc, "Maximum live storage")
    if capacity <= 0:
        raise WaterAvailabilityError("Maximum live storage must be positive.")
    if live_storage < 0:
        raise WaterAvailabilityError("Latest live storage cannot be negative.")

    reference_date = pd.Timestamp(as_of or date.today()).normalize()
    age = int((reference_date - observation_date).days)
    if age < 0:
        raise WaterAvailabilityError(
            "Latest storage observation is later than the selected reference date."
        )
    storage_pct = live_storage / capacity * 100.0
    remaining = capacity - live_storage
    date_indexed = ordered.set_index("calendar_date")

    def change(days: int) -> float | None:
        previous = _storage_at_date(
            date_indexed.reset_index(),
            observation_date - pd.Timedelta(days=days),
        )
        return None if previous is None else live_storage - previous

    return StorageStatus(
        observation_date=observation_date,
        observation_age_days=age,
        live_storage_tmc=live_storage,
        max_live_storage_tmc=capacity,
        storage_pct=storage_pct,
        remaining_live_storage_tmc=remaining,
        change_1d_tmc=change(1),
        change_7d_tmc=change(7),
        change_30d_tmc=change(30),
        is_outside_capacity=live_storage > capacity,
    )


def _validated_daily_rainfall(rainfall: pd.DataFrame) -> pd.DataFrame:
    required = {"Monitoring Date", "Rainfall_MM", "Stations_Reporting"}
    missing = required - set(rainfall.columns)
    if missing:
        raise WaterAvailabilityError(
            f"Rainfall data is missing columns: {sorted(missing)}."
        )
    rows = rainfall.loc[:, sorted(required)].copy()
    rows["Monitoring Date"] = pd.to_datetime(
        rows["Monitoring Date"], format="%Y-%m-%d", errors="coerce"
    ).dt.normalize()
    if rows["Monitoring Date"].isna().any():
        raise WaterAvailabilityError("Rainfall data contains invalid dates.")
    if rows["Monitoring Date"].duplicated().any():
        raise WaterAvailabilityError("Rainfall data has duplicate dates.")
    for column in ("Rainfall_MM", "Stations_Reporting"):
        raw = rows[column]
        rows[column] = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & rows[column].isna()).any():
            raise WaterAvailabilityError(
                f"Rainfall column '{column}' contains non-numeric values."
            )
    if rows["Rainfall_MM"].lt(0).any():
        raise WaterAvailabilityError("Rainfall measurements cannot be negative.")
    return rows.sort_values("Monitoring Date").reset_index(drop=True)


def summarize_rainfall(rainfall: pd.DataFrame) -> dict[str, Any]:
    """Summarize the latest available rainfall date and its trailing periods."""
    daily = _validated_daily_rainfall(rainfall)
    measured = daily.loc[daily["Rainfall_MM"].notna()]
    if daily.empty or measured.empty:
        return {
            "latest_date": None,
            "latest_mm": None,
            "latest_stations_reporting": None,
            "source_start_date": None,
            "source_end_date": None,
            "weekly": {"sum_mm": None, "days_available": 0, "days_expected": 7},
            "monthly": {
                "sum_mm": None,
                "days_available": 0,
                "days_expected": 0,
                "period_start": None,
                "period_end": None,
            },
            "daily": daily,
            "monthly_history": pd.DataFrame(),
        }

    latest = measured.iloc[-1]
    latest_date = pd.Timestamp(latest["Monitoring Date"])
    indexed = daily.set_index("Monitoring Date").asfreq("D")
    weekly_dates = pd.date_range(
        latest_date - pd.Timedelta(days=6), latest_date, freq="D"
    )
    weekly_values = indexed["Rainfall_MM"].reindex(weekly_dates)
    weekly_complete = len(weekly_values) == 7 and weekly_values.notna().all()
    weekly_summary = {
        "sum_mm": float(weekly_values.sum()) if weekly_complete else None,
        "days_available": int(weekly_values.notna().sum()),
        "days_expected": 7,
        "period_start": weekly_dates[0],
        "period_end": weekly_dates[-1],
    }

    month_start = latest_date.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_dates = pd.date_range(month_start, month_end, freq="D")
    month_values = indexed["Rainfall_MM"].reindex(month_dates)
    month_complete = (
        latest_date == month_end
        and len(month_values) == len(month_dates)
        and month_values.notna().all()
    )
    monthly_summary = {
        "sum_mm": float(month_values.sum()) if month_complete else None,
        "days_available": int(month_values.notna().sum()),
        "days_expected": int(len(month_dates)),
        "period_start": month_start,
        "period_end": month_end,
    }
    monthly_history = (
        daily.set_index("Monitoring Date")["Rainfall_MM"]
        .resample("MS")
        .agg(["sum", "count"])
        .rename(columns={"sum": "rainfall_sum_mm", "count": "days_available"})
        .reset_index()
        .rename(columns={"Monitoring Date": "month_start"})
    )
    monthly_history["days_expected"] = (
        monthly_history["month_start"] + pd.offsets.MonthEnd(0)
    ).dt.day
    monthly_history["is_complete"] = (
        monthly_history["days_available"].eq(monthly_history["days_expected"])
    )
    monthly_history.loc[~monthly_history["is_complete"], "rainfall_sum_mm"] = np.nan
    return {
        "latest_date": latest_date,
        "latest_mm": float(latest["Rainfall_MM"]),
        "latest_stations_reporting": (
            int(latest["Stations_Reporting"])
            if pd.notna(latest["Stations_Reporting"])
            else None
        ),
        "source_start_date": pd.Timestamp(daily["Monitoring Date"].min()),
        "source_end_date": pd.Timestamp(daily["Monitoring Date"].max()),
        "weekly": weekly_summary,
        "monthly": monthly_summary,
        "daily": daily,
        "monthly_history": monthly_history,
    }


def derive_low_inflow_reference(
    history: pd.DataFrame,
    *,
    train_end: str = "2022-12-31",
) -> float | None:
    """Return a training-period 25th percentile reference, not an official limit."""
    return derive_inflow_references(history, train_end=train_end)["low"]


def derive_inflow_references(
    history: pd.DataFrame,
    *,
    train_end: str = "2022-12-31",
) -> dict[str, float | None]:
    """Return historical training-period inflow quartiles for relative comparisons."""
    if not {"calendar_date", "inflow_cusecs"}.issubset(history.columns):
        return {"low": None, "median": None, "high": None}
    dates = pd.to_datetime(history["calendar_date"], errors="coerce")
    inflow = pd.to_numeric(history["inflow_cusecs"], errors="coerce")
    training = inflow.loc[
        dates.le(pd.Timestamp(train_end)) & inflow.notna() & inflow.ge(0)
    ]
    if training.empty:
        return {"low": None, "median": None, "high": None}
    quantiles = training.quantile([0.25, 0.50, 0.75])
    return {
        "low": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "high": float(quantiles.loc[0.75]),
    }


def assess_water_alert(
    status: StorageStatus,
    *,
    predicted_inflow_cusecs: float | None,
    prediction_date: pd.Timestamp | None = None,
    low_inflow_reference_cusecs: float | None,
    thresholds: dict[str, float] | None = None,
) -> AlertResult:
    """Assign a transparent, provisional level using storage and relative inflow."""
    configured = dict(DEFAULT_STORAGE_THRESHOLDS)
    if thresholds:
        configured.update(thresholds)
    moderate = _finite_value(configured["moderate_pct"], "Moderate threshold")
    low = _finite_value(configured["low_pct"], "Low threshold")
    critical = _finite_value(configured["critical_pct"], "Critical threshold")
    if not (0 <= critical < low < moderate <= 100):
        raise WaterAvailabilityError(
            "Storage alert thresholds must satisfy 0 <= critical < low < moderate <= 100."
        )
    triggered_values: list[str] = [
        f"Live storage: {status.live_storage_tmc:.3f} TMC "
        f"({status.storage_pct:.1f}% of the derived {status.max_live_storage_tmc:.3f} TMC live capacity), "
        f"observed {status.observation_date.date()}."
    ]
    reasons: list[str] = []
    if status.is_outside_capacity:
        return AlertResult(
            level="Insufficient data",
            reasons=(
                "The reported live storage is above the derived maximum live capacity; "
                "verify the source record and capacity definition before classifying availability.",
            ),
            triggered_values=tuple(triggered_values),
            provisional_thresholds=configured,
        )

    if status.storage_pct <= critical:
        level = "Critical Warning"
        reasons.append(
            f"Live storage is at or below the configured provisional critical level "
            f"({critical:.1f}%)."
        )
    elif status.storage_pct <= low:
        level = "Low Storage Warning"
        reasons.append(
            f"Live storage is at or below the configured provisional low level ({low:.1f}%)."
        )
    elif status.storage_pct <= moderate:
        level = "Moderate Warning"
        reasons.append(
            f"Live storage is at or below the configured provisional watch level ({moderate:.1f}%)."
        )
    else:
        level = "Normal"

    if (
        level == "Normal"
        and status.change_7d_tmc is not None
        and status.change_7d_tmc < 0
        and predicted_inflow_cusecs is not None
        and low_inflow_reference_cusecs is not None
        and np.isfinite(predicted_inflow_cusecs)
        and predicted_inflow_cusecs < low_inflow_reference_cusecs
    ):
        level = "Moderate Warning"
        reasons.append(
            "Storage decreased over the previous 7 days and predicted next-day "
            "inflow is below the historical training-period 25th percentile."
        )
    if status.change_7d_tmc is not None:
        triggered_values.append(
            f"7-day live-storage change: {status.change_7d_tmc:+.3f} TMC."
        )
    if predicted_inflow_cusecs is not None and np.isfinite(predicted_inflow_cusecs):
        triggered_values.append(
            f"Predicted next-day inflow: {predicted_inflow_cusecs:,.0f} cusecs"
            + (
                f" for {pd.Timestamp(prediction_date).date()}."
                if prediction_date is not None
                else "."
            )
        )
    if low_inflow_reference_cusecs is not None and np.isfinite(
        low_inflow_reference_cusecs
    ):
        triggered_values.append(
            f"Historical training-period 25th percentile inflow (relative reference, "
            f"not an operating limit): {low_inflow_reference_cusecs:,.0f} cusecs."
        )
    if not reasons:
        reasons.append(
            "Current live storage is above the configured provisional watch level; "
            "this rule result does not guarantee future water availability."
        )
    return AlertResult(
        level=level,
        reasons=tuple(reasons),
        triggered_values=tuple(triggered_values),
        provisional_thresholds=configured,
    )


def _build_legacy_conservation_recommendations(
    status: StorageStatus,
    alert: AlertResult,
    *,
    predicted_inflow_cusecs: float | None,
    prediction_date: pd.Timestamp | None = None,
    low_inflow_reference_cusecs: float | None,
    latest_rainfall_date: pd.Timestamp | None,
    latest_rainfall_mm: float | None = None,
    low_rainfall_reference_mm: float | None = None,
    reference_date: date | pd.Timestamp,
    max_observation_age_days: int,
    max_rainfall_age_days: int,
) -> list[Recommendation]:
    """Create only condition-triggered, explainable non-operational guidance."""
    recommendations: list[Recommendation] = []
    today = pd.Timestamp(reference_date).normalize()

    if status.observation_age_days > max_observation_age_days:
        recommendations.append(
            Recommendation(
                title="Refresh reservoir observations",
                reason=(
                    f"The latest dam observation is {status.observation_age_days} days old, "
                    f"older than the dashboard's {max_observation_age_days}-day freshness setting."
                ),
                supporting_values=(
                    f"Latest observation: {status.observation_date.date()}; "
                    f"dashboard reference date: {today.date()}."
                ),
                priority="Data freshness",
            )
        )

    if alert.level == "Critical Warning":
        recommendations.append(
            Recommendation(
                title="Review essential water allocations with the responsible authority",
                reason=(
                    "Storage crossed the configured provisional critical threshold. "
                    "Allocation decisions must follow approved operating and water-rights rules."
                ),
                supporting_values=(
                    f"Live storage: {status.live_storage_tmc:.3f} TMC "
                    f"({status.storage_pct:.1f}%)."
                ),
                priority="High",
            )
        )
    elif alert.level == "Low Storage Warning":
        recommendations.append(
            Recommendation(
                title="Review upcoming allocations against approved plans",
                reason=(
                    "Storage crossed the configured provisional low threshold; "
                    "the dashboard cannot determine legal or operational allocations."
                ),
                supporting_values=(
                    f"Live storage: {status.live_storage_tmc:.3f} TMC "
                    f"({status.storage_pct:.1f}%)."
                ),
                priority="High",
            )
        )
    elif alert.level == "Moderate Warning":
        inflow_text = (
            f"Predicted next-day inflow: {predicted_inflow_cusecs:,.0f} cusecs"
            + (
                f" for {pd.Timestamp(prediction_date).date()}."
                if prediction_date is not None
                else "."
            )
            if predicted_inflow_cusecs is not None
            and np.isfinite(predicted_inflow_cusecs)
            else "A usable next-day inflow prediction is unavailable."
        )
        reference_text = (
            f"Historical 25th-percentile reference: "
            f"{low_inflow_reference_cusecs:,.0f} cusecs."
            if low_inflow_reference_cusecs is not None
            else "Historical low-inflow reference is unavailable."
        )
        recommendations.append(
            Recommendation(
                title=(
                    "Monitor storage and review planned withdrawals"
                    if status.change_7d_tmc is not None
                    and status.change_7d_tmc < 0
                    else "Review planned allocations and monitor storage"
                ),
                reason=(
                    "Observed live storage has declined over the previous 7 days, "
                    "and the configured rules raised a moderate warning."
                    if status.change_7d_tmc is not None
                    and status.change_7d_tmc < 0
                    else "Storage crossed the configured provisional moderate "
                    "threshold; review plans using approved allocation rules."
                ),
                supporting_values=(
                    (
                        f"7-day change: {status.change_7d_tmc:+.3f} TMC. "
                        if status.change_7d_tmc is not None
                        else "7-day storage change is unavailable. "
                    )
                    + f"{inflow_text} {reference_text}"
                ),
                priority="Moderate",
            )
        )

    if latest_rainfall_date is None:
        recommendations.append(
            Recommendation(
                title="Obtain a rainfall update before rainfall-informed review",
                reason="No measured rainfall observations are available in the loaded archive.",
                supporting_values="Rainfall condition: unavailable.",
                priority="Data availability",
            )
        )
    else:
        rainfall_age = int((today - pd.Timestamp(latest_rainfall_date).normalize()).days)
        if rainfall_age < 0 or rainfall_age > max_rainfall_age_days:
            recommendations.append(
                Recommendation(
                    title="Refresh rainfall observations",
                    reason=(
                        f"The latest rainfall observation is {rainfall_age} days old "
                        "and must not be interpreted as current rainfall."
                    ),
                    supporting_values=(
                        f"Last measured rainfall date: "
                        f"{pd.Timestamp(latest_rainfall_date).date()}."
                    ),
                    priority="Data availability",
                )
            )
        elif (
            latest_rainfall_mm is not None
            and low_rainfall_reference_mm is not None
            and predicted_inflow_cusecs is not None
            and low_inflow_reference_cusecs is not None
            and np.isfinite(latest_rainfall_mm)
            and np.isfinite(low_rainfall_reference_mm)
            and np.isfinite(predicted_inflow_cusecs)
            and np.isfinite(low_inflow_reference_cusecs)
            and latest_rainfall_mm <= low_rainfall_reference_mm
            and predicted_inflow_cusecs < low_inflow_reference_cusecs
        ):
            recommendations.append(
                Recommendation(
                    title="Prepare for potentially reduced water availability",
                    reason=(
                        "Fresh measured rainfall is at or below its historical "
                        "10th-percentile reference and predicted inflow is below "
                        "its training-period 25th-percentile reference. These are "
                        "relative indicators, not a shortage forecast."
                    ),
                    supporting_values=(
                        f"Rainfall {latest_rainfall_mm:.2f} mm on "
                        f"{pd.Timestamp(latest_rainfall_date).date()} "
                        f"(historical 10th percentile {low_rainfall_reference_mm:.2f} mm); "
                        f"predicted inflow {predicted_inflow_cusecs:,.0f} cusecs"
                        + (
                            f" for {pd.Timestamp(prediction_date).date()} "
                            if prediction_date is not None
                            else " "
                        )
                        + f"(historical 25th percentile "
                        f"{low_inflow_reference_cusecs:,.0f} cusecs)."
                    ),
                    priority="Moderate",
                )
            )

    if (
        not recommendations
        and alert.level == "Normal"
        and status.change_7d_tmc is not None
        and status.change_7d_tmc >= 0
        and status.observation_age_days <= max_observation_age_days
    ):
        recommendations.append(
            Recommendation(
                title="Continue routine storage monitoring",
                reason=(
                    "The latest observation is fresh, storage is above the configured "
                    "provisional watch level, and the 7-day live-storage change is non-negative."
                ),
                supporting_values=(
                    f"Storage: {status.storage_pct:.1f}% of derived live capacity; "
                    f"7-day change: {status.change_7d_tmc:+.3f} TMC."
                ),
                priority="Routine",
            )
        )
    return recommendations


def build_conservation_recommendations(
    status: StorageStatus,
    alert: AlertResult,
    *,
    predicted_inflow_cusecs: float | None,
    prediction_date: pd.Timestamp | None = None,
    low_inflow_reference_cusecs: float | None,
    high_inflow_reference_cusecs: float | None = None,
    latest_inflow_cusecs: float | None = None,
    latest_inflow_date: pd.Timestamp | None = None,
    latest_outflow_cusecs: float | None = None,
    latest_outflow_date: pd.Timestamp | None = None,
    latest_rainfall_date: pd.Timestamp | None,
    latest_rainfall_mm: float | None = None,
    low_rainfall_reference_mm: float | None = None,
    high_rainfall_reference_mm: float | None = None,
    reference_date: date | pd.Timestamp,
    max_observation_age_days: int,
    max_rainfall_age_days: int,
) -> list[Recommendation]:
    """Generate one explainable storage/inflow recommendation plus data notices."""
    today = pd.Timestamp(reference_date).normalize()
    dam_is_fresh = status.observation_age_days <= max_observation_age_days
    valid_prediction = (
        predicted_inflow_cusecs is not None
        and np.isfinite(predicted_inflow_cusecs)
        and predicted_inflow_cusecs >= 0
    )
    valid_low_reference = (
        low_inflow_reference_cusecs is not None
        and np.isfinite(low_inflow_reference_cusecs)
    )
    valid_high_reference = (
        high_inflow_reference_cusecs is not None
        and np.isfinite(high_inflow_reference_cusecs)
    )
    valid_latest_inflow = (
        latest_inflow_cusecs is not None
        and np.isfinite(latest_inflow_cusecs)
        and latest_inflow_cusecs >= 0
    )
    valid_outflow = (
        latest_outflow_cusecs is not None
        and np.isfinite(latest_outflow_cusecs)
        and latest_outflow_cusecs >= 0
    )

    rainfall_age = (
        int((today - pd.Timestamp(latest_rainfall_date).normalize()).days)
        if latest_rainfall_date is not None
        else None
    )
    rainfall_is_fresh = (
        rainfall_age is not None
        and 0 <= rainfall_age <= max_rainfall_age_days
        and latest_rainfall_mm is not None
        and np.isfinite(latest_rainfall_mm)
    )
    if latest_rainfall_date is None:
        rainfall_evidence = (
            "Rainfall unavailable; it was not used to characterize current conditions."
        )
    elif latest_rainfall_mm is None or not np.isfinite(latest_rainfall_mm):
        rainfall_evidence = (
            f"Rainfall measurement unavailable for {pd.Timestamp(latest_rainfall_date).date()}."
        )
    elif rainfall_is_fresh:
        rainfall_evidence = (
            f"Measured rainfall {latest_rainfall_mm:.2f} mm on "
            f"{pd.Timestamp(latest_rainfall_date).date()} "
            "(recent context only; not a forecast)."
        )
    else:
        rainfall_evidence = (
            f"Latest measured rainfall {latest_rainfall_mm:.2f} mm on "
            f"{pd.Timestamp(latest_rainfall_date).date()} is stale and was not "
            "treated as current."
        )

    storage_evidence = (
        f"Recorded storage {status.live_storage_tmc:.3f} TMC "
        f"({status.storage_pct:.1f}%) on {status.observation_date.date()}."
    )
    change_evidence = (
        (
            f"Observed 7-day storage change {status.change_7d_tmc:+.3f} TMC."
            if status.change_7d_tmc is not None
            else "Observed 7-day storage change unavailable."
        )
        + (
            f" Observed 30-day storage change {status.change_30d_tmc:+.3f} TMC."
            if status.change_30d_tmc is not None
            else " Observed 30-day storage change unavailable."
        )
    )
    if valid_prediction:
        date_text = (
            f" for {pd.Timestamp(prediction_date).date()}"
            if prediction_date is not None
            else " (prediction date unavailable)"
        )
        prediction_evidence = (
            f"Predicted next-day inflow {predicted_inflow_cusecs:,.0f} cusecs"
            f"{date_text} (model estimate, not a measurement)."
        )
    else:
        prediction_evidence = (
            "Predicted next-day inflow unavailable; no value was substituted."
        )
    if valid_outflow:
        outflow_date_text = (
            f" on {pd.Timestamp(latest_outflow_date).date()}"
            if latest_outflow_date is not None
            else " (date unavailable)"
        )
        outflow_evidence = (
            f"Latest observed outflow {latest_outflow_cusecs:,.0f} cusecs"
            f"{outflow_date_text} (measured, not forecast)."
        )
    else:
        outflow_evidence = "Latest observed outflow unavailable."
    if valid_latest_inflow:
        inflow_date_text = (
            f" on {pd.Timestamp(latest_inflow_date).date()}"
            if latest_inflow_date is not None
            else " (date unavailable)"
        )
        observed_inflow_evidence = (
            f"Latest recorded inflow {latest_inflow_cusecs:,.0f} cusecs"
            f"{inflow_date_text} (measured, not forecast)."
        )
    else:
        observed_inflow_evidence = "Latest recorded inflow unavailable."
    if valid_low_reference and valid_high_reference:
        historical_evidence = (
            f"Historical training-period inflow references: 25th percentile "
            f"{low_inflow_reference_cusecs:,.0f}, 75th percentile "
            f"{high_inflow_reference_cusecs:,.0f} cusecs; relative comparisons, "
            "not operating limits."
        )
    elif valid_low_reference:
        historical_evidence = (
            f"Historical training-period 25th-percentile inflow reference "
            f"{low_inflow_reference_cusecs:,.0f} cusecs; high reference unavailable."
        )
    else:
        historical_evidence = "Historical inflow references unavailable."
    evidence = " ".join(
        (
            storage_evidence,
            change_evidence,
            prediction_evidence,
            outflow_evidence,
            observed_inflow_evidence,
            historical_evidence,
            rainfall_evidence,
        )
    )

    thresholds = alert.provisional_thresholds
    low_storage = status.storage_pct <= thresholds["low_pct"]
    watch_storage = status.storage_pct <= thresholds["moderate_pct"]
    high_storage = not watch_storage
    declining = status.change_7d_tmc is not None and status.change_7d_tmc < 0
    rising = status.change_7d_tmc is not None and status.change_7d_tmc > 0
    inflow_low = (
        valid_prediction
        and valid_low_reference
        and predicted_inflow_cusecs < low_inflow_reference_cusecs
    )
    inflow_high = (
        valid_prediction
        and valid_high_reference
        and predicted_inflow_cusecs >= high_inflow_reference_cusecs
    )
    below_latest_outflow = (
        valid_prediction
        and valid_outflow
        and predicted_inflow_cusecs < latest_outflow_cusecs
    )
    heavy_rain = (
        rainfall_is_fresh
        and high_rainfall_reference_mm is not None
        and np.isfinite(high_rainfall_reference_mm)
        and latest_rainfall_mm >= high_rainfall_reference_mm
    )

    category = "Storage-based recommendation"
    title = "Continue monitoring reservoir storage"
    action = (
        "Use approved operating procedures and reassess as new measurements arrive."
    )
    reason = (
        "The available values do not meet a more specific storage/inflow combination "
        "rule; no release or allocation change is inferred."
    )
    priority = "Routine"

    if valid_prediction and (valid_low_reference or valid_high_reference):
        category = "Combined water conservation recommendation"
        if heavy_rain and inflow_high:
            title = "Monitor measured reservoir levels after high rainfall and inflow"
            action = (
                "Track actual reservoir levels and follow approved reservoir and "
                "flood-management procedures; do not infer a release action from "
                "rainfall or the model estimate."
            )
            reason = (
                "Recent measured rainfall is at or above the archive 90th-percentile "
                "reference and predicted inflow is at or above the training-period "
                "75th-percentile reference. This co-occurrence does not establish "
                "that rainfall caused the predicted inflow."
            )
            priority = "Moderate"
        elif low_storage and inflow_low:
            title = "Prepare for potentially reduced water availability"
            action = (
                "Review planned allocations with the responsible authority and "
                "prepare contingency plans under approved rules."
            )
            reason = (
                "Storage is at or below the configured provisional low threshold "
                "and predicted inflow is below the historical training-period "
                "25th-percentile reference. This is a relative planning signal, "
                "not a shortage forecast."
            )
            priority = "High"
        elif low_storage and inflow_high:
            title = "Wait for predicted inflow to be measured before revising plans"
            action = (
                "Monitor measured storage after the prediction date; do not count "
                "forecast inflow as water already stored."
            )
            reason = (
                "Storage is at or below the configured provisional low threshold, "
                "while predicted inflow is at or above the historical 75th-percentile "
                "reference. The estimate may indicate improvement, but no inflow "
                "has been confirmed."
            )
            priority = "Moderate"
        elif declining and below_latest_outflow:
            title = "Review planned releases and water allocations"
            action = (
                "Compare release and allocation plans with approved operating rules; "
                "this dashboard does not recommend changing a release."
            )
            reason = (
                "Observed storage declined over seven days and predicted next-day "
                "inflow is below the latest measured outflow. Those values refer to "
                "different dates, so this is a review signal only."
            )
            priority = "High"
        elif high_storage and inflow_low:
            title = "Use water efficiently and monitor future inflow"
            action = (
                "Continue efficient water use and reassess after the next measured "
                "inflow and storage update."
            )
            reason = (
                "Storage is above the configured provisional watch threshold, "
                "but predicted inflow is below the historical 25th-percentile reference."
            )
            priority = "Moderate"
        elif high_storage and inflow_high:
            title = "Monitor storage capacity after the predicted inflow"
            action = (
                "Track measured storage after the prediction date and follow approved "
                "reservoir-management procedures; do not treat the forecast as received water."
            )
            reason = (
                "Storage is above the configured provisional watch threshold and "
                "predicted inflow is at or above the historical 75th-percentile reference."
                + (
                    " The observed seven-day storage trend is also positive."
                    if rising
                    else " A positive seven-day storage trend was not established."
                )
            )
            priority = "Moderate"
        elif not low_storage and not high_storage and inflow_high and rising:
            title = "Continue efficient water management and monitor capacity"
            action = (
                "Continue efficient water use and compare capacity with measured "
                "storage after the forecast target date."
            )
            reason = (
                "The observed seven-day storage change is positive and predicted "
                "inflow is at or above the historical 75th-percentile reference, "
                "while storage remains between the configured provisional low and "
                "watch thresholds."
            )
            priority = "Routine"
        elif (
            not low_storage
            and not high_storage
            and not inflow_low
            and not inflow_high
            and rising
        ):
            category = "Combined water conservation recommendation"
            title = "Continue efficient water management and monitor capacity"
            action = (
                "Maintain efficient water use and review measured storage after "
                "the forecast target date."
            )
            reason = (
                "Storage remains between the configured provisional low and watch "
                "thresholds, seven-day storage change is positive, and predicted "
                "inflow is within historical training-period quartile references."
            )
            priority = "Routine"
        elif inflow_low:
            category = "Inflow-prediction-based recommendation"
            title = "Conserve water and monitor the next inflow observation"
            action = (
                "Continue efficient water use and reassess after the prediction date "
                "using measured data."
            )
            reason = (
                "Predicted inflow is below the historical training-period "
                "25th-percentile reference; this is not a shortage forecast or "
                "operating limit."
            )
            priority = "Moderate"
        elif inflow_high:
            category = "Inflow-prediction-based recommendation"
            title = "Confirm the predicted inflow with a measured update"
            action = (
                "Reassess storage and plans after the predicted inflow is recorded; "
                "do not act as though it has already occurred."
            )
            reason = (
                "Predicted inflow is at or above the historical training-period "
                "75th-percentile reference."
            )
            priority = "Routine"
        else:
            stable_conditions = (
                alert.level == "Normal"
                and not declining
                and valid_low_reference
                and valid_high_reference
                and low_inflow_reference_cusecs
                <= predicted_inflow_cusecs
                < high_inflow_reference_cusecs
                and dam_is_fresh
            )
            if stable_conditions:
                category = "Water Conservation Actions"
                title = (
                    "Reservoir conditions are relatively stable. Continue efficient "
                    "water management and regular monitoring."
                )
                action = title
                reason = (
                    "The storage alert is normal, the seven-day storage trend is not "
                    "declining, and the predicted inflow lies between historical "
                    "training-period quartile references."
                )
                priority = "Routine"
            else:
                title = "Monitor storage and verify the forecast against measurements"
                action = (
                    "Continue approved conservation practices and compare the next "
                    "measured inflow/storage update with this forecast."
                )
                reason = (
                    "Predicted inflow falls between the available historical low/high "
                    "quartile references and is not classified here as unusually low "
                    "or high. The recommendation is to wait for measured confirmation."
                )
                priority = "Routine"
    elif low_storage:
        title = "Review upcoming allocations against approved plans"
        action = (
            "Review planned allocations with the responsible authority under "
            "approved operating and water-rights rules."
        )
        reason = (
            "Storage is at or below the configured provisional low threshold, but "
            "a valid forecast/history comparison is unavailable."
        )
        priority = "High"
    elif alert.level == "Moderate Warning":
        title = "Monitor storage and review planned allocations"
        action = "Review plans against approved rules using verified measurements."
        reason = (
            "Storage is at or below the configured provisional watch threshold. "
            "Prediction-based conditions could not be fully evaluated."
        )
        priority = "Moderate"
    elif declining:
        title = "Monitor the observed storage decline"
        action = "Review upcoming plans using approved procedures and updated measurements."
        reason = "The observed seven-day storage change is negative."
        priority = "Moderate"

    action_items = []
    reason_items = []
    if watch_storage:
        action_items.append(
            "Review planned water allocations under approved rules, with priority "
            "for essential needs determined by the responsible authority."
        )
        reason_items.append(
            f"Recorded storage is {status.storage_pct:.1f}%, at or below the "
            f"configured provisional watch threshold of "
            f"{thresholds['moderate_pct']:.1f}%."
        )
    if rising:
        action_items.append(
            "Continue monitoring storage; the recent seven-day trend is positive, "
            "but it does not guarantee future availability."
        )
        reason_items.append(
            f"The seven-day storage trend is positive: storage increased by "
            f"{status.change_7d_tmc:+.3f} TMC."
        )
    elif declining:
        action_items.append(
            "Review planned withdrawals and allocations against approved rules "
            "using updated measurements."
        )
        reason_items.append(
            f"Observed storage decreased by {status.change_7d_tmc:.3f} TMC "
            "over seven days."
        )
    if (
        inflow_low
        and rainfall_is_fresh
        and low_rainfall_reference_mm is not None
        and np.isfinite(low_rainfall_reference_mm)
        and latest_rainfall_mm <= low_rainfall_reference_mm
    ):
        reason_items.append(
            "Recent measured rainfall is also at or below its historical "
            "10th-percentile reference; rainfall remains context, not a causal "
            "or shortage determination."
        )
    if heavy_rain and inflow_high:
        reason_items.append(
            "Recent measured rainfall is high relative to its archive reference, "
            "but this does not establish that rainfall caused the predicted inflow."
        )
    if inflow_low:
        action_items.append(
            "Conserve water and compare the estimate with the next measured inflow; "
            "if allocation changes are considered, use approved plans and prioritize "
            "essential needs through the responsible authority."
        )
        reason_items.append(
            f"Predicted inflow of {predicted_inflow_cusecs:,.0f} cusecs is below "
            f"the historical 25th-percentile reference of "
            f"{low_inflow_reference_cusecs:,.0f} cusecs; this is a relative signal, "
            "not a shortage forecast."
        )
    elif inflow_high:
        action_items.append(
            "Compare the estimate with the next actual inflow and measured storage; "
            "do not count forecast inflow as water already stored."
        )
        reason_items.append(
            f"Predicted inflow of {predicted_inflow_cusecs:,.0f} cusecs is at or "
            f"above the historical 75th-percentile reference of "
            f"{high_inflow_reference_cusecs:,.0f} cusecs."
        )
        if heavy_rain:
            action_items.append(
                "Follow approved reservoir and flood-management procedures; do not "
                "infer a release action from rainfall or the model estimate."
            )
    elif valid_prediction and valid_low_reference and valid_high_reference:
        action_items.append(
            "Compare predicted inflow with the next actual measurement before "
            "revising water plans."
        )
        reason_items.append(
            f"Predicted inflow of {predicted_inflow_cusecs:,.0f} cusecs is within "
            f"the historical 25th–75th percentile range "
            f"({low_inflow_reference_cusecs:,.0f}–"
            f"{high_inflow_reference_cusecs:,.0f} cusecs), so it is not unusually "
            "low or high relative to that reference period."
        )
    else:
        action_items.append(
            "Continue efficient water use and reassess when a valid forecast and "
            "new measurements are available."
        )
        reason_items.append(
            "A valid forecast comparison with historical inflow references is "
            "unavailable."
        )

    action = "\n".join(f"- {item}" for item in action_items)
    reason = " ".join(reason_items)
    if watch_storage:
        title = "Review water allocation and monitor reservoir availability."
        priority = "High" if low_storage or inflow_low else "Moderate"
        category = "Combined water conservation recommendation"
    elif heavy_rain and inflow_high:
        title = "Monitor measured reservoir levels after high rainfall and inflow"
        priority = "Moderate"
    elif high_storage and inflow_low:
        title = "Use water efficiently and monitor future inflow"
        priority = "Moderate"
    elif high_storage and inflow_high:
        title = "Monitor storage capacity after the predicted inflow"
        priority = "Moderate"
    elif declining and below_latest_outflow:
        title = "Review planned releases and water allocations"
        priority = "High"
    elif inflow_low:
        title = "Conserve water and monitor the next inflow observation"
        priority = "Moderate"
    elif inflow_high:
        title = "Confirm the predicted inflow with a measured update"
        priority = "Routine"
    elif rising:
        title = "Acknowledge the positive storage trend and monitor availability"
        priority = "Routine"
    else:
        title = "Continue efficient water use and monitor reservoir availability"
        priority = "Routine"

    recommendations = [
        Recommendation(
            category=category,
            title=title,
            action=action,
            reason=reason,
            supporting_values=evidence,
            priority=priority,
        )
    ]
    if not dam_is_fresh:
        recommendations.append(
            Recommendation(
                category="Data freshness",
                title="Refresh reservoir observations",
                action=(
                    "Obtain newer verified reservoir observations before making "
                    "current operational decisions."
                ),
                reason=(
                    f"The latest reservoir record is {status.observation_age_days} "
                    f"days old, exceeding the configured {max_observation_age_days}-day "
                    "freshness setting."
                ),
                supporting_values=evidence,
                priority="Data freshness",
            )
        )
    if valid_prediction and (
        prediction_date is None
        or pd.Timestamp(prediction_date).normalize() <= today
    ):
        recommendations.append(
            Recommendation(
                category="Data freshness",
                title="Forecast target date has passed",
                action=(
                    "Use a forecast based on the latest available observations for "
                    "current planning; compare this historical estimate with its "
                    "actual measurement when available."
                ),
                reason=(
                    "The prediction target date is not later than the dashboard "
                    "reference date, so the estimate is historical rather than a "
                    "current next-day forecast."
                ),
                supporting_values=evidence,
                priority="Data freshness",
            )
        )
    if latest_rainfall_date is None:
        recommendations.append(
            Recommendation(
                category="Data availability",
                title="Rainfall context is unavailable",
                action="Obtain a verified rainfall update before rainfall-informed review.",
                reason="No measured rainfall observations are available.",
                supporting_values=evidence,
                priority="Data availability",
            )
        )
    elif rainfall_age is not None and (
        rainfall_age < 0 or rainfall_age > max_rainfall_age_days
    ):
        recommendations.append(
            Recommendation(
                category="Data freshness",
                title="Refresh rainfall observations",
                action="Use a recent measured rainfall record for rainfall-informed review.",
                reason=(
                    f"The latest rainfall observation is {rainfall_age} days old "
                    "and was not treated as current."
                ),
                supporting_values=evidence,
                priority="Data freshness",
            )
        )
    elif not rainfall_is_fresh:
        recommendations.append(
            Recommendation(
                category="Data availability",
                title="Rainfall measurement is unavailable",
                action="Check the rainfall source before rainfall-informed planning.",
                reason="A rainfall date exists but its measured value is missing or invalid.",
                supporting_values=evidence,
                priority="Data availability",
            )
        )
    return recommendations
