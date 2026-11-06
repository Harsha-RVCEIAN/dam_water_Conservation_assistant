"""
Canonical water-balance module for K.R.Sagara Dam.

All flow rates are assumed to represent the mean rate over the exact interval.
A complete balance requires aligned dates, valid storage and flow measurements,
and verified evaporation. A discrepancy is not proof of dam leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from src.utils.unit_conversions import cusecs_to_tmc


WATER_BALANCE_CONFIG_PATH = Path("config/water_balance.yaml")
AUDIT_REPORT_PATH = Path("reports/water_balance_audit.csv")
SUPPORTED_STORAGE_BASES = {"live_above_cill_tmc", "gross_capacity_tmc"}


def load_water_balance_config(
    config_path: Path = WATER_BALANCE_CONFIG_PATH,
) -> Dict[str, Any]:
    """Load configuration with conservative defaults."""
    defaults = {
        "balance": {
            "flow_conversion_factor": 0.0000864,
            "storage_basis": "live_above_cill_tmc",
            "evaporation_status": "convention_unverified",
            "allow_incomplete_discrepancy": False,
            "discrepancy_thresholds": {"moderate_tmc": 2.0, "extreme_tmc": 5.0},
            "leakage_disclaimer": "Water-balance discrepancy is not proof of leakage.",
        }
    }
    if not config_path.exists():
        return defaults

    with open(config_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Water-balance configuration must be a YAML mapping.")

    result = {"balance": defaults["balance"].copy()}
    configured = loaded.get("balance", {})
    if isinstance(configured, dict):
        result["balance"].update(configured)
    return result


@dataclass
class WaterBalanceResult:
    start_date: str
    end_date: str
    elapsed_days: int
    storage_basis: str
    observed_storage_change_tmc: Optional[float]
    inflow_tmc: Optional[float]
    river_outflow_tmc: Optional[float]
    canal_withdrawal_tmc: Optional[float]
    evaporation_tmc: Optional[float]
    expected_storage_change_tmc: Optional[float]
    balance_discrepancy_tmc: Optional[float]
    diagnostic_net_flow_excluding_evap_tmc: Optional[float]
    diagnostic_discrepancy_excluding_evap_tmc: Optional[float]
    status: str
    reasons: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    disclaimer: str = "Water-balance discrepancy is not proof of physical dam leakage."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _valid_nonnegative(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(number) and number >= 0)


def _as_valid_float(value: Any) -> Optional[float]:
    return float(value) if _valid_nonnegative(value) else None


def _date_label(value: Any) -> str:
    try:
        parsed = pd.to_datetime(value, errors="raise")
        return "INVALID_DATE" if pd.isna(parsed) else str(parsed.date())
    except (TypeError, ValueError, OverflowError):
        return "INVALID_DATE"


def compute_water_balance_interval(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    start_storage_tmc: Optional[float],
    end_storage_tmc: Optional[float],
    inflow_cusecs: Optional[float],
    river_outflow_cusecs: Optional[float],
    canal_withdrawal_cusecs: Optional[float],
    evaporation_cusecs: Optional[float] = None,
    storage_basis: str = "live_above_cill_tmc",
    expected_days: int = 1,
    evaporation_verified: bool = False,
) -> WaterBalanceResult:
    """Compute a balance only when every required term is valid and aligned."""
    reasons: List[str] = []
    flags: List[str] = []

    if storage_basis not in SUPPORTED_STORAGE_BASES:
        reasons.append("unsupported_storage_basis")

    expected_days_valid = (
        not isinstance(expected_days, bool)
        and isinstance(expected_days, (int, np.integer))
        and expected_days > 0
    )
    if not expected_days_valid:
        reasons.append("invalid_expected_days")

    dates_valid = False
    elapsed_days = 0
    try:
        if start_date is None or end_date is None:
            reasons.append("invalid_or_missing_interval_dates")
        else:
            start_ts = pd.to_datetime(start_date, errors="raise")
            end_ts = pd.to_datetime(end_date, errors="raise")
            if pd.isna(start_ts) or pd.isna(end_ts):
                reasons.append("invalid_or_missing_interval_dates")
            else:
                elapsed_days = int((end_ts.normalize() - start_ts.normalize()).days)
                if elapsed_days == 0:
                    reasons.append("same_start_and_end_date")
                elif elapsed_days < 0:
                    reasons.append("reversed_interval_dates")
                else:
                    dates_valid = True
    except (TypeError, ValueError, OverflowError):
        elapsed_days = 0
        dates_valid = False
        reasons.append("invalid_or_missing_interval_dates")

    if dates_valid and expected_days_valid and elapsed_days != expected_days:
        reasons.append(
            f"elapsed_days_mismatch (expected {expected_days}, observed {elapsed_days})"
        )
    interval_aligned = (
        dates_valid and expected_days_valid and elapsed_days == expected_days
    )

    start_storage = _as_valid_float(start_storage_tmc)
    end_storage = _as_valid_float(end_storage_tmc)

    for name, original, valid in (
        ("start_storage", start_storage_tmc, start_storage),
        ("end_storage", end_storage_tmc, end_storage),
    ):
        if original is None or pd.isna(original):
            reasons.append(f"missing_{name}")
        elif valid is None:
            reasons.append(f"invalid_{name}")
            flags.append(f"invalid_{name}")

    flow_values = {
        "inflow": _as_valid_float(inflow_cusecs),
        "river_outflow": _as_valid_float(river_outflow_cusecs),
        "canal_withdrawal": _as_valid_float(canal_withdrawal_cusecs),
    }
    original_flows = {
        "inflow": inflow_cusecs,
        "river_outflow": river_outflow_cusecs,
        "canal_withdrawal": canal_withdrawal_cusecs,
    }
    for name, original in original_flows.items():
        if original is None or (
            isinstance(original, (float, np.floating)) and np.isnan(original)
        ):
            reasons.append(f"missing_{name}")
        elif flow_values[name] is None:
            reasons.append(f"invalid_{name}")
            flags.append(f"invalid_{name}")

    evaporation = _as_valid_float(evaporation_cusecs)
    if evaporation_cusecs is None or (
        isinstance(evaporation_cusecs, (float, np.floating))
        and np.isnan(evaporation_cusecs)
    ):
        reasons.append("missing_evaporation")
    elif evaporation is None:
        reasons.append("invalid_evaporation")
        flags.append("invalid_evaporation")
    elif not evaporation_verified:
        reasons.append("evaporation_source_convention_unverified")

    observed_change = None
    if start_storage is not None and end_storage is not None:
        observed_change = round(end_storage - start_storage, 5)

    in_tmc = out_tmc = with_tmc = evap_tmc = None
    if interval_aligned:
        hours = elapsed_days * 24.0
        if flow_values["inflow"] is not None:
            in_tmc = cusecs_to_tmc(flow_values["inflow"], hours=hours)
        if flow_values["river_outflow"] is not None:
            out_tmc = cusecs_to_tmc(flow_values["river_outflow"], hours=hours)
        if flow_values["canal_withdrawal"] is not None:
            with_tmc = cusecs_to_tmc(flow_values["canal_withdrawal"], hours=hours)
        if evaporation is not None and evaporation_verified:
            evap_tmc = cusecs_to_tmc(evaporation, hours=hours)

    # This is diagnostic only and is intentionally separate from the complete
    # balance. Do not calculate it across a missing or mismatched interval.
    diagnostic_net = diagnostic_discrepancy = None
    if (
        interval_aligned
        and in_tmc is not None
        and out_tmc is not None
        and with_tmc is not None
    ):
        diagnostic_net = round(in_tmc - out_tmc - with_tmc, 5)
        if observed_change is not None:
            diagnostic_discrepancy = round(observed_change - diagnostic_net, 5)

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        status = (
            "UNVERIFIED_EVAPORATION"
            if reasons == ["evaporation_source_convention_unverified"]
            else "INCOMPLETE"
        )
        return WaterBalanceResult(
            start_date=_date_label(start_date),
            end_date=_date_label(end_date),
            elapsed_days=elapsed_days,
            storage_basis=storage_basis,
            observed_storage_change_tmc=observed_change,
            inflow_tmc=round(in_tmc, 5) if in_tmc is not None else None,
            river_outflow_tmc=round(out_tmc, 5) if out_tmc is not None else None,
            canal_withdrawal_tmc=round(with_tmc, 5) if with_tmc is not None else None,
            evaporation_tmc=round(evap_tmc, 5) if evap_tmc is not None else None,
            expected_storage_change_tmc=None,
            balance_discrepancy_tmc=None,
            diagnostic_net_flow_excluding_evap_tmc=diagnostic_net,
            diagnostic_discrepancy_excluding_evap_tmc=diagnostic_discrepancy,
            status=status,
            reasons=reasons,
            quality_flags=flags,
        )

    expected_change = round(in_tmc - out_tmc - with_tmc - evap_tmc, 5)
    discrepancy = (
        round(observed_change - expected_change, 5)
        if observed_change is not None
        else None
    )
    return WaterBalanceResult(
        start_date=_date_label(start_date),
        end_date=_date_label(end_date),
        elapsed_days=elapsed_days,
        storage_basis=storage_basis,
        observed_storage_change_tmc=observed_change,
        inflow_tmc=round(in_tmc, 5),
        river_outflow_tmc=round(out_tmc, 5),
        canal_withdrawal_tmc=round(with_tmc, 5),
        evaporation_tmc=round(evap_tmc, 5),
        expected_storage_change_tmc=expected_change,
        balance_discrepancy_tmc=discrepancy,
        diagnostic_net_flow_excluding_evap_tmc=diagnostic_net,
        diagnostic_discrepancy_excluding_evap_tmc=diagnostic_discrepancy,
        status="COMPLETE",
        reasons=[],
        quality_flags=flags,
    )


def audit_historical_water_balance(
    df: pd.DataFrame,
    storage_basis: str = "live_above_cill_tmc",
    output_path: Path = AUDIT_REPORT_PATH,
) -> pd.DataFrame:
    """Audit consecutive records; no missing dates are interpolated."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required = ["inflow_cusecs", "outflow_cusecs", "withdrawal_cusecs"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required audit columns: " + ", ".join(missing))

    working = df.copy()
    if "calendar_date" not in working.columns:
        if "monitoring_date" not in working.columns:
            raise ValueError(
                "DataFrame must contain 'calendar_date' or 'monitoring_date'."
            )
        from src.data.validate import parse_monitoring_dates
        working["calendar_date"] = parse_monitoring_dates(
            working["monitoring_date"]
        )

    working["calendar_date"] = pd.to_datetime(
        working["calendar_date"], errors="coerce"
    ).dt.normalize()
    working = (
        working.dropna(subset=["calendar_date"])
        .sort_values("calendar_date")
        .reset_index(drop=True)
    )

    if storage_basis not in working.columns:
        raise ValueError(f"Storage column '{storage_basis}' is missing.")

    records: List[Dict[str, Any]] = []
    for index in range(1, len(working)):
        previous = working.iloc[index - 1]
        current = working.iloc[index]

        result = compute_water_balance_interval(
            start_date=previous["calendar_date"],
            end_date=current["calendar_date"],
            start_storage_tmc=previous.get(storage_basis),
            end_storage_tmc=current.get(storage_basis),
            inflow_cusecs=current.get("inflow_cusecs"),
            river_outflow_cusecs=current.get("outflow_cusecs"),
            canal_withdrawal_cusecs=current.get("withdrawal_cusecs"),
            evaporation_cusecs=current.get("evaporation_cusecs"),
            storage_basis=storage_basis,
            expected_days=1,
            evaporation_verified=False,
        )
        record = result.to_dict()
        record["reasons"] = "; ".join(result.reasons)
        record["quality_flags"] = "; ".join(result.quality_flags)
        records.append(record)

    audit_df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    return audit_df
