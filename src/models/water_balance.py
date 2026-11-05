"""Canonical water-balance module for K.R.Sagara Dam.

Implements simplified mass-balance accounting, interval alignment, completeness checking,
and diagnostic discrepancy auditing.

Formulas:
  volume_tmc = flow_cusecs * 0.0000864 * (elapsed_hours / 24)
  expected_storage_change_tmc = inflow_tmc - river_outflow_tmc - canal_withdrawal_tmc - evaporation_tmc
  observed_storage_change_tmc = storage_end_tmc - storage_start_tmc
  balance_discrepancy_tmc = observed_storage_change_tmc - expected_storage_change_tmc

Safety & Physics Rules:
1. All terms must cover the exact same elapsed time interval and use compatible TMC units.
2. Missing evaporation or missing dates cannot produce a falsely complete balance.
3. Unverified evaporation convention blocks a complete balance; discrepancy remains None.
4. Departures from simplified water balance reflect unmeasured runoff, rainfall, timing delays,
   or gauge calibration; they do NOT prove dam leakage.
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


def load_water_balance_config(config_path: Path = WATER_BALANCE_CONFIG_PATH) -> Dict[str, Any]:
    """Load water balance configuration."""
    if not config_path.exists():
        return {
            "balance": {
                "flow_conversion_factor": 0.0000864,
                "storage_basis": "live_above_cill_tmc",
                "evaporation_status": "convention_unverified",
                "allow_incomplete_discrepancy": False,
                "discrepancy_thresholds": {"moderate_tmc": 2.0, "extreme_tmc": 5.0},
                "leakage_disclaimer": "Water balance discrepancy is not proof of leakage.",
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    # Diagnostic balance excluding evaporation for engineering analysis
    diagnostic_net_flow_excluding_evap_tmc: Optional[float]
    diagnostic_discrepancy_excluding_evap_tmc: Optional[float]
    status: str  # "COMPLETE", "INCOMPLETE", "UNVERIFIED_EVAPORATION"
    reasons: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    disclaimer: str = "Discrepancy is not proof of dam leakage."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    """Compute simplified water balance across an aligned observation interval.

    Enforces strict interval duration and marks balance incomplete if any required
    flow or storage reading is missing or unverified.
    """
    t_start = pd.to_datetime(start_date)
    t_end = pd.to_datetime(end_date)
    elapsed_days = int((t_end.normalize() - t_start.normalize()).days)

    reasons: List[str] = []
    flags: List[str] = []

    # Check elapsed interval
    if elapsed_days != expected_days:
        reasons.append(f"elapsed_days_mismatch (expected {expected_days}, observed {elapsed_days})")

    # Check storage inputs
    if start_storage_tmc is None or pd.isna(start_storage_tmc):
        reasons.append("missing_start_storage")
    if end_storage_tmc is None or pd.isna(end_storage_tmc):
        reasons.append("missing_end_storage")

    # Check flow inputs
    if inflow_cusecs is None or pd.isna(inflow_cusecs):
        reasons.append("missing_inflow")
    elif inflow_cusecs < 0:
        flags.append("negative_inflow_invalid")

    if river_outflow_cusecs is None or pd.isna(river_outflow_cusecs):
        reasons.append("missing_river_outflow")
    elif river_outflow_cusecs < 0:
        flags.append("negative_river_outflow_invalid")

    if canal_withdrawal_cusecs is None or pd.isna(canal_withdrawal_cusecs):
        reasons.append("missing_canal_withdrawal")
    elif canal_withdrawal_cusecs < 0:
        flags.append("negative_canal_withdrawal_invalid")

    # Check evaporation
    if evaporation_cusecs is None or pd.isna(evaporation_cusecs):
        reasons.append("missing_evaporation")
    elif not evaporation_verified:
        reasons.append("evaporation_source_convention_unverified")

    # Calculate volumes over elapsed hours
    elapsed_hours = elapsed_days * 24.0
    in_tmc = cusecs_to_tmc(inflow_cusecs, hours=elapsed_hours) if (inflow_cusecs is not None and pd.notna(inflow_cusecs) and inflow_cusecs >= 0) else None
    out_tmc = cusecs_to_tmc(river_outflow_cusecs, hours=elapsed_hours) if (river_outflow_cusecs is not None and pd.notna(river_outflow_cusecs) and river_outflow_cusecs >= 0) else None
    with_tmc = cusecs_to_tmc(canal_withdrawal_cusecs, hours=elapsed_hours) if (canal_withdrawal_cusecs is not None and pd.notna(canal_withdrawal_cusecs) and canal_withdrawal_cusecs >= 0) else None
    evap_tmc = cusecs_to_tmc(evaporation_cusecs, hours=elapsed_hours) if (evaporation_cusecs is not None and pd.notna(evaporation_cusecs) and evaporation_cusecs >= 0 and evaporation_verified) else None

    # Observed storage change
    if start_storage_tmc is not None and end_storage_tmc is not None and not pd.isna(start_storage_tmc) and not pd.isna(end_storage_tmc):
        obs_change = float(round(end_storage_tmc - start_storage_tmc, 5))
    else:
        obs_change = None

    # Diagnostic net flow excluding evaporation
    if in_tmc is not None and out_tmc is not None and with_tmc is not None:
        diag_net = float(round(in_tmc - out_tmc - with_tmc, 5))
        diag_disc = float(round(obs_change - diag_net, 5)) if obs_change is not None else None
    else:
        diag_net = None
        diag_disc = None

    # If unverified evaporation is the only blocker:
    if len(reasons) > 0:
        status = "UNVERIFIED_EVAPORATION" if set(reasons) == {"evaporation_source_convention_unverified"} else "INCOMPLETE"
        return WaterBalanceResult(
            start_date=str(t_start.date()),
            end_date=str(t_end.date()),
            elapsed_days=elapsed_days,
            storage_basis=storage_basis,
            observed_storage_change_tmc=obs_change,
            inflow_tmc=round(in_tmc, 5) if in_tmc is not None else None,
            river_outflow_tmc=round(out_tmc, 5) if out_tmc is not None else None,
            canal_withdrawal_tmc=round(with_tmc, 5) if with_tmc is not None else None,
            evaporation_tmc=round(evap_tmc, 5) if evap_tmc is not None else None,
            expected_storage_change_tmc=None,
            balance_discrepancy_tmc=None,  # Incomplete interval never receives numeric discrepancy
            diagnostic_net_flow_excluding_evap_tmc=diag_net,
            diagnostic_discrepancy_excluding_evap_tmc=diag_disc,
            status=status,
            reasons=reasons,
            quality_flags=flags,
        )

    # Complete balance when all terms valid and verified
    exp_change = float(round(in_tmc - out_tmc - with_tmc - evap_tmc, 5))
    discrepancy = float(round(obs_change - exp_change, 5)) if obs_change is not None else None

    return WaterBalanceResult(
        start_date=str(t_start.date()),
        end_date=str(t_end.date()),
        elapsed_days=elapsed_days,
        storage_basis=storage_basis,
        observed_storage_change_tmc=obs_change,
        inflow_tmc=round(in_tmc, 5),
        river_outflow_tmc=round(out_tmc, 5),
        canal_withdrawal_tmc=round(with_tmc, 5),
        evaporation_tmc=round(evap_tmc, 5),
        expected_storage_change_tmc=exp_change,
        balance_discrepancy_tmc=discrepancy,
        diagnostic_net_flow_excluding_evap_tmc=diag_net,
        diagnostic_discrepancy_excluding_evap_tmc=diag_disc,
        status="COMPLETE",
        reasons=[],
        quality_flags=flags,
    )


def audit_historical_water_balance(
    df: pd.DataFrame,
    storage_basis: str = "live_above_cill_tmc",
    output_path: Path = AUDIT_REPORT_PATH,
) -> pd.DataFrame:
    """Run sequential historical water balance audit across whole dataset.

    Enforces interval integrity across calendar gaps and exports reports/water_balance_audit.csv.
    """
    df = df.copy()
    if "calendar_date" not in df.columns:
        from src.data.validate import parse_monitoring_dates
        df["calendar_date"] = parse_monitoring_dates(df["monitoring_date"]).dt.normalize()
    df = df.sort_values("calendar_date").reset_index(drop=True)

    records = []
    for i in range(1, len(df)):
        row_prev = df.iloc[i - 1]
        row_curr = df.iloc[i]

        res = compute_water_balance_interval(
            start_date=row_prev["calendar_date"],
            end_date=row_curr["calendar_date"],
            start_storage_tmc=row_prev.get(storage_basis),
            end_storage_tmc=row_curr.get(storage_basis),
            inflow_cusecs=row_curr.get("inflow_cusecs"),
            river_outflow_cusecs=row_curr.get("outflow_cusecs"),
            canal_withdrawal_cusecs=row_curr.get("withdrawal_cusecs"),
            evaporation_cusecs=row_curr.get("evaporation_cusecs"),
            storage_basis=storage_basis,
            expected_days=1,
            evaporation_verified=False,  # Unverified per findings
        )
        records.append({
            "start_date": res.start_date,
            "end_date": res.end_date,
            "elapsed_days": res.elapsed_days,
            "storage_basis": res.storage_basis,
            "observed_storage_change_tmc": res.observed_storage_change_tmc,
            "inflow_tmc": res.inflow_tmc,
            "river_outflow_tmc": res.river_outflow_tmc,
            "canal_withdrawal_tmc": res.canal_withdrawal_tmc,
            "evaporation_tmc": res.evaporation_tmc,
            "expected_storage_change_tmc": res.expected_storage_change_tmc,
            "balance_discrepancy_tmc": res.balance_discrepancy_tmc,
            "diagnostic_net_flow_excluding_evap_tmc": res.diagnostic_net_flow_excluding_evap_tmc,
            "diagnostic_discrepancy_excluding_evap_tmc": res.diagnostic_discrepancy_excluding_evap_tmc,
            "status": res.status,
            "reasons": "; ".join(res.reasons),
        })

    audit_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    return audit_df
