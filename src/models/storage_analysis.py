"""Storage and capacity consistency module for K.R.Sagara Dam.

Implements centralized, validated storage calculations, explicit capacity reference
metadata management, and strict elapsed-interval change tracking.

Rules:
- Gross capacity (gross_capacity_tmc) and live storage above cill (live_above_cill_tmc)
  are distinct physical quantities and must not be conflated.
- Dead storage is approximately 4.401 TMC (difference between gross and live above sill).
- Storage percentage remains unavailable (None / N/A) until an authoritative,
  current operational live-capacity denominator is verified and documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("config/reservoir_reference.yaml")


def load_reservoir_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load reservoir and capacity reference metadata from YAML configuration."""
    if not config_path.exists():
        # Fallback to default constants if config file is absent
        return {
            "reservoir": {"name": "Krishna Raja Sagara (KRS)"},
            "capacity_references": [
                {"name": "design_gross", "value_tmc": 49.452, "type": "design", "status": "reference_only"},
                {"name": "design_live", "value_tmc": 45.051, "type": "design", "status": "reference_only"},
                {"name": "surveyed_live_2009", "value_tmc": 42.99, "type": "survey", "year": 2009, "status": "historical_reference"},
            ],
            "storage": {
                "candidate_live_column": "live_above_cill_tmc",
                "observed_gross_column": "gross_capacity_tmc",
                "dead_storage_tmc": 4.401,
                "percentage_capacity_reference": None,
            },
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class StorageIntervalResult:
    start_date: Optional[str]
    end_date: Optional[str]
    elapsed_days: Optional[int]
    start_storage_tmc: Optional[float]
    end_storage_tmc: Optional[float]
    storage_change_tmc: Optional[float]
    storage_basis: str
    is_valid_daily_interval: bool
    status: str  # "COMPLETE", "INCOMPLETE_MISSING_DATA", "INVALID_INTERVAL"
    reasons: List[str] = field(default_factory=list)


def calculate_storage_percentage(
    storage_tmc: Optional[float],
    capacity_reference_tmc: Optional[float] = None,
) -> Optional[float]:
    """Calculate storage percentage only if a verified capacity denominator is supplied.

    Returns None if denominator is None or unverified.
    """
    if storage_tmc is None or capacity_reference_tmc is None:
        return None
    if capacity_reference_tmc <= 0:
        return None
    return float(round((storage_tmc / capacity_reference_tmc) * 100.0, 2))


def compute_interval_storage_change(
    start_storage_tmc: Optional[float],
    end_storage_tmc: Optional[float],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    storage_basis: str = "live_above_cill_tmc",
    expected_days: int = 1,
) -> StorageIntervalResult:
    """Compute observed storage change between two calendar dates with interval validation.

    Prevents false daily change derivation across multi-day calendar gaps.
    """
    t_start = pd.to_datetime(start_date)
    t_end = pd.to_datetime(end_date)
    elapsed_days = int((t_end.normalize() - t_start.normalize()).days)

    reasons = []
    if start_storage_tmc is None or pd.isna(start_storage_tmc):
        reasons.append("missing_start_storage")
    if end_storage_tmc is None or pd.isna(end_storage_tmc):
        reasons.append("missing_end_storage")

    if elapsed_days != expected_days:
        reasons.append(f"elapsed_days_mismatch (expected {expected_days}, observed {elapsed_days})")

    is_daily = (elapsed_days == 1)

    if reasons:
        return StorageIntervalResult(
            start_date=str(t_start.date()),
            end_date=str(t_end.date()),
            elapsed_days=elapsed_days,
            start_storage_tmc=float(start_storage_tmc) if start_storage_tmc is not None and not pd.isna(start_storage_tmc) else None,
            end_storage_tmc=float(end_storage_tmc) if end_storage_tmc is not None and not pd.isna(end_storage_tmc) else None,
            storage_change_tmc=None,
            storage_basis=storage_basis,
            is_valid_daily_interval=is_daily and len(reasons) == 0,
            status="INCOMPLETE_INTERVAL" if ("missing_start_storage" in reasons or "missing_end_storage" in reasons) else "INVALID_INTERVAL_DURATION",
            reasons=reasons,
        )

    delta = float(round(end_storage_tmc - start_storage_tmc, 5))
    return StorageIntervalResult(
        start_date=str(t_start.date()),
        end_date=str(t_end.date()),
        elapsed_days=elapsed_days,
        start_storage_tmc=float(round(start_storage_tmc, 4)),
        end_storage_tmc=float(round(end_storage_tmc, 4)),
        storage_change_tmc=delta,
        storage_basis=storage_basis,
        is_valid_daily_interval=is_daily,
        status="COMPLETE",
        reasons=[],
    )


def extract_storage_summary(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract validated current storage metrics and capacity reference metadata."""
    if config is None:
        config = load_reservoir_config()

    storage_cfg = config.get("storage", {})
    cap_refs = config.get("capacity_references", [])
    primary_col = storage_cfg.get("candidate_live_column", "live_above_cill_tmc")
    gross_col = storage_cfg.get("observed_gross_column", "gross_capacity_tmc")
    denom_ref = storage_cfg.get("percentage_capacity_reference")

    if df.empty:
        return {"status": "NO_DATA"}

    latest_row = df.iloc[-1]
    curr_live = float(latest_row[primary_col]) if primary_col in latest_row and pd.notna(latest_row[primary_col]) else None
    curr_gross = float(latest_row[gross_col]) if gross_col in latest_row and pd.notna(latest_row[gross_col]) else None

    # Calculate percentage only if denominator reference is explicitly configured and verified
    pct_full = calculate_storage_percentage(curr_live, denom_ref)

    # Calculate yesterday-to-today change with interval check
    prev_row = df.iloc[-2] if len(df) > 1 else None
    if prev_row is not None and primary_col in df.columns:
        date_curr = latest_row["calendar_date"] if "calendar_date" in latest_row else latest_row["monitoring_date"]
        date_prev = prev_row["calendar_date"] if "calendar_date" in prev_row else prev_row["monitoring_date"]
        daily_change = compute_interval_storage_change(
            start_storage_tmc=prev_row[primary_col],
            end_storage_tmc=latest_row[primary_col],
            start_date=date_prev,
            end_date=date_curr,
            storage_basis=primary_col,
            expected_days=1,
        )
    else:
        daily_change = None

    return {
        "reservoir_name": config.get("reservoir", {}).get("name", "KRS"),
        "primary_storage_column": primary_col,
        "current_live_storage_tmc": curr_live,
        "current_gross_storage_tmc": curr_gross,
        "percentage_full": pct_full,
        "percentage_status": "AVAILABLE" if pct_full is not None else "UNVERIFIED_DENOMINATOR_NA",
        "capacity_references": cap_refs,
        "daily_change": daily_change.__dict__ if daily_change else None,
    }

