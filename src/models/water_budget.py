"""Water-budget logic, unit conversions, and hypothetical conservation scenario modeling.

This module provides backward-compatible wrappers around the canonical
`src.models.water_balance` and `src.utils.unit_conversions` modules.

Formulas:
- Daily volume (TMC) = Mean daily flow (cusecs) * 0.0000864
- Expected storage change (TMC) ≈ Inflow - Outflow - Withdrawal - Evaporation
- Conservation savings (TMC) = (Baseline Withdrawal - Scenario Withdrawal) * days * 0.0000864

All scenarios are explicitly designated as hypothetical decision-support simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.unit_conversions import (
    CUSEC_DAY_TO_TMC_FACTOR,
    cusecs_to_tmc,
    tmc_to_cusecs,
)

# Physical constants
CUSEC_DAY_TO_TMC = CUSEC_DAY_TO_TMC_FACTOR  # 1 cusec * 86400 seconds / 10^9 cu ft
TMC_TO_CUSEC_DAY = 1.0 / CUSEC_DAY_TO_TMC
KRS_DESIGN_GROSS_CAPACITY_TMC = 49.452
KRS_DESIGN_LIVE_CAPACITY_TMC = 45.051
KRS_DEAD_STORAGE_TMC = 4.401  # Reference dead storage below sill level (reservoir_reference.yaml)


def cusec_day_to_tmc(cusecs: float) -> float:
    """Convert daily mean flow rate (cusecs) to daily volume in TMC."""
    return float(cusecs_to_tmc(cusecs, hours=24.0))


def tmc_to_cusec_day(tmc: float) -> float:
    """Convert volume in TMC to equivalent daily mean flow rate in cusecs."""
    return float(tmc_to_cusecs(tmc, hours=24.0))


@dataclass
class DailyWaterBalance:
    inflow_cusecs: float
    outflow_cusecs: float
    withdrawal_cusecs: float
    evaporation_cusecs: float
    inflow_vol_tmc: float
    outflow_vol_tmc: float
    withdrawal_vol_tmc: float
    evaporation_vol_tmc: float
    net_storage_change_tmc: float


def compute_daily_water_balance(
    inflow_cusecs: float,
    outflow_cusecs: float,
    withdrawal_cusecs: float,
    evaporation_cusecs: float = 0.0,
) -> DailyWaterBalance:
    """Compute daily volume components and expected storage change in TMC."""
    in_vol = cusec_day_to_tmc(inflow_cusecs)
    out_vol = cusec_day_to_tmc(outflow_cusecs)
    with_vol = cusec_day_to_tmc(withdrawal_cusecs)
    evap_vol = cusec_day_to_tmc(evaporation_cusecs)

    net_change = in_vol - out_vol - with_vol - evap_vol

    return DailyWaterBalance(
        inflow_cusecs=inflow_cusecs,
        outflow_cusecs=outflow_cusecs,
        withdrawal_cusecs=withdrawal_cusecs,
        evaporation_cusecs=evaporation_cusecs,
        inflow_vol_tmc=in_vol,
        outflow_vol_tmc=out_vol,
        withdrawal_vol_tmc=with_vol,
        evaporation_vol_tmc=evap_vol,
        net_storage_change_tmc=net_change,
    )


def evaluate_balance_discrepancy(
    actual_storage_change_tmc: float,
    expected_storage_change_tmc: float,
    warning_threshold_tmc: float = 2.0,
    extreme_threshold_tmc: float = 5.0,
) -> Dict[str, Any]:
    """Compare observed storage change against water balance.

    Flagging does NOT indicate leakage; it flags unmodeled factors
    (intense rain, catchment runoff, timing lags, gate calibration) for review.
    """
    discrepancy = actual_storage_change_tmc - expected_storage_change_tmc
    abs_disc = abs(discrepancy)

    if abs_disc >= extreme_threshold_tmc:
        status = "HIGH_DISCREPANCY"
        note = "Unusual discrepancy flagged for review. Likely unmodeled runoff or flood routing delay. Not proof of leakage."
    elif abs_disc >= warning_threshold_tmc:
        status = "MODERATE_DISCREPANCY"
        note = "Moderate deviation from simple water balance."
    else:
        status = "NORMAL"
        note = "Storage change closely matches expected water balance."

    return {
        "actual_storage_change_tmc": actual_storage_change_tmc,
        "expected_storage_change_tmc": expected_storage_change_tmc,
        "discrepancy_tmc": discrepancy,
        "abs_discrepancy_tmc": abs_disc,
        "status": status,
        "explanation": note,
    }


@dataclass
class ConservationScenarioResult:
    scenario_name: str
    simulation_days: int
    current_storage_tmc: float
    assumed_inflow_cusecs: float
    planned_outflow_cusecs: float
    baseline_withdrawal_cusecs: float
    conservation_withdrawal_cusecs: float
    reduction_percentage: float
    evaporation_cusecs: float
    water_saved_tmc: float
    savings_as_pct_capacity: float
    projected_baseline_storage_tmc: float
    projected_scenario_storage_tmc: float
    disclaimer: str
    raw_projected_baseline_tmc: float = 0.0
    raw_projected_scenario_tmc: float = 0.0
    baseline_exceeds_bounds: bool = False
    scenario_exceeds_bounds: bool = False
    assumptions: Dict[str, Any] = None

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "simulation_days": self.simulation_days,
            "current_storage_tmc": self.current_storage_tmc,
            "assumed_inflow_cusecs": self.assumed_inflow_cusecs,
            "planned_outflow_cusecs": self.planned_outflow_cusecs,
            "baseline_withdrawal_cusecs": self.baseline_withdrawal_cusecs,
            "conservation_withdrawal_cusecs": self.conservation_withdrawal_cusecs,
            "reduction_percentage": self.reduction_percentage,
            "water_saved_tmc": self.water_saved_tmc,
            "savings_as_pct_capacity": self.savings_as_pct_capacity,
            "projected_baseline_storage_tmc": self.projected_baseline_storage_tmc,
            "projected_scenario_storage_tmc": self.projected_scenario_storage_tmc,
            "raw_projected_baseline_tmc": self.raw_projected_baseline_tmc,
            "raw_projected_scenario_tmc": self.raw_projected_scenario_tmc,
            "baseline_exceeds_bounds": self.baseline_exceeds_bounds,
            "scenario_exceeds_bounds": self.scenario_exceeds_bounds,
            "assumptions": self.assumptions,
            "disclaimer": self.disclaimer,
        }


def simulate_conservation_scenario(
    current_storage_tmc: float,
    assumed_inflow_cusecs: float,
    planned_outflow_cusecs: float,
    baseline_withdrawal_cusecs: float,
    reduction_percentage: float,
    simulation_days: int = 7,
    evaporation_cusecs: float = 150.0,
    design_capacity_tmc: Optional[float] = None,
    storage_basis: str = "live_above_cill_tmc",
    scenario_name: str = "Irrigation Canal Rationing Scenario",
    inflow_source: str = "Observed / Assumed constant flow",
    outflow_source: str = "Planned constant release",
) -> ConservationScenarioResult:
    """Simulate a hypothetical water-conservation scenario over a specified horizon.

    Compares baseline vs reduced canal withdrawal while keeping other terms constant.
    Aligns capacity reference to storage basis (live vs gross).
    Storage is bounded between 0 and design capacity, with raw unclipped values retained.
    """
    if simulation_days <= 0:
        raise ValueError("Simulation days must be positive")
    if not (0.0 <= reduction_percentage <= 100.0):
        raise ValueError("Reduction percentage must be between 0 and 100%")
    if current_storage_tmc < 0:
        raise ValueError("Current storage cannot be negative")
    if assumed_inflow_cusecs < 0:
        raise ValueError("Assumed inflow cannot be negative")
    if planned_outflow_cusecs < 0:
        raise ValueError("Planned outflow cannot be negative")
    if baseline_withdrawal_cusecs < 0:
        raise ValueError("Baseline withdrawal cannot be negative")

    # Resolve capacity reference according to storage basis
    if design_capacity_tmc is None:
        capacity_limit = (
            KRS_DESIGN_LIVE_CAPACITY_TMC
            if storage_basis == "live_above_cill_tmc"
            else KRS_DESIGN_GROSS_CAPACITY_TMC
        )
    else:
        capacity_limit = design_capacity_tmc

    if capacity_limit <= 0:
        raise ValueError("Design capacity limit must be positive")

    # Conservation withdrawal rate
    conservation_withdrawal = baseline_withdrawal_cusecs * (1.0 - (reduction_percentage / 100.0))

    # Daily volumes in TMC
    inflow_daily_tmc = cusec_day_to_tmc(assumed_inflow_cusecs)
    outflow_daily_tmc = cusec_day_to_tmc(planned_outflow_cusecs)
    evap_daily_tmc = cusec_day_to_tmc(evaporation_cusecs)

    baseline_withdrawal_daily_tmc = cusec_day_to_tmc(baseline_withdrawal_cusecs)
    conservation_withdrawal_daily_tmc = cusec_day_to_tmc(conservation_withdrawal)

    # Net daily storage change
    baseline_net_daily = inflow_daily_tmc - outflow_daily_tmc - baseline_withdrawal_daily_tmc - evap_daily_tmc
    scenario_net_daily = inflow_daily_tmc - outflow_daily_tmc - conservation_withdrawal_daily_tmc - evap_daily_tmc

    # Total savings over simulation period
    daily_saved_tmc = baseline_withdrawal_daily_tmc - conservation_withdrawal_daily_tmc
    total_saved_tmc = daily_saved_tmc * simulation_days
    savings_pct_cap = (total_saved_tmc / capacity_limit) * 100.0

    # Projected storages (raw mathematical projection)
    raw_proj_baseline = current_storage_tmc + (baseline_net_daily * simulation_days)
    raw_proj_scenario = current_storage_tmc + (scenario_net_daily * simulation_days)

    baseline_exceeds = (raw_proj_baseline < 0.0) or (raw_proj_baseline > capacity_limit)
    scenario_exceeds = (raw_proj_scenario < 0.0) or (raw_proj_scenario > capacity_limit)

    # Clamped for display safety while retaining raw
    projected_baseline = float(min(max(0.0, raw_proj_baseline), capacity_limit))
    projected_scenario = float(min(max(0.0, raw_proj_scenario), capacity_limit))

    assumptions = {
        "simulation_horizon_days": simulation_days,
        "storage_basis": storage_basis,
        "capacity_denominator_tmc": capacity_limit,
        "capacity_basis_alignment": f"Storage basis '{storage_basis}' evaluated against capacity limit of {capacity_limit} TMC",
        "constant_inflow_assumption": f"Flow of {assumed_inflow_cusecs:.1f} cusecs held constant across {simulation_days} days ({inflow_source})",
        "constant_outflow_assumption": f"River outflow of {planned_outflow_cusecs:.1f} cusecs held constant ({outflow_source})",
        "evaporation_handling": f"{evaporation_cusecs:.1f} cusecs daily assumption (provisional estimate)",
    }

    disclaimer = (
        "Hypothetical educational simulation based on simplified daily mass balance. "
        "Does not account for downstream irrigation water rights, river treaty requirements, "
        "or localized rainfall. Not an operational dam-control instruction."
    )

    return ConservationScenarioResult(
        scenario_name=scenario_name,
        simulation_days=simulation_days,
        current_storage_tmc=current_storage_tmc,
        assumed_inflow_cusecs=assumed_inflow_cusecs,
        planned_outflow_cusecs=planned_outflow_cusecs,
        baseline_withdrawal_cusecs=baseline_withdrawal_cusecs,
        conservation_withdrawal_cusecs=conservation_withdrawal,
        reduction_percentage=reduction_percentage,
        evaporation_cusecs=evaporation_cusecs,
        water_saved_tmc=round(total_saved_tmc, 5),
        savings_as_pct_capacity=round(savings_pct_cap, 3),
        projected_baseline_storage_tmc=round(projected_baseline, 4),
        projected_scenario_storage_tmc=round(projected_scenario, 4),
        raw_projected_baseline_tmc=round(raw_proj_baseline, 4),
        raw_projected_scenario_tmc=round(raw_proj_scenario, 4),
        baseline_exceeds_bounds=baseline_exceeds,
        scenario_exceeds_bounds=scenario_exceeds,
        assumptions=assumptions,
        disclaimer=disclaimer,
    )


def compare_conservation_scenarios(
    current_storage_tmc: float,
    assumed_inflow_cusecs: float,
    planned_outflow_cusecs: float,
    baseline_withdrawal_cusecs: float,
    reductions: Optional[List[float]] = None,
    simulation_days: int = 7,
    evaporation_cusecs: float = 0.0,
    design_capacity_tmc: Optional[float] = None,
    storage_basis: str = "live_above_cill_tmc",
    inflow_source: str = "Observed historical baseline",
) -> List[ConservationScenarioResult]:
    """Compare multiple standardized withdrawal rationing scenarios (5%, 10%, 15%, 20%)."""
    if reductions is None:
        reductions = [5.0, 10.0, 15.0, 20.0]

    results: List[ConservationScenarioResult] = []
    for r in sorted(reductions):
        res = simulate_conservation_scenario(
            current_storage_tmc=current_storage_tmc,
            assumed_inflow_cusecs=assumed_inflow_cusecs,
            planned_outflow_cusecs=planned_outflow_cusecs,
            baseline_withdrawal_cusecs=baseline_withdrawal_cusecs,
            reduction_percentage=r,
            simulation_days=simulation_days,
            evaporation_cusecs=evaporation_cusecs,
            design_capacity_tmc=design_capacity_tmc,
            storage_basis=storage_basis,
            scenario_name=f"{int(r)}% Canal Withdrawal Rationing",
            inflow_source=inflow_source,
        )
        results.append(res)
    return results
