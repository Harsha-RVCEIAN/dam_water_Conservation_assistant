"""Water-budget logic, unit conversions, and hypothetical conservation scenario modeling.

Formulas:
- Daily volume (TMC) = Mean daily flow (cusecs) * 0.0000864
- Expected storage change (TMC) ≈ Inflow - Outflow - Withdrawal - Evaporation
- Conservation savings (TMC) = (Baseline Withdrawal - Scenario Withdrawal) * days * 0.0000864

All scenarios are explicitly designated as hypothetical decision-support simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Physical constants
CUSEC_DAY_TO_TMC = 0.0000864  # 1 cusec * 86400 seconds / 10^9 cu ft
TMC_TO_CUSEC_DAY = 1.0 / CUSEC_DAY_TO_TMC
KRS_DESIGN_GROSS_CAPACITY_TMC = 49.452
KRS_DEAD_STORAGE_TMC = 4.379  # Approx dead storage below cill level


def cusec_day_to_tmc(cusecs: float) -> float:
    """Convert daily mean flow rate (cusecs) to daily volume in TMC."""
    if cusecs < 0:
        raise ValueError(f"Flow rate cannot be negative: {cusecs}")
    return float(cusecs * CUSEC_DAY_TO_TMC)


def tmc_to_cusec_day(tmc: float) -> float:
    """Convert volume in TMC to equivalent daily mean flow rate in cusecs."""
    if tmc < 0:
        raise ValueError(f"Volume cannot be negative: {tmc}")
    return float(tmc * TMC_TO_CUSEC_DAY)


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
    design_capacity_tmc: float = KRS_DESIGN_GROSS_CAPACITY_TMC,
    scenario_name: str = "Irrigation Canal Rationing Scenario",
) -> ConservationScenarioResult:
    """Simulate a hypothetical water-conservation scenario over a specified horizon.

    Compares baseline vs reduced canal withdrawal while keeping other terms constant.
    Storage is bounded between 0 and design capacity.
    """
    if simulation_days <= 0:
        raise ValueError("Simulation days must be positive")
    if not (0.0 <= reduction_percentage <= 100.0):
        raise ValueError("Reduction percentage must be between 0 and 100%")
    if current_storage_tmc < 0:
        raise ValueError("Current storage cannot be negative")

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
    savings_pct_cap = (total_saved_tmc / design_capacity_tmc) * 100.0

    # Projected storages (clamped to physical bounds)
    raw_proj_baseline = current_storage_tmc + (baseline_net_daily * simulation_days)
    raw_proj_scenario = current_storage_tmc + (scenario_net_daily * simulation_days)

    projected_baseline = float(min(max(0.0, raw_proj_baseline), design_capacity_tmc))
    projected_scenario = float(min(max(0.0, raw_proj_scenario), design_capacity_tmc))

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
        disclaimer=disclaimer,
    )
