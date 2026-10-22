"""Tests for physical unit conversions and water-budget mass balance logic."""

import pytest

from src.models.water_budget import (
    CUSEC_DAY_TO_TMC,
    KRS_DESIGN_GROSS_CAPACITY_TMC,
    compute_daily_water_balance,
    cusec_day_to_tmc,
    evaluate_balance_discrepancy,
    simulate_conservation_scenario,
    tmc_to_cusec_day,
)


def test_unit_conversion_constants():
    # 1 cusec for 1 day = 86,400 cu ft = 0.0000864 * 10^9 cu ft = 0.0000864 TMC
    assert CUSEC_DAY_TO_TMC == 0.0000864
    assert cusec_day_to_tmc(1.0) == pytest.approx(0.0000864, rel=1e-7)
    assert tmc_to_cusec_day(0.0000864) == pytest.approx(1.0, rel=1e-7)


def test_negative_flow_raises_value_error():
    with pytest.raises(ValueError, match="cannot be negative"):
        cusec_day_to_tmc(-500.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        tmc_to_cusec_day(-1.0)


def test_daily_water_balance_known_arithmetic():
    # Inflow: 10,000 cusecs -> 0.864 TMC
    # Outflow: 5,000 cusecs -> 0.432 TMC
    # Withdrawal: 2,000 cusecs -> 0.1728 TMC
    # Evaporation: 500 cusecs -> 0.0432 TMC
    # Net change: 0.864 - 0.432 - 0.1728 - 0.0432 = 0.216 TMC
    balance = compute_daily_water_balance(
        inflow_cusecs=10000.0,
        outflow_cusecs=5000.0,
        withdrawal_cusecs=2000.0,
        evaporation_cusecs=500.0,
    )
    assert balance.inflow_vol_tmc == pytest.approx(0.864, rel=1e-5)
    assert balance.outflow_vol_tmc == pytest.approx(0.432, rel=1e-5)
    assert balance.withdrawal_vol_tmc == pytest.approx(0.1728, rel=1e-5)
    assert balance.evaporation_vol_tmc == pytest.approx(0.0432, rel=1e-5)
    assert balance.net_storage_change_tmc == pytest.approx(0.216, rel=1e-5)


def test_balance_discrepancy_flags():
    normal = evaluate_balance_discrepancy(actual_storage_change_tmc=0.217, expected_storage_change_tmc=0.216)
    assert normal["status"] == "NORMAL"

    warning = evaluate_balance_discrepancy(actual_storage_change_tmc=3.0, expected_storage_change_tmc=0.2)
    assert warning["status"] == "MODERATE_DISCREPANCY"

    extreme = evaluate_balance_discrepancy(actual_storage_change_tmc=10.0, expected_storage_change_tmc=0.0)
    assert extreme["status"] == "HIGH_DISCREPANCY"
    assert "Not proof of leakage" in extreme["explanation"]


def test_conservation_scenario_simulation():
    # Baseline withdrawal: 2,000 cusecs
    # 20% reduction: 1,600 cusecs (saving 400 cusecs/day)
    # Over 7 days: 400 * 0.0000864 * 7 = 0.24192 TMC
    res = simulate_conservation_scenario(
        current_storage_tmc=30.0,
        assumed_inflow_cusecs=3000.0,
        planned_outflow_cusecs=1000.0,
        baseline_withdrawal_cusecs=2000.0,
        reduction_percentage=20.0,
        simulation_days=7,
        evaporation_cusecs=100.0,
    )

    expected_saving = 400.0 * 0.0000864 * 7
    assert res.water_saved_tmc == pytest.approx(expected_saving, rel=1e-4)
    assert res.projected_scenario_storage_tmc > res.projected_baseline_storage_tmc
    assert res.projected_scenario_storage_tmc <= KRS_DESIGN_GROSS_CAPACITY_TMC
    assert "Hypothetical educational simulation" in res.disclaimer


def test_conservation_scenario_invalid_inputs():
    with pytest.raises(ValueError, match="Simulation days must be positive"):
        simulate_conservation_scenario(30.0, 1000.0, 500.0, 500.0, 10.0, simulation_days=0)

    with pytest.raises(ValueError, match="between 0 and 100%"):
        simulate_conservation_scenario(30.0, 1000.0, 500.0, 500.0, 120.0, simulation_days=7)
