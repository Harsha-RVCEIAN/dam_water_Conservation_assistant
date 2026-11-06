"""Tests for Layer B scenario comparison, input validations, and mathematical bounds checks."""

import pytest

from src.models.water_budget import (
    KRS_DESIGN_GROSS_CAPACITY_TMC,
    KRS_DESIGN_LIVE_CAPACITY_TMC,
    compare_conservation_scenarios,
    simulate_conservation_scenario,
)


def test_compare_conservation_scenarios_standard_fractions():
    # Test 5%, 10%, 15%, 20% comparison
    results = compare_conservation_scenarios(
        current_storage_tmc=30.0,
        assumed_inflow_cusecs=2000.0,
        planned_outflow_cusecs=500.0,
        baseline_withdrawal_cusecs=2000.0,
        reductions=[5.0, 10.0, 15.0, 20.0],
        simulation_days=10,
        evaporation_cusecs=0.0,
        storage_basis="live_above_cill_tmc",
    )

    assert len(results) == 4
    # Check savings monotonically increase with reduction %
    savings = [r.water_saved_tmc for r in results]
    assert savings[0] < savings[1] < savings[2] < savings[3]

    # Verify 10% saving exact formula:
    # 2000 * 0.10 = 200 cusecs saving * 0.0000864 * 10 days = 0.1728 TMC
    assert results[1].water_saved_tmc == pytest.approx(0.1728, rel=1e-4)

    for r in results:
        assert r.raw_projected_scenario_tmc >= r.raw_projected_baseline_tmc
        assert r.assumptions["storage_basis"] == "live_above_cill_tmc"
        assert r.assumptions["capacity_denominator_tmc"] == KRS_DESIGN_LIVE_CAPACITY_TMC


def test_scenario_detects_bounds_exceeded_without_silent_corruption():
    # Simulate a scenario with extreme deficit that drains storage below 0
    res = simulate_conservation_scenario(
        current_storage_tmc=1.0,
        assumed_inflow_cusecs=0.0,
        planned_outflow_cusecs=5000.0,
        baseline_withdrawal_cusecs=5000.0,
        reduction_percentage=5.0,
        simulation_days=10,
        evaporation_cusecs=0.0,
        storage_basis="live_above_cill_tmc",
    )
    # Raw mathematical projection is negative
    assert res.raw_projected_baseline_tmc < 0.0
    assert res.baseline_exceeds_bounds is True
    # Clamped for display safety is 0.0
    assert res.projected_baseline_storage_tmc == 0.0


def test_scenario_negative_flow_inputs_rejected():
    with pytest.raises(ValueError, match="Assumed inflow cannot be negative"):
        simulate_conservation_scenario(30.0, -100.0, 500.0, 500.0, 10.0)

    with pytest.raises(ValueError, match="Planned outflow cannot be negative"):
        simulate_conservation_scenario(30.0, 1000.0, -500.0, 500.0, 10.0)

    with pytest.raises(ValueError, match="Baseline withdrawal cannot be negative"):
        simulate_conservation_scenario(30.0, 1000.0, 500.0, -500.0, 10.0)


def test_scenario_capacity_basis_alignment():
    # Live storage basis uses live design capacity (45.051 TMC)
    res_live = simulate_conservation_scenario(
        current_storage_tmc=20.0,
        assumed_inflow_cusecs=1000.0,
        planned_outflow_cusecs=500.0,
        baseline_withdrawal_cusecs=500.0,
        reduction_percentage=10.0,
        storage_basis="live_above_cill_tmc",
    )
    assert res_live.assumptions["capacity_denominator_tmc"] == KRS_DESIGN_LIVE_CAPACITY_TMC

    # Gross storage basis uses gross design capacity (49.452 TMC)
    res_gross = simulate_conservation_scenario(
        current_storage_tmc=25.0,
        assumed_inflow_cusecs=1000.0,
        planned_outflow_cusecs=500.0,
        baseline_withdrawal_cusecs=500.0,
        reduction_percentage=10.0,
        storage_basis="gross_capacity_tmc",
    )
    assert res_gross.assumptions["capacity_denominator_tmc"] == KRS_DESIGN_GROSS_CAPACITY_TMC


def test_scenario_initial_storage_exceeds_capacity():
    # Initial live storage > 45.051 TMC must raise ValueError
    with pytest.raises(ValueError, match="exceeds capacity limit"):
        simulate_conservation_scenario(
            current_storage_tmc=48.0,
            assumed_inflow_cusecs=1000.0,
            planned_outflow_cusecs=500.0,
            baseline_withdrawal_cusecs=500.0,
            reduction_percentage=10.0,
            storage_basis="live_above_cill_tmc",
        )


def test_scenario_nan_and_infinite_inputs_rejected():
    import numpy as np

    with pytest.raises(ValueError, match="must be a finite number"):
        simulate_conservation_scenario(np.nan, 1000.0, 500.0, 500.0, 10.0)

    with pytest.raises(ValueError, match="must be a finite number"):
        simulate_conservation_scenario(30.0, np.inf, 500.0, 500.0, 10.0)

    with pytest.raises(ValueError, match="must be a finite number"):
        simulate_conservation_scenario(30.0, 1000.0, 500.0, 500.0, 10.0, evaporation_cusecs=np.nan)


def test_scenario_invalid_storage_basis_and_capacity():
    with pytest.raises(ValueError, match="Unsupported storage basis"):
        simulate_conservation_scenario(
            30.0, 1000.0, 500.0, 500.0, 10.0, storage_basis="unsupported_basis"
        )

    with pytest.raises(ValueError, match="Design capacity limit must be positive"):
        simulate_conservation_scenario(
            30.0, 1000.0, 500.0, 500.0, 10.0, design_capacity_tmc=-10.0
        )

