"""Tests for Layer B scenario comparison and mathematical bounds checks."""

import pytest

from src.models.water_budget import (
    KRS_DESIGN_GROSS_CAPACITY_TMC,
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
        assert "constant_inflow_assumption" in r.assumptions


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
    )
    # Raw mathematical projection is negative
    assert res.raw_projected_baseline_tmc < 0.0
    assert res.baseline_exceeds_bounds is True
    # Clamped for display safety is 0.0
    assert res.projected_baseline_storage_tmc == 0.0

