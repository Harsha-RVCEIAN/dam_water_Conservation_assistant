"""Unit tests for canonical water-balance module."""

import pytest

from src.models.water_balance import (
    compute_water_balance_interval,
    load_water_balance_config,
)


def test_water_balance_sign_convention_hand_calculated():
    # Inflow: 10,000 cusecs * 0.0000864 = 0.864 TMC
    # Outflow to River: 3,000 cusecs * 0.0000864 = 0.2592 TMC
    # Canal Withdrawal: 2,000 cusecs * 0.0000864 = 0.1728 TMC
    # Evaporation: 500 cusecs * 0.0000864 = 0.0432 TMC (assuming verified for test)
    # Expected Storage Change: 0.864 - 0.2592 - 0.1728 - 0.0432 = 0.3888 TMC
    # Observed Start: 30.0 TMC, End: 30.4 TMC => Observed change: +0.4000 TMC
    # Balance Discrepancy: +0.4000 - 0.3888 = +0.0112 TMC (positive means observed > expected)
    res = compute_water_balance_interval(
        start_date="2020-01-01",
        end_date="2020-01-02",
        start_storage_tmc=30.0,
        end_storage_tmc=30.4,
        inflow_cusecs=10000.0,
        river_outflow_cusecs=3000.0,
        canal_withdrawal_cusecs=2000.0,
        evaporation_cusecs=500.0,
        evaporation_verified=True,
    )
    assert res.status == "COMPLETE"
    assert res.inflow_tmc == pytest.approx(0.864, rel=1e-4)
    assert res.river_outflow_tmc == pytest.approx(0.2592, rel=1e-4)
    assert res.canal_withdrawal_tmc == pytest.approx(0.1728, rel=1e-4)
    assert res.evaporation_tmc == pytest.approx(0.0432, rel=1e-4)
    assert res.expected_storage_change_tmc == pytest.approx(0.3888, rel=1e-4)
    assert res.observed_storage_change_tmc == pytest.approx(0.4000, rel=1e-4)
    assert res.balance_discrepancy_tmc == pytest.approx(0.0112, rel=1e-4)
    assert "leak" not in res.disclaimer.lower() or "not proof of" in res.disclaimer.lower()


def test_unverified_evaporation_blocks_complete_balance():
    # If evaporation convention is unverified, discrepancy must remain None
    res = compute_water_balance_interval(
        start_date="2020-01-01",
        end_date="2020-01-02",
        start_storage_tmc=30.0,
        end_storage_tmc=30.4,
        inflow_cusecs=10000.0,
        river_outflow_cusecs=3000.0,
        canal_withdrawal_cusecs=2000.0,
        evaporation_cusecs=500.0,
        evaporation_verified=False,  # Unverified
    )
    assert res.status == "UNVERIFIED_EVAPORATION"
    assert res.expected_storage_change_tmc is None
    assert res.balance_discrepancy_tmc is None
    assert "evaporation_source_convention_unverified" in res.reasons
    # But diagnostic balance excluding evaporation is computed for engineering inspection
    assert res.diagnostic_net_flow_excluding_evap_tmc is not None


def test_missing_evaporation_preserves_incomplete():
    res = compute_water_balance_interval(
        start_date="2020-01-01",
        end_date="2020-01-02",
        start_storage_tmc=30.0,
        end_storage_tmc=30.4,
        inflow_cusecs=10000.0,
        river_outflow_cusecs=3000.0,
        canal_withdrawal_cusecs=2000.0,
        evaporation_cusecs=None,  # Missing
    )
    assert res.status == "INCOMPLETE"
    assert res.balance_discrepancy_tmc is None
    assert "missing_evaporation" in res.reasons


def test_two_day_interval_fails_daily_validation():
    # 2 days elapsed (e.g. across 2020-02-11 gap)
    res = compute_water_balance_interval(
        start_date="2020-02-10",
        end_date="2020-02-12",
        start_storage_tmc=30.0,
        end_storage_tmc=30.4,
        inflow_cusecs=10000.0,
        river_outflow_cusecs=3000.0,
        canal_withdrawal_cusecs=2000.0,
        expected_days=1,
    )
    assert res.status == "INCOMPLETE"
    assert "elapsed_days_mismatch (expected 1, observed 2)" in res.reasons
    assert res.balance_discrepancy_tmc is None


def test_river_and_canal_are_both_subtracted_separately():
    res = compute_water_balance_interval(
        start_date="2020-01-01",
        end_date="2020-01-02",
        start_storage_tmc=30.0,
        end_storage_tmc=30.0,
        inflow_cusecs=1000.0,
        river_outflow_cusecs=600.0,
        canal_withdrawal_cusecs=400.0,
        evaporation_cusecs=0.0,
        evaporation_verified=True,
    )
    # Inflow (0.0864) - River (0.05184) - Canal (0.03456) = 0.0 TMC
    assert res.expected_storage_change_tmc == pytest.approx(0.0, abs=1e-6)
    assert res.observed_storage_change_tmc == pytest.approx(0.0, abs=1e-6)
    assert res.balance_discrepancy_tmc == pytest.approx(0.0, abs=1e-6)


def test_negative_and_nan_inputs_yield_incomplete():
    import numpy as np

    # Negative inflow
    res_inflow = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", 30.0, 30.0, -100.0, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_inflow.status == "INCOMPLETE"
    assert "invalid_inflow" in res_inflow.reasons

    # Negative outflow
    res_outflow = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", 30.0, 30.0, 100.0, -50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_outflow.status == "INCOMPLETE"
    assert "invalid_river_outflow" in res_outflow.reasons

    # Negative withdrawal
    res_with = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", 30.0, 30.0, 100.0, 50.0, -50.0, 0.0, evaporation_verified=True
    )
    assert res_with.status == "INCOMPLETE"
    assert "invalid_canal_withdrawal" in res_with.reasons

    # Negative evaporation
    res_evap = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", 30.0, 30.0, 100.0, 50.0, 50.0, -10.0, evaporation_verified=True
    )
    assert res_evap.status == "INCOMPLETE"
    assert "invalid_evaporation" in res_evap.reasons

    # Negative start storage
    res_start = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", -5.0, 30.0, 100.0, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_start.status == "INCOMPLETE"
    assert "invalid_start_storage" in res_start.reasons

    # NaN / infinite inputs
    res_nan = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", np.nan, 30.0, np.inf, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_nan.status == "INCOMPLETE"
    assert any("start_storage" in r for r in res_nan.reasons)
    assert "invalid_inflow" in res_nan.reasons


def test_invalid_date_conditions_yield_incomplete():
    # Same start and end date
    res_same = compute_water_balance_interval(
        "2020-01-01", "2020-01-01", 30.0, 30.0, 100.0, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_same.status == "INCOMPLETE"
    assert "same_start_and_end_date" in res_same.reasons

    # Reversed dates
    res_rev = compute_water_balance_interval(
        "2020-01-05", "2020-01-01", 30.0, 30.0, 100.0, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_rev.status == "INCOMPLETE"
    assert "reversed_interval_dates" in res_rev.reasons

    # Missing dates
    res_miss = compute_water_balance_interval(
        None, "2020-01-02", 30.0, 30.0, 100.0, 50.0, 50.0, 0.0, evaporation_verified=True
    )
    assert res_miss.status == "INCOMPLETE"
    assert "invalid_or_missing_interval_dates" in res_miss.reasons

    # Unsupported storage basis
    res_basis = compute_water_balance_interval(
        "2020-01-01", "2020-01-02", 30.0, 30.0, 100.0, 50.0, 50.0, 0.0, storage_basis="unsupported_basis"
    )
    assert res_basis.status == "INCOMPLETE"
    assert "unsupported_storage_basis" in res_basis.reasons


