"""Unit tests for storage and capacity consistency module."""

import pandas as pd
import pytest

from src.models.storage_analysis import (
    calculate_storage_percentage,
    compute_interval_storage_change,
    extract_storage_summary,
    load_reservoir_config,
)


def test_capacity_references_loaded_correctly():
    cfg = load_reservoir_config()
    assert cfg["reservoir"]["name"] == "Krishna Raja Sagara (KRS)"
    names = [r["name"] for r in cfg["capacity_references"]]
    assert "design_gross" in names
    assert "design_live" in names
    assert "surveyed_live_2009" in names


def test_storage_percentage_none_when_denominator_unverified():
    # When denominator is None, return None
    pct = calculate_storage_percentage(25.0, None)
    assert pct is None

    # When denominator is valid positive number, return percentage
    pct_valid = calculate_storage_percentage(25.0, 50.0)
    assert pct_valid == 50.0


def test_storage_change_across_consecutive_days():
    res = compute_interval_storage_change(
        start_storage_tmc=20.0,
        end_storage_tmc=21.5,
        start_date="2020-01-01",
        end_date="2020-01-02",
        expected_days=1,
    )
    assert res.status == "COMPLETE"
    assert res.is_valid_daily_interval is True
    assert res.elapsed_days == 1
    assert res.storage_change_tmc == 1.5


def test_storage_change_rejects_multi_day_gap_as_daily():
    # 2 days elapsed (e.g. across missing date 2020-02-11)
    res = compute_interval_storage_change(
        start_storage_tmc=20.0,
        end_storage_tmc=21.5,
        start_date="2020-02-10",
        end_date="2020-02-12",
        expected_days=1,
    )
    assert res.is_valid_daily_interval is False
    assert res.elapsed_days == 2
    assert "elapsed_days_mismatch" in res.reasons[0]


def test_missing_endpoints_yield_incomplete():
    res = compute_interval_storage_change(
        start_storage_tmc=None,
        end_storage_tmc=21.5,
        start_date="2020-01-01",
        end_date="2020-01-02",
    )
    assert res.status == "INCOMPLETE_INTERVAL"
    assert res.storage_change_tmc is None
    assert "missing_start_storage" in res.reasons


def test_extract_storage_summary_distinguishes_gross_and_live():
    df = pd.DataFrame({
        "calendar_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
        "gross_capacity_tmc": [35.0, 35.5],
        "live_above_cill_tmc": [30.6, 31.1],
    })
    summary = extract_storage_summary(df)
    assert summary["current_live_storage_tmc"] == 31.1
    assert summary["current_gross_storage_tmc"] == 35.5
    # Distinct columns
    assert summary["current_live_storage_tmc"] != summary["current_gross_storage_tmc"]
    # Denominator unverified => N/A
    assert summary["percentage_full"] is None
    assert summary["percentage_status"] == "UNVERIFIED_DENOMINATOR_NA"
    assert len(summary["capacity_references"]) >= 3

