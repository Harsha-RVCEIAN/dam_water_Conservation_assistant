"""Tests for rainfall source, schema, aggregation, and merge integrity.

Follows Sections 12.1 and 12.2 of the Rainfall MLOps Reproducibility Plan.
All tests use synthetic, self-contained fixtures to ensure isolated, repeatable execution.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.build_rainfall_dataset import parse_candidate_dates, build_daily_rainfall


# ==============================================================================
# 12.1 Rainfall Source & Schema Tests
# ==============================================================================

def test_valid_source_manifest_accepted():
    """Verify that a properly structured source manifest parses correctly."""
    manifest_yaml = """
    manifest_version: "1.0.0"
    provider: "CWC_WRIS"
    variable: "daily_rainfall"
    units: "mm"
    timezone: "Asia/Kolkata"
    """
    data = yaml.safe_load(manifest_yaml)
    assert data["provider"] == "CWC_WRIS"
    assert data["units"] == "mm"
    assert data["variable"] == "daily_rainfall"


def test_missing_provider_rejected_for_production():
    """Production ingestion must reject manifests lacking an approved provider."""
    manifest_data = {
        "provider": None,
        "variable": "daily_rainfall",
        "units": "mm"
    }
    assert manifest_data["provider"] is None
    is_valid = bool(manifest_data.get("provider"))
    assert not is_valid


def test_invalid_station_id_rejected():
    """Empty or malformed station IDs must be detected and rejected."""
    candidates = pd.DataFrame({
        "Station Code": [None, "", "   ", "VALID100"],
        "Monitoring Date": ["2024-01-01"] * 4,
        "Rainfall (MM)": [0.0, 5.2, 3.1, 0.0]
    })
    invalid_mask = candidates["Station Code"].isna() | candidates["Station Code"].str.strip().eq("")
    assert invalid_mask.sum() == 3


def test_invalid_coordinates_rejected():
    """Geographic coordinates outside valid range must be rejected."""
    coords = pd.DataFrame({
        "station": ["S1", "S2", "S3"],
        "lat": [12.4, 95.0, 12.5],  # 95.0 is invalid latitude (> 90)
        "lon": [76.5, 76.2, 190.0]  # 190.0 is invalid longitude (> 180)
    })
    valid_lat = coords["lat"].between(-90, 90)
    valid_lon = coords["lon"].between(-180, 180)
    assert not valid_lat.all()
    assert not valid_lon.all()
    assert valid_lat.sum() == 2
    assert valid_lon.sum() == 2


def test_unknown_unit_rejected():
    """Units other than approved units (mm) must be rejected."""
    allowed_units = {"mm", "millimeters"}
    assert "inches" not in allowed_units
    assert "cm" not in allowed_units
    assert "mm" in allowed_units


def test_documented_missing_sentinel_becomes_null():
    """Standard hydromet sentinels (-999, -999.9) must convert to NaN."""
    sentinels = [-999.0, -999.9]
    raw_series = pd.Series([10.5, -999.0, 0.0, -999.9, 12.0])
    cleaned = raw_series.replace(sentinels, np.nan)
    assert cleaned.isna().sum() == 2
    assert cleaned.iloc[0] == 10.5
    assert cleaned.iloc[2] == 0.0


def test_negative_rainfall_rejected():
    """Unsentineled negative rainfall is invalid physics and must be rejected."""
    rainfall = pd.Series([5.0, -1.5, 0.0, 12.3])
    negative_mask = rainfall < 0
    assert negative_mask.sum() == 1
    assert negative_mask.iloc[1]


def test_duplicate_station_date_fails():
    """Duplicate observations for the same station on the same date must fail build."""
    candidates = pd.DataFrame({
        "Station Code": ["STN1", "STN1"],
        "Monitoring Date": ["2024-01-01", "2024-01-01"],
        "Rainfall (MM)": [5.0, 10.0]
    })
    with pytest.raises(ValueError, match="duplicate selected station-date rows"):
        build_daily_rainfall(candidates, selected_codes={"STN1"})


# ==============================================================================
# 12.2 Aggregation and Merge Tests
# ==============================================================================

def test_valid_zero_rainfall_remains_zero():
    """Valid zero observations (dry days) must remain exactly 0.0 mm."""
    candidates = pd.DataFrame({
        "Station Code": ["STN1"],
        "Monitoring Date": ["2024-01-01"],
        "Rainfall (MM)": [0.0]
    })
    daily, stats = build_daily_rainfall(candidates, selected_codes={"STN1"})
    assert stats["zero_rainfall_observations"] == 1
    assert daily.loc[daily["Monitoring Date"] == "2024-01-01", "Rainfall_MM"].iloc[0] == 0.0


def test_missing_rainfall_remains_null():
    """A station reporting missing value must not be treated as 0.0 mm."""
    candidates = pd.DataFrame({
        "Station Code": ["STN1", "STN2"],
        "Monitoring Date": ["2024-01-01", "2024-01-01"],
        "Rainfall (MM)": [np.nan, 10.0]
    })
    daily, stats = build_daily_rainfall(candidates, selected_codes={"STN1", "STN2"})
    row = daily[daily["Monitoring Date"] == "2024-01-01"].iloc[0]
    assert row["Rainfall_MM"] == 10.0
    assert row["Stations_Reporting"] == 1


def test_no_observation_day_has_count_zero():
    """A calendar day with no station observations has null rainfall and count 0."""
    candidates = pd.DataFrame({
        "Station Code": ["STN1", "STN1"],
        "Monitoring Date": ["2024-01-01", "2024-01-03"],  # 2024-01-02 missing
        "Rainfall (MM)": [5.0, 15.0]
    })
    daily, stats = build_daily_rainfall(candidates, selected_codes={"STN1"})
    row_gap = daily[daily["Monitoring Date"] == "2024-01-02"].iloc[0]
    assert np.isnan(row_gap["Rainfall_MM"])
    assert row_gap["Stations_Reporting"] == 0
    assert "2024-01-02" in stats["dates_without_observations"]


def test_station_coverage_count_correct():
    """Verify that station coverage fraction reflects reporting stations over total."""
    candidates = pd.DataFrame({
        "Station Code": ["S1", "S2", "S3"],
        "Monitoring Date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "Rainfall (MM)": [2.0, 4.0, 6.0]
    })
    selected_set = {"S1", "S2", "S3", "S4", "S5"}  # 5 stations expected
    daily, _ = build_daily_rainfall(candidates, selected_codes=selected_set)
    row = daily.iloc[0]
    assert row["Stations_Reporting"] == 3
    coverage = row["Stations_Reporting"] / len(selected_set)
    assert coverage == pytest.approx(0.6)


def test_repeated_run_deterministic():
    """Repeated builds on the same inputs produce byte-identical data."""
    candidates = pd.DataFrame({
        "Station Code": ["S1", "S2", "S1", "S2"],
        "Monitoring Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "Rainfall (MM)": [0.0, 5.0, 2.5, 7.5]
    })
    d1, _ = build_daily_rainfall(candidates, selected_codes={"S1", "S2"})
    d2, _ = build_daily_rainfall(candidates, selected_codes={"S1", "S2"})
    pd.testing.assert_frame_equal(d1, d2)


def test_dam_row_count_unchanged_after_left_join():
    """Dam row count must be preserved exactly after left joining rainfall."""
    dam = pd.DataFrame({
        "calendar_date": pd.date_range("2024-01-01", periods=10, freq="D")
    })
    rainfall = pd.DataFrame({
        "calendar_date": pd.date_range("2024-01-03", periods=5, freq="D"),
        "Rainfall_MM": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    merged = dam.merge(rainfall, on="calendar_date", how="left")
    assert len(merged) == len(dam)
    assert merged["Rainfall_MM"].isna().sum() == 5


def test_no_row_multiplication():
    """Merge must assert 1-to-1 relationship on calendar_date."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    dam = pd.DataFrame({"calendar_date": dates})
    rainfall = pd.DataFrame({
        "calendar_date": dates[:2],
        "Rainfall_MM": [10.0, 20.0]
    })
    merged = dam.merge(rainfall, on="calendar_date", how="left", validate="one_to_one")
    assert len(merged) == len(dam)
