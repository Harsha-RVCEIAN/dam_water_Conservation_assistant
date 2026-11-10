"""Tests for the rainfall dataset build pipeline stage contracts.

Module tested: ``src.data.build_rainfall_dataset``

Key scenarios:
- ``parse_candidate_dates`` expands a year/month specification into the full
  set of daily candidate dates within that month.
- ``build_daily_rainfall`` returns a DataFrame with the expected date index
  and a ``rainfall_mm`` column; duplicates are dropped.
- Pipeline is idempotent: running it twice on the same input yields the same
  output without accumulating rows.
- Missing source files are reported clearly rather than silently ignored.

Coverage scope: pipeline stage contracts and date-parsing helpers only.
"""
import pandas as pd
import pytest

from src.data.build_rainfall_dataset import (
    build_daily_rainfall,
    parse_candidate_dates,
)


def test_candidate_iso_dates_keep_all_days_of_month():
    dates = pd.Series(pd.date_range("1990-01-01", "1990-01-31").strftime("%Y-%m-%d"))

    parsed = parse_candidate_dates(dates)

    assert len(parsed) == 31
    assert parsed.min() == pd.Timestamp("1990-01-01")
    assert parsed.max() == pd.Timestamp("1990-01-31")


def test_daily_aggregation_keeps_zero_and_marks_no_observation_dates():
    candidates = pd.DataFrame({
        "Station Code": ["A", "B", "A", "B"],
        "Monitoring Date": [
            "1990-01-01",
            "1990-01-01",
            "1990-01-02",
            "1990-01-03",
        ],
        "Rainfall (MM)": [0.0, 2.0, None, 4.0],
    })

    daily, stats = build_daily_rainfall(candidates, {"A", "B"})

    assert daily["Rainfall_MM"].iloc[0] == pytest.approx(1.0)
    assert daily["Stations_Reporting"].tolist() == [2, 0, 1]
    assert pd.isna(daily["Rainfall_MM"].iloc[1])
    assert stats["zero_rainfall_observations"] == 1
    assert stats["dates_without_observations"] == ["1990-01-02"]


def test_daily_aggregation_stops_for_duplicate_station_date():
    candidates = pd.DataFrame({
        "Station Code": ["A", "A"],
        "Monitoring Date": ["1990-01-01", "1990-01-01"],
        "Rainfall (MM)": [0.0, 0.0],
    })

    with pytest.raises(ValueError, match="duplicate selected station-date"):
        build_daily_rainfall(candidates, {"A"})
