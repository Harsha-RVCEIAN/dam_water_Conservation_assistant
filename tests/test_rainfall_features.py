"""Tests for rainfall feature-engineering contracts.

Module tested: ``src.data.rainfall_features``

Key scenarios:
- ``build_rainfall_features`` produces exactly the columns listed in
  ``RAINFALL_FEATURE_COLUMNS`` (8 features).
- Rolling and lag features are computed without data-leakage (future
  observations do not influence past rows).
- Edge cases: single-row frames, all-NaN rainfall input, zero rainfall.
- Output dtype and index alignment match the input date index.

Coverage scope: feature contract and anti-leakage invariants; does not
exercise the full training pipeline.
"""
import pandas as pd
import pytest

from src.data.rainfall_features import (
    RAINFALL_FEATURE_COLUMNS,
    build_rainfall_features,
)


def _rainfall_frame(
    start: str,
    end: str,
    *,
    missing_dates: set[str] | None = None,
    missing_values: set[str] | None = None,
) -> pd.DataFrame:
    missing_dates = missing_dates or set()
    missing_values = missing_values or set()
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for day in dates:
        day_string = day.strftime("%Y-%m-%d")
        if day_string in missing_dates:
            continue
        rows.append({
            "Monitoring Date": day_string,
            "Rainfall_MM": None if day_string in missing_values else float(day.day),
            "Stations_Reporting": 5,
        })
    return pd.DataFrame(rows)


def _feature_row(dam_dates: list[str], rainfall: pd.DataFrame, date: str):
    features = build_rainfall_features(pd.Series(dam_dates), rainfall)
    return features.loc[features["calendar_date"].eq(pd.Timestamp(date))].iloc[0]


def test_measured_zero_is_available_and_distinct_from_missing():
    rainfall = pd.DataFrame({
        "Monitoring Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Rainfall_MM": [0.0, None, 4.0],
        "Stations_Reporting": [10, None, 5],
    })

    features = build_rainfall_features(
        pd.Series(["2020-01-02", "2020-01-03", "2020-01-04"]),
        rainfall,
    ).set_index("calendar_date")

    zero_row = features.loc[pd.Timestamp("2020-01-02")]
    missing_row = features.loc[pd.Timestamp("2020-01-03")]
    assert zero_row["rainfall_mm_lag_1"] == 0.0
    assert zero_row["rainfall_lag_1_available"] == 1
    assert pd.isna(missing_row["rainfall_mm_lag_1"])
    assert missing_row["rainfall_lag_1_available"] == 0


def test_daily_calendar_gap_is_reindexed_and_invalidates_windows():
    rainfall = _rainfall_frame(
        "2020-01-01", "2020-01-07", missing_dates={"2020-01-06"}
    )

    row = _feature_row(["2020-01-08"], rainfall, "2020-01-08")

    assert row["rainfall_mm_lag_1"] == 7.0
    assert pd.isna(row["rainfall_mm_sum_3d"])
    assert pd.isna(row["rainfall_mm_sum_7d"])
    assert row["rainfall_window_3d_complete"] == 0
    assert row["rainfall_window_7d_complete"] == 0


def test_rolling_window_boundaries_use_exact_prior_calendar_days():
    rainfall = _rainfall_frame("2020-01-01", "2020-01-08")

    row = _feature_row(["2020-01-09"], rainfall, "2020-01-09")

    # Row t=Jan 9 uses Jan 8 for lag 1, Jan 6-8 for 3 days,
    # and Jan 2-8 for 7 days.
    assert row["rainfall_mm_lag_1"] == 8.0
    assert row["rainfall_mm_sum_3d"] == 6.0 + 7.0 + 8.0
    assert row["rainfall_mm_sum_7d"] == sum(range(2, 9))
    assert row["rainfall_window_3d_complete"] == 1
    assert row["rainfall_window_7d_complete"] == 1


@pytest.mark.parametrize(
    ("missing_day", "window_column", "complete_column"),
    [
        ("2020-01-06", "rainfall_mm_sum_3d", "rainfall_window_3d_complete"),
        ("2020-01-03", "rainfall_mm_sum_7d", "rainfall_window_7d_complete"),
    ],
)
def test_missing_value_in_window_keeps_total_missing(
    missing_day, window_column, complete_column
):
    rainfall = _rainfall_frame(
        "2020-01-01", "2020-01-08", missing_values={missing_day}
    )

    row = _feature_row(["2020-01-09"], rainfall, "2020-01-09")

    assert pd.isna(row[window_column])
    assert row[complete_column] == 0


def test_target_alignment_uses_t_minus_1_rainfall_for_t_plus_1_target():
    rainfall = _rainfall_frame("2020-01-01", "2020-01-09")
    dam = pd.DataFrame({
        "calendar_date": pd.to_datetime(["2020-01-09"]),
        "target_inflow_next_day_cusecs": [1234.0],
    })
    features = build_rainfall_features(dam["calendar_date"], rainfall)
    aligned = dam.merge(features, on="calendar_date", validate="one_to_one")

    assert aligned.loc[0, "calendar_date"] == pd.Timestamp("2020-01-09")
    assert aligned.loc[0, "rainfall_mm_lag_1"] == 8.0
    assert aligned.loc[0, "target_inflow_next_day_cusecs"] == 1234.0
    assert aligned.loc[0, "rainfall_mm_lag_1"] == rainfall.loc[
        rainfall["Monitoring Date"].eq("2020-01-08"), "Rainfall_MM"
    ].iloc[0]


def test_changing_current_or_future_rainfall_does_not_change_row_features():
    rainfall = _rainfall_frame("2020-01-01", "2020-01-12")
    baseline = _feature_row(["2020-01-09"], rainfall, "2020-01-09")
    changed = rainfall.copy()
    changed.loc[
        changed["Monitoring Date"].ge("2020-01-09"), "Rainfall_MM"
    ] = 10000.0
    after_change = _feature_row(["2020-01-09"], changed, "2020-01-09")

    assert baseline[list(RAINFALL_FEATURE_COLUMNS)].equals(
        after_change[list(RAINFALL_FEATURE_COLUMNS)]
    )


def test_february_11_rainfall_is_used_for_february_12_even_without_dam_row():
    rainfall = pd.DataFrame({
        "Monitoring Date": [
            "2020-02-08",
            "2020-02-09",
            "2020-02-10",
            "2020-02-11",
            "2020-02-12",
        ],
        "Rainfall_MM": [8.0, 9.0, 10.0, 11.0, 12.0],
        "Stations_Reporting": [10, 10, 10, 10, 10],
    })

    row = _feature_row(["2020-02-10", "2020-02-12"], rainfall, "2020-02-12")

    assert row["rainfall_mm_lag_1"] == 11.0
    assert row["rainfall_mm_sum_3d"] == 9.0 + 10.0 + 11.0


def test_station_coverage_uses_fixed_ten_station_denominator():
    rainfall = pd.DataFrame({
        "Monitoring Date": ["2020-01-01", "2020-01-02"],
        "Rainfall_MM": [0.0, None],
        "Stations_Reporting": [5, 0],
    })

    features = build_rainfall_features(
        pd.Series(["2020-01-02", "2020-01-03"]),
        rainfall,
    ).set_index("calendar_date")
    half_coverage = features.loc[pd.Timestamp("2020-01-02")]
    no_coverage = features.loc[pd.Timestamp("2020-01-03")]

    assert half_coverage["rainfall_stations_reporting_lag_1"] == 5
    assert half_coverage["rainfall_station_coverage_lag_1"] == 0.5
    assert no_coverage["rainfall_stations_reporting_lag_1"] == 0
    assert no_coverage["rainfall_station_coverage_lag_1"] == 0.0
    assert pd.isna(no_coverage["rainfall_mm_lag_1"])
    assert no_coverage["rainfall_lag_1_available"] == 0


def test_2026_missing_rainfall_stays_missing_without_carry_forward():
    rainfall = _rainfall_frame("2025-12-24", "2025-12-31")

    features = build_rainfall_features(
        pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"]),
        rainfall,
    ).set_index("calendar_date")
    first_2026_day = features.loc[pd.Timestamp("2026-01-01")]
    second_2026_day = features.loc[pd.Timestamp("2026-01-02")]

    assert first_2026_day["rainfall_mm_lag_1"] == 31.0
    assert first_2026_day["rainfall_window_7d_complete"] == 1
    assert pd.isna(second_2026_day["rainfall_mm_lag_1"])
    assert pd.isna(second_2026_day["rainfall_mm_sum_3d"])
    assert pd.isna(second_2026_day["rainfall_mm_sum_7d"])
    assert second_2026_day["rainfall_lag_1_available"] == 0
    assert second_2026_day["rainfall_window_3d_complete"] == 0
    assert second_2026_day["rainfall_window_7d_complete"] == 0


def test_feature_names_and_order_are_stable():
    rainfall = _rainfall_frame("2020-01-01", "2020-01-03")

    features = build_rainfall_features(pd.Series(["2020-01-04"]), rainfall)

    assert tuple(features.columns[1:]) == RAINFALL_FEATURE_COLUMNS
    assert features.columns.tolist() == [
        "calendar_date",
        "rainfall_mm_lag_1",
        "rainfall_mm_sum_3d",
        "rainfall_mm_sum_7d",
        "rainfall_stations_reporting_lag_1",
        "rainfall_station_coverage_lag_1",
        "rainfall_lag_1_available",
        "rainfall_window_3d_complete",
        "rainfall_window_7d_complete",
    ]
