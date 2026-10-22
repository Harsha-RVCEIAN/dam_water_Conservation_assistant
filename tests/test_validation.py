"""Tests for data validation, header cleaning, and date parsing."""

import pandas as pd
import pytest

from src.data.validate import (
    CANONICAL_COLUMNS,
    clean_header_name,
    normalize_dataframe_columns,
    parse_monitoring_dates,
    run_validation,
)


def test_clean_header_name_normalizes_newlines_and_spaces():
    assert clean_header_name("Sub \nBasin") == "sub_basin"
    assert clean_header_name("Sub  Basin") == "sub_basin"
    assert clean_header_name("Live Capacity \n(TMC)") == "live_capacity_tmc"
    assert clean_header_name("Cumulative Evaporation \n(TMC)") == "cumulative_evaporation_tmc"
    assert clean_header_name("Inflow (Cusecs)") == "inflow_cusecs"


def test_normalize_dataframe_columns():
    raw_df = pd.DataFrame(columns=["Reservoir Name", "Sub \nBasin", "Live Capacity \n(TMC)"])
    clean_df, mapping = normalize_dataframe_columns(raw_df)
    assert "reservoir_name" in clean_df.columns
    assert "sub_basin" in clean_df.columns
    assert "live_capacity_tmc" in clean_df.columns
    assert len(mapping) == 3


def test_parse_monitoring_dates():
    sample_dates = pd.Series([
        "01-01-1990 12:01:00",
        "15-08-2020 12:00:00",
        "invalid_date_str",
    ])
    parsed = parse_monitoring_dates(sample_dates)
    assert parsed.iloc[0] == pd.Timestamp("1990-01-01 12:01:00")
    assert parsed.iloc[1] == pd.Timestamp("2020-08-15 12:00:00")
    assert pd.isna(parsed.iloc[2])


def test_validation_report_on_synthetic_data(tmp_path):
    # Construct a small synthetic raw CSV with an intentional gap and duplicate check
    synth_file = tmp_path / "synthetic_raw.csv"
    data = {
        "Reservoir Name": ["K.R.Sagara Dam", "K.R.Sagara Dam", "K.R.Sagara Dam"],
        "Basin": ["Cauvery", "Cauvery", "Cauvery"],
        "Sub \nBasin": ["Upper Cauvery", "Upper Cauvery", "Upper Cauvery"],
        "River": ["Cauvery", "Cauvery", "Cauvery"],
        "Monitoring Date": ["01-01-2020 12:00:00", "02-01-2020 12:00:00", "04-01-2020 12:00:00"],  # Jan 3 missing
        "Percentage Full": [50.0, None, 52.0],
        "Reservoir Level (ft)": [2450.0, 2451.0, 15000.0],  # Level anomaly at row 2
        "Design Gross Capacity (TMC)": [49.452, 49.452, 49.452],
        "Gross Capacity (TMC)": [25.0, 25.5, 26.0],
        "Live Capacity \n(TMC)": [20.0, 20.5, 21.0],
        "Live Above Cill (TMC)": [20.0, 20.5, 21.0],
        "Inflow (Cusecs)": [1000.0, None, 1200.0],
        "Outflow to River (Cusecs)": [500.0, 500.0, 500.0],
        "Canal Withdrawal (Cusecs)": [200.0, 200.0, 200.0],
        "Evaporation (Cusecs)": [50.0, 50.0, 50.0],
        "Cumulative Inflow (TMC)": [10.0, 10.1, 10.2],
        "Cumulative Outflow (TMC)": [5.0, 5.05, 5.1],
        "Cumulative Withdrawal (TMC)": [2.0, 2.02, 2.04],
        "Cumulative Evaporation \n(TMC)": [0.5, 0.51, 0.52],
    }
    pd.DataFrame(data).to_csv(synth_file, index=False)

    report = run_validation(synth_file)
    assert report.total_rows == 3
    assert report.date_parse_failures == 0
    assert report.missing_calendar_days == ["2020-01-03"]
    assert report.missing_values_by_column["inflow_cusecs"] == 1
    assert any("15000.0 ft" in f for f in report.flagged_anomalies)


def test_validation_fails_on_missing_file():
    with pytest.raises(FileNotFoundError):
        run_validation(pd.Path("non_existent_file.csv") if hasattr(pd, "Path") else pytest.importorskip("pathlib").Path("non_existent.csv"))
