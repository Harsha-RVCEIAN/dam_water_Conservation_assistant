"""Build cutoff-safe rainfall features for experimental dam forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAM_DATA_PATH = PROJECT_ROOT / "data/processed/krs_dam_processed.csv"
RAINFALL_DATA_PATH = PROJECT_ROOT / "data/processed/krs_daily_rainfall.csv"
STATION_COUNT = 10

RAINFALL_FEATURE_COLUMNS = (
    "rainfall_mm_lag_1",
    "rainfall_mm_sum_3d",
    "rainfall_mm_sum_7d",
    "rainfall_stations_reporting_lag_1",
    "rainfall_station_coverage_lag_1",
    "rainfall_lag_1_available",
    "rainfall_window_3d_complete",
    "rainfall_window_7d_complete",
)


def _normalize_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{name} contains {int(invalid.sum())} invalid dates; examples: {examples}"
        )
    return parsed.dt.normalize()


def build_rainfall_features(
    dam_dates: pd.Series,
    rainfall: pd.DataFrame,
) -> pd.DataFrame:
    """Return date-keyed rainfall features using only observations through t-1.

    The returned frame has one row per supplied dam date, in the same order as
    ``dam_dates``. Rainfall is first reindexed over its full continuous calendar,
    independently of the dam dates, so rainfall dates without dam observations
    remain available to lag and rolling windows.
    """
    required_columns = {"Monitoring Date", "Rainfall_MM", "Stations_Reporting"}
    missing_columns = required_columns - set(rainfall.columns)
    if missing_columns:
        raise ValueError(
            f"Rainfall data is missing columns: {sorted(missing_columns)}"
        )

    dam_calendar_dates = _normalize_dates(
        pd.Series(dam_dates).reset_index(drop=True), "Dam dates"
    )
    if dam_calendar_dates.empty:
        raise ValueError("At least one dam date is required to build rainfall features.")

    rain = rainfall.loc[
        :, ["Monitoring Date", "Rainfall_MM", "Stations_Reporting"]
    ].copy()
    # The daily rainfall file is written in ISO format. Parse explicitly to
    # avoid ambiguous day-first interpretation.
    raw_rain_dates = rain["Monitoring Date"].astype("string")
    parsed_rain_dates = pd.to_datetime(
        raw_rain_dates, format="%Y-%m-%d", errors="coerce"
    )
    invalid_rain_dates = parsed_rain_dates.isna()
    if invalid_rain_dates.any():
        examples = (
            raw_rain_dates.loc[invalid_rain_dates]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Rainfall data contains {int(invalid_rain_dates.sum())} invalid "
            f"ISO dates; examples: {examples}"
        )
    rain["Monitoring Date"] = parsed_rain_dates.dt.normalize()

    if rain["Monitoring Date"].duplicated().any():
        duplicates = (
            rain.loc[
                rain["Monitoring Date"].duplicated(keep=False), "Monitoring Date"
            ]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(f"Rainfall data has duplicate dates: {duplicates}")

    raw_rainfall = rain["Rainfall_MM"]
    rain["Rainfall_MM"] = pd.to_numeric(raw_rainfall, errors="coerce")
    invalid_rainfall = raw_rainfall.notna() & rain["Rainfall_MM"].isna()
    if invalid_rainfall.any():
        examples = raw_rainfall.loc[invalid_rainfall].astype(str).head(5).tolist()
        raise ValueError(
            f"Rainfall_MM contains non-numeric values; examples: {examples}"
        )
    if rain["Rainfall_MM"].lt(0).any():
        raise ValueError("Rainfall_MM contains negative measurements.")

    raw_reporting = rain["Stations_Reporting"]
    rain["Stations_Reporting"] = pd.to_numeric(raw_reporting, errors="coerce")
    invalid_reporting = raw_reporting.notna() & rain["Stations_Reporting"].isna()
    if invalid_reporting.any():
        examples = raw_reporting.loc[invalid_reporting].astype(str).head(5).tolist()
        raise ValueError(
            "Stations_Reporting contains non-numeric values; "
            f"examples: {examples}"
        )
    reporting = rain["Stations_Reporting"].dropna()
    if (
        reporting.lt(0).any()
        or reporting.gt(STATION_COUNT).any()
        or reporting.mod(1).ne(0).any()
    ):
        raise ValueError(
            f"Stations_Reporting must be a whole number between 0 and {STATION_COUNT}."
        )

    rain = rain.set_index("Monitoring Date").sort_index()
    start_date = min(
        rain.index.min(),
        dam_calendar_dates.min() - pd.Timedelta(days=7),
    )
    end_date = max(rain.index.max(), dam_calendar_dates.max())
    full_calendar = pd.date_range(start_date, end_date, freq="D", name="calendar_date")
    daily = rain.reindex(full_calendar)

    rainfall_through_previous_day = daily["Rainfall_MM"].shift(1)
    reporting_through_previous_day = daily["Stations_Reporting"].shift(1)

    feature_calendar = pd.DataFrame(index=full_calendar)
    feature_calendar["rainfall_mm_lag_1"] = rainfall_through_previous_day
    feature_calendar["rainfall_mm_sum_3d"] = (
        rainfall_through_previous_day.rolling(window=3, min_periods=3).sum()
    )
    feature_calendar["rainfall_mm_sum_7d"] = (
        rainfall_through_previous_day.rolling(window=7, min_periods=7).sum()
    )
    feature_calendar["rainfall_stations_reporting_lag_1"] = (
        reporting_through_previous_day
    )
    feature_calendar["rainfall_station_coverage_lag_1"] = (
        reporting_through_previous_day / STATION_COUNT
    )
    feature_calendar["rainfall_lag_1_available"] = (
        rainfall_through_previous_day.notna().astype("int8")
    )
    feature_calendar["rainfall_window_3d_complete"] = (
        rainfall_through_previous_day.notna()
        .rolling(window=3, min_periods=3)
        .sum()
        .eq(3)
        .astype("int8")
    )
    feature_calendar["rainfall_window_7d_complete"] = (
        rainfall_through_previous_day.notna()
        .rolling(window=7, min_periods=7)
        .sum()
        .eq(7)
        .astype("int8")
    )

    dam_rows = pd.DataFrame({
        "calendar_date": dam_calendar_dates,
        "_input_order": range(len(dam_calendar_dates)),
    })
    result = dam_rows.merge(
        feature_calendar.reset_index(),
        on="calendar_date",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.sort_values("_input_order").drop(columns="_input_order")
    result["rainfall_lag_1_available"] = (
        result["rainfall_lag_1_available"].fillna(0).astype("int8")
    )
    result["rainfall_window_3d_complete"] = (
        result["rainfall_window_3d_complete"].fillna(0).astype("int8")
    )
    result["rainfall_window_7d_complete"] = (
        result["rainfall_window_7d_complete"].fillna(0).astype("int8")
    )
    return result[["calendar_date", *RAINFALL_FEATURE_COLUMNS]].reset_index(drop=True)


def load_rainfall_features(
    dam_data_path: Path = DAM_DATA_PATH,
    rainfall_data_path: Path = RAINFALL_DATA_PATH,
) -> pd.DataFrame:
    """Load project datasets and build rainfall features for the dam dates."""
    if not dam_data_path.exists():
        raise FileNotFoundError(f"Dam dataset not found: {dam_data_path}")
    if not rainfall_data_path.exists():
        raise FileNotFoundError(f"Daily rainfall dataset not found: {rainfall_data_path}")

    dam_dates = pd.read_csv(dam_data_path, usecols=["calendar_date"])["calendar_date"]
    rainfall = pd.read_csv(
        rainfall_data_path,
        usecols=["Monitoring Date", "Rainfall_MM", "Stations_Reporting"],
        dtype={"Monitoring Date": "string"},
    )
    return build_rainfall_features(dam_dates, rainfall)
