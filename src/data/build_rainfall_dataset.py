from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/interim/krs_rainfall_candidates.csv"
STATION_FILE = PROJECT_ROOT / "data/interim/selected_rainfall_stations.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/krs_daily_rainfall.csv"

REQUIRED_COLUMNS = {"Station Code", "Monitoring Date", "Rainfall (MM)"}
START_DATE = pd.Timestamp("1990-01-01")
END_DATE = pd.Timestamp("2026-10-21")


def parse_candidate_dates(values: pd.Series) -> pd.Series:
    """Parse the ISO dates written by filter_rainfall.py."""
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{int(invalid.sum())} candidate dates are not ISO YYYY-MM-DD values; "
            f"examples: {examples}"
        )
    return parsed.dt.normalize()


def build_daily_rainfall(
    candidates: pd.DataFrame,
    selected_codes: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate unique valid selected-station observations onto a daily index."""
    candidates = candidates.copy()
    candidates.columns = candidates.columns.astype(str).str.strip()
    missing_columns = REQUIRED_COLUMNS - set(candidates.columns)
    if missing_columns:
        raise ValueError(f"Candidate data is missing columns: {sorted(missing_columns)}")
    if not selected_codes:
        raise ValueError("The selected station list is empty.")

    candidates["Station Code"] = (
        candidates["Station Code"].astype("string").str.strip()
    )
    candidates["Monitoring Date"] = parse_candidate_dates(
        candidates["Monitoring Date"].astype("string")
    )
    raw_rainfall_values = candidates["Rainfall (MM)"].copy()
    candidates["Rainfall (MM)"] = pd.to_numeric(
        candidates["Rainfall (MM)"], errors="coerce"
    )

    selected = candidates[
        candidates["Station Code"].isin(selected_codes)
        & candidates["Monitoring Date"].between(START_DATE, END_DATE)
    ].copy()
    if selected.empty:
        raise ValueError("No matching selected-station records found in candidates.")

    duplicate_mask = selected.duplicated(
        subset=["Station Code", "Monitoring Date"], keep=False
    )
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        examples = selected.loc[
            duplicate_mask, ["Station Code", "Monitoring Date", "Rainfall (MM)"]
        ].head(10)
        raise ValueError(
            f"Found {duplicate_rows} duplicate selected station-date rows. "
            "Aggregation stopped so duplicate measurements can be reviewed:\n"
            f"{examples.to_string(index=False)}"
        )

    source_rainfall = raw_rainfall_values.loc[selected.index]
    missing_measurements = int(source_rainfall.isna().sum())
    negative_measurements = int(selected["Rainfall (MM)"].lt(0).sum())
    invalid_numeric_count = int(
        (source_rainfall.notna() & selected["Rainfall (MM)"].isna()).sum()
    )

    valid = selected[
        selected["Rainfall (MM)"].notna()
        & selected["Rainfall (MM)"].ge(0)
    ].copy()
    if valid.empty:
        raise ValueError("Selected stations have no valid non-negative rainfall values.")

    valid_dates = valid["Monitoring Date"]
    observed_daily = valid.groupby("Monitoring Date").agg(
        Rainfall_MM=("Rainfall (MM)", "mean"),
        Stations_Reporting=("Station Code", "nunique"),
    )

    calendar = pd.date_range(
        valid_dates.min(), valid_dates.max(), freq="D", name="Monitoring Date"
    )
    daily = observed_daily.reindex(calendar)
    daily["Stations_Reporting"] = (
        daily["Stations_Reporting"].fillna(0).astype("int64")
    )
    daily = daily.reset_index()

    no_observation_dates = daily.loc[
        daily["Stations_Reporting"].eq(0), "Monitoring Date"
    ]
    stats: dict[str, object] = {
        "candidate_records": len(candidates),
        "selected_station_count": len(selected_codes),
        "selected_candidate_records": len(selected),
        "selected_station_date_observations": len(valid),
        "duplicate_station_date_rows": duplicate_rows,
        "missing_measurements": missing_measurements,
        "invalid_numeric_measurements": invalid_numeric_count,
        "negative_measurements": negative_measurements,
        "zero_rainfall_observations": int(valid["Rainfall (MM)"].eq(0).sum()),
        "unique_rainfall_dates": int(valid_dates.nunique()),
        "first_valid_date": valid_dates.min(),
        "last_valid_date": valid_dates.max(),
        "daily_calendar_rows": len(daily),
        "dates_without_observations": no_observation_dates.dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
    }
    return daily, stats


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Candidate rainfall file not found: {INPUT_FILE}")
    if not STATION_FILE.exists():
        raise FileNotFoundError(f"Selected station file not found: {STATION_FILE}")

    stations = pd.read_csv(STATION_FILE, dtype={"Station Code": "string"})
    if "Station Code" not in stations:
        raise ValueError(f"Selected station file has no 'Station Code' column: {STATION_FILE}")
    station_codes = stations["Station Code"].dropna().str.strip()
    if station_codes.duplicated().any():
        duplicates = station_codes[station_codes.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Selected station file contains duplicate codes: {duplicates}")
    selected_codes = set(station_codes)

    candidates = pd.read_csv(
        INPUT_FILE,
        dtype={"Station Code": "string", "Monitoring Date": "string"},
        low_memory=False,
    )
    daily, stats = build_daily_rainfall(candidates, selected_codes)

    temporary_output = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.tmp{OUTPUT_FILE.suffix}"
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(temporary_output, index=False, date_format="%Y-%m-%d")
    temporary_output.replace(OUTPUT_FILE)

    print(f"Candidate rainfall records: {stats['candidate_records']:,}")
    print(f"Selected stations: {stats['selected_station_count']:,}")
    print(f"Selected candidate records: {stats['selected_candidate_records']:,}")
    print(
        "Valid selected station-date observations: "
        f"{stats['selected_station_date_observations']:,}"
    )
    print(
        "Duplicate selected station-date rows: "
        f"{stats['duplicate_station_date_rows']:,} (none removed; duplicates stop the build)"
    )
    print(f"Missing selected measurements: {stats['missing_measurements']:,}")
    print(f"Non-numeric selected measurements: {stats['invalid_numeric_measurements']:,}")
    print(f"Negative selected measurements excluded: {stats['negative_measurements']:,}")
    print(f"Valid zero-rainfall observations: {stats['zero_rainfall_observations']:,}")
    print(f"Unique dates with valid rainfall: {stats['unique_rainfall_dates']:,}")
    print(f"First valid date: {stats['first_valid_date']:%Y-%m-%d}")
    print(f"Last valid date: {stats['last_valid_date']:%Y-%m-%d}")
    print(f"Daily calendar rows written: {stats['daily_calendar_rows']:,}")

    missing_dates = stats["dates_without_observations"]
    print(f"Calendar dates with no observations: {len(missing_dates):,}")
    print(
        "Dates with no observations: "
        + (", ".join(missing_dates) if missing_dates else "None")
    )
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
