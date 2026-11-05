from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAINFALL_DIR = PROJECT_ROOT / "data" / "raw" / "rainfall"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_FILE = OUTPUT_DIR / "krs_rainfall_candidates.csv"
STATIONS_FILE = OUTPUT_DIR / "krs_rainfall_candidate_stations.csv"
FILTERED_TEMP_FILE = FILTERED_FILE.with_name(f"{FILTERED_FILE.stem}.tmp{FILTERED_FILE.suffix}")
STATIONS_TEMP_FILE = STATIONS_FILE.with_name(f"{STATIONS_FILE.stem}.tmp{STATIONS_FILE.suffix}")

# --------------------------------------------------
# 2. Date range matching the existing dam dataset
# --------------------------------------------------

START_DATE = pd.Timestamp("1990-01-01")
END_DATE = pd.Timestamp("2026-10-21")

# Initial geographic selection.
# These are candidate districts, not the final station selection.
SELECTED_DISTRICTS = {
    "kodagu",
    "hassan",
    "mysuru",
    "mandya",
}

# --------------------------------------------------
# 3. Find the rainfall CSV files
# --------------------------------------------------

csv_files = sorted(RAINFALL_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {RAINFALL_DIR}"
    )

print(f"Found {len(csv_files)} rainfall CSV files.")

# --------------------------------------------------
# 4. Prepare temporary output files. Existing outputs are replaced only
# after every source file has been processed successfully.
# --------------------------------------------------

for temporary_file in (FILTERED_TEMP_FILE, STATIONS_TEMP_FILE):
    if temporary_file.exists():
        temporary_file.unlink()

station_parts = []
total_rows_read = 0
total_invalid_dates = 0
total_geographic_matches = 0
total_geographic_date_matches = 0
total_rows_kept = 0

# --------------------------------------------------
# 5. Process files in chunks to limit memory usage
# --------------------------------------------------

for file in csv_files:
    print(f"\nProcessing: {file.name}")

    try:
        reader = pd.read_csv(
            file,
            chunksize=100_000,
            low_memory=False,
        )

        for chunk in reader:
            # Normalize column names.
            chunk.columns = (
                chunk.columns.astype(str).str.strip()
            )

            required_columns = {
                "Station Code",
                "Location Name",
                "State",
                "District",
                "Basin",
                "Sub Basin",
                "Latitude",
                "Longitude",
                "Monitoring Date",
                "Rainfall (MM)",
            }

            missing_columns = (
                required_columns - set(chunk.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"{file.name} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            total_rows_read += len(chunk)
            chunk["Station Code"] = (
                chunk["Station Code"].astype("string").str.strip()
            )

            # Normalize text fields for filtering.
            state = (
                chunk["State"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            basin = (
                chunk["Basin"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            sub_basin = (
                chunk["Sub Basin"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            district = (
                chunk["District"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            # Parse dates. The portal may use DD-MM-YYYY.
            dates = pd.to_datetime(
                chunk["Monitoring Date"],
                format="%d-%m-%Y",
                errors="coerce",
            )
            total_invalid_dates += int(dates.isna().sum())

            # Geographic filters.
            mask = (
                state.eq("karnataka")
                & basin.eq("cauvery")
                & sub_basin.eq("upper cauvery")
                & district.isin(SELECTED_DISTRICTS)
            )
            total_geographic_matches += int(mask.sum())

            # Date filters.
            date_mask = dates.between(
                START_DATE,
                END_DATE,
                inclusive="both",
            )
            total_geographic_date_matches += int((mask & date_mask).sum())
            mask &= date_mask

            filtered = chunk.loc[mask].copy()
            filtered["Monitoring Date"] = dates.loc[mask]

            if filtered.empty:
                continue

            # Convert rainfall to numeric without inventing values.
            filtered["Rainfall (MM)"] = pd.to_numeric(
                filtered["Rainfall (MM)"],
                errors="coerce",
            )

            # Preserve missing rainfall as missing.
            # Zero rainfall remains a valid observation.

            filtered.to_csv(
                FILTERED_TEMP_FILE,
                mode="a",
                header=not FILTERED_TEMP_FILE.exists(),
                index=False,
            )

            station_columns = [
                "Station Code",
                "Location Name",
                "District",
                "Taluk",
                "Basin",
                "Sub Basin",
                "Latitude",
                "Longitude",
            ]

            available_station_columns = [
                col
                for col in station_columns
                if col in filtered.columns
            ]

            station_parts.append(
                filtered[available_station_columns].drop_duplicates()
            )

            total_rows_kept += len(filtered)

    except Exception as exc:
        raise RuntimeError(
            f"Failed while processing {file.name}: {exc}"
        ) from exc

# --------------------------------------------------
# 6. Save the candidate station list
# --------------------------------------------------

if total_rows_kept == 0:
    raise ValueError(
        "No rows matched the filters. Check the actual "
        "column values and date format in the source CSV files."
    )

stations = pd.concat(
    station_parts,
    ignore_index=True,
).drop_duplicates()

stations.to_csv(STATIONS_TEMP_FILE, index=False)
FILTERED_TEMP_FILE.replace(FILTERED_FILE)
STATIONS_TEMP_FILE.replace(STATIONS_FILE)

# --------------------------------------------------
# 7. Report
# --------------------------------------------------

print("\nFiltering complete.")
print(f"Total source rows read: {total_rows_read:,}")
print(f"Unparseable source dates: {total_invalid_dates:,}")
print(f"Rows matching geographic filters: {total_geographic_matches:,}")
print(
    "Rows matching geography and inclusive date range: "
    f"{total_geographic_date_matches:,}"
)
print(f"Candidate rainfall records written: {total_rows_kept:,}")
print(f"Unique candidate stations: {len(stations):,}")

print(f"\nFiltered records saved to:\n{FILTERED_FILE}")
print(f"\nCandidate stations saved to:\n{STATIONS_FILE}")

print("\nDistrict counts:")
print(stations["District"].value_counts(dropna=False))

print("\nFirst candidate stations:")
print(stations.head(20).to_string(index=False))