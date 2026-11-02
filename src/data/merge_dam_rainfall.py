from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DAM_FILE = ROOT / "data/processed/krs_dam_processed.csv"
RAINFALL_FILE = ROOT / "data/processed/krs_daily_rainfall.csv"
OUTPUT_FILE = ROOT / "data/interim/krs_dam_rainfall_merged.csv"


def main():
    if not DAM_FILE.exists():
        raise FileNotFoundError(f"Processed dam dataset not found: {DAM_FILE}")
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Daily rainfall dataset not found: {RAINFALL_FILE}")

    dam = pd.read_csv(DAM_FILE)
    rainfall = pd.read_csv(RAINFALL_FILE)
    dam_row_count = len(dam)

    required_dam_columns = {"calendar_date"}
    required_rainfall_columns = {
        "Monitoring Date",
        "Rainfall_MM",
        "Stations_Reporting",
    }
    missing_dam_columns = required_dam_columns - set(dam.columns)
    if missing_dam_columns:
        raise ValueError(
            f"Dam dataset is missing columns: {sorted(missing_dam_columns)}"
        )
    missing_rainfall_columns = required_rainfall_columns - set(rainfall.columns)
    if missing_rainfall_columns:
        raise ValueError(
            "Rainfall dataset is missing columns: "
            f"{sorted(missing_rainfall_columns)}"
        )

    # Standardize the dates
    dam["calendar_date"] = pd.to_datetime(
        dam["calendar_date"], errors="coerce"
    ).dt.normalize()

    rainfall["Monitoring Date"] = pd.to_datetime(
        rainfall["Monitoring Date"], errors="coerce"
    ).dt.normalize()

    if dam["calendar_date"].isna().any():
        raise ValueError("Invalid dates found in dam dataset.")

    if rainfall["Monitoring Date"].isna().any():
        raise ValueError("Invalid dates found in rainfall dataset.")

    # Each dataset must have one row per normalized date.
    if dam["calendar_date"].duplicated().any():
        raise ValueError("Duplicate dates found in dam dataset.")

    if rainfall["Monitoring Date"].duplicated().any():
        raise ValueError("Duplicate dates found in rainfall dataset.")

    rainfall["Rainfall_MM"] = pd.to_numeric(
        rainfall["Rainfall_MM"], errors="coerce"
    )
    rainfall["Stations_Reporting"] = pd.to_numeric(
        rainfall["Stations_Reporting"], errors="raise"
    )
    if (rainfall["Stations_Reporting"] < 0).any():
        raise ValueError("Negative Stations_Reporting values found.")

    # Rename rainfall date to match dam date
    rainfall = rainfall.rename(
        columns={"Monitoring Date": "calendar_date"}
    )

    # Preserve every dam record
    merged = dam.merge(
        rainfall[
            ["calendar_date", "Rainfall_MM", "Stations_Reporting"]
        ],
        on="calendar_date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = int((merged["_merge"] == "both").sum())
    matched_with_observation = int(merged["Rainfall_MM"].notna().sum())
    without_observation = dam_row_count - matched_with_observation

    merged = merged.drop(columns="_merge")
    merged = merged.sort_values("calendar_date").reset_index(drop=True)
    if len(merged) != dam_row_count:
        raise AssertionError(
            f"Left join changed row count: dam={dam_row_count}, merged={len(merged)}"
        )
    if merged["calendar_date"].duplicated().any():
        raise AssertionError("Left join duplicated one or more dam dates.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.tmp{OUTPUT_FILE.suffix}"
    )
    merged.to_csv(temporary_output, index=False)
    temporary_output.replace(OUTPUT_FILE)

    print("Dam rows before merge:", dam_row_count)
    print("Merged rows:", len(merged))
    print("Dam date keys joined to rainfall calendar:", matched)
    print(
        "Dam dates with valid rainfall observations:",
        f"{matched_with_observation} ({matched_with_observation / dam_row_count:.2%})",
    )
    print(
        "Dam dates without valid rainfall observations:",
        f"{without_observation} ({without_observation / dam_row_count:.2%})",
    )
    print("Dam rows lost:", dam_row_count - len(merged))
    print("Duplicate dam dates after merge:", merged["calendar_date"].duplicated().sum())
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()  