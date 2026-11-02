from pathlib import Path
import pandas as pd
from math import radians, sin, cos, asin, sqrt

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAINFALL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "krs_rainfall_candidates.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rainfall_station_summary.csv"

# Approximate coordinates of KRS Dam
KRS_LATITUDE = 12.424
KRS_LONGITUDE = 76.573

# --------------------------------------------------
# 2. Distance calculation
# --------------------------------------------------

def distance_km(lat1, lon1, lat2, lon2):
    """Calculate approximate straight-line distance in km."""
    if pd.isna(lat1) or pd.isna(lon1):
        return float("nan")

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 6371 * 2 * asin(sqrt(a))


# --------------------------------------------------
# 3. Read rainfall data in chunks
# --------------------------------------------------

if not RAINFALL_FILE.exists():
    raise FileNotFoundError(
        f"Rainfall file not found: {RAINFALL_FILE}"
    )

station_stats = {}

for chunk in pd.read_csv(
    RAINFALL_FILE,
    chunksize=100_000,
    low_memory=False,
):
    chunk.columns = chunk.columns.str.strip()

    chunk["Monitoring Date"] = pd.to_datetime(
        chunk["Monitoring Date"],
        errors="coerce",
    )

    chunk["Rainfall (MM)"] = pd.to_numeric(
        chunk["Rainfall (MM)"],
        errors="coerce",
    )

    for station_code, group in chunk.groupby("Station Code"):
        if pd.isna(station_code):
            continue

        code = str(station_code)

        if code not in station_stats:
            first_row = group.iloc[0]

            station_stats[code] = {
                "Station Code": code,
                "Location Name": first_row["Location Name"],
                "District": first_row["District"],
                "Taluk": first_row.get("Taluk", ""),
                "Latitude": pd.to_numeric(
                    first_row["Latitude"], errors="coerce"
                ),
                "Longitude": pd.to_numeric(
                    first_row["Longitude"], errors="coerce"
                ),
                "First Date": group["Monitoring Date"].min(),
                "Last Date": group["Monitoring Date"].max(),
                "Record Count": 0,
                "Missing Rainfall Count": 0,
                "Zero Rainfall Count": 0,
            }

        stats = station_stats[code]

        stats["First Date"] = min(
            stats["First Date"],
            group["Monitoring Date"].min(),
        )

        stats["Last Date"] = max(
            stats["Last Date"],
            group["Monitoring Date"].max(),
        )

        stats["Record Count"] += len(group)

        stats["Missing Rainfall Count"] += (
            group["Rainfall (MM)"].isna().sum()
        )

        stats["Zero Rainfall Count"] += (
            group["Rainfall (MM)"].eq(0).sum()
        )


# --------------------------------------------------
# 4. Create station summary
# --------------------------------------------------

summary = pd.DataFrame(station_stats.values())

summary["Distance from KRS (km)"] = summary.apply(
    lambda row: distance_km(
        row["Latitude"],
        row["Longitude"],
        KRS_LATITUDE,
        KRS_LONGITUDE,
    ),
    axis=1,
)

summary["Missing Rainfall (%)"] = (
    summary["Missing Rainfall Count"]
    / summary["Record Count"].replace(0, pd.NA)
    * 100
)

summary = summary.sort_values(
    "Distance from KRS (km)"
)

summary.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\nStation inspection complete.")
print(f"Number of stations: {len(summary)}")
print(f"Summary saved to: {OUTPUT_FILE}")

print("\nClosest 20 stations:")
print(
    summary[
        [
            "Station Code",
            "Location Name",
            "District",
            "Taluk",
            "Latitude",
            "Longitude",
            "Distance from KRS (km)",
            "First Date",
            "Last Date",
            "Record Count",
            "Missing Rainfall (%)",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nStations by district:")
print(summary.groupby("District").size())

print("\nHistorical coverage:")
print(
    summary[
        [
            "Station Code",
            "Location Name",
            "First Date",
            "Last Date",
            "Record Count",
        ]
    ]
    .sort_values("First Date")
    .to_string(index=False)
)