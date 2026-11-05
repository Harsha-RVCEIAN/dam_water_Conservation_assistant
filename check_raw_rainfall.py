from pathlib import Path
import pandas as pd

folder = Path("data/raw/rainfall")
station = "KC2IIMYS1009"

total_rows = 0
found_dates = set()

print("Searching raw rainfall files...")

for file in folder.glob("*.csv"):
    print(f"Checking: {file.name}")

    for chunk in pd.read_csv(
        file,
        dtype={"Station Code": str},
        chunksize=100000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Station Code" not in chunk.columns:
            continue

        if "Monitoring Date" not in chunk.columns:
            continue

        rows = chunk[
            chunk["Station Code"].astype(str).str.strip() == station
        ]

        if rows.empty:
            continue

        dates = pd.to_datetime(
            rows["Monitoring Date"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        mask = dates.between("1990-01-01", "1990-01-31")
        january_rows = rows.loc[mask]

        if not january_rows.empty:
            january_dates = pd.to_datetime(
                january_rows["Monitoring Date"],
                format="%d-%m-%Y",
                errors="coerce"
            )

            total_rows += len(january_rows)
            found_dates.update(
                january_dates.dropna().dt.strftime("%Y-%m-%d")
            )

print("\n========== RESULTS ==========")
print("Station:", station)
print("Raw records in January 1990:", total_rows)
print("Number of distinct dates:", len(found_dates))
print("Dates found:")

for date in sorted(found_dates):
    print(date)