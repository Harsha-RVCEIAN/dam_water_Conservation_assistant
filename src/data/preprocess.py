"""Reproducible data preprocessing pipeline for K.R.Sagara Dam dataset.

Loads raw observations, normalizes schemas, parses chronological dates,
derives physical hydrological volumes and balances, flags known anomalies,
constructs strictly non-leaking feature lags, and outputs a validated
processed dataset to data/processed/krs_dam_processed.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data.validate import (
    RAW_DATA_PATH,
    normalize_dataframe_columns,
    parse_monitoring_dates,
)

PROCESSED_DATA_PATH = Path("data/processed/krs_dam_processed.csv")
CUSEC_DAY_TO_TMC = 0.0000864  # Exact daily volume conversion: 86400 cu ft / 10^9


def load_and_normalize(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and normalize headers and date types."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path.resolve()}")

    df_raw = pd.read_csv(raw_path)
    df, _ = normalize_dataframe_columns(df_raw)

    # Parse timestamps and standardize chronological ordering
    parsed_dates = parse_monitoring_dates(df["monitoring_date"])
    df["date"] = parsed_dates
    df["calendar_date"] = parsed_dates.dt.normalize()

    # Sort strictly chronologically
    df = df.sort_values("calendar_date").reset_index(drop=True)
    return df


def add_hydrological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived volumes, consistent percentage full, and water balance terms."""
    df = df.copy()

    # Consistent percentage full derived from actual Gross Capacity and constant Design Gross Capacity
    # Matches the formula verified on mid-2019+ data: (Gross / Design Gross) * 100
    df["percentage_full_calculated"] = (
        df["gross_capacity_tmc"] / df["design_gross_capacity_tmc"]
    ) * 100.0

    # Retain original percentage_full, but provide a verified continuous column
    df["percentage_full_continuous"] = df["percentage_full"].fillna(
        df["percentage_full_calculated"]
    )

    # Daily flow volumes in TMC
    df["inflow_vol_tmc"] = df["inflow_cusecs"] * CUSEC_DAY_TO_TMC
    df["outflow_vol_tmc"] = df["outflow_cusecs"] * CUSEC_DAY_TO_TMC
    df["withdrawal_vol_tmc"] = df["withdrawal_cusecs"] * CUSEC_DAY_TO_TMC
    df["evaporation_vol_tmc"] = df["evaporation_cusecs"] * CUSEC_DAY_TO_TMC

    # Net flow volume: Inflow - Outflow - Withdrawal - Evaporation (evaporation nulls treated as 0 for balance)
    evap_vol_filled = df["evaporation_vol_tmc"].fillna(0.0)
    df["net_flow_tmc"] = (
        df["inflow_vol_tmc"]
        - df["outflow_vol_tmc"]
        - df["withdrawal_vol_tmc"]
        - evap_vol_filled
    )

    # Actual daily storage change observed in Gross Capacity
    df["storage_change_tmc"] = df["gross_capacity_tmc"].diff()

    # Discrepancy between actual change and net flow balance
    df["balance_discrepancy_tmc"] = df["storage_change_tmc"] - df["net_flow_tmc"]

    # Anomaly flags
    # 1. Level anomaly: FRL is ~2482 ft MSL. Any level > 2500 ft is flagged.
    df["is_level_anomalous"] = df["reservoir_level_ft"] > 2500.0

    # Cleaned reservoir level (interpolating only the single known transcription typo, leaving others untouched)
    df["reservoir_level_ft_clean"] = df["reservoir_level_ft"].copy()
    anom_mask = df["is_level_anomalous"]
    if anom_mask.any():
        # Replace anomalous levels with linear interpolation between adjacent days
        df.loc[anom_mask, "reservoir_level_ft_clean"] = np.nan
        df["reservoir_level_ft_clean"] = df["reservoir_level_ft_clean"].interpolate(method="linear")

    # 2. Water balance anomaly flag (|discrepancy| > 5.0 TMC)
    # Serves as an operational review flag, NOT proof of leakage
    df["is_balance_anomalous"] = df["balance_discrepancy_tmc"].abs() > 5.0

    # Calendar features
    df["year"] = df["calendar_date"].dt.year
    df["month"] = df["calendar_date"].dt.month
    df["day_of_month"] = df["calendar_date"].dt.day
    df["day_of_year"] = df["calendar_date"].dt.dayofyear
    df["day_of_week"] = df["calendar_date"].dt.dayofweek
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    return df


def add_forecasting_lags(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (1, 2, 3, 7, 14),
    rolling_windows: Tuple[int, ...] = (7, 14, 30),
) -> pd.DataFrame:
    """Generate strictly backward-looking lag and rolling features to prevent leakage."""
    df = df.copy()

    # The dataset has exactly one missing day (2020-02-11). To ensure lag_1 is truly
    # the previous calendar day, we reindex against a full daily calendar range, compute lags,
    # and retain the exact observations.
    date_indexed = df.set_index("calendar_date")
    full_idx = pd.date_range(df["calendar_date"].min(), df["calendar_date"].max(), freq="D")
    reindexed = date_indexed.reindex(full_idx)

    # Inflow lags (pure historical values)
    for lag in lags:
        reindexed[f"inflow_lag_{lag}"] = reindexed["inflow_cusecs"].shift(lag)
        reindexed[f"storage_lag_{lag}"] = reindexed["gross_capacity_tmc"].shift(lag)
        reindexed[f"outflow_lag_{lag}"] = reindexed["outflow_cusecs"].shift(lag)

    # Rolling statistics based on past observations (shifted by 1 to exclude current day)
    shifted_inflow = reindexed["inflow_cusecs"].shift(1)
    for w in rolling_windows:
        reindexed[f"inflow_roll_mean_{w}"] = shifted_inflow.rolling(window=w, min_periods=max(1, w // 2)).mean()
        reindexed[f"inflow_roll_std_{w}"] = shifted_inflow.rolling(window=w, min_periods=max(1, w // 2)).std()
        reindexed[f"inflow_roll_max_{w}"] = shifted_inflow.rolling(window=w, min_periods=max(1, w // 2)).max()

    # Next-day forecasting target: inflow at t+1
    reindexed["target_inflow_next_day_cusecs"] = reindexed["inflow_cusecs"].shift(-1)

    # Restore original index rows only (preserving raw row integrity without fabricating observations)
    processed = reindexed.loc[df["calendar_date"]].reset_index()
    processed = processed.rename(columns={"index": "calendar_date"})

    return processed


def run_preprocessing_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Execute complete reproducible preprocessing pipeline and write output."""
    print(f"Starting preprocessing pipeline from {raw_path}...")
    df = load_and_normalize(raw_path)
    initial_rows = len(df)

    df = add_hydrological_features(df)
    df = add_forecasting_lags(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Preprocessing completed successfully:")
    print(f"  Input rows: {initial_rows:,}")
    print(f"  Output rows: {len(df):,}")
    print(f"  Output columns: {len(df.columns)}")
    print(f"  Saved to: {output_path}")
    return df


def main() -> None:
    """CLI entrypoint for preprocessing script."""
    run_preprocessing_pipeline()


if __name__ == "__main__":
    main()
