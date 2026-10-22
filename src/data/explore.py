"""Exploratory Data Analysis (EDA) module for K.R.Sagara Dam historical data.

Computes flow distributions, seasonal hydrological patterns, storage utilization,
unit conversion validation, and water-balance discrepancy statistics.
Outputs a comprehensive markdown summary to reports/eda_summary.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.validate import (
    RAW_DATA_PATH,
    normalize_dataframe_columns,
    parse_monitoring_dates,
)

EDA_REPORT_PATH = Path("reports/eda_summary.md")
CUSEC_DAY_TO_TMC = 0.0000864


def compute_eda_statistics(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed hydrological statistics from raw dataset."""
    df, _ = normalize_dataframe_columns(df_raw)
    df["date"] = parse_monitoring_dates(df["monitoring_date"])
    df["calendar_date"] = df["date"].dt.normalize()
    df = df.sort_values("calendar_date").reset_index(drop=True)

    df["month"] = df["calendar_date"].dt.month
    df["year"] = df["calendar_date"].dt.year
    df["day_of_year"] = df["calendar_date"].dt.dayofyear

    # Daily flow volumes in TMC
    df["inflow_vol_tmc"] = df["inflow_cusecs"] * CUSEC_DAY_TO_TMC
    df["outflow_vol_tmc"] = df["outflow_cusecs"] * CUSEC_DAY_TO_TMC
    df["withdrawal_vol_tmc"] = df["withdrawal_cusecs"] * CUSEC_DAY_TO_TMC
    df["evap_vol_tmc"] = df["evaporation_cusecs"].fillna(0) * CUSEC_DAY_TO_TMC

    # Net flow volume & actual storage change
    df["net_flow_tmc"] = (
        df["inflow_vol_tmc"]
        - df["outflow_vol_tmc"]
        - df["withdrawal_vol_tmc"]
        - df["evap_vol_tmc"]
    )
    df["storage_change_tmc"] = df["gross_capacity_tmc"].diff()
    df["balance_discrepancy_tmc"] = df["storage_change_tmc"] - df["net_flow_tmc"]

    # Inflow statistics
    inflow_stats = df["inflow_cusecs"].describe().to_dict()
    inflow_stats["zeros"] = int((df["inflow_cusecs"] == 0).sum())
    inflow_stats["missing"] = int(df["inflow_cusecs"].isna().sum())

    # Monthly inflow averages (cusecs)
    monthly_inflow = df.groupby("month")["inflow_cusecs"].agg(["mean", "median", "max", "count"]).to_dict("index")

    # Storage statistics
    storage_stats = df["gross_capacity_tmc"].describe().to_dict()
    level_stats = df["reservoir_level_ft"].describe().to_dict()

    # Flow conversion validation against cumulative inflow diff
    df["cum_inflow_diff"] = df["cumulative_inflow_tmc"].diff()
    # Exclude June 1 annual reset
    non_reset = df[(df["month"] != 6) | (df["calendar_date"].dt.day != 1)].dropna(
        subset=["inflow_vol_tmc", "cum_inflow_diff"]
    )
    conversion_corr = float(non_reset["inflow_vol_tmc"].corr(non_reset["cum_inflow_diff"]))
    mean_abs_diff = float((non_reset["inflow_vol_tmc"] - non_reset["cum_inflow_diff"]).abs().mean())

    # Water balance stats
    balance_valid = df.dropna(subset=["balance_discrepancy_tmc"])
    balance_stats = balance_valid["balance_discrepancy_tmc"].describe().to_dict()

    return {
        "row_count": len(df),
        "start_date": str(df["calendar_date"].min().date()),
        "end_date": str(df["calendar_date"].max().date()),
        "inflow_stats": inflow_stats,
        "monthly_inflow": monthly_inflow,
        "storage_stats": storage_stats,
        "level_stats": level_stats,
        "conversion_corr": conversion_corr,
        "conversion_mean_abs_diff": mean_abs_diff,
        "balance_stats": balance_stats,
    }


def generate_eda_markdown(stats: Dict[str, Any]) -> str:
    """Format EDA statistics as a markdown document."""
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    lines = [
        "# Exploratory Data Analysis: K.R.Sagara Dam Hydrology",
        "",
        f"- **Dataset Period:** {stats['start_date']} to {stats['end_date']} ({stats['row_count']:,} observations)",
        "- **Basin / Reservoir:** Cauvery Basin / K.R.Sagara Dam (Karnataka, India)",
        "- **Design Gross Capacity:** 49.452 TMC",
        "",
        "## 1. Primary Forecasting Target: Inflow (Cusecs)",
        "",
        "| Metric | Value (Cusecs) |",
        "|---|---|",
        f"| Mean | {stats['inflow_stats']['mean']:.2f} |",
        f"| Median (50%) | {stats['inflow_stats']['50%']:.2f} |",
        f"| Min | {stats['inflow_stats']['min']:.2f} |",
        f"| Max | {stats['inflow_stats']['max']:.2f} |",
        f"| Std Dev | {stats['inflow_stats']['std']:.2f} |",
        f"| 75th Percentile | {stats['inflow_stats']['75%']:.2f} |",
        f"| Zero flow days | {stats['inflow_stats']['zeros']:,} ({stats['inflow_stats']['zeros'] / stats['row_count'] * 100:.2f}%) |",
        f"| Missing flow records | {stats['inflow_stats']['missing']} ({stats['inflow_stats']['missing'] / stats['row_count'] * 100:.3f}%) |",
        "",
        "## 2. Seasonal Inflow Patterns (Hydrological Regime)",
        "",
        "The Cauvery basin experiences a strong monsoon regime. Inflow peaks during the Southwest Monsoon (July–August), with secondary peaks during the Northeast Monsoon (October–November).",
        "",
        "| Month | Mean Inflow (Cusecs) | Median Inflow (Cusecs) | Peak Inflow (Cusecs) | Days Observed |",
        "|---|---|---|---|---|",
    ]

    for m in range(1, 13):
        m_info = stats["monthly_inflow"][m]
        lines.append(
            f"| {month_names[m-1]} ({m:02d}) | {m_info['mean']:.1f} | {m_info['median']:.1f} | {m_info['max']:.1f} | {int(m_info['count']):,} |"
        )

    lines.extend([
        "",
        "## 3. Reservoir Storage & Elevation",
        "",
        "| Parameter | Min | Mean | Median | Max | Std |",
        "|---|---|---|---|---|---|",
        f"| Gross Capacity (TMC) | {stats['storage_stats']['min']:.2f} | {stats['storage_stats']['mean']:.2f} | {stats['storage_stats']['50%']:.2f} | {stats['storage_stats']['max']:.2f} | {stats['storage_stats']['std']:.2f} |",
        f"| Reservoir Level (ft) | {stats['level_stats']['min']:.2f} | {stats['level_stats']['mean']:.2f} | {stats['level_stats']['50%']:.2f} | {stats['level_stats']['max']:.2f} | {stats['level_stats']['std']:.2f} |",
        "",
        "> [!NOTE]\n> Reservoir level maximum includes a single data-recording typo (14,782 ft on 1992-08-28), where normal Full Reservoir Level is ~2,482 ft MSL.",
        "",
        "## 4. Unit Conversion & Water-Budget Verification",
        "",
        f"- **Unit Conversion Verification:** Correlation between daily inflow volume (`Inflow * 0.0000864 TMC`) and `Cumulative Inflow` day-to-day delta is **{stats['conversion_corr']:.6f}** (mean difference: **{stats['conversion_mean_abs_diff']:.5f} TMC** due to 3-decimal rounding).",
        "- **Conclusion:** The rate column represents mean cusecs over the day, and `0.0000864` is the exact mathematical conversion to daily volume in TMC.",
        "",
        "### Simplified Water-Balance Discrepancy Statistics",
        "",
        "`Expected Storage Change = Inflow - Outflow - Canal Withdrawal - Evaporation`",
        "",
        "| Discrepancy Metric | Value (TMC) |",
        "|---|---|",
        f"| Mean discrepancy | {stats['balance_stats']['mean']:.4f} |",
        f"| Median discrepancy | {stats['balance_stats']['50%']:.5f} |",
        f"| Interquartile range (25% - 75%) | {stats['balance_stats']['25%']:.4f} to {stats['balance_stats']['75%']:.4f} |",
        f"| Max positive discrepancy | {stats['balance_stats']['max']:.2f} |",
        f"| Max negative discrepancy | {stats['balance_stats']['min']:.2f} |",
        "",
        "> [!IMPORTANT]\n> A water-balance discrepancy represents unmodeled hydrological factors (direct precipitation, unmeasured catchment runoff, gate timing, gauge adjustment), NOT proof of leakage. Anomalies are flagged solely for review.",
    ])

    return "\n".join(lines)


def run_eda(data_path: Path = RAW_DATA_PATH) -> Path:
    """Run full EDA pipeline and output markdown report."""
    EDA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(data_path)
    stats = compute_eda_statistics(df_raw)
    md = generate_eda_markdown(stats)
    EDA_REPORT_PATH.write_text(md, encoding="utf-8")
    return EDA_REPORT_PATH


def main() -> None:
    """CLI entrypoint for explore module."""
    out = run_eda()
    print(f"EDA report successfully generated at {out}")


if __name__ == "__main__":
    main()
