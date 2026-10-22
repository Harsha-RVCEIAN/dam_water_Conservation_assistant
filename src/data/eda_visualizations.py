"""EDA Visualizations Module for K.R.Sagara Dam Hydrological Dataset.

Generates 5 high-value diagnostic and exploratory figures:
1. Seasonal Hydrology Regime & Inflow Climatology (Southwest & Northeast Monsoons)
2. Inflow vs Discharge Operational Mass Balance (Buffering & Drawdown Cycles)
3. Elevation-Storage Bathymetric Rating Curve & Sensor Anomaly Detection
4. Autoregressive Temporal Memory & Autocorrelation Structure (ACF / PACF)
5. Physical Water-Balance Discrepancy & Sanity Distribution (TMC conservation)

Saves publication-quality figures to reports/figures/ and brain artifact directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

REPORTS_FIG_DIR = Path("reports/figures")
BRAIN_FIG_DIR = Path(r"C:\Users\harsh\.gemini\antigravity-ide\brain\bb1ce316-45e9-47cf-9b75-e772b241fecb\figures")
PROCESSED_DATA_PATH = Path("data/processed/krs_dam_processed.csv")
RAW_DATA_PATH = Path("data/raw/krs_dam_dataset.csv")

KRS_DESIGN_CAPACITY_TMC = 49.452
KRS_FRL_FT = 2482.0
KRS_DEAD_STORAGE_FT = 2406.8


def ensure_dirs():
    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    BRAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)


def copy_to_brain(filename: str):
    src = REPORTS_FIG_DIR / filename
    dst = BRAIN_FIG_DIR / filename
    if src.exists():
        shutil.copy2(src, dst)


def plot_01_seasonal_regime(df: pd.DataFrame) -> Path:
    """Graph 1: Seasonal Hydrology Regime & Inflow Dynamics."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Calculate monthly summary statistics
    monthly_stats = df.groupby("month")["inflow_cusecs"].agg(
        p10=lambda s: s.quantile(0.10),
        p25=lambda s: s.quantile(0.25),
        median="median",
        mean="mean",
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99),
        max_val="max",
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    x = np.arange(1, 13)

    # Highlight Monsoon Seasons
    ax.axvspan(5.5, 9.5, color="#3498db", alpha=0.12, label="Southwest Monsoon (Jun-Sep)")
    ax.axvspan(9.5, 11.5, color="#2ecc71", alpha=0.10, label="Northeast Monsoon (Oct-Nov)")
    ax.axvspan(0.5, 5.5, color="#f39c12", alpha=0.07, label="Dry / Lean Season (Dec-May)")
    ax.axvspan(11.5, 12.5, color="#f39c12", alpha=0.07)

    # Shaded percentiles (P10-P90 & P25-P75)
    ax.fill_between(x, monthly_stats["p10"], monthly_stats["p90"], color="#1f77b4", alpha=0.18, label="10th - 90th Percentile Range")
    ax.fill_between(x, monthly_stats["p25"], monthly_stats["p75"], color="#1f77b4", alpha=0.35, label="Interquartile Range (IQR 25-75%)")

    # Lines for Median, Mean, Max
    ax.plot(x, monthly_stats["median"], color="#0b3c5d", marker="o", linewidth=2.5, label="Median Inflow (50%)")
    ax.plot(x, monthly_stats["mean"], color="#d9534f", linestyle="--", marker="s", linewidth=2.0, label="Mean Inflow (Skewed by Floods)")
    ax.scatter(x, monthly_stats["max_val"], color="#8e44ad", s=45, zorder=5, label="Historical Record Peak")

    # Annotate key peaks and lean periods
    aug_mean = monthly_stats.loc[monthly_stats["month"] == 8, "mean"].values[0]
    aug_max = monthly_stats.loc[monthly_stats["month"] == 8, "max_val"].values[0]
    mar_med = monthly_stats.loc[monthly_stats["month"] == 3, "median"].values[0]

    ax.annotate(
        f"Aug Max Peak: {aug_max:,.0f} cfs\n(Aug Mean: {aug_mean:,.0f} cfs)",
        xy=(8, aug_max),
        xytext=(8.3, aug_max * 0.85),
        arrowprops=dict(facecolor="#8e44ad", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#8e44ad",
    )

    ax.annotate(
        f"Lean Baseline: Mar Median {mar_med:.0f} cfs",
        xy=(3, mar_med),
        xytext=(1.8, 18000),
        arrowprops=dict(facecolor="#0b3c5d", shrink=0.05, width=1.2, headwidth=5),
        fontsize=9,
        fontweight="bold",
        color="#0b3c5d",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontweight="semibold")
    ax.set_xlim(0.5, 12.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.set_ylim(50, 300000)

    ax.set_title("Graph 1: Monthly Inflow Climatology & Monsoon Hydrological Regime (1990–2026)\nLog-scale Inflow Distribution showing Southwest Monsoon Dominance", pad=12)
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Inflow Rate (Cusecs, Log Scale)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = REPORTS_FIG_DIR / "01_seasonal_hydrology_regime.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    copy_to_brain("01_seasonal_hydrology_regime.png")
    return out_path


def plot_02_inflow_outflow_balance(df: pd.DataFrame) -> Path:
    """Graph 2: Inflow vs Outflows Operational Balance."""
    # Monthly averages across all years
    monthly_flows = df.groupby("month").agg(
        inflow=("inflow_cusecs", "mean"),
        outflow=("outflow_cusecs", "mean"),
        withdrawal=("withdrawal_cusecs", "mean"),
        gross_storage=("gross_capacity_tmc", "mean"),
    ).reset_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    ax2 = ax1.twinx()

    # Flow components
    width = 0.35
    total_discharge = monthly_flows["outflow"] + monthly_flows["withdrawal"]

    # Shaded background showing replenishment vs drawdown
    ax1.axvspan(5.5, 9.5, color="#e8f8f5", alpha=0.6, label="Filling / Replenishment Window")
    ax1.axvspan(9.5, 12.5, color="#fef9e7", alpha=0.6, label="Drawdown & Conservation Window")
    ax1.axvspan(0.5, 5.5, color="#fef9e7", alpha=0.6)

    # Bars for outflow components
    p1 = ax1.bar(x - width/2, monthly_flows["outflow"], width, label="River Release Outflow (cusecs)", color="#3498db", alpha=0.85)
    p2 = ax1.bar(x - width/2, monthly_flows["withdrawal"], width, bottom=monthly_flows["outflow"], label="Canal Irrigation/City Withdrawal (cusecs)", color="#1abc9c", alpha=0.85)

    # Bar for Inflow
    p3 = ax1.bar(x + width/2, monthly_flows["inflow"], width, label="Mean Inflow (cusecs)", color="#2c3e50", alpha=0.90)

    # Storage Line on Secondary Axis
    p4 = ax2.plot(x, monthly_flows["gross_storage"], color="#e74c3c", linewidth=2.8, marker="o", label="Mean Reservoir Storage (TMC)")
    ax2.axhline(KRS_DESIGN_CAPACITY_TMC, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8, label="Design Capacity (49.45 TMC)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(month_names, fontweight="semibold")
    ax1.set_xlim(0.5, 12.5)
    ax1.set_xlabel("Month of Year")
    ax1.set_ylabel("Mean Flow Rate (Cusecs)")
    ax2.set_ylabel("Reservoir Storage (TMC)", color="#c0392b", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 65)
    ax1.set_ylim(0, 18000)

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    # Combined legend placed cleanly at upper left
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=2)

    ax1.set_title("Graph 2: Operational Flow Balance & Reservoir Storage Dynamics\nInflow Buffering in Monsoon vs. Continuous Downstream Supply in Dry Months", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = REPORTS_FIG_DIR / "02_inflow_outflow_balance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    copy_to_brain("02_inflow_outflow_balance.png")
    return out_path


def plot_03_reservoir_elevation_storage_curve(df: pd.DataFrame) -> Path:
    """Graph 3: Elevation-Storage Bathymetric Rating Curve & Anomaly Detection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left plot: Raw vs Cleaned showing data entry anomaly
    df_raw = pd.read_csv(RAW_DATA_PATH)
    raw_level_col = "Reservoir Level (ft)"
    raw_storage_col = "Gross Capacity (TMC)"
    
    ax1.scatter(df_raw[raw_level_col], df_raw[raw_storage_col], s=6, color="#e74c3c", alpha=0.5, label="Raw Daily Observations")
    ax1.set_title("Raw Sensor Readings: Outlier / Typo Detection", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Reservoir Level (ft MSL)")
    ax1.set_ylabel("Gross Capacity (TMC)")
    ax1.axvline(2482, color="#2980b9", linestyle="--", label="Normal Full Reservoir Level (~2,482 ft)")
    ax1.annotate(
        "Clerical Typo: 14,782 ft\n(1992-08-28)",
        xy=(14782, 49.4),
        xytext=(9500, 30),
        arrowprops=dict(facecolor="#c0392b", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#c0392b",
    )
    ax1.set_ylim(0, 75)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right plot: Cleaned physical bathymetric rating curve
    valid_mask = (df["reservoir_level_ft_clean"] >= 2400) & (df["reservoir_level_ft_clean"] <= 2505)
    sub = df[valid_mask].sample(min(4000, len(df)), random_state=42)

    sc = ax2.scatter(
        sub["reservoir_level_ft_clean"],
        sub["gross_capacity_tmc"],
        c=sub["year"],
        cmap="viridis",
        s=8,
        alpha=0.6,
        label="Daily Observations (Cleaned)",
    )
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Observation Year (1990 - 2026)", rotation=270, labelpad=15)

    # Physical reference lines
    ax2.axvline(KRS_FRL_FT, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Full Reservoir Level ({KRS_FRL_FT} ft)")
    ax2.axvline(KRS_DEAD_STORAGE_FT, color="#7f8c8d", linestyle=":", linewidth=1.5, label=f"Dead Sill Level ({KRS_DEAD_STORAGE_FT} ft)")
    ax2.axhline(KRS_DESIGN_CAPACITY_TMC, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Design Capacity ({KRS_DESIGN_CAPACITY_TMC} TMC)")

    ax2.set_title("Clean Physical Bathymetry Curve (Elevation vs Storage)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Reservoir Elevation (ft MSL, Cleaned)")
    ax2.set_ylabel("Gross Storage (TMC)")
    ax2.set_xlim(2400, 2490)
    ax2.set_ylim(0, 55)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Graph 3: Reservoir Bathymetry & Level-Storage Elevation Rating Curve\nValidates Non-linear Physical Capacity & Catches Data Logging Typos", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = REPORTS_FIG_DIR / "03_reservoir_elevation_storage_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    copy_to_brain("03_reservoir_elevation_storage_curve.png")
    return out_path


def plot_04_inflow_autocorrelation_memory(df: pd.DataFrame) -> Path:
    """Graph 4: Autocorrelation (ACF) & Lagged Memory Decay of Inflow."""
    s = df["inflow_cusecs"].ffill()
    max_lag = 40
    lags = np.arange(1, max_lag + 1)
    acf_vals = [float(s.autocorr(lag=l)) for l in lags]

    # Approximate 95% confidence interval for white noise: +/- 1.96 / sqrt(N)
    ci_bound = 1.96 / np.sqrt(len(s))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # 95% confidence interval
    ax.axhspan(-ci_bound, ci_bound, color="#bdc3c7", alpha=0.5, label="95% White Noise Confidence Interval")
    ax.axhline(0, color="#7f8c8d", linewidth=1.0)

    # Stem plot for ACF
    markerline, stemlines, baseline = ax.stem(lags, acf_vals, basefmt=" ", linefmt="#2980b9", markerfmt="o")
    plt.setp(markerline, markersize=5, color="#1b4f72")
    plt.setp(stemlines, linewidth=1.5, color="#2980b9")

    # Annotate key lag features used in ML models
    key_lags = [1, 2, 7, 14, 30]
    for kl in key_lags:
        val = acf_vals[kl - 1]
        ax.scatter([kl], [val], color="#e74c3c", s=40, zorder=5)
        offset_y = 0.05 if val > 0.4 else 0.07
        ax.annotate(
            f"Lag {kl}: {val:.3f}",
            xy=(kl, val),
            xytext=(kl + 0.4, val + offset_y),
            arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.0, headwidth=4),
            fontsize=8.5,
            fontweight="bold",
            color="#922b21",
        )

    ax.set_title("Graph 4: Autocorrelation Function (ACF) of Daily Inflow (Lags 1 to 40)\nEmpirical Justification for Autoregressive Feature Engineering in ML Models", pad=12)
    ax.set_xlabel("Time Lag (Days)")
    ax.set_ylabel("Autocorrelation Coefficient (r)")
    ax.set_xlim(0, max_lag + 1)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks([1, 2, 3, 5, 7, 10, 14, 21, 28, 30, 35, 40])
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = REPORTS_FIG_DIR / "04_inflow_autocorrelation_memory.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    copy_to_brain("04_inflow_autocorrelation_memory.png")
    return out_path


def plot_05_water_balance_discrepancy(df: pd.DataFrame) -> Path:
    """Graph 5: Water-Balance Discrepancy & Operational Sanity Distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    disc = df["balance_discrepancy_tmc"].dropna()

    # Left: Histogram of core distribution (-0.05 to +0.05 TMC)
    core = disc[(disc >= -0.02) & (disc <= 0.02)]
    ax1.hist(core, bins=60, color="#16a085", edgecolor="white", alpha=0.85, density=True)
    ax1.axvline(disc.median(), color="#c0392b", linestyle="--", linewidth=1.5, label=f"Median: {disc.median():.6f} TMC")
    ax1.axvline(disc.quantile(0.25), color="#2980b9", linestyle=":", linewidth=1.2, label=f"IQR (25-75%): {disc.quantile(0.25):.4f} to {disc.quantile(0.75):.4f}")
    ax1.axvline(disc.quantile(0.75), color="#2980b9", linestyle=":", linewidth=1.2)

    ax1.set_title("Daily Balance Discrepancy (Core Normal Distribution)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Discrepancy (TMC) = Storage Change - Net Flow")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Full timeline of discrepancy anomalies across 1990-2026
    df["cal_date"] = pd.to_datetime(df["calendar_date"])
    normal_mask = df["balance_discrepancy_tmc"].abs() <= 2.0
    anom_mask = df["balance_discrepancy_tmc"].abs() > 2.0

    ax2.scatter(df.loc[normal_mask, "cal_date"], df.loc[normal_mask, "balance_discrepancy_tmc"], color="#bdc3c7", s=3, alpha=0.4, label="Normal Operation (|Δ| ≤ 2 TMC)")
    ax2.scatter(df.loc[anom_mask, "cal_date"], df.loc[anom_mask, "balance_discrepancy_tmc"], color="#e74c3c", s=25, alpha=0.9, label=f"Flagged Anomalies ({anom_mask.sum()} days)")

    ax2.axhline(2.0, color="#e67e22", linestyle="--", linewidth=1.2, label="Anomaly Threshold (±2 TMC)")
    ax2.axhline(-2.0, color="#e67e22", linestyle="--", linewidth=1.2)
    ax2.axhline(0.0, color="#2c3e50", linestyle="-", linewidth=0.8)

    ax2.set_title("Timeline of Operational Balance Discrepancies (1990–2026)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Observation Date")
    ax2.set_ylabel("Discrepancy (TMC)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Graph 5: Daily Water-Balance Verification & Operational Anomaly Detection\nVerifies Physical Unit Consistency (1 cusec-day = 0.0000864 TMC) and Flags Sensor Outliers", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = REPORTS_FIG_DIR / "05_water_balance_discrepancy.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    copy_to_brain("05_water_balance_discrepancy.png")
    return out_path


def generate_all_eda_plots() -> list[Path]:
    """Generate all 5 high-value EDA figures and save them."""
    ensure_dirs()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    paths = [
        plot_01_seasonal_regime(df),
        plot_02_inflow_outflow_balance(df),
        plot_03_reservoir_elevation_storage_curve(df),
        plot_04_inflow_autocorrelation_memory(df),
        plot_05_water_balance_discrepancy(df),
    ]
    return paths


if __name__ == "__main__":
    generated = generate_all_eda_plots()
    for p in generated:
        print(f"Generated: {p}")
