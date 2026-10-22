# Exploratory Data Analysis: K.R.Sagara Dam Hydrology

- **Dataset Period:** 1990-01-01 to 2026-10-21 (13,442 observations)
- **Basin / Reservoir:** Cauvery Basin / K.R.Sagara Dam (Karnataka, India)
- **Design Gross Capacity:** 49.452 TMC

## 1. Primary Forecasting Target: Inflow (Cusecs)

| Metric | Value (Cusecs) |
|---|---|
| Mean | 4822.14 |
| Median (50%) | 1555.00 |
| Min | 0.00 |
| Max | 204200.00 |
| Std Dev | 9988.10 |
| 75th Percentile | 5019.00 |
| Zero flow days | 389 (2.89%) |
| Missing flow records | 7 (0.052%) |

## 2. Seasonal Inflow Patterns (Hydrological Regime)

The Cauvery basin experiences a strong monsoon regime. Inflow peaks during the Southwest Monsoon (July–August), with secondary peaks during the Northeast Monsoon (October–November).

| Month | Mean Inflow (Cusecs) | Median Inflow (Cusecs) | Peak Inflow (Cusecs) | Days Observed |
|---|---|---|---|---|
| Jan (01) | 675.0 | 450.0 | 5717.0 | 1,145 |
| Feb (02) | 404.8 | 227.5 | 11157.0 | 1,042 |
| Mar (03) | 326.5 | 241.0 | 4392.0 | 1,146 |
| Apr (04) | 664.3 | 315.5 | 8257.0 | 1,110 |
| May (05) | 1092.1 | 650.0 | 26424.0 | 1,147 |
| Jun (06) | 3937.4 | 1664.5 | 73811.0 | 1,110 |
| Jul (07) | 14111.8 | 7522.0 | 168762.0 | 1,147 |
| Aug (08) | 15565.6 | 8840.0 | 204200.0 | 1,147 |
| Sep (09) | 7817.3 | 5413.0 | 55774.0 | 1,109 |
| Oct (10) | 6635.8 | 5189.5 | 65371.0 | 1,136 |
| Nov (11) | 3857.3 | 2694.5 | 37166.0 | 1,080 |
| Dec (12) | 2188.7 | 1815.0 | 10925.0 | 1,116 |

## 3. Reservoir Storage & Elevation

| Parameter | Min | Mean | Median | Max | Std |
|---|---|---|---|---|---|
| Gross Capacity (TMC) | 0.10 | 29.54 | 29.41 | 69.32 | 14.37 |
| Reservoir Level (ft) | 2406.80 | 2449.22 | 2451.64 | 14782.00 | 107.78 |

> [!NOTE]
> Reservoir level maximum includes a single data-recording typo (14,782 ft on 1992-08-28), where normal Full Reservoir Level is ~2,482 ft MSL.

## 4. Unit Conversion & Water-Budget Verification

- **Unit Conversion Verification:** Correlation between daily inflow volume (`Inflow * 0.0000864 TMC`) and `Cumulative Inflow` day-to-day delta is **1.000000** (mean difference: **0.00032 TMC** due to 3-decimal rounding).
- **Conclusion:** The rate column represents mean cusecs over the day, and `0.0000864` is the exact mathematical conversion to daily volume in TMC.

### Simplified Water-Balance Discrepancy Statistics

`Expected Storage Change = Inflow - Outflow - Canal Withdrawal - Evaporation`

| Discrepancy Metric | Value (TMC) |
|---|---|
| Mean discrepancy | -0.0077 |
| Median discrepancy | -0.00003 |
| Interquartile range (25% - 75%) | -0.0010 to 0.0008 |
| Max positive discrepancy | 33.00 |
| Max negative discrepancy | -33.02 |

> [!IMPORTANT]
> A water-balance discrepancy represents unmodeled hydrological factors (direct precipitation, unmeasured catchment runoff, gate timing, gauge adjustment), NOT proof of leakage. Anomalies are flagged solely for review.