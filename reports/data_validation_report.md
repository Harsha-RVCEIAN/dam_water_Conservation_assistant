# Data Validation Report: K.R.Sagara Dam Dataset

- **Input file:** `data\raw\krs_dam_dataset.csv`
- **Total rows:** 13,442
- **Total columns:** 19
- **Observed date range:** 1990-01-01 to 2026-10-21
- **Strict chronological order:** Yes
- **Duplicate timestamps:** 0
- **Duplicate calendar days:** 0
- **Date parse failures:** 0
- **Missing calendar days:** 1 (2020-02-11)

## 1. Schema & Column Normalization

| Original Raw Header | Normalized Name |
|---|---|
| `Reservoir Name` | `reservoir_name` |
| `Basin` | `basin` |
| `Sub \nBasin` | `sub_basin` |
| `River` | `river` |
| `Monitoring Date` | `monitoring_date` |
| `Percentage Full` | `percentage_full` |
| `Reservoir Level (ft)` | `reservoir_level_ft` |
| `Design Gross Capacity (TMC)` | `design_gross_capacity_tmc` |
| `Gross Capacity (TMC)` | `gross_capacity_tmc` |
| `Live Capacity \n(TMC)` | `live_capacity_tmc` |
| `Live Above Cill (TMC)` | `live_above_cill_tmc` |
| `Inflow (Cusecs)` | `inflow_cusecs` |
| `Outflow to River (Cusecs)` | `outflow_cusecs` |
| `Canal Withdrawal (Cusecs)` | `withdrawal_cusecs` |
| `Evaporation (Cusecs)` | `evaporation_cusecs` |
| `Cumulative Inflow (TMC)` | `cumulative_inflow_tmc` |
| `Cumulative Outflow (TMC)` | `cumulative_outflow_tmc` |
| `Cumulative Withdrawal (TMC)` | `cumulative_withdrawal_tmc` |
| `Cumulative Evaporation \n(TMC)` | `cumulative_evaporation_tmc` |

## 2. Missing Values Analysis

| Column | Missing Count | Missing Percentage |
|---|---|---|
| `reservoir_name` | 0 | 0.00% |
| `basin` | 0 | 0.00% |
| `sub_basin` | 0 | 0.00% |
| `river` | 0 | 0.00% |
| `monitoring_date` | 0 | 0.00% |
| `percentage_full` | 10,763 | 80.07% |
| `reservoir_level_ft` | 0 | 0.00% |
| `design_gross_capacity_tmc` | 0 | 0.00% |
| `gross_capacity_tmc` | 0 | 0.00% |
| `live_capacity_tmc` | 0 | 0.00% |
| `live_above_cill_tmc` | 0 | 0.00% |
| `inflow_cusecs` | 7 | 0.05% |
| `outflow_cusecs` | 0 | 0.00% |
| `withdrawal_cusecs` | 0 | 0.00% |
| `evaporation_cusecs` | 546 | 4.06% |
| `cumulative_inflow_tmc` | 0 | 0.00% |
| `cumulative_outflow_tmc` | 0 | 0.00% |
| `cumulative_withdrawal_tmc` | 0 | 0.00% |
| `cumulative_evaporation_tmc` | 365 | 2.72% |

## 3. Physical Plausibility & Anomaly Inspection

- Physically implausible Reservoir Level `14782.0 ft` at row 970 (28-08-1992 12:08:00). Normal FRL is ~2482 ft MSL. Likely transcription error.

## 4. Constant & Categorical Fields

- `reservoir_name`: Constant value `K.R.Sagara Dam` across all records.
- `basin`: Constant value `Cauvery` across all records.
- `sub_basin`: Constant value `Upper Cauvery` across all records.
- `river`: Constant value `Cauvery` across all records.
- `design_gross_capacity_tmc`: Constant value `49.452` across all records.

## 5. Cumulative Series Resets

- Cumulative flow columns reset annually on Month: `[6]` (June 1st, corresponding to the start of the Indian water year / southwest monsoon).

## 6. Recommendations for Pipeline

- **Inflow target:** Has only 7 missing values out of 13,442 rows (99.95% complete). Highly suitable as primary forecasting target.
- **Percentage Full:** Missing 80.1% of records (only populated from mid-2019 onward). Pipeline should derive Percentage Full directly via `(gross_capacity_tmc / design_gross_capacity_tmc) * 100`.
- **Single Missing Date:** `2020-02-11` is missing. Preserve gap in raw; for lag generation, align on calendar dates.
- **Flagged Reservoir Level Anomaly:** Row at `1992-08-28` records `14782.0 ft` (transcription error for ~2468.48 ft). Flag for review and keep raw intact.