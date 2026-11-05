# Data Validation Report: K.R.Sagara Dam Dataset

- **Input file:** `data\raw\krs_dam_dataset.csv`
- **Total rows:** 13,442
- **Total columns:** 19
- **Observed date range:** 1990-01-01 to 2026-10-21
- **Strict chronological order:** Yes
- **Data age from current execution:** 15 days (HISTORICAL_ARCHIVE)
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

| Column | Missing Count | Missing Percentage | Policy |
|---|---|---|---|
| `reservoir_name` | 0 | 0.00% | Complete |
| `basin` | 0 | 0.00% | Complete |
| `sub_basin` | 0 | 0.00% | Complete |
| `river` | 0 | 0.00% | Complete |
| `monitoring_date` | 0 | 0.00% | Complete |
| `percentage_full` | 10,763 | 80.07% | Preserved NaN (never zeroed) |
| `reservoir_level_ft` | 0 | 0.00% | Complete |
| `design_gross_capacity_tmc` | 0 | 0.00% | Complete |
| `gross_capacity_tmc` | 0 | 0.00% | Complete |
| `live_capacity_tmc` | 0 | 0.00% | Complete |
| `live_above_cill_tmc` | 0 | 0.00% | Complete |
| `inflow_cusecs` | 7 | 0.05% | Preserved NaN (never zeroed) |
| `outflow_cusecs` | 0 | 0.00% | Complete |
| `withdrawal_cusecs` | 0 | 0.00% | Complete |
| `evaporation_cusecs` | 546 | 4.06% | Preserved NaN (never zeroed) |
| `cumulative_inflow_tmc` | 0 | 0.00% | Complete |
| `cumulative_outflow_tmc` | 0 | 0.00% | Complete |
| `cumulative_withdrawal_tmc` | 0 | 0.00% | Complete |
| `cumulative_evaporation_tmc` | 365 | 2.72% | Preserved NaN (never zeroed) |

## 3. Physical Plausibility & Suspicious Record Review

- Physically implausible Reservoir Level `14782.0 ft` at row 970 (28-08-1992 12:08:00). Normal FRL is ~2482 ft MSL. Likely transcription error.
- Found 5 records where gross storage exceeds design capacity (49.452 TMC), peaking at 69.322 TMC. Preserved for review.

### Detailed Suspicious Records Audit

| Row Index | Date | Column | Observed Value | Review Reason | Action |
|---|---|---|---|---|---|
| 970 | `28-08-1992 12:08:00` | `reservoir_level_ft` | `14782.0` | Elevation exceeds 2500 ft (normal FRL ~2482 ft MSL) | Flagged & Preserved |
| 1739 | `06-10-1994 12:10:00` | `gross_capacity_tmc` | `49.552` | Exceeds KRS design gross capacity (49.452 TMC) | Flagged & Preserved |
| 3343 | `26-02-1999 12:02:00` | `gross_capacity_tmc` | `58.903` | Exceeds KRS design gross capacity (49.452 TMC) | Flagged & Preserved |
| 7714 | `14-02-2011 12:02:00` | `gross_capacity_tmc` | `69.322` | Exceeds KRS design gross capacity (49.452 TMC) | Flagged & Preserved |
| 7715 | `15-02-2011 12:02:00` | `gross_capacity_tmc` | `68.971` | Exceeds KRS design gross capacity (49.452 TMC) | Flagged & Preserved |
| 7716 | `16-02-2011 12:02:00` | `gross_capacity_tmc` | `68.631` | Exceeds KRS design gross capacity (49.452 TMC) | Flagged & Preserved |

## 4. Constant & Categorical Fields

- `reservoir_name`: Constant value `K.R.Sagara Dam` across all records.
- `basin`: Constant value `Cauvery` across all records.
- `sub_basin`: Constant value `Upper Cauvery` across all records.
- `river`: Constant value `Cauvery` across all records.
- `design_gross_capacity_tmc`: Constant value `49.452` across all records.

## 5. Cumulative Series Resets

- Cumulative flow columns reset annually on Month: `[6]` (June 1st, water year start).

## 6. Recommendations & Policy Enforcement

- **Inflow target:** Complete (99.95%). Highly suitable for forecasting.
- **Evaporation:** 546 missing observations preserved as NaN. Evaporation convention is unverified.
- **High Capacity Observations:** 5 records exceed design capacity (49.452 TMC). Preserved and audited.
- **Calendar Gap:** 2020-02-11 is missing. Interval arithmetic must not treat 2-day gap as 1-day step.