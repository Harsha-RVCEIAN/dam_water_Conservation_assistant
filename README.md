# Dam Water Conservation Assistant --- MLOps Project

An explainable, reproducible student MLOps project analyzing 36 years of historical hydrological observations from **K.R.Sagara Dam** (Cauvery Basin, Karnataka, India). The project provides daily inflow forecasting, data-quality anomaly auditing, and hypothetical water-conservation decision simulation.

---

## 1. Quickstart & Exact Windows Commands

All commands run from the project root using the project virtual environment (`.venv`):

```powershell
# 1. Verify Python environment & dependencies
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check

# 2. Run data validation
.\.venv\Scripts\python.exe -m src.data.validate

# 3. Run exploratory hydrological analysis (EDA)
.\.venv\Scripts\python.exe -m src.data.explore

# 4. Run reproducible preprocessing pipeline
.\.venv\Scripts\python.exe -m src.data.preprocess

# 5. Train baseline & Random Forest model, log to MLflow
.\.venv\Scripts\python.exe -m src.models.train

# 6. Run automated test suite
.\.venv\Scripts\python.exe -m pytest -q

# 7. Launch interactive Streamlit dashboard
.\.venv\Scripts\python.exe -m streamlit run src/app.py

# 8. View MLflow experiment runs (optional)
.\.venv\Scripts\python.exe -m mlflow ui
```

---

## 2. Project Architecture & Directory Layout

```text
dam_mlops/
├── data/
│   ├── raw/
│   │   ├── krs_dam_dataset.csv          # 13,442 raw observations (1990-2026); never modified
│   │   └── krs_dam_dataset.csv.dvc      # DVC data tracking pointer
│   └── processed/
│       └── krs_dam_processed.csv        # Preprocessed features, lags, & volume conversions
├── src/
│   ├── data/
│   │   ├── validate.py                  # Schema normalization, date parsing, & sanity checks
│   │   ├── explore.py                   # Hydrological regimes, seasonality, & unit verification
│   │   └── preprocess.py                # Pipeline generating leakage-free lags & volumes
│   ├── models/
│   │   ├── water_budget.py              # Physical unit conversions & scenario simulation
│   │   ├── train.py                     # Chronological split, baseline, RF model, & MLflow logging
│   │   └── predict.py                   # Production inference wrapper
│   └── app.py                           # Interactive Streamlit dashboard
├── tests/
│   ├── test_validation.py               # Schema & date parsing tests
│   ├── test_units_and_budget.py         # Unit conversion & mass balance tests
│   ├── test_forecasting.py              # Leakage check & model loading tests
│   └── test_app_smoke.py                # Dashboard & contract smoke tests
├── reports/
│   ├── data_validation_report.md        # Factual raw data audit report
│   └── eda_summary.md                   # Detailed exploratory analysis report
├── models/
│   ├── inflow_forecaster.joblib         # Serialized scikit-learn pipeline
│   └── model_metadata.json              # Model parameters, split dates, & held-out scores
├── dvc.yaml                             # Reproducible pipeline definition
├── requirements.txt                     # Pinned project dependencies
└── README.md
```

---

## 3. Hydrological Data & Unit Conversions

- **Raw Observations:** 13,442 rows from `1990-01-01` to `2026-10-21`.
- **Observation Frequency:** Daily. Exactly 1 missing calendar day across 36 years (`2020-02-11`), preserved without synthetic interpolation.
- **Unit Conversion Rule:**
  $$\text{Daily Volume (TMC)} = \text{Mean Flow (cusecs)} \times 0.0000864$$
  *(Verified with $r = 0.999999$ correlation against day-to-day cumulative inflow delta).*
- **Simplified Daily Water Balance:**
  $$\text{Expected Storage Change (TMC)} \approx \text{Inflow} - \text{River Outflow} - \text{Canal Withdrawal} - \text{Evaporation}$$
- **Data Quality Anomaly:** Single transcription typo flagged on `1992-08-28` (`14,782.0 ft` vs normal Full Reservoir Level of ~2,482 ft MSL). Raw data preserved intact.

---

## 4. Modeling & Held-Out Test Evaluation

- **Forecasting Target:** Next-Day Inflow (`Inflow (t+1)` in cusecs).
- **Evaluation Split:** Strictly chronological to prevent time-series data leakage:
  - **Train:** 1990-01-02 to 2022-12-31 (12,035 days)
  - **Validation:** 2023-01-01 to 2024-12-31 (731 days)
  - **Held-Out Test:** 2025-01-01 to 2026-10-20 (658 days)

### Test Set Performance

| Model | MAE (cusecs) | RMSE (cusecs) | $R^2$ |
|---|---|---|---|
| **Persistence Baseline ($Inflow_t$)** | 2,278.40 | 6,685.56 | 0.4796 |
| **7-Day Moving Average** | 2,996.67 | 7,361.00 | 0.3691 |
| **Random Forest Regressor** | **2,347.07** | **5,929.95** | **0.5906** |

The Random Forest model achieves an **11.30% reduction in RMSE** over the persistence baseline and improves the variance explained ($R^2$) from 0.48 to 0.59.

---

## 5. MLOps Lifecycle

1. **Data Versioning (DVC):**
   - Configured with local cache (`data/raw/krs_dam_dataset.csv.dvc`).
   - Pipeline defined in `dvc.yaml` for stages `validate`, `explore`, `preprocess`, and `train`.
   - *Note: Local cache only; no remote cloud storage is configured.*
2. **Experiment Tracking (MLflow):**
   - Local tracking store at `mlruns/`.
   - Logs hyperparameters, chronological split bounds, baseline metrics, test metrics, and pipeline artifacts.
3. **Automated Testing (pytest):**
   - 17 unit and smoke tests covering header normalization, date parsing, unit conversions, water balance arithmetic, chronological ordering, leakage prevention, and app imports.

---

## 6. Limitations & Disclaimer

> [!IMPORTANT]
> - **Educational Prototype:** This tool is an academic demonstration for decision-support modeling. It is **not** an official dam safety, flood warning, or gate control system.
> - **Unmodeled Hydrology:** Weather forecasts, localized heavy rainfall, upstream radar data, and dynamic flood routing waves are not included in statistical lag models.
> - **Water Balance Discrepancies:** Departures from the simplified daily mass balance represent unmeasured catchment runoff, river timing lags, or gauge calibration factors. **They do not prove dam leakage.**
