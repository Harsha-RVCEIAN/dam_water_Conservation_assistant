# Dam Water Conservation Assistant --- MLOps Project

An explainable, reproducible student MLOps project analyzing 36 years of historical hydrological observations from **K.R.Sagara Dam** (Cauvery Basin, Karnataka, India). The project provides daily inflow forecasting, data-quality anomaly auditing, and hypothetical water-conservation decision simulation.

---

## 1. Quickstart & Exact Windows Commands

All commands run from the project root using the project virtual environment (`.venv`):

The raw dam CSV is DVC-tracked. If it is unavailable, validation and EDA use the
existing processed dataset read-only. Preprocessing preserves the existing
processed file instead of overwriting it. To regenerate processed data, restore
the original raw CSV from the configured DVC source first.

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

# 8. View MLflow experiment runs and Model Registry
.\.venv\Scripts\python.exe -m mlflow ui

# 9. Register production candidate in MLflow Model Registry (Phase 2 Step 2)
.\.venv\Scripts\python.exe -m src.models.model_registry

# 10. Run Model Registry tests
.\.venv\Scripts\python.exe -m pytest tests/test_model_registry.py -v

# 11. Launch FastAPI Model Serving service (Phase 2 Step 3)
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 12. Run FastAPI serving test suite
.\.venv\Scripts\python.exe -m pytest tests/test_api_schemas.py tests/test_api.py -v
```

### FastAPI Model Serving Operations

The project exposes the registered model (`KRS_Dam_Inflow_Forecaster`, Version 1) over HTTP via FastAPI:

```powershell
# 1. Start the HTTP inference service
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 2. Check service liveness
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method GET

# 3. Check model readiness
Invoke-RestMethod -Uri http://127.0.0.1:8000/ready -Method GET

# 4. Inspect model metadata & quality status
Invoke-RestMethod -Uri http://127.0.0.1:8000/model-info -Method GET

# 5. Send an inflow prediction request
$body = @{
    inflow_lag_1 = 1500.0; inflow_lag_2 = 1450.0; inflow_lag_3 = 1400.0
    inflow_lag_7 = 1300.0; inflow_lag_14 = 1200.0; inflow_roll_mean_7 = 1420.0
    inflow_roll_std_7 = 75.0; inflow_roll_max_7 = 1550.0; inflow_roll_mean_14 = 1350.0
    inflow_roll_mean_30 = 1300.0; storage_lag_1 = 35.5; storage_lag_7 = 36.0
    outflow_lag_1 = 2200.0; outflow_lag_7 = 2150.0; month = 7; day_of_year = 195
    is_monsoon = 1
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method POST -Body $body -ContentType "application/json"

# 6. View interactive Swagger UI documentation
# Open http://127.0.0.1:8000/docs in your browser
```

### MLflow Model Registry Operations

The project manages model versioning through the native MLflow Model Registry under `KRS_Dam_Inflow_Forecaster`:

```powershell
# Start MLflow UI to view tracking runs & Model Registry
.\.venv\Scripts\python.exe -m mlflow ui

# Inspect registered models and active versions
.\.venv\Scripts\python.exe -c "import mlflow; client = mlflow.MlflowClient(); [print(f'Model: {m.name}, Versions: {[v.version for v in m.latest_versions]}') for m in client.search_registered_models()]"

# Load registered model version 1 directly in Python
.\.venv\Scripts\python.exe -c "import mlflow.sklearn; m = mlflow.sklearn.load_model('models:/KRS_Dam_Inflow_Forecaster/1'); print(m)"

# Load registered model using candidate alias (@candidate)
.\.venv\Scripts\python.exe -c "import mlflow.sklearn; m = mlflow.sklearn.load_model('models:/KRS_Dam_Inflow_Forecaster@candidate'); print(m)"

# Back up tracking and registry store (non-destructive)
Copy-Item -Path mlruns -Destination mlruns_backup -Recurse
```

### Rainfall experiment tracking

The completed seven-model rainfall experiment can be logged without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.models.log_rainfall_mlflow
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns_rainfall_enhanced
```

Its separate local MLflow tracking store is `mlruns_rainfall_enhanced/`; this
does not reuse or replace the original trainer's `mlruns/` store.

### Rainfall model comparison dashboard

Launch the separate dashboard for the seven saved rainfall experiment methods:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/rainfall_dashboard.py
```

It reads the saved metrics and metadata, and loads the already-trained rainfall
artifacts for next-day scenario forecasts; it never retrains them. The dashboard
has overview, historical explorer, forecast studio, model performance lab,
feature explorer, and experiment pages. Explore selectable series and rolling
trends, filter validation/test comparisons, and inspect feature distributions.
The forecast studio lets you change 14 dam/flow inputs and choose recorded,
dry, typical-historical, wet-historical, or custom rainfall scenarios. Calendar
fields and rainfall availability indicators are derived. Missing rainfall
remains missing unless you explicitly choose a hypothetical scenario. The
dashboard uses the project's existing Streamlit and Plotly dependencies.

### Rainfall enrichment pipeline

Run these commands from the project root to rebuild the approved rainfall
candidates, daily rainfall series, and left-joined interim dataset:

```powershell
.\.venv\Scripts\python.exe src\data\filter_rainfall.py
.\.venv\Scripts\python.exe src\data\build_rainfall_dataset.py
.\.venv\Scripts\python.exe src\data\merge_dam_rainfall.py
```

The raw rainfall date format is `DD-MM-YYYY`; the filter writes candidate dates
as ISO `YYYY-MM-DD`, which the daily builder parses explicitly. Missing rainfall
observations remain missing (not zero-filled). The merge writes only to
`data/interim/krs_dam_rainfall_merged.csv`; it does not overwrite the processed
dam dataset or retrain the model.

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

### Test Set Performance (Authoritative Held-Out Test Split: 2025-01-01 to 2026-10-20, 658 days)

| Model | MAE (cusecs) | RMSE (cusecs) | $R^2$ | WAPE |
|---|---|---|---|---|
| **Persistence Baseline ($Inflow_t$)** | 2,278.40 | 6,685.56 | 0.4796 | 44.58% |
| **7-Day Moving Average** | 2,996.67 | 7,361.00 | 0.3691 | 58.63% |
| **Tuned Random Forest Regressor** | **2,335.17** | **5,935.09** | **0.5899** | **45.69%** |

The Tuned Random Forest model achieves an **11.23% reduction in RMSE** over the persistence baseline and improves the variance explained ($R^2$) from 0.48 to 0.5899.

*(Note: An earlier exploratory 2024 test window [295 days: 2024-01-01 to 2024-10-21] produced MAE 2,347.07 cusecs, RMSE 5,929.95 cusecs. All production reports and metadata now reflect the authoritative 658-day held-out test split.)*

---

## 5. MLOps Lifecycle

1. **Data Versioning (DVC):**
   - Configured with local cache (`data/raw/krs_dam_dataset.csv.dvc`).
   - Pipeline defined in `dvc.yaml` for stages `validate`, `explore`, `preprocess`, and `train`.
   - *Note: Local cache only; no remote cloud storage is configured.*
2. **Experiment Tracking (MLflow):**
   - Local tracking store at `mlruns/`.
   - Logs hyperparameters, chronological split bounds, baseline metrics, test metrics, and pipeline artifacts.
3. **Model Quality Evaluation & Acceptance (Phase 2 Step 1):**
   - Evaluates performance across low/normal/high flow regimes, seasons, and 80% empirical prediction intervals.
   - Acceptance status: `CONDITIONAL` (RMSE improvement +11.23% vs persistence; MAE slightly higher in dry-season baseflow).
4. **Model Registry (Phase 2 Step 2):**
   - Centralized model management under registered name `KRS_Dam_Inflow_Forecaster` (Version 1, `@candidate`).
   - Lineage preserved from original training run `157ee4e92e284c82a63f385d6c4c9469` without retraining.
   - Verified 100% prediction equivalence against disk artifact (`max_diff = 0.0` cfs).
5. **FastAPI Model Serving (Phase 2 Step 3):**
   - High-performance HTTP inference service loading registered Version 1 at startup via lifespan context manager.
   - Strict 19-feature Pydantic schema validation (`extra='forbid'`, rejecting NaN/inf).
   - Probes for process liveness (`/health`) and model readiness (`/ready`), plus metadata (`/model-info`) and forecasting (`/predict`).
   - Mathematical prediction equivalence verified against direct MLflow predictions (`max_diff <= 1e-5` cusecs).
6. **Automated Testing (pytest):**
   - 220+ unit, integration, schema, and API lifecycle tests covering preprocessing, feature engineering, water balance mass accounting, model quality, model registry, and FastAPI serving contracts.

---

## 6. Limitations & Disclaimer

> [!IMPORTANT]
> - **Educational Prototype:** This tool is an academic demonstration for decision-support modeling. It is **not** an official dam safety, flood warning, or gate control system.
> - **Unmodeled Hydrology:** Weather forecasts, localized heavy rainfall, upstream radar data, and dynamic flood routing waves are not included in statistical lag models.
> - **Water Balance Discrepancies:** Departures from the simplified daily mass balance represent unmeasured catchment runoff, river timing lags, or gauge calibration factors. **They do not prove dam leakage.**

---

## 7. Layer B — Water Conservation Decision-Support Assistant

Layer B adds a deterministic decision-support rule engine on top of Layer A's hydrological forecasting and mass-balance diagnostics:

1. **Reservoir Condition Assessment:** Evaluates live storage trajectory, inflow, river outflow, and canal withdrawal trends across recent observations without inventing unverified capacity percentages.
2. **Abnormal Flow & Water-Balance Review:** Applies robust Median Absolute Deviation (MAD) rolling filters to highlight unusual spikes while clearly distinguishing data-quality gaps from physical anomalies. Water-balance residuals are treated as mass-accounting flags, never as proof of physical leakage.
3. **Shortage Risk Screening:** Identifies potential water stress from multi-day storage depletion and withdrawal/inflow imbalances without extrapolating one-day ML forecasts into unsupported long-term drought predictions.
4. **Advisory Conservation Recommendations:** Produces prioritized, traceable recommendations (reviewing canal records, auditing stage-discharge curves, or checking telemetry) without prescribing gate operations or official allocation cuts.
5. **Standardized Rationing Scenarios:** Compares hypothetical 5%, 10%, 15%, and 20% canal withdrawal reductions, calculating conserved volume ($TMC = \Delta flow \times 0.0000864 \times N$) and unclipped mathematical projections with boundary flags.

### Launching the Assistant

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```
Open tab **"🛡️ Water Conservation Assistant"** to view real-time condition summaries, risk alerts, flow flags, and scenario comparisons.
