# Antigravity Instructions: Improve Phase 1 Dam Inflow Prediction Model

## Project context

We are building a **Dam Water Conservation Assistant** using historical K.R. Sagara Dam data. Phase 1 has a working inflow forecasting pipeline, preprocessing, MLflow tracking, a Streamlit dashboard, and automated tests.

Current reported results:
- Persistence baseline: MAE 2,278.40 cusecs; RMSE 6,685.56 cusecs; R² 0.4796
- 7-day moving-average baseline: MAE 2,996.67 cusecs; RMSE 7,361.00 cusecs; R² 0.3691
- Random Forest: MAE 2,347.07 cusecs; RMSE 5,929.95 cusecs; R² 0.5906

Random Forest improves RMSE by approximately 11.3% versus persistence, but persistence has slightly better MAE. Do not claim the model is already excellent or guarantee a 9.5/10 score. The goal is to strengthen validation, reliability, and reproducibility.

## Main instruction to Antigravity

Inspect the existing repository and scripts first. **Modify the current project rather than creating a new project or duplicating files.** Preserve existing working functionality. Before making changes, briefly explain which files need modification and why. Then implement the steps below in priority order.

Do not fabricate data, silently change raw data, or report improvements unless experiments actually demonstrate them. Keep the raw CSV unchanged.

---

## Step 1 — Audit and correct the chronological data split

The reported setup uses training data from 1990–2022 and a test period from 2025–2026, while 2023–2024 appear to be excluded. Investigate why.

Required actions:
1. Inspect the training script, dataset date range, filtering logic, and split logic.
2. Report row counts and date ranges for training, validation, and test sets.
3. Unless the data or project requirements justify a different approach, use:
   - Training: 1990-01-01 through 2022-12-31
   - Validation: 2023-01-01 through 2024-12-31
   - Final test: 2025-01-01 through the latest valid date in the dataset
4. Keep the final test set untouched while selecting features and tuning models.
5. Handle missing calendar days explicitly; do not assume every date exists.
6. Ensure feature generation does not remove or misalign target dates.
7. If there is a valid reason for a different split, document it rather than forcing these dates.

Acceptance criteria:
- No unexplained exclusion of 2023–2024.
- Chronological split and row counts are logged and saved.
- No training target dates occur after validation or test target dates.

---

## Step 2 — Audit and prevent data leakage

Inspect every model feature and classify it as available at forecast time, available only afterward, or uncertain/requires confirmation.

Required actions:
1. Confirm lag features use only previous observations.
2. Confirm rolling features are shifted appropriately so the target value is excluded.
3. Check for direct or indirect leakage through current-day inflow, cumulative totals, water-balance columns, future outflow, or storage values.
4. For multi-step forecasts, ensure later steps do not use actual future inflow values unavailable at the initial forecast time.
5. Fit preprocessing, imputation, scaling, and feature selection on training data only where applicable.
6. Add a feature-availability table to model documentation.
7. Add tests for common leakage patterns and chronological splitting.

Acceptance criteria:
- Every retained feature has a documented availability rationale.
- No target-derived or future-only data leaks into training or inference.
- Automated tests cover lag and rolling-feature alignment.

---

## Step 3 — Build fair forecasting baselines and model comparisons

Evaluate candidates using the same validation target dates and forecast setup.

Models to compare:
1. Persistence / last-observation baseline
2. Seasonal-naive baseline, if a seasonal period is defensible
3. Existing 7-day moving-average baseline
4. Ridge Regression or another simple linear model
5. Random Forest
6. HistGradientBoosting or XGBoost, only if available and appropriate

Required metrics:
- MAE in cusecs
- RMSE in cusecs
- R²
- Optional: WAPE or another suitable scale-independent metric. Use MAPE only if zero and near-zero values are handled safely.

Required actions:
1. Use identical validation dates for each candidate wherever possible.
2. Clearly distinguish one-day-ahead prediction from multi-day forecasting.
3. Ensure baselines use only information available at the same forecast origin.
4. Save a comparison table as a CSV or report artifact.
5. Do not select a model solely because it has better R² or RMSE; consider MAE and high-flow behavior too.

Acceptance criteria:
- A reproducible comparison table exists.
- Model selection uses validation data only.
- Baseline implementations are tested.

---

## Step 4 — Tune Random Forest and candidate models correctly

Required actions:
1. Use time-ordered validation or rolling-origin cross-validation. Never randomly shuffle time-series data for model selection.
2. Tune a reasonable search space for `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`.
3. Avoid an excessively large search that is slow or likely to overfit validation.
4. Log parameters, validation metrics, feature configuration, and artifacts to MLflow for each meaningful experiment.
5. Select the final candidate using validation results.
6. After selection, retrain using permitted pre-test data if appropriate, then evaluate once on the untouched test period.
7. Keep test metrics separate from validation metrics and label them clearly.

Acceptance criteria:
- Hyperparameter selection uses training/validation data only.
- MLflow contains comparable runs with parameters and metrics.
- Final test results are not used for tuning.

---

## Step 5 — Improve forecasting features safely

Inspect the existing 64-feature pipeline before changing it. Keep only features operationally available and useful in chronological validation.

Candidate features:
- Previous inflow lags: 1, 7, 14, and 30 days, where appropriate
- Trailing inflow means over 7 and 30 days, shifted to exclude the target date
- Month and day-of-year seasonality features
- Recent storage and outflow only if genuinely available at prediction time
- Other features only if present in source data and justified

Required actions:
1. Establish a baseline using current safe features.
2. Add candidate features in small groups and compare validation metrics.
3. Keep a feature only if it improves validation performance or provides a documented operational benefit.
4. Check missing values and date gaps so lag alignment is correct.
5. Do not invent rainfall, weather, upstream telemetry, or other absent data.
6. Do not automatically include all 64 features; audit each feature's usefulness and availability.

Acceptance criteria:
- Feature changes are supported by validation evidence.
- Retained features are reproducible at prediction time.
- Feature list and generation rules are saved with the model.

---

## Step 6 — Evaluate performance across water conditions

Overall metrics may hide poor performance during high inflow or sudden spikes.

Required actions:
1. Define low, normal, and high inflow thresholds using training data or documented official criteria, never the final test set.
2. Report MAE and RMSE by inflow group.
3. Report performance by relevant season or month, including monsoon periods if appropriate for this location and dataset.
4. Identify large-error dates and investigate spikes, data-quality issues, and feature limitations.
5. Keep anomaly flags separate from claims of physical leakage. Flow anomalies can trigger investigation but do not prove leakage.

Acceptance criteria:
- A performance breakdown by flow regime and season is saved.
- High-flow limitations are documented.
- Claims about abnormal flow are appropriately cautious.

---

## Step 7 — Investigate data quality and physical assumptions

The existing audit reported:
- One missing calendar day: 2020-02-11
- Seven missing inflow values
- A suspicious reservoir-level value of 14,782.0 ft on 1992-08-28
- Annual cumulative values that appear to reset around June 1

These are investigation items, not confirmed errors.

Required actions:
1. Check suspicious values against the original/official source if available.
2. Do not overwrite the raw CSV.
3. If a correction is verified, implement it as a documented preprocessing rule with provenance; otherwise flag or exclude it using a transparent policy.
4. Document how missing inflow values are handled and whether imputation is appropriate.
5. Verify the conversion assumption: `daily TMC = mean daily flow in cusecs × 0.0000864`. This is valid when the cusecs value represents a mean flow rate over the day. Confirm what the source variable represents.
6. Investigate annual cumulative resets and ensure cumulative values are not treated as continuous across reset boundaries.
7. Add tests for units, missing values, suspicious values, and date handling.

Acceptance criteria:
- Raw data remains unchanged.
- Cleaning decisions are documented and testable.
- Unit conversions are justified by the source-variable meaning.

---

## Step 8 — Improve forecast usefulness for water conservation

Enhance the current dashboard only after the forecasting evaluation is reliable.

Required actions:
1. Clearly display forecast horizon and forecast-generation date/time.
2. Display predicted inflow in cusecs and explain the unit.
3. Add uncertainty intervals only if estimated and evaluated using a defensible method; do not invent confidence levels.
4. Warn when a forecast is outside the validated data range or conditions differ substantially from training.
5. Tie conservation recommendations to transparent rules and available data, and state assumptions.
6. Explain that abnormal inflow/outflow patterns can trigger investigation but cannot establish that the dam is leaking.
7. Do not present the tool as a replacement for dam operators or official safety procedures.

Acceptance criteria:
- Forecast horizon and units are visible.
- Any uncertainty estimate is evidence-based.
- Recommendations and anomaly warnings are explainable and appropriately limited.

---

## Step 9 — Improve reproducibility, testing, and MLflow tracking

Required actions:
1. Log model parameters, features, date splits, metrics, and relevant artifacts in MLflow.
2. Save the selected model with metadata recording the training cutoff, feature names, target, forecast horizon, and preprocessing version.
3. Save validation comparisons and final test metrics as artifacts.
4. Add or update tests for:
   - Chronological splitting
   - Leakage-sensitive lag and rolling features
   - Missing values and missing dates
   - Prediction output shape and finite values
   - Unit conversion and water-budget calculations
   - Reproducible inference from the saved model
5. Run the existing test suite and report the actual command and result.
6. Run the dashboard smoke test and verify existing tabs still work.
7. Do not claim success for tests or dashboard checks that were not actually run.

Acceptance criteria:
- The saved model loads and performs inference.
- Evaluation can be rerun using documented commands.
- MLflow records make model selection auditable.
- Existing working features are preserved.

---

## Final model acceptance criteria

Do not assign a 9.5/10 rating merely because the code runs. Aim for evidence:

| Area | Required evidence |
|---|---|
| Data leakage | No known leakage; feature availability documented |
| Validation | Chronological, reproducible train/validation/test periods |
| Baselines | Fair comparisons using matching forecast dates |
| Accuracy | Consistent validation improvement over relevant baselines |
| High inflow | Error analysis for high-flow periods and spikes |
| Reliability | Limitations and uncertainty documented |
| MLOps | Model, features, metrics, splits, and experiment IDs reproducible |

There is no universal RMSE or R² threshold that automatically earns a 9.5/10 rating. If the model does not beat a baseline on every metric, report that honestly and explain the trade-off.

## Implementation workflow

1. Inspect the repository and report the relevant files.
2. Present a short plan of the changes.
3. Implement steps in priority order, keeping changes focused.
4. Run targeted tests after each major change.
5. Run the full test suite and dashboard smoke test.
6. Summarize:
   - Files changed
   - What changed and why
   - Validation results
   - Final test results
   - MLflow experiment locations/IDs
   - Remaining limitations
7. Do not claim the model achieved a higher score unless actual evaluation results support it.

**Important:** Preserve the existing project. Do not rebuild it from scratch, delete useful features without justification, or fabricate improved metrics.
