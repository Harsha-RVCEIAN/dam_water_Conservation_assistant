# Dam Water Conservation Assistant --- One-Run Build Guide for Antigravity

**Project goal:** Build a working, explainable student project that uses
historical K.R.Sagara Dam data to forecast inflow and support
water-conservation decisions, with a reproducible MLOps workflow.

**How to use this file:** Put this file in the project root and give
Antigravity this single instruction:

> Read `BUILD_GUIDE.md` completely and follow it as the single source of
> project instructions. First inspect the existing repository and
> current installation status. Then complete the project end-to-end in
> small, testable stages without asking me to approve every stage.
> Follow all safety rules and data limitations in the guide. Do not stop
> after writing a plan; implement, test, document, and report the
> finished project. If a real blocker prevents progress, explain the
> exact blocker and continue with all independent work that can safely
> be completed.

------------------------------------------------------------------------

## 1. Current state and priority

The user reports that these steps have already been performed:

-   `.gitignore` created.
-   Git repository initialized.
-   `.venv` created with Python 3.11.
-   `requirements.txt` written.
-   Package installation started inside `.venv`.

**First action:** Inspect the current terminal/repository state and
determine whether installation has finished. Do not start a second
installation while one is still running. Reuse the existing `.venv`; do
not delete or recreate it unless it is demonstrably broken and the
user's work is preserved.

The raw dataset may be named `krs_dam_dataset.csv` or
`krs_dam_only.csv`. Discover the actual file and use the real path.
Expected location is `data/raw/`. Never rename, overwrite, clean in
place, or delete the raw dataset.

The repository may already contain instructions, a report PDF, data,
scripts, or other work. Inspect before creating or replacing anything.

## 2. Definition of done

Complete the project as far as the available dataset and environment
allow. The finished repository should include:

-   A verified Python environment and minimal, compatible dependencies.
-   Raw-data validation and an exploratory-data report.
-   A reproducible preprocessing pipeline that preserves missingness and
    documents decisions.
-   A processed dataset generated from the raw file, never manually
    edited.
-   A chronological baseline and a machine-learning inflow forecasting
    model.
-   Evaluation metrics on a held-out time period, compared with a simple
    baseline.
-   MLflow experiment tracking, using a local tracking store.
-   DVC data/pipeline versioning if it can be configured locally without
    remote credentials. Do not claim cloud storage is configured unless
    it actually is.
-   A water-budget/scenario calculator with explicit assumptions and
    unit checks.
-   A Streamlit dashboard that clearly separates observed data, model
    forecasts, and hypothetical scenarios.
-   Automated tests for data validation, unit conversions, water-budget
    arithmetic, and prediction input handling.
-   A README with exact setup/run commands, project limitations, and
    reproducibility instructions.
-   A concise final report stating what was implemented, tests run,
    actual measured model scores, and anything not completed.

Do not spend time adding unnecessary infrastructure. Prefer a reliable
local demo over an overengineered deployment.

## 3. Mandatory safety and implementation rules

### Repository and files

1.  Inspect `git status`, the directory tree, `.gitignore`,
    `requirements.txt`, and the data files before making changes.
2.  Keep changes inside this project directory.
3.  Preserve all existing user work. Do not blindly replace existing
    files; inspect and update them carefully.
4.  Never overwrite or modify the raw CSV. Write generated data to
    `data/processed/`.
5.  Do not delete the existing `venv` folder. Use `.venv` for this
    project.
6.  Do not commit automatically. Show `git status` at the end.
7.  Do not use destructive commands such as `git reset --hard`,
    `git clean -fd`, recursive deletion, or force-pushing.
8.  Never expose secrets, credentials, or local environment variables in
    source code or logs.
9.  Do not install packages globally. Use
    `.\.venv\Scripts\python.exe -m pip ...` on Windows.
10. If an operation risks losing user data, stop that operation,
    preserve the files, and explain the issue.

### Code quality

-   Use readable, modular Python with clear function names, type hints
    where useful, and useful error messages.
-   Use `pathlib` and paths relative to the project root, not hard-coded
    user-specific absolute paths.
-   Set random seeds where applicable.
-   Avoid duplicate logic across scripts and dashboard.
-   Keep training, preprocessing, and UI logic separate.
-   Do not suppress exceptions or fake successful outputs.
-   Do not fabricate data, metrics, charts, model artifacts, test
    results, or completed MLOps stages.
-   If a dependency fails, diagnose the specific compatibility issue; do
    not repeatedly reinstall every package.
-   Avoid adding packages unless they are needed.
-   Run the relevant tests after each major stage and fix failures
    before proceeding.

## 4. Data understanding and validation

The dataset is believed to contain historical observations for
**K.R.Sagara Dam**. Previously observed/claimed columns include:

-   `Reservoir Name`
-   `Basin`
-   `Sub Basin` (may contain an embedded newline in the original header)
-   `River`
-   `Monitoring Date`
-   `Percentage Full (%)`
-   `Reservoir Level (ft)`
-   `Design Gross Capacity (TMC)`
-   `Gross Capacity (TMC)`
-   `Live Capacity (TMC)`
-   `Live Above Cill (TMC)`
-   `Inflow (Cusecs)`
-   `Outflow to River (Cusecs)`
-   `Canal Withdrawal (Cusecs)`
-   `Evaporation (Cusecs)`
-   `Cumulative Inflow (TMC)`
-   `Cumulative Outflow (TMC)`
-   `Cumulative Withdrawal (TMC)`
-   `Cumulative Evaporation (TMC)`

Treat this as a provisional schema, not a guarantee. Inspect actual
headers, row count, data types, missingness, duplicates, date range,
unique reservoir names, and numeric ranges first. Clean
whitespace/newline artifacts in column names consistently and document
the original-to-cleaned mapping.

### Validation checks

-   Parse `Monitoring Date` using the actual format. Report parse
    failures.
-   Check whether rows are chronological; sort only in the processed
    pipeline and record that action.
-   Check duplicate timestamps and duplicate records.
-   Verify the observed time frequency rather than assuming it is daily.
    List date gaps and irregular intervals.
-   Check missing values by column and suspicious numeric values.
-   Check whether columns that appear constant really are constant.
-   Check for negative flow/storage values and physically implausible
    values. Flag them for review rather than silently deleting them.
-   Verify whether capacity columns are consistent and whether
    `Gross Capacity (TMC)` is appropriate for any percentage-full
    calculation. Do not infer a formula without evidence.
-   Check cumulative series for decreases/resets and explain them; do
    not automatically treat every decrease as an error.
-   Write a validation/EDA summary under `reports/`, including row
    counts, date range, missingness, duplicate counts, frequency
    findings, and flagged values.
-   If the actual file differs from this expected schema, adapt to the
    real file and document the difference.

### Missing data policy

-   Do not fill all missing values with zero.
-   Do not interpolate flows or storage without a defensible, documented
    reason.
-   Do not invent a missing daily observation just to make the series
    look continuous.
-   Preserve the raw dataset unchanged.
-   For modelling, use rows with valid target values and required
    features. If limited imputation is justified for a feature, fit that
    imputer on training data only (inside a pipeline) and document it.
-   If a date gap exists, report it. A gap is not automatically a data
    error.
-   If a key target column has too few valid values, stop model training
    and explain the limitation rather than fabricating a substitute.

## 5. Water-flow units and water-budget logic

Flow columns such as inflow/outflow/withdrawal/evaporation may be in
**cusecs**, which is a rate. TMC is a volume. Only convert a rate to a
volume when the measurement interval and meaning support that
conversion.

For a flow representing the **mean cusecs over a full day**, the
commonly used conversion is:

`Daily volume (TMC) = Mean flow (cusecs) × 0.0000864`

Use this conversion only after checking that each observation represents
a daily mean or a suitable daily rate. If the data are instantaneous
readings or the observation interval is unknown, label the conversion as
an approximation and do not present it as an authoritative measured
volume.

A simplified daily balance may be represented as:

`Expected storage change ≈ inflow volume − river outflow volume − canal withdrawal volume − evaporation volume`

This is only a simplified balance. Actual reservoir storage can also be
affected by precipitation, direct runoff, releases, seepage, measurement
timing, operational practices, and other factors. Confirm the
definitions and interval alignment of the available columns before using
this equation. If comparing storage changes, convert all terms to
compatible volume units over the same interval.

**Abnormal balance is not proof of leakage.** At most, the project may
flag unusual or unexplained discrepancies for further investigation. Do
not label an anomaly as a confirmed leak or claim the dataset directly
measures leakage unless it contains a verified leakage measurement.

Do not claim real water savings unless the calculation compares a
clearly defined baseline and scenario using defensible assumptions.
Label scenario results as hypothetical.

## 6. Forecasting approach

### Target

Use `Inflow (Cusecs)` as the primary prediction target if validation
confirms that it exists and contains sufficient usable data. The
prediction horizon must match the verified observation frequency. If the
dataset is daily, a next-day forecast may be used; otherwise adapt the
horizon and document it.

### Features

Use only information that would genuinely be available at prediction
time. Reasonable candidates, depending on data quality, include lagged
inflow, prior observations of storage/level, outflow, withdrawal, and
calendar features. Avoid using future values or columns that directly
reveal the target for the same timestamp.

Avoid leakage: - Never randomly split time-series rows. - Split
chronologically into train, validation (if sufficient data), and test
periods. - Fit preprocessing and imputers on training data only. - Do
not use cumulative or derived columns without checking whether they
include information from the target period. - Do not evaluate on
training data and report it as test performance.

### Models

1.  Implement a simple baseline first, such as persistence (next
    prediction equals the latest available inflow), or another justified
    naive forecast.
2.  Train one modest model, such as a scikit-learn
    RandomForestRegressor, using a chronological split and a
    reproducible feature pipeline.
3.  Compare the ML model against the baseline on the exact same held-out
    test dates.
4.  Report MAE and RMSE, sample count, test date range, and baseline
    scores. If useful, include R² but do not use it as the only metric.
5.  Save the model and feature/preprocessing pipeline with joblib only
    after successful evaluation.
6.  If the model does not beat the baseline, say so honestly. Do not
    tune endlessly or invent a better score.
7.  If the dataset or target does not support a reliable model, deliver
    a working baseline and clearly state the limitation.

## 7. MLOps requirements

### Data versioning --- DVC

-   Configure DVC only after confirming it is installed in `.venv`.
-   Track the raw data or the appropriate pipeline inputs without
    copying sensitive or huge files unnecessarily.
-   Use a local DVC cache by default unless an actual remote is
    configured.
-   Do not claim remote storage or collaboration is set up without
    testing it.
-   Keep Git metadata and DVC metadata consistent.
-   Do not force-add ignored files without a reason.
-   Document the exact commands used and any limitation.

### Reproducible pipeline

Create a straightforward pipeline with stages for
validation/preprocessing and model training where practical. Ensure
commands work from the repository root and outputs are deterministic
where possible. If DVC pipeline configuration becomes a disproportionate
blocker, keep the scripts reproducible and document what is missing
rather than faking a successful DVC run.

### Experiment tracking --- MLflow

-   Use a local MLflow tracking URI/store inside the project, for
    example `mlruns/` or a documented local SQLite database.
-   Ensure training records parameters, metrics, model details, and
    relevant dataset/version identifiers.
-   Make the UI/run instructions explicit.
-   Verify the run appears in the local tracking store before claiming
    tracking works.
-   Do not include personal absolute paths or secrets in tracked
    parameters.

## 8. Water-conservation assistant and dashboard

Build a small Streamlit dashboard that runs locally and is
understandable during a student demonstration. Include the following
sections if supported by the data:

1.  **Reservoir overview:** latest valid observed date,
    storage/level/capacity fields that pass validation, and data
    freshness caveat.
2.  **Historical trends:** inflow, outflow, withdrawal, level/storage
    trends, with units on every chart.
3.  **Inflow forecast:** model forecast alongside recent historical
    inflow. Identify the forecast horizon and show that it is a
    prediction, not an observation.
4.  **Unusual-pattern flags:** flag statistical or water-balance
    deviations only when the logic is implemented and tested. Show the
    rule/threshold and describe the result as a flag for review, not
    proof of leakage.
5.  **Conservation scenario calculator:** allow users to change explicit
    hypothetical inputs (for example, an assumed reduction in canal
    withdrawal or an assumed inflow scenario) and calculate the implied
    volume difference only when units and time intervals match.
6.  **Data-quality and limitations panel:** missing values, date gaps,
    model test metrics, and warnings about interpretation.

Dashboard rules: - Distinguish **Observed**, **Predicted**, and
**Hypothetical scenario** visually and in text. - Never imply the
dashboard is connected to live dam telemetry unless a real live data
source is implemented and verified. - Do not invent official operating
thresholds, alert colours, legal release rules, or conservation
recommendations. - Do not present a model output as an operational
instruction for dam operators. - Show units, dates, assumptions, and
uncertainty/limitations near the relevant outputs. - Use graceful error
messages when files or models are missing. - Keep the UI clean and
demo-friendly; do not spend time on decorative complexity.

## 9. Automated tests

Create and run tests for at least:

-   Date parsing and column-name normalization.
-   Required-column checks and clear failure messages.
-   Duplicate/missing date reporting.
-   Unit conversion: `1 cusec-day = 0.0000864 TMC`, if using that
    conversion.
-   Water-budget arithmetic using a small known example with consistent
    units.
-   Negative/impossible input handling where relevant.
-   Chronological train/test split (training dates must precede test
    dates).
-   No feature/target leakage in the selected feature construction.
-   Model artifact loading and prediction on a valid sample, if a
    trained artifact exists.
-   Dashboard startup/import smoke test where feasible.

Tests must use small synthetic fixtures when possible; do not require
the full raw dataset for every unit test. Report actual test results.

## 10. Suggested project structure

Adapt to existing files instead of forcing this exact structure:

``` text
project-root/
├── data/
│   ├── raw/                 # original input; never modified by scripts
│   └── processed/           # generated validated/cleaned outputs
├── src/
│   ├── data/
│   │   ├── validate.py
│   │   ├── explore.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── water_budget.py
│   └── app.py
├── tests/
├── reports/
├── models/                  # generated model artifacts; generally not committed if large
├── docs/
├── .dvc/                    # if DVC is configured
├── .gitignore
├── requirements.txt
├── dvc.yaml                 # if pipeline is configured
├── README.md
└── BUILD_GUIDE.md
```

## 11. Dependency policy

Use Python 3.11 and the existing `.venv`. Inspect the current
`requirements.txt` first. Keep only dependencies the implementation
actually needs. Likely packages include:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `streamlit`
-   `plotly` (only if used)
-   `mlflow`
-   `dvc`
-   `joblib` (usually available through scikit-learn but may be declared
    explicitly)
-   `pytest`

Add `openpyxl` only if Excel files are actually read or written. Avoid
heavyweight packages not required by the project. Use compatible
versions, install inside `.venv`, and verify with `pip check`. If
installation is currently running, wait for that process and inspect its
result before starting another install.

Windows activation:

``` powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy, do not change
machine-wide execution policy just for this project; invoke the
environment directly:

``` powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Run scripts using the `.venv` interpreter to avoid accidentally using
global Python.

## 12. Recommended execution order

Proceed through all stages in one overall run, but verify each stage
before moving on:

### Stage A --- Inspect and verify environment

-   Inspect repository and `git status`.
-   Confirm package installation state; wait if still running.
-   Verify `.venv` Python, `pip check`, and imports for required
    packages.
-   Resolve only actual dependency/environment errors.

### Stage B --- Validate and explore data

-   Find the actual raw CSV.
-   Build validation and EDA scripts.
-   Run them against the real file.
-   Save a factual report and inspect warnings.
-   Do not proceed with unsupported assumptions about frequency or
    units.

### Stage C --- Preprocess

-   Build a reproducible script from raw to processed.
-   Clean headers and parse dates.
-   Handle missingness according to the policy above.
-   Preserve the raw file and record row counts/transformations.
-   Run validation tests.

### Stage D --- Train and evaluate

-   Build a baseline and one ML model.
-   Use chronological evaluation.
-   Log actual metrics to MLflow.
-   Save artifacts only after successful runs.
-   Confirm the model beats or does not beat the baseline honestly.

### Stage E --- Version and track

-   Configure DVC and reproducible pipeline where feasible.
-   Verify the actual commands work.
-   Do not claim a remote, CI/CD, or deployment that does not exist.

### Stage F --- Build dashboard

-   Connect it to the processed data, saved model, and tested
    water-budget functions.
-   Separate observed, predicted, and scenario values.
-   Test core calculations and error handling.

### Stage G --- Final verification

-   Run the entire preprocessing pipeline from the documented command.
-   Run training and confirm metrics/artifacts/MLflow records.
-   Run all tests.
-   Start the dashboard and confirm it loads.
-   Inspect Git diff/status and ensure raw data is unchanged.
-   Update README with exact commands and known limitations.

Do not merely create files; execute the project and verify outputs. If a
stage fails, fix the root cause and rerun the relevant checks. If
blocked by missing data or an external credential, continue with
independent local work and clearly document the blocker.

## 13. README commands

Use commands that match the actual filenames and implementation. Do not
copy this section blindly if script names differ.

Typical Windows commands:

``` powershell
# Verify environment
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check

# Explore/validate data
.\.venv\Scripts\python.exe -m src.data.explore

# Preprocess
.\.venv\Scripts\python.exe -m src.data.preprocess

# Train and log experiment
.\.venv\Scripts\python.exe -m src.models.train

# Run tests
.\.venv\Scripts\python.exe -m pytest -q

# Launch dashboard
.\.venv\Scripts\python.exe -m streamlit run src/app.py

# View MLflow runs, if configured
.\.venv\Scripts\python.exe -m mlflow ui
```

If module execution requires package files such as `__init__.py`, create
them as needed. Document any different command that is actually tested.
Do not include commands that fail or files that do not exist.

## 14. Final report Antigravity must provide

At completion, return a concise checklist containing:

-   [ ] Environment verified; package versions and `pip check` result.
-   [ ] Actual raw file path, row count, columns, and observed date
    range.
-   [ ] Missing values, duplicate records, date gaps, and important
    data-quality findings.
-   [ ] Processed dataset path and transformations applied.
-   [ ] Target, forecast horizon, features, model, and chronological
    split dates.
-   [ ] Actual baseline and model MAE/RMSE on held-out data.
-   [ ] MLflow tracking location and confirmation that a run exists.
-   [ ] DVC status and whether a remote is configured (never guess).
-   [ ] Dashboard command and tested functionality.
-   [ ] Test command and real pass/fail summary.
-   [ ] Files changed and final Git status.
-   [ ] Known limitations and anything not completed.

Do not claim the project is complete if any of the core commands have
not been run. State failures plainly and provide the shortest next
action to fix each one.

------------------------------------------------------------------------

## Final instruction to Antigravity

**Execute this guide end-to-end using the existing project and
`.venv`.** Be efficient: inspect once, reuse what exists, avoid
unnecessary explanations and repeated package installs, and do not ask
for approval at every stage. Preserve the raw data and user work. Do not
fabricate results. Finish with a working, tested local demonstration and
a factual completion report. If the current installation is still
running, wait for it to finish before taking any conflicting action.
