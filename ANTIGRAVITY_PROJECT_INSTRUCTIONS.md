# Antigravity Instructions --- Dam Water Conservation Assistant

## Deadline and goal

Build a working, explainable student MLOps project demo by tomorrow.
Implement the end-to-end workflow from raw data exploration through
preprocessing, data versioning, model training, experiment tracking,
testing/evaluation, and a Streamlit dashboard.

## Required presentation sections

The app or project documentation must support explaining: 1.
Introduction 2. Problem Statement 3. Objectives 4. MLOps Lifecycle 5.
Tools Used 6. Data Exploration & Preparation 7. Data Pipeline 8. Data
Versioning & Storage 9. Model Development 10. Experiment Tracking 11.
Testing & Evaluation

## First: inspect, do not code

Inspect the repository and `data/raw/` and `docs/`. Report the raw file
names, file formats, dimensions, column names, data types, date range if
applicable, and likely target variables. Read the project brief and
reference report if present. Do not fabricate columns or measurements.
Do not modify raw data. Then propose the shortest implementation plan
with no more than 5 tasks. Keep the report concise and wait for the next
instruction.

## Project requirements

-   The project is a Dam Water Conservation Assistant, not only a
    water-level predictor.
-   Use actual supplied raw dam data. Do not generate fake measurements
    to make the app look complete.
-   The dam choice, source, units, and operating thresholds must be
    verified or explicitly marked provisional.
-   Distinguish observed values, model predictions, and hypothetical
    scenario assumptions.
-   Do not claim an unexplained water-balance difference proves dam
    leakage.
-   Never mix incompatible units or time intervals. Convert flow rates
    to volume correctly before combining them with storage.
-   Never use future records to train a model evaluated on earlier
    records. Use chronological train/test splitting.
-   Do not invent model scores, water savings, thresholds, or data
    sources.
-   Preserve the raw data unchanged. Store cleaned/derived data
    separately.
-   Use clear caveats where the data does not support a requested
    feature.

## Required implementation order

Complete one task at a time. After each task, run it or test it before
proceeding.

### Task 1 --- Environment and dependencies

-   Detect operating system and Python availability.
-   Create a virtual environment `.venv`.
-   Install only needed packages using pip.
-   Create `requirements.txt`.
-   Include, as appropriate: pandas, numpy, matplotlib, seaborn (only if
    useful), scikit-learn, streamlit, plotly, mlflow, dvc, joblib,
    openpyxl, pytest.
-   Avoid optional packages unless actually needed.
-   Provide exact Windows PowerShell activation, install, and run
    commands in README.
-   Do not assume installation succeeded; verify imports and versions.

### Task 2 --- Data exploration and preparation

-   Write reproducible scripts for loading raw CSV/Excel files, schema
    inspection, descriptive statistics, missing values, duplicates, date
    validation, outlier review, and unit checks.
-   Save concise EDA outputs/plots under `reports/`.
-   Create a preprocessing script that reads raw data and writes cleaned
    data to `data/processed/`.
-   Document every cleaning decision. Do not silently drop records or
    fill missing values without justification.
-   Add a data dictionary and data-quality report.
-   Ensure processed data is reproducible by rerunning the script.

### Task 3 --- Data pipeline and DVC

-   Create a simple pipeline: raw data -\> validation/EDA -\>
    preprocessing -\> training/evaluation.
-   Use DVC to track dataset/pipeline stages where feasible.
-   Initialize Git if not initialized. Add `.gitignore` for `.venv/`,
    MLflow local files if appropriate, cache files, secrets, and
    generated large files as appropriate.
-   Do not commit secrets or credentials.
-   Configure a local DVC remote only if needed and document that a
    local remote is not off-machine backup.
-   Test DVC commands and do not claim remote storage is configured
    unless verified.

### Task 4 --- Model development and MLflow

-   Select a prediction target only after inspecting the real data.
    Prefer future storage if valid; otherwise use a supported target
    such as inflow.
-   Start with a simple baseline and one suitable scikit-learn model.
-   Use chronological train/test splitting. Avoid leakage in feature
    engineering.
-   Report MAE and RMSE when valid; compare to baseline where practical.
-   Log parameters, metrics, and model artifacts in MLflow.
-   Verify the actual tracking URI and give a working command to view
    runs.
-   Keep model complexity proportionate to the dataset and deadline.

### Task 5 --- Testing, dashboard, and documentation

-   Add tests for data loading, preprocessing, unit/logic checks, and
    prediction pipeline.
-   Build a simple Streamlit dashboard with: Overview, Data Exploration,
    Forecast, Conservation Scenario, Data Quality, and
    About/Limitations.
-   Show actual data and explicit prediction horizon. Display metrics
    only when computed from a valid test set.
-   The conservation simulator must expose assumptions and only
    calculate savings when units and water-budget logic are compatible.
    Otherwise label the feature as illustrative and do not present fake
    numeric savings.
-   Add readable errors for missing files or columns.
-   Update README with architecture, setup, exact commands, pipeline,
    model results, MLflow instructions, DVC instructions, limitations,
    and demo steps.

## Suggested structure

Create only what is needed, using this structure unless the existing
repository warrants a simpler one:

data/ raw/ \# original input data, never modified processed/ \#
generated by preprocessing src/ data/ explore.py preprocess.py
validate.py models/ train.py predict.py pipeline.py app.py \# Streamlit
entry point tests/ reports/ \# generated EDA and evaluation reports
models/ \# saved model artifacts if needed mlruns/ \# local MLflow
tracking only if selected dvc.yaml \# only if a DVC pipeline is
configured .dvc/ .gitignore requirements.txt README.md PROJECT_BRIEF.md

Do not create empty files just to match the tree. Avoid duplicate
scripts.

## Token and time efficiency

-   Inspect only files relevant to the current task.
-   Work on one task at a time and make the smallest useful changes.
-   Do not repeat this full brief in responses.
-   Do not rewrite working code.
-   Do not print whole files or huge logs. Summarize outcomes in at most
    5 bullets.
-   Do not install or upgrade packages repeatedly. Check the environment
    first.
-   Run focused tests after each change.
-   If a command fails, show the short error and fix that specific
    issue.
-   Ask a question only if an essential decision cannot be inferred from
    supplied files.
-   Do not start all tasks at once. After initial inspection, wait for
    the user to approve Task 1.

## Definition of done

-   A clean setup can install dependencies from requirements.txt.
-   The complete raw-to-processed pipeline can be rerun.
-   EDA and cleaning decisions can be shown and explained.
-   DVC tracks data/pipeline versioning as claimed.
-   Model training and chronological evaluation run successfully.
-   MLflow records actual runs, metrics, parameters, and artifacts.
-   Tests pass or known failures are documented.
-   Dashboard launches with a documented command.
-   README supports a live demonstration and honest explanation of
    limitations.

## First message to the user

After inspection, provide only: 1. Dataset inventory and columns. 2.
Python/environment status. 3. Proposed project structure. 4. The first
exact commands to run for environment setup. 5. Any critical blocker.
Then wait. Do not code or install anything until the user approves Task
1.
