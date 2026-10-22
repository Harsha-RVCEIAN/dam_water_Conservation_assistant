# Project Brief --- Dam Water Conservation Assistant

## Introduction

A student MLOps project that analyzes historical dam data, forecasts a
supported water-availability variable, and compares clearly labeled
hypothetical conservation scenarios.

## Problem Statement

Dam operators and water planners need to understand possible changes in
reservoir water availability. Historical data can be analyzed and
forecasts can support planning, but the quality and coverage of the
supplied data limit what can be concluded.

## Objectives

-   Explore and validate the supplied raw dataset.
-   Build a reproducible data-preparation pipeline.
-   Track data versions using DVC where feasible.
-   Train and evaluate a suitable baseline and machine-learning model.
-   Track experiments with MLflow.
-   Compare a baseline water-use scenario with a hypothetical
    conservation scenario when the data and units support it.
-   Present findings, predictions, and limitations in a dashboard.

## MLOps Lifecycle

Data exploration -\> validation and preprocessing -\> data/pipeline
versioning -\> model training -\> experiment tracking -\> testing and
evaluation -\> dashboard/demo.

## Tools

Python, Pandas, NumPy, Scikit-learn, Streamlit, DVC, MLflow, Git, and
pytest. Install only packages that are actually needed.

## Data Exploration & Preparation

Use the original raw dataset. Inspect columns, types, date range,
missing values, duplicates, outliers, and units. Document cleaning
decisions and generate a separate processed dataset using reproducible
scripts.

## Data Pipeline

Raw data -\> validation/EDA -\> preprocessing -\> model
training/evaluation -\> dashboard.

## Data Versioning & Storage

Use Git for code and DVC for dataset/pipeline versioning where feasible.
Document the storage location and whether any remote backup is actually
configured.

## Model Development

Choose the target only after inspecting the actual dataset. Use a simple
baseline and a suitable model. For time-dependent data, use
chronological train/test splitting and avoid leakage.

## Experiment Tracking

Use MLflow to log real model parameters, evaluation metrics, and
artifacts. Document the tracking URI and command to view experiments.

## Testing & Evaluation

Test data loading, preprocessing, pipeline execution, and prediction
behavior. Report metrics only when computed correctly on a held-out
chronological test set. Compare with a baseline where practical.

## Conservation Assistant Behavior

Clearly distinguish observed data, model predictions, and hypothetical
assumptions. Do not invent dam measurements, savings, or operating
thresholds. Do not claim a water-balance anomaly proves leakage. Only
combine water volumes and flow rates after correct unit and time
conversion. Any scenario must respect official operating constraints and
essential water needs.

## Limitations

The selected dam, source, units, and available target variables must be
verified from the actual supplied data and official documentation. The
prototype supports decision-making education; it is not an official
dam-control or safety system.
