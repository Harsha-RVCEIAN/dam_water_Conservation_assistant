"""K.R.S. Dam water-conservation dashboard using saved rainfall-model results."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.rainfall_features import (
    RAINFALL_DATA_PATH,
    RAINFALL_FEATURE_COLUMNS,
)
from src.models.rainfall_inference import (
    MODEL_ARTIFACT_FILES,
    MODEL_DIR,
    RainfallInferenceError,
    feature_values_for_row,
    load_forecast_data,
    load_forecast_models,
    predict_all_models,
    eligible_prediction_issue_dates,
    predict_for_issue_date,
)
from src.models.water_availability import (
    DEFAULT_STORAGE_THRESHOLDS,
    MAX_LIVE_STORAGE_TMC,
    WaterAvailabilityError,
    assess_water_alert,
    build_conservation_recommendations,
    calculate_storage_status,
    derive_inflow_references,
    summarize_rainfall,
)
from src.models.water_budget import (
    KRS_DEAD_STORAGE_TMC,
    KRS_DESIGN_GROSS_CAPACITY_TMC,
    simulate_conservation_scenario,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports/rainfall_enhanced_model_metrics.csv"
DAM_DATA_PATH = PROJECT_ROOT / "data/processed/krs_dam_processed.csv"
RAINFALL_SOURCE_PATH = PROJECT_ROOT / RAINFALL_DATA_PATH.relative_to(
    PROJECT_ROOT
)

EXPECTED_MODELS = (
    "Persistence",
    "7-Day Moving Average",
    "Seasonal Naive",
    "Ridge Regression",
    "Random Forest",
    "Tuned Random Forest",
    "HistGradientBoosting",
)
MODEL_METADATA_FILES = {
    "Persistence": "rainfall_persistence_v1.json",
    "7-Day Moving Average": "rainfall_7day_moving_average_v1.json",
    "Seasonal Naive": "rainfall_seasonal_naive_v1.json",
    "Ridge Regression": "rainfall_ridge_regression_v1.json",
    "Random Forest": "rainfall_random_forest_v1.json",
    "Tuned Random Forest": "rainfall_tuned_random_forest_v1.json",
    "HistGradientBoosting": "rainfall_histgradientboosting_v1.json",
}
REQUIRED_METRIC_COLUMNS = {
    "model_name",
    "model_type",
    "algorithm",
    "split",
    "sample_count",
    "mae",
    "rmse",
    "r2",
    "feature_count",
    "rainfall_features_used",
}
NUMERIC_COLUMNS = ("sample_count", "mae", "rmse", "r2", "feature_count")


class DashboardDataError(ValueError):
    """Raised when saved metrics or metadata cannot be safely presented."""


def _parse_boolean(value: Any, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise DashboardDataError(
        f"Metrics column '{column}' contains an invalid boolean value: {value!r}."
    )


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    """Load and validate the saved validation/test metrics CSV."""
    if not path.is_file():
        raise DashboardDataError(f"Metrics report not found: {path}")
    try:
        metrics = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read metrics report {path}: {exc}") from exc

    missing_columns = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise DashboardDataError(
            f"Metrics report is missing required columns: {sorted(missing_columns)}."
        )
    if metrics.empty:
        raise DashboardDataError("Metrics report contains no result rows.")

    for column in ("model_name", "model_type", "algorithm", "split"):
        if metrics[column].isna().any():
            raise DashboardDataError(
                f"Metrics report contains missing values in '{column}'."
            )
        metrics[column] = metrics[column].astype(str).str.strip()
        if metrics[column].eq("").any():
            raise DashboardDataError(
                f"Metrics report contains empty values in '{column}'."
            )

    metrics["split"] = metrics["split"].str.lower()
    if set(metrics["split"]) != {"validation", "test"}:
        raise DashboardDataError(
            "Metrics report must contain validation and test rows only."
        )
    if set(metrics["model_name"]) != set(EXPECTED_MODELS):
        raise DashboardDataError(
            "Metrics report model names do not match the seven expected methods. "
            f"Found: {sorted(metrics['model_name'].unique())}."
        )
    if metrics.duplicated(["model_name", "split"]).any():
        raise DashboardDataError(
            "Metrics report has duplicate model/split rows."
        )
    expected_pairs = len(EXPECTED_MODELS) * 2
    if len(metrics) != expected_pairs:
        raise DashboardDataError(
            f"Metrics report must contain {expected_pairs} model/split rows; "
            f"found {len(metrics)}."
        )

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(metrics[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy()).all():
            raise DashboardDataError(
                f"Metrics column '{column}' contains missing or invalid values."
            )
        metrics[column] = converted

    if metrics["sample_count"].le(0).any() or metrics["feature_count"].lt(0).any():
        raise DashboardDataError(
            "Sample counts must be positive and feature counts cannot be negative."
        )
    if metrics["sample_count"].mod(1).ne(0).any() or metrics["feature_count"].mod(1).ne(0).any():
        raise DashboardDataError("Sample and feature counts must be whole numbers.")
    if metrics["mae"].lt(0).any() or metrics["rmse"].lt(0).any():
        raise DashboardDataError("MAE and RMSE cannot be negative.")

    metrics["rainfall_features_used"] = metrics[
        "rainfall_features_used"
    ].map(lambda value: _parse_boolean(value, column="rainfall_features_used"))

    inconsistent = metrics.groupby("model_name").agg(
        model_types=("model_type", "nunique"),
        algorithms=("algorithm", "nunique"),
        feature_counts=("feature_count", "nunique"),
        rainfall_flags=("rainfall_features_used", "nunique"),
    )
    if inconsistent.gt(1).any().any():
        raise DashboardDataError(
            "Model type, algorithm, feature count, or rainfall usage varies "
            "between validation and test rows."
        )

    expected_rainfall_flags = {
        "Persistence": False,
        "7-Day Moving Average": False,
        "Seasonal Naive": False,
        "Ridge Regression": True,
        "Random Forest": True,
        "Tuned Random Forest": True,
        "HistGradientBoosting": True,
    }
    for model_name, expected in expected_rainfall_flags.items():
        actual = bool(
            metrics.loc[
                metrics["model_name"].eq(model_name), "rainfall_features_used"
            ].iloc[0]
        )
        if actual is not expected:
            raise DashboardDataError(
                f"Rainfall feature usage is inconsistent for {model_name}."
            )

    return metrics.sort_values(
        ["model_name", "split"], key=lambda values: values.map(
            {"validation": 0, "test": 1}
        ) if values.name == "split" else values
    ).reset_index(drop=True)


def load_model_metadata(
    model_dir: Path = MODEL_DIR,
) -> dict[str, dict[str, Any]]:
    """Read model metadata and validate names, types, and split dates."""
    metadata_by_model: dict[str, dict[str, Any]] = {}
    for model_name, filename in MODEL_METADATA_FILES.items():
        path = model_dir / filename
        if not path.is_file():
            raise DashboardDataError(f"Model metadata file not found: {path}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise DashboardDataError(
                f"Could not read model metadata {path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise DashboardDataError(f"Model metadata must be a JSON object: {path}")
        if metadata.get("model_name") != model_name:
            raise DashboardDataError(
                f"Model name in metadata {path} does not match {model_name!r}."
            )
        required = {
            "model_type",
            "target",
            "feature_count",
            "rainfall_features_used",
            "actual_date_ranges",
            "sample_counts",
        }
        missing = required - set(metadata)
        if missing:
            raise DashboardDataError(
                f"Model metadata {path} is missing: {sorted(missing)}."
            )
        if metadata["target"] != "target_inflow_next_day_cusecs":
            raise DashboardDataError(f"Unexpected target in metadata: {path}")
        if not isinstance(metadata["actual_date_ranges"], dict):
            raise DashboardDataError(f"Invalid date ranges in metadata: {path}")
        if not isinstance(metadata["sample_counts"], dict):
            raise DashboardDataError(f"Invalid sample counts in metadata: {path}")
        for split in ("train", "validation", "test"):
            date_range = metadata["actual_date_ranges"].get(split)
            sample_count = metadata["sample_counts"].get(split)
            if (
                not isinstance(date_range, dict)
                or not date_range.get("start")
                or not date_range.get("end")
                or not isinstance(sample_count, int)
                or sample_count < 1
            ):
                raise DashboardDataError(
                    f"Invalid {split} range or sample count in metadata: {path}"
                )
        metadata_by_model[model_name] = metadata

    reference = metadata_by_model[EXPECTED_MODELS[0]]
    shared_ranges = reference["actual_date_ranges"]
    shared_samples = reference["sample_counts"]
    for model_name, metadata in metadata_by_model.items():
        if metadata["actual_date_ranges"] != shared_ranges:
            raise DashboardDataError(
                f"Date ranges disagree across model metadata ({model_name})."
            )
        if metadata["sample_counts"] != shared_samples:
            raise DashboardDataError(
                f"Sample counts disagree across model metadata ({model_name})."
            )
    return metadata_by_model


def validate_metrics_metadata(
    metrics: pd.DataFrame,
    metadata_by_model: dict[str, dict[str, Any]],
) -> None:
    """Ensure displayed metrics and metadata describe the same saved experiment."""
    for model_name, metadata in metadata_by_model.items():
        rows = metrics.loc[metrics["model_name"].eq(model_name)]
        if set(rows["model_type"]) != {metadata["model_type"]}:
            raise DashboardDataError(
                f"Model type differs between metrics and metadata for {model_name}."
            )
        if set(rows["feature_count"].astype(int)) != {int(metadata["feature_count"])}:
            raise DashboardDataError(
                f"Feature count differs between metrics and metadata for {model_name}."
            )
        if set(rows["rainfall_features_used"]) != {
            bool(metadata["rainfall_features_used"])
        }:
            raise DashboardDataError(
                f"Rainfall feature usage differs between metrics and metadata "
                f"for {model_name}."
            )
        for split in ("validation", "test"):
            row = rows.loc[rows["split"].eq(split)]
            expected_metrics = metadata.get("metrics", {}).get(split)
            if len(row) != 1 or not isinstance(expected_metrics, dict):
                raise DashboardDataError(
                    f"Missing {split} metrics or metadata for {model_name}."
                )
            for column in ("sample_count", "mae", "rmse", "r2"):
                actual = float(row.iloc[0][column])
                expected = expected_metrics.get(column)
                try:
                    same = np.isfinite(float(expected)) and np.isclose(
                        actual, float(expected), rtol=0.0, atol=1e-8
                    )
                except (TypeError, ValueError):
                    same = False
                if not same:
                    raise DashboardDataError(
                        f"{split} {column} differs between metrics and metadata "
                        f"for {model_name}."
                    )


def prepare_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build validation/test columns and a test-only average-rank ordering.

    Lower MAE and RMSE are better; higher R² is better. The overall ordering is
    computed as the mean of the three metric ranks, not from hardcoded names.
    """
    required = REQUIRED_METRIC_COLUMNS
    missing = required - set(metrics.columns)
    if missing:
        raise DashboardDataError(
            f"Cannot prepare comparison; columns missing: {sorted(missing)}."
        )
    test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
    if set(test_metrics["model_name"]) != set(EXPECTED_MODELS):
        raise DashboardDataError("Test metrics do not include all seven models.")

    for column, ascending in (
        ("mae", True),
        ("rmse", True),
        ("r2", False),
    ):
        test_metrics[f"{column}_rank"] = test_metrics[column].rank(
            method="min", ascending=ascending
        )
    test_metrics["average_test_rank"] = test_metrics[
        ["mae_rank", "rmse_rank", "r2_rank"]
    ].mean(axis=1)
    test_metrics = test_metrics.sort_values(
        ["average_test_rank", "mae", "rmse", "model_name"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    best_model = str(test_metrics.iloc[0]["model_name"])

    value_columns = ("sample_count", "mae", "rmse", "r2")
    table_parts = []
    for split in ("validation", "test"):
        part = metrics.loc[metrics["split"].eq(split)].set_index("model_name")
        part = part.loc[list(EXPECTED_MODELS), list(value_columns)]
        part.columns = pd.MultiIndex.from_product([[split.title()], part.columns])
        table_parts.append(part)
    comparison = pd.concat(table_parts, axis=1)
    ranks = test_metrics.set_index("model_name")[
        ["average_test_rank", "mae_rank", "rmse_rank", "r2_rank"]
    ]
    ranks.columns = pd.MultiIndex.from_product([["Test ranking"], ranks.columns])
    comparison = comparison.join(ranks)
    comparison = comparison.sort_values(
        [
            ("Test ranking", "average_test_rank"),
            ("Test", "mae"),
            ("Test", "rmse"),
        ],
        ascending=[True, True, True],
    )
    return comparison, best_model


def _common_test_period(
    metadata_by_model: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    period = metadata_by_model[EXPECTED_MODELS[0]]["actual_date_ranges"]["test"]
    return str(period["start"]), str(period["end"])


CALENDAR_FEATURES = {
    "month",
    "day_of_year",
    "is_monsoon",
    "sin_day_of_year",
    "cos_day_of_year",
}
DAM_FEATURE_LABELS = {
    "inflow_lag_1": "Inflow 1 day before (cusecs)",
    "inflow_lag_2": "Inflow 2 days before (cusecs)",
    "inflow_lag_3": "Inflow 3 days before (cusecs)",
    "inflow_lag_7": "Inflow 7 days before (cusecs)",
    "inflow_lag_14": "Inflow 14 days before (cusecs)",
    "inflow_roll_mean_7": "7-day mean inflow (cusecs)",
    "inflow_roll_std_7": "7-day inflow standard deviation (cusecs)",
    "inflow_roll_max_7": "7-day maximum inflow (cusecs)",
    "inflow_roll_mean_14": "14-day mean inflow (cusecs)",
    "inflow_roll_mean_30": "30-day mean inflow (cusecs)",
    "storage_lag_1": "Storage 1 day before (TMC)",
    "storage_lag_7": "Storage 7 days before (TMC)",
    "outflow_lag_1": "Outflow 1 day before (cusecs)",
    "outflow_lag_7": "Outflow 7 days before (cusecs)",
}
RAINFALL_FEATURE_LABELS = {
    "rainfall_mm_lag_1": "Rainfall 1 day before (mm)",
    "rainfall_mm_sum_3d": "Complete 3-day rainfall sum (mm)",
    "rainfall_mm_sum_7d": "Complete 7-day rainfall sum (mm)",
    "rainfall_stations_reporting_lag_1": "Stations reporting 1 day before",
}
FEATURE_DESCRIPTIONS = {
    **DAM_FEATURE_LABELS,
    "month": "Calendar month of the selected observation date; derived from date.",
    "day_of_year": "Day number in the selected observation year; derived from date.",
    "is_monsoon": "Existing seasonal indicator from the dam dataset.",
    "sin_day_of_year": "Sine encoding of day of year; derived from the dam dataset.",
    "cos_day_of_year": "Cosine encoding of day of year; derived from the dam dataset.",
    **RAINFALL_FEATURE_LABELS,
    "rainfall_station_coverage_lag_1": "Reporting stations divided by 10; derived.",
    "rainfall_lag_1_available": "1 when previous-day rainfall is available, otherwise 0.",
    "rainfall_window_3d_complete": "1 when all three rainfall days are available.",
    "rainfall_window_7d_complete": "1 when all seven rainfall days are available.",
}
DAM_EDITABLE_FEATURES = tuple(DAM_FEATURE_LABELS)
RAINFALL_EDITABLE_FEATURES = tuple(RAINFALL_FEATURE_LABELS)
STATION_COUNT = 10


@st.cache_data
def _cached_forecast_data() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    return load_forecast_data()


@st.cache_resource
def _cached_forecast_models() -> dict[str, Any]:
    return load_forecast_models()


@st.cache_data
def _cached_dam_history() -> pd.DataFrame:
    if not DAM_DATA_PATH.is_file():
        raise DashboardDataError(f"Processed dam data not found: {DAM_DATA_PATH}")
    try:
        dam = pd.read_csv(DAM_DATA_PATH, parse_dates=["calendar_date"])
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read processed dam data: {exc}") from exc
    required = {
        "calendar_date",
        "gross_capacity_tmc",
        "live_above_cill_tmc",
        "design_gross_capacity_tmc",
        "inflow_cusecs",
        "outflow_cusecs",
        "withdrawal_cusecs",
    }
    missing = required - set(dam.columns)
    if missing:
        raise DashboardDataError(
            f"Processed dam data is missing columns: {sorted(missing)}."
        )
    if dam["calendar_date"].isna().any() or dam["calendar_date"].duplicated().any():
        raise DashboardDataError("Processed dam dates must be valid and unique.")
    return dam.sort_values("calendar_date").reset_index(drop=True)


@st.cache_data
def _cached_daily_rainfall() -> pd.DataFrame:
    if not RAINFALL_SOURCE_PATH.is_file():
        raise DashboardDataError(
            f"Processed rainfall data not found: {RAINFALL_SOURCE_PATH}"
        )
    try:
        return pd.read_csv(
            RAINFALL_SOURCE_PATH,
            usecols=["Monitoring Date", "Rainfall_MM", "Stations_Reporting"],
            dtype={"Monitoring Date": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(
            f"Could not read processed rainfall data: {exc}"
        ) from exc


def _latest_saved_forecast(
    dam_history: pd.DataFrame,
    model_data: pd.DataFrame,
    feature_names: tuple[str, ...],
    dam_features: tuple[str, ...],
    models: dict[str, Any],
    best_model: str,
) -> tuple[float | None, pd.Timestamp, str | None, tuple[str, ...]]:
    """Forecast from the latest observed dam row only; never synthesize inputs."""
    feature_date = pd.Timestamp(dam_history["calendar_date"].iloc[-1]).normalize()
    forecast_date = feature_date + pd.Timedelta(days=1)
    matching = model_data.loc[
        pd.to_datetime(model_data["calendar_date"]).dt.normalize().eq(feature_date)
    ]
    if len(matching) != 1:
        return None, forecast_date, (
            "The latest dam observation does not align to exactly one trained "
            "feature row; no prediction was generated."
        ), ()
    missing_rainfall_features = tuple(
        name
        for name in RAINFALL_FEATURE_COLUMNS
        if pd.isna(matching.iloc[0][name])
    )
    try:
        inputs = feature_values_for_row(matching.iloc[0], feature_names)
    except RainfallInferenceError as exc:
        return (
            None,
            forecast_date,
            f"Required prediction inputs are unavailable: {exc}",
            missing_rainfall_features,
        )
    missing_dam = [name for name in dam_features if not np.isfinite(inputs[name])]
    if missing_dam:
        return None, forecast_date, (
            "Dam/calendar prediction features are missing: "
            f"{missing_dam}. No prediction was generated."
        ), missing_rainfall_features
    try:
        forecasts = predict_all_models(inputs, models)
    except RainfallInferenceError as exc:
        return (
            None,
            forecast_date,
            f"Saved model could not make a prediction: {exc}",
            missing_rainfall_features,
        )
    selected = forecasts.loc[
        forecasts["model_name"].eq(best_model), "forecast_cusecs"
    ]
    if len(selected) != 1 or not np.isfinite(float(selected.iloc[0])):
        return None, forecast_date, (
            f"The selected saved model {best_model!r} did not return a valid prediction."
        ), missing_rainfall_features
    return float(selected.iloc[0]), forecast_date, None, missing_rainfall_features


def _render_conservation_overview(
    status,
    *,
    predicted_inflow: float | None,
    prediction_error: str | None,
    missing_rainfall_features: tuple[str, ...],
    prediction_date: pd.Timestamp,
    selected_model: str,
    alert,
    rainfall_summary: dict[str, Any],
    dam_freshness_days: int,
) -> None:
    st.subheader("Current available water")
    st.caption(
        f"Latest dam archive observation: **{status.observation_date:%Y-%m-%d}** "
        f"({status.observation_age_days} days old as of today)."
    )
    if status.observation_age_days > dam_freshness_days:
        st.warning(
            "This is an outdated archive observation, not current reservoir telemetry. "
            "Update dam records before using this for operational decisions."
        )
    columns = st.columns(4)
    columns[0].metric("Live water above cill", f"{status.live_storage_tmc:.2f} TMC")
    columns[1].metric("Live storage percentage", f"{status.storage_pct:.1f}%")
    columns[2].metric(
        "Remaining live capacity",
        f"{status.remaining_live_storage_tmc:.2f} TMC",
    )
    columns[3].metric(
        "Maximum live capacity",
        f"{status.max_live_storage_tmc:.3f} TMC",
    )
    if status.is_outside_capacity:
        st.error(
            "Recorded live storage exceeds the derived live-capacity basis. "
            "The value was not clipped; verify the source observation."
        )
    if predicted_inflow is not None:
        next_cols = st.columns(3)
        next_cols[0].metric(
            f"Predicted inflow · {prediction_date:%Y-%m-%d}",
            f"{predicted_inflow:,.0f} cusecs",
        )
        next_cols[1].metric("Selected saved model", selected_model)
        alert_label = (
            alert.level
            if status.observation_age_days <= dam_freshness_days
            else f"Stale-data result · {alert.level}"
        )
        next_cols[2].metric("Water availability alert", alert_label)
        if missing_rainfall_features:
            st.warning(
                f"{len(missing_rainfall_features)} of {len(RAINFALL_FEATURE_COLUMNS)} "
                "rainfall inputs are unavailable for this forecast date. The saved "
                "learned-model pipeline uses its training-fitted imputer for missing "
                "numeric values; this is not a forecast based on fresh rainfall."
            )
    else:
        st.warning(
            "Next-day inflow prediction is unavailable. The dashboard will not "
            "substitute or fabricate missing prediction inputs. "
            f"Details: {prediction_error or 'No prediction was returned.'}"
        )
    st.markdown("#### Recent storage direction")
    trend_cols = st.columns(3)
    for column, period, change in (
        (trend_cols[0], "1 day", status.change_1d_tmc),
        (trend_cols[1], "7 days", status.change_7d_tmc),
        (trend_cols[2], "30 days", status.change_30d_tmc),
    ):
        column.metric(
            f"Live storage change · {period}",
            "Unavailable" if change is None else f"{change:+.3f} TMC",
        )
    latest_rainfall_date = rainfall_summary["latest_date"]
    if latest_rainfall_date is None:
        st.info("Rainfall observations are unavailable.")
    else:
        age = (pd.Timestamp(date.today()) - latest_rainfall_date).days
        st.info(
            f"Rainfall is historical: latest measured value was "
            f"{rainfall_summary['latest_mm']:.2f} mm on "
            f"{latest_rainfall_date:%Y-%m-%d} ({age} days ago)."
        )


def _render_water_storage_trends(dam_history: pd.DataFrame) -> None:
    st.subheader("Water storage history")
    data = dam_history.copy()
    data["calendar_date"] = pd.to_datetime(data["calendar_date"])
    years = sorted(data["calendar_date"].dt.year.unique().tolist())
    selected_period = st.select_slider(
        "Select storage period",
        options=years,
        value=(years[max(0, len(years) - 5)], years[-1]),
        key="water_storage_year_range",
    )
    start_year, end_year = selected_period
    data = data.loc[
        data["calendar_date"].dt.year.between(start_year, end_year)
    ].copy()
    data["live_storage_pct"] = (
        pd.to_numeric(data["live_above_cill_tmc"], errors="coerce")
        / MAX_LIVE_STORAGE_TMC
        * 100
    )
    display = st.radio(
        "Storage measure",
        ["Live water (TMC)", "Live storage (%)"],
        horizontal=True,
        key="water_storage_measure",
    )
    y_column = (
        "live_above_cill_tmc"
        if display == "Live water (TMC)"
        else "live_storage_pct"
    )
    y_label = "Live water above cill (TMC)" if y_column == "live_above_cill_tmc" else "Live storage (%)"
    figure = px.line(
        data,
        x="calendar_date",
        y=y_column,
        labels={"calendar_date": "Observation date", y_column: y_label},
        title=f"Observed reservoir storage · {start_year}–{end_year}",
    )
    figure.update_traces(connectgaps=False, line={"color": "#45d6b2", "width": 2})
    if y_column == "live_storage_pct":
        figure.add_hline(
            y=100,
            line_dash="dash",
            line_color="#e5a000",
            annotation_text="Derived maximum live capacity",
        )
    figure.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)
    over_capacity = int(
        pd.to_numeric(dam_history["live_above_cill_tmc"], errors="coerce")
        .gt(MAX_LIVE_STORAGE_TMC)
        .sum()
    )
    if over_capacity:
        st.warning(
            f"{over_capacity} historical live-above-cill values exceed the derived "
            f"{MAX_LIVE_STORAGE_TMC:.3f} TMC maximum. They remain visible as recorded."
        )
    st.caption(
        f"Percentage = live water above cill / {MAX_LIVE_STORAGE_TMC:.3f} TMC. "
        f"The maximum is derived from {KRS_DESIGN_GROSS_CAPACITY_TMC:.3f} TMC "
        f"design gross capacity less {KRS_DEAD_STORAGE_TMC:.3f} TMC approximate "
        "dead storage; it is not an independently verified operating limit."
    )


def _render_rainfall_conditions(summary: dict[str, Any]) -> None:
    st.subheader("Measured rainfall conditions")
    if summary["latest_date"] is None:
        st.error("No measured rainfall observations are available.")
        return
    latest_date = summary["latest_date"]
    age = (pd.Timestamp(date.today()) - latest_date).days
    st.caption(
        f"Latest measured rainfall: {latest_date:%Y-%m-%d} · "
        f"rainfall archive ends {summary['source_end_date']:%Y-%m-%d}"
    )
    if age > 7:
        st.warning(
            f"Rainfall data is {age} days old. These are historical observations, "
            "not current rainfall conditions."
        )
    columns = st.columns(4)
    columns[0].metric("Latest measured day", f"{summary['latest_mm']:.2f} mm")
    stations = summary["latest_stations_reporting"]
    columns[1].metric(
        "Stations reporting",
        "Unavailable" if stations is None else f"{stations} / 10",
    )
    weekly = summary["weekly"]
    columns[2].metric(
        "Latest complete 7-day rainfall",
        "Unavailable"
        if weekly["sum_mm"] is None
        else f"{weekly['sum_mm']:.2f} mm",
    )
    monthly = summary["monthly"]
    columns[3].metric(
        "Latest complete month rainfall",
        "Unavailable"
        if monthly["sum_mm"] is None
        else f"{monthly['sum_mm']:.2f} mm",
    )
    if weekly["sum_mm"] is not None:
        st.caption(
            f"7-day period: {weekly['period_start']:%Y-%m-%d} to "
            f"{weekly['period_end']:%Y-%m-%d}."
        )
    else:
        st.caption(
            f"Latest 7-day window is incomplete: "
            f"{weekly['days_available']}/{weekly['days_expected']} measured days."
        )
    if monthly["sum_mm"] is not None:
        st.caption(f"Monthly period: {monthly['period_start']:%B %Y}.")
    else:
        st.caption(
            f"Latest month is incomplete: {monthly['days_available']}/"
            f"{monthly['days_expected']} measured days."
        )
    st.info(
        "A measured 0 mm is a real zero. Missing values and incomplete periods "
        "are unavailable, not zero."
    )

    daily = summary["daily"].copy()
    daily["Monitoring Date"] = pd.to_datetime(daily["Monitoring Date"])
    years = sorted(daily["Monitoring Date"].dt.year.unique().tolist())
    year_range = st.select_slider(
        "Rainfall trend years",
        options=years,
        value=(years[max(0, len(years) - 5)], years[-1]),
        key="water_rainfall_year_range",
    )
    start_year, end_year = year_range
    daily = daily.loc[
        daily["Monitoring Date"].dt.year.between(start_year, end_year)
    ].copy()
    view = st.radio(
        "Rainfall chart",
        ["Daily rainfall", "Monthly rainfall totals"],
        horizontal=True,
        key="water_rainfall_chart",
    )
    observed_all = pd.to_numeric(summary["daily"]["Rainfall_MM"], errors="coerce").dropna()
    low_threshold = float(observed_all.quantile(0.10))
    heavy_threshold = float(observed_all.quantile(0.90))
    if view == "Daily rainfall":
        daily["Rainfall_MM"] = pd.to_numeric(daily["Rainfall_MM"], errors="coerce")
        observed = daily.dropna(subset=["Rainfall_MM"])
        figure = px.bar(
            observed,
            x="Monitoring Date",
            y="Rainfall_MM",
            labels={
                "Monitoring Date": "Observation date",
                "Rainfall_MM": "Measured rainfall (mm)",
            },
            title=f"Daily measured rainfall · {start_year}–{end_year}",
        )
        figure.add_hline(
            y=low_threshold,
            line_dash="dot",
            line_color="#e5a000",
            annotation_text=f"Archive 10th percentile · {low_threshold:.2f} mm",
        )
        figure.add_hline(
            y=heavy_threshold,
            line_dash="dash",
            line_color="#cf5c36",
            annotation_text=f"Archive 90th percentile · {heavy_threshold:.2f} mm",
        )
        figure.update_layout(height=430, hovermode="x")
        st.plotly_chart(figure, use_container_width=True)
        low_days = observed.loc[observed["Rainfall_MM"].le(low_threshold)]
        heavy_days = observed.loc[observed["Rainfall_MM"].ge(heavy_threshold)]
        summary_cols = st.columns(2)
        summary_cols[0].metric(
            "Low-rainfall days in selected period", f"{len(low_days):,}"
        )
        summary_cols[1].metric(
            "Heavy-rainfall days in selected period", f"{len(heavy_days):,}"
        )
        st.caption(
            "Low and heavy periods are descriptive archive-wide 10th/90th percentile "
            "comparisons, not official weather alerts."
        )
    else:
        monthly_history = summary["monthly_history"].copy()
        monthly_history["month_start"] = pd.to_datetime(monthly_history["month_start"])
        selected_months = monthly_history.loc[
            monthly_history["month_start"].dt.year.between(start_year, end_year)
            & monthly_history["is_complete"]
        ]
        if selected_months.empty:
            st.info("No complete measured months are available for this range.")
        else:
            figure = px.bar(
                selected_months,
                x="month_start",
                y="rainfall_sum_mm",
                labels={
                    "month_start": "Month",
                    "rainfall_sum_mm": "Measured rainfall total (mm)",
                },
                title=f"Complete monthly rainfall · {start_year}–{end_year}",
            )
            figure.update_layout(height=430, hovermode="x")
            st.plotly_chart(figure, use_container_width=True)
        omitted = int((~monthly_history["is_complete"]).sum())
        if omitted:
            st.caption(
                f"{omitted} incomplete months are omitted, not converted to zero."
            )
    st.markdown("#### Rainfall used for the inflow model")
    st.write(
        "The selected rainfall-enhanced learned model uses rainfall through the "
        "day before its feature date: previous-day rainfall, complete 3-day and "
        "7-day sums, station reporting/coverage, and availability indicators. "
        f"The rainfall source ends on {summary['source_end_date']:%Y-%m-%d}; "
        "later dates remain missing."
    )


def _render_water_alert_and_recommendations(
    status,
    alert,
    recommendations,
    *,
    prediction: float | None,
    prediction_date: pd.Timestamp,
    prediction_error: str | None,
    missing_rainfall_features: tuple[str, ...],
    inflow_references: dict[str, float | None],
    latest_inflow: float | None,
    latest_outflow: float | None,
    rainfall_summary: dict[str, Any],
    dam_freshness_days: int,
    rainfall_freshness_days: int,
) -> None:
    st.subheader("Water availability alert")
    if status.observation_age_days > dam_freshness_days:
        st.warning(
            f"### {alert.level} — historical result; reservoir data is stale"
        )
    elif alert.level == "Critical Warning":
        st.error(f"### {alert.level}")
    elif alert.level in {"Low Storage Warning", "Moderate Warning"}:
        st.warning(f"### {alert.level}")
    elif alert.level == "Insufficient data":
        st.error(f"### {alert.level}")
    else:
        st.success(f"### {alert.level} — no configured storage warning triggered")
    st.caption(
        "Alert levels use configurable provisional thresholds; no verified official "
        "KRS shortage thresholds or allocation rules were found in this project."
    )
    for reason in alert.reasons:
        st.write(f"- {reason}")
    with st.expander("Alert evidence and dates"):
        for value in alert.triggered_values:
            st.write(f"- {value}")
        st.write(
            f"- Latest dam observation: {status.observation_date:%Y-%m-%d} "
            f"({status.observation_age_days} days old)."
        )
    st.caption(
        f"Configured storage thresholds: moderate ≤ "
        f"{alert.provisional_thresholds['moderate_pct']:.0f}%, low ≤ "
        f"{alert.provisional_thresholds['low_pct']:.0f}%, critical ≤ "
        f"{alert.provisional_thresholds['critical_pct']:.0f}%. "
        f"Freshness settings: dam {dam_freshness_days} days, rainfall "
        f"{rainfall_freshness_days} days."
    )
    st.markdown("#### A. Water Conservation Actions")
    actions = [
        item
        for item in recommendations
        if item.priority not in {"Data freshness", "Data availability"}
    ]
    if not actions:
        st.info(
            "Reservoir conditions are relatively stable. Continue efficient water "
            "management and regular monitoring."
        )
    for item in actions:
        with st.container(border=True):
            st.markdown(f"**{item.category}** · **Priority: {item.priority}**")
            st.markdown(f"### {item.title}")
            st.markdown("**Recommended action:**")
            st.markdown(item.action)
            st.markdown(f"**Why this was generated:** {item.reason}")
            st.caption(f"**Evidence:** {item.supporting_values}")

    st.markdown("#### B. Inflow Prediction Insights")
    if prediction is None:
        st.warning(
            "Predicted inflow is unavailable; prediction-based planning insights "
            "cannot be generated."
        )
        if prediction_error:
            st.caption(prediction_error)
    else:
        insights = st.columns(4)
        insights[0].metric(
            f"Predicted inflow · {prediction_date:%Y-%m-%d}",
            f"{prediction:,.0f} cusecs",
        )
        insights[1].metric(
            "Historical inflow P25",
            "Unavailable"
            if inflow_references["low"] is None
            else f"{inflow_references['low']:,.0f} cusecs",
        )
        insights[2].metric(
            "Historical inflow P75",
            "Unavailable"
            if inflow_references["high"] is None
            else f"{inflow_references['high']:,.0f} cusecs",
        )
        if (
            inflow_references["low"] is None
            or inflow_references["high"] is None
        ):
            st.info("Relative historical inflow classification is unavailable.")
        elif prediction < inflow_references["low"]:
            st.info(
                "The estimate is relatively low compared with training-period "
                "inflow history; consider it a planning signal, not a shortage forecast."
            )
        elif prediction >= inflow_references["high"]:
            st.info(
                "The estimate is relatively high compared with training-period "
                "inflow history; confirm it with measured inflow before revising plans."
            )
        else:
            st.info(
                "The estimate is within the historical 25th–75th percentile range; "
                "compare it with the next measured observation."
            )
        insights[3].metric(
            "Latest recorded inflow",
            "Unavailable" if latest_inflow is None else f"{latest_inflow:,.0f} cusecs",
        )
        st.caption(
            f"Latest observed outflow: "
            f"{'Unavailable' if latest_outflow is None else f'{latest_outflow:,.0f} cusecs'} "
            f"on {status.observation_date:%Y-%m-%d}. Predicted values are estimates; "
            "recorded inflow/outflow are measurements."
        )
        rainfall_date = rainfall_summary["latest_date"]
        if rainfall_date is None:
            st.caption("Rainfall availability: no measured rainfall records.")
        else:
            rainfall_age = (
                pd.Timestamp(date.today()) - pd.Timestamp(rainfall_date)
            ).days
            st.caption(
                f"Rainfall availability: latest measurement "
                f"{rainfall_summary['latest_mm']:.2f} mm on "
                f"{pd.Timestamp(rainfall_date):%Y-%m-%d} "
                f"({rainfall_age} days old; "
                f"{'recent context' if rainfall_age <= rainfall_freshness_days else 'stale, not current'})."
            )
        if missing_rainfall_features:
            st.caption(
                f"The current prediction used the saved fitted imputer for "
                f"{len(missing_rainfall_features)} missing rainfall features."
            )
        st.caption(
            "Historical percentiles are relative training-period references, not "
            "official dam operating limits. Rainfall and inflow are not assumed to "
            "have a direct causal relationship."
        )

    st.markdown("#### C. Data Reliability Warnings")
    reliability_warnings = [
        item
        for item in recommendations
        if item.priority in {"Data freshness", "Data availability"}
    ]
    warning_count = 0
    for item in reliability_warnings:
        warning_count += 1
        st.markdown(
            f"- **{item.title}:** {item.reason} Supporting data: "
            f"{item.supporting_values} **Action:** {item.action}"
        )
    if prediction_error:
        warning_count += 1
        st.warning(f"Prediction limitation: {prediction_error}")
    if missing_rainfall_features:
        warning_count += 1
        st.warning(
            f"{len(missing_rainfall_features)} rainfall model inputs are unavailable "
            "for the latest issue date; the fitted preprocessing pipeline imputed "
            "them. No rainfall values were fabricated."
        )
    if warning_count == 0:
        st.info("No data reliability warnings are currently triggered.")
    st.warning(
        "Recommendations are advisory only. They do not control gates, releases, "
        "or allocations and never override official operating procedures."
    )


def _render_conservation_scenario(latest: pd.Series) -> None:
    st.subheader("Hypothetical conservation scenario")
    st.caption(
        "Simplified scenario only. This does not recommend an actual release or "
        "withdrawal and must not replace approved dam operations."
    )
    labels = {
        "Gross storage": pd.to_numeric(latest.get("gross_capacity_tmc"), errors="coerce"),
        "Inflow": pd.to_numeric(latest.get("inflow_cusecs"), errors="coerce"),
        "Outflow": pd.to_numeric(latest.get("outflow_cusecs"), errors="coerce"),
        "Withdrawal": pd.to_numeric(latest.get("withdrawal_cusecs"), errors="coerce"),
        "Evaporation": pd.to_numeric(latest.get("evaporation_cusecs"), errors="coerce"),
    }
    missing = [label for label, value in labels.items() if pd.isna(value)]
    invalid = [label for label, value in labels.items() if pd.notna(value) and value < 0]
    if missing or invalid:
        st.warning(
            "Scenario defaults are incomplete or invalid; no values are fabricated. "
            f"Missing: {missing or 'none'}; invalid: {invalid or 'none'}."
        )
        return
    if labels["Gross storage"] > KRS_DESIGN_GROSS_CAPACITY_TMC:
        st.warning(
            "Recorded gross storage exceeds design gross capacity. Verify the record "
            "before using the scenario calculator."
        )
        return
    left, right = st.columns(2)
    with left:
        days = st.slider("Scenario horizon (days)", 1, 30, 7, key="water_scenario_days")
        reduction = st.slider(
            "Hypothetical canal withdrawal reduction (%)",
            0,
            50,
            15,
            step=5,
            key="water_scenario_reduction",
        )
        start_storage = st.number_input(
            "Starting gross storage (TMC)",
            min_value=0.0,
            max_value=KRS_DESIGN_GROSS_CAPACITY_TMC,
            value=float(labels["Gross storage"]),
            step=0.1,
            key="water_scenario_storage",
        )
    with right:
        assumed_inflow = st.number_input(
            "Assumed inflow (cusecs)",
            min_value=0.0,
            value=float(labels["Inflow"]),
            step=100.0,
            key="water_scenario_inflow",
        )
        assumed_outflow = st.number_input(
            "Assumed river outflow (cusecs)",
            min_value=0.0,
            value=float(labels["Outflow"]),
            step=100.0,
            key="water_scenario_outflow",
        )
        assumed_withdrawal = st.number_input(
            "Assumed canal withdrawal (cusecs)",
            min_value=0.0,
            value=float(labels["Withdrawal"]),
            step=100.0,
            key="water_scenario_withdrawal",
        )
        assumed_evaporation = st.number_input(
            "Assumed evaporation (cusecs)",
            min_value=0.0,
            value=float(labels["Evaporation"]),
            step=10.0,
            key="water_scenario_evaporation",
        )
    result = simulate_conservation_scenario(
        current_storage_tmc=float(start_storage),
        assumed_inflow_cusecs=float(assumed_inflow),
        planned_outflow_cusecs=float(assumed_outflow),
        baseline_withdrawal_cusecs=float(assumed_withdrawal),
        reduction_percentage=float(reduction),
        simulation_days=int(days),
        evaporation_cusecs=float(assumed_evaporation),
        design_capacity_tmc=KRS_DESIGN_GROSS_CAPACITY_TMC,
    )
    output = st.columns(3)
    output[0].metric("Hypothetical water saved", f"{result.water_saved_tmc:.4f} TMC")
    output[1].metric(
        "Baseline ending gross storage",
        f"{result.projected_baseline_storage_tmc:.2f} TMC",
    )
    output[2].metric(
        "Scenario ending gross storage",
        f"{result.projected_scenario_storage_tmc:.2f} TMC",
    )
    st.info(result.disclaimer)


def _render_forecast_and_model(
    *,
    model_data: pd.DataFrame,
    dam_history: pd.DataFrame,
    feature_names: tuple[str, ...],
    dam_features: tuple[str, ...],
    models: dict[str, Any],
    selected_model: str,
    metrics: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> None:
    st.subheader("Date-selectable next-day inflow prediction")
    try:
        eligible_dates = eligible_prediction_issue_dates(
            model_data,
            dam_history,
            dam_features,
        )
    except RainfallInferenceError as exc:
        st.error(f"Cannot offer prediction dates: {exc}")
        return
    issue_date = st.date_input(
        "Prediction issue date (observations available through this date)",
        value=eligible_dates[-1].date(),
        min_value=eligible_dates[0].date(),
        max_value=eligible_dates[-1].date(),
        key="next_day_prediction_issue_date",
        help=(
            "The selected date is the feature cutoff. The forecast target is the "
            "following calendar day. Dates with unavailable required dam lag "
            "features are rejected rather than replaced."
        ),
    )
    st.caption(
        f"Eligible issue-date range: {eligible_dates[0]:%Y-%m-%d} to "
        f"{eligible_dates[-1]:%Y-%m-%d}; dates must also exist in the dam archive "
        "and have complete required dam/calendar model inputs."
    )
    try:
        selected_forecast = predict_for_issue_date(
            issue_date,
            model_data=model_data,
            dam_history=dam_history,
            feature_names=feature_names,
            dam_features=dam_features,
            models=models,
            model_name=selected_model,
        )
    except RainfallInferenceError as exc:
        st.error(f"No prediction was generated for the selected date: {exc}")
    else:
        selected_reference_end = min(
            selected_forecast.issue_date,
            pd.Timestamp(
                metadata[selected_model]["requested_date_ranges"]["train"]["end"]
            ),
        ).strftime("%Y-%m-%d")
        selected_inflow_references = derive_inflow_references(
            dam_history,
            train_end=selected_reference_end,
        )
        st.write(f"**Selected issue date:** {selected_forecast.issue_date:%Y-%m-%d}")
        st.write(f"**Prediction target date:** {selected_forecast.target_date:%Y-%m-%d}")
        columns = st.columns(3)
        columns[0].metric(
            "Predicted next-day inflow",
            f"{selected_forecast.predicted_inflow_cusecs:,.0f} cusecs",
        )
        columns[1].metric(
            "Actual target-day inflow",
            "Unavailable"
            if selected_forecast.actual_inflow_cusecs is None
            else f"{selected_forecast.actual_inflow_cusecs:,.0f} cusecs",
        )
        columns[2].metric(
            "Prediction error (predicted − actual)",
            "Unavailable"
            if selected_forecast.signed_error_cusecs is None
            else f"{selected_forecast.signed_error_cusecs:+,.0f} cusecs",
        )
        if selected_forecast.actual_inflow_cusecs is None:
            st.info(
                "No measured inflow observation is available for the target date; "
                "actual inflow and prediction error are not shown."
            )
        else:
            st.caption(
                "Actual inflow is the measured observation on the target date. "
                "It is displayed only for comparison and is never passed into "
                "the selected issue-date prediction."
            )

        if selected_forecast.missing_rainfall_features:
            st.warning(
                f"Rainfall is unavailable for this issue date: "
                f"{len(selected_forecast.missing_rainfall_features)} of "
                f"{len(RAINFALL_FEATURE_COLUMNS)} engineered rainfall inputs are "
                "missing. The saved Tuned Random Forest pipeline's fitted training "
                "imputer handled those missing inputs; rainfall was not replaced by zero."
            )
        else:
            st.success(
                "All engineered rainfall inputs required for this prediction are "
                "available from observations before or on the issue date."
            )

        selected_low_reference = selected_inflow_references["low"]
        selected_high_reference = selected_inflow_references["high"]
        if selected_low_reference is None or selected_high_reference is None:
            st.info(
                "Historical inflow comparison is unavailable because issue-date "
                "quartile references could not be calculated."
            )
        elif selected_forecast.predicted_inflow_cusecs < selected_low_reference:
            st.info(
                f"Relative inflow level: low — below the issue-date historical "
                f"25th percentile ({selected_low_reference:,.0f} cusecs). This is a "
                "comparison reference, not an operating limit."
            )
        elif selected_forecast.predicted_inflow_cusecs >= selected_high_reference:
            st.info(
                f"Relative inflow level: high — at or above the issue-date "
                f"historical 75th percentile ({selected_high_reference:,.0f} "
                "cusecs). This is a comparison reference, not an operating limit."
            )
        else:
            st.info(
                f"Relative inflow level: within the issue-date historical "
                f"25th–75th percentile range ({selected_low_reference:,.0f}–"
                f"{selected_high_reference:,.0f} "
                "cusecs); these are comparison references, not operating limits."
            )

        with st.expander("Prediction inputs used"):
            input_rows = []
            rainfall_names = set(RAINFALL_FEATURE_COLUMNS)
            for name in feature_names:
                value = selected_forecast.feature_values[name]
                input_rows.append(
                    {
                        "Feature": name,
                        "Value": (
                            "Unavailable — fitted training imputer used"
                            if name in rainfall_names and not np.isfinite(value)
                            else float(value)
                        ),
                        "Input type": (
                            "Rainfall"
                            if name in rainfall_names
                            else "Dam/calendar"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(input_rows), hide_index=True, use_container_width=True)
        st.caption(
            f"Model: **{selected_model}** · issue-date features are cutoff-safe "
            "and the target date is exactly one calendar day later."
        )
    st.caption(
        "Prediction uses the existing saved Tuned Random Forest artifact; no model "
        "is retrained. Validation and test metrics are kept in the technical appendix."
    )
    with st.expander("Technical appendix: seven-model evaluation"):
        st.info(
            "Validation results support model development. Test results are the "
            "final held-out evaluation and are not used to tune this prediction."
        )
        st.markdown("#### Validation and held-out test results")
        report = metrics.pivot(index="model_name", columns="split")
        report.columns = report.columns.swaplevel(0, 1)
        report = report.reindex(EXPECTED_MODELS)
        columns = [
            ("validation", "mae"),
            ("validation", "rmse"),
            ("validation", "r2"),
            ("validation", "sample_count"),
            ("test", "mae"),
            ("test", "rmse"),
            ("test", "r2"),
            ("test", "sample_count"),
        ]
        report = report.loc[:, columns]
        report.columns = [
            "Validation MAE (cusecs)",
            "Validation RMSE (cusecs)",
            "Validation R²",
            "Validation samples",
            "Test MAE (cusecs)",
            "Test RMSE (cusecs)",
            "Test R²",
            "Test samples",
        ]
        st.dataframe(report, use_container_width=True)
        test_start, test_end = _common_test_period(metadata)
        st.caption(
            f"Held-out test period: {test_start} to {test_end}. "
            f"Metrics source: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`."
        )
        st.markdown("#### Forecasting methods")
        methods = metrics.loc[metrics["split"].eq("test")].set_index("model_name")
        for model_name in EXPECTED_MODELS:
            row = methods.loc[model_name]
            rainfall_used = bool(row["rainfall_features_used"])
            description = (
                f"**{model_name}** · {row['algorithm']} · "
                f"{'learned model; 27 inputs include rainfall features' if rainfall_used else 'rule-based; original formula retained and rainfall is not used'}"
            )
            st.markdown(description)
        st.caption(
            "MAE is average absolute prediction error in cusecs. RMSE also uses "
            "cusecs and penalizes large errors more strongly. R² compares explained "
            "target variation with a mean-prediction baseline; it is not prediction "
            "accuracy or a percentage of correct predictions."
        )
        st.caption(
            f"Saved artifact: "
            f"`{(MODEL_DIR / MODEL_ARTIFACT_FILES[selected_model]).relative_to(PROJECT_ROOT)}`. "
            "The three rule-based methods retain their original formulas and do "
            "not learn rainfall effects."
        )


def _inject_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rain-ink: #e8f3f4;
            --rain-muted: #9eb4bb;
            --rain-panel: #102b35;
            --rain-panel-2: #153944;
            --rain-accent: #45d6b2;
            --rain-blue: #58a6ff;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(32, 132, 130, .20), transparent 34%),
                linear-gradient(160deg, #071820 0%, #0a2029 48%, #0b1c29 100%);
            color: var(--rain-ink);
        }
        [data-testid="stHeader"] { background: rgba(7, 24, 32, .78); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b222b 0%, #102d38 100%);
            border-right: 1px solid rgba(126, 202, 195, .16);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 12px;
            padding: .45rem .65rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(69, 214, 178, .10);
        }
        .rain-hero {
            padding: 1.5rem 1.7rem;
            border-radius: 22px;
            margin: .25rem 0 1.25rem 0;
            color: #f3fbfb;
            background:
                radial-gradient(circle at 85% 20%, rgba(69, 214, 178, .22), transparent 28%),
                linear-gradient(115deg, #123f4a 0%, #12645e 58%, #174e67 100%);
            border: 1px solid rgba(174, 241, 220, .22);
            box-shadow: 0 18px 48px rgba(0, 0, 0, .20);
        }
        .rain-hero h1 { margin: 0; font-size: clamp(1.7rem, 3.2vw, 2.7rem); }
        .rain-hero p { color: #c9e6e5; margin: .5rem 0 0 0; }
        .rain-eyebrow {
            color: #8ef0d2;
            text-transform: uppercase;
            letter-spacing: .16em;
            font-size: .73rem;
            font-weight: 800;
            margin-bottom: .45rem;
        }
        .rain-card {
            background: linear-gradient(145deg, rgba(20, 54, 65, .96), rgba(13, 39, 49, .96));
            border: 1px solid rgba(133, 190, 195, .18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 104px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, .14);
        }
        .rain-card-label { color: #a9c5ca; font-size: .82rem; }
        .rain-card-value { color: #effafa; font-size: 1.55rem; font-weight: 750; margin-top: .32rem; }
        .rain-card-note { color: #71dfbd; font-size: .78rem; margin-top: .18rem; }
        div[data-testid="stMetric"] {
            background: rgba(17, 47, 57, .76);
            border: 1px solid rgba(133, 190, 195, .17);
            border-radius: 15px;
            padding: .9rem 1rem;
        }
        div[data-testid="stMetricValue"] { color: #f3fbfb; }
        div[data-testid="stTabs"] button { border-radius: 12px; }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(133, 190, 195, .18);
            border-radius: 14px;
            background: rgba(13, 39, 49, .38);
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(69, 214, 178, .52);
            background: linear-gradient(100deg, #1d7f70, #17677b);
            color: white;
            font-weight: 700;
        }
        .stButton > button:hover { border-color: #8ef0d2; color: white; }
        hr { border-color: rgba(150, 207, 204, .15); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _edit_dam_features(
    source: pd.Series,
    dam_features: tuple[str, ...],
    selected_date: pd.Timestamp,
) -> dict[str, float]:
    """Render scenario controls for observed dam features, preserving date features."""
    scenario = {
        name: float(source[name]) if pd.notna(source[name]) else float("nan")
        for name in dam_features
    }
    editable = [name for name in dam_features if name not in CALENDAR_FEATURES]
    st.markdown("#### Dam and flow inputs")
    st.caption(
        "These controls start from the selected archived observation. Changes are "
        "hypothetical scenarios and do not update the source dataset."
    )
    columns = st.columns(2)
    for index, feature in enumerate(editable):
        default = scenario[feature]
        if not np.isfinite(default) or default < 0:
            st.error(
                f"Cannot safely initialize '{feature}' from the selected archive row."
            )
            continue
        scenario[feature] = columns[index % 2].number_input(
            DAM_FEATURE_LABELS[feature],
            min_value=0.0,
            value=default,
            step=max(1.0, abs(default) * 0.05),
            format="%.3f",
            key=f"rain_scenario_{selected_date.date()}_{feature}",
            help="Non-negative scenario value; the original observation is unchanged.",
        )
    for feature in CALENDAR_FEATURES.intersection(dam_features):
        scenario[feature] = (
            float(source[feature])
            if pd.notna(source[feature])
            else float("nan")
        )
    return scenario


def _edit_rainfall_features(
    source: pd.Series,
    selected_date: pd.Timestamp,
    data: pd.DataFrame,
) -> tuple[dict[str, float], bool]:
    """Choose archived rainfall or an explicit, clearly labeled scenario profile."""
    recorded = {
        feature: float(source[feature]) if pd.notna(source[feature]) else float("nan")
        for feature in RAINFALL_FEATURE_COLUMNS
    }
    valid_inputs = True
    st.markdown("#### Rainfall inputs")
    st.caption(
        "Rainfall features are observations through the day before the selected "
        "feature date. Scenario profiles are hypothetical and do not alter the "
        "historical dataset."
    )
    profile = st.selectbox(
        "Rainfall scenario profile",
        (
            "Recorded / preserve missing",
            "Dry scenario (assumed zero rain)",
            "Typical historical scenario (median)",
            "Wet historical scenario (90th percentile)",
            "Custom rainfall values",
        ),
        key=f"rain_scenario_profile_{selected_date.date()}",
        help=(
            "Historical median and 90th-percentile values summarize the available "
            "rainfall archive; they are illustrative scenarios, not a weather forecast."
        ),
    )
    rainfall = recorded.copy()
    if profile == "Dry scenario (assumed zero rain)":
        rainfall["rainfall_mm_lag_1"] = 0.0
        rainfall["rainfall_mm_sum_3d"] = 0.0
        rainfall["rainfall_mm_sum_7d"] = 0.0
        rainfall["rainfall_stations_reporting_lag_1"] = float(STATION_COUNT)
        st.info("Scenario assumption: zero rainfall with all 10 stations reporting.")
    elif profile in {
        "Typical historical scenario (median)",
        "Wet historical scenario (90th percentile)",
    }:
        quantile = (
            0.5
            if profile == "Typical historical scenario (median)"
            else 0.9
        )
        for feature in RAINFALL_EDITABLE_FEATURES:
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            rainfall[feature] = (
                float(values.quantile(quantile)) if not values.empty else float("nan")
            )
        station_feature = "rainfall_stations_reporting_lag_1"
        if np.isfinite(rainfall[station_feature]):
            rainfall[station_feature] = float(
                np.clip(np.rint(rainfall[station_feature]), 0, STATION_COUNT)
            )
        label = "median" if quantile == 0.5 else "90th percentile"
        st.info(
            f"Scenario assumption: rainfall feature values at the historical "
            f"{label} of the available source observations."
        )
    elif profile == "Custom rainfall values":
        st.caption(
            "Blank values remain missing. Entered rolling sums are treated as "
            "complete 3-day or 7-day windows."
        )
        columns = st.columns(2)
        for index, feature in enumerate(RAINFALL_EDITABLE_FEATURES):
            actual_value = recorded[feature]
            default_text = (
                f"{actual_value:.3f}" if np.isfinite(actual_value) else ""
            )
            entered = columns[index % 2].text_input(
                RAINFALL_FEATURE_LABELS[feature],
                value=default_text,
                placeholder="No source observation; leave blank",
                key=f"rain_scenario_{selected_date.date()}_{feature}",
                help="Enter a finite, non-negative scenario value.",
            ).strip()
            if not entered:
                rainfall[feature] = float("nan")
                continue
            try:
                value = float(entered)
            except ValueError:
                st.error(f"{RAINFALL_FEATURE_LABELS[feature]} must be a number.")
                valid_inputs = False
                continue
            if not np.isfinite(value) or value < 0:
                st.error(
                    f"{RAINFALL_FEATURE_LABELS[feature]} must be finite and non-negative."
                )
                valid_inputs = False
                continue
            if (
                feature == "rainfall_stations_reporting_lag_1"
                and (value > STATION_COUNT or not value.is_integer())
            ):
                st.error(
                    f"Stations reporting must be a whole number from 0 to {STATION_COUNT}."
                )
                valid_inputs = False
                continue
            rainfall[feature] = value

    rainfall["rainfall_station_coverage_lag_1"] = (
        rainfall["rainfall_stations_reporting_lag_1"] / STATION_COUNT
        if np.isfinite(rainfall["rainfall_stations_reporting_lag_1"])
        else float("nan")
    )
    rainfall["rainfall_lag_1_available"] = float(
        np.isfinite(rainfall["rainfall_mm_lag_1"])
    )
    rainfall["rainfall_window_3d_complete"] = float(
        np.isfinite(rainfall["rainfall_mm_sum_3d"])
    )
    rainfall["rainfall_window_7d_complete"] = float(
        np.isfinite(rainfall["rainfall_mm_sum_7d"])
    )
    return rainfall, valid_inputs


def _render_overview(data: pd.DataFrame, test_end: str) -> None:
    st.subheader("Latest archived reservoir and flow status")
    latest = data.iloc[-1]
    date = pd.Timestamp(latest["calendar_date"])
    st.caption(f"Latest dam observation in the archive: **{date:%Y-%m-%d}**")
    if date > pd.Timestamp(test_end):
        st.warning(
            "The latest dam observation is later than the rainfall experiment's "
            f"test period ending {test_end}. Forecasts from this date are "
            "demonstrations outside the experiment's evaluated date range."
        )
    kpis = (
        ("gross_capacity_tmc", "Gross storage", "TMC"),
        ("reservoir_level_ft", "Reservoir level", "ft"),
        ("inflow_cusecs", "Observed inflow", "cusecs"),
        ("outflow_cusecs", "Observed outflow", "cusecs"),
    )
    columns = st.columns(4)
    for column, (key, label, unit) in zip(columns, kpis):
        if key in data.columns and pd.notna(latest[key]):
            column.metric(label, f"{latest[key]:,.2f} {unit}")
    st.markdown("#### Recent water-flow pulse")
    recent = data.tail(30).copy()
    recent["calendar_date"] = pd.to_datetime(recent["calendar_date"])
    flow_columns = [
        column
        for column in ("inflow_cusecs", "outflow_cusecs", "withdrawal_cusecs")
        if column in recent.columns
    ]
    if flow_columns:
        selected_flows = st.multiselect(
            "Show flow series",
            flow_columns,
            default=flow_columns[:2],
            format_func=lambda value: value.replace("_cusecs", "").replace("_", " ").title(),
            key="rain_overview_flow_select",
        )
        if selected_flows:
            figure = px.line(
                recent,
                x="calendar_date",
                y=selected_flows,
                markers=True,
                labels={
                    "calendar_date": "Date",
                    "value": "Daily flow (cusecs)",
                    "variable": "Series",
                },
            )
            figure.update_layout(
                height=360,
                hovermode="x unified",
                legend_title_text="",
                margin=dict(l=12, r=12, t=18, b=12),
            )
            st.plotly_chart(figure, use_container_width=True)
    rain_available = int(data["rainfall_lag_1_available"].sum())
    st.progress(
        rain_available / len(data),
        text=f"Rainfall lag available for {rain_available:,} of {len(data):,} archived dates",
    )
    st.info(
        "This is an educational decision-support prototype using a historical "
        "archive. It is not connected to live dam telemetry and is not an "
        "operational flood or safety system."
    )


def _render_historical_trends(data: pd.DataFrame) -> None:
    st.subheader("Historical reservoir and inflow trends")
    dated = data.copy()
    dated["calendar_date"] = pd.to_datetime(dated["calendar_date"])
    years = sorted(dated["calendar_date"].dt.year.unique().tolist())
    period = st.select_slider(
        "Explore date range",
        options=years,
        value=(years[max(0, len(years) - 5)], years[-1]),
        key="rain_dashboard_history_years",
    )
    start_year, end_year = period
    year_rows = dated.loc[
        dated["calendar_date"].dt.year.between(start_year, end_year)
    ]
    col1, col2 = st.columns([1, 2])
    with col1:
        group = st.selectbox(
            "Measure group",
            ["Flow (cusecs)", "Storage (TMC)", "Rainfall (mm)"],
            key="rain_history_measure_group",
        )
        options_by_group = {
            "Flow (cusecs)": [
                "inflow_cusecs",
                "outflow_cusecs",
                "withdrawal_cusecs",
            ],
            "Storage (TMC)": ["gross_capacity_tmc", "live_capacity_tmc"],
            "Rainfall (mm)": [
                "rainfall_mm_lag_1",
                "rainfall_mm_sum_3d",
                "rainfall_mm_sum_7d",
            ],
        }
        selected_series = st.multiselect(
            "Choose series",
            options_by_group[group],
            default=options_by_group[group][:2],
            format_func=lambda value: value.replace("_", " ").title(),
            key="rain_history_series",
        )
        aggregation = st.radio(
            "View",
            ["Daily", "30-day rolling mean"],
            horizontal=True,
            key="rain_history_aggregation",
        )
        if aggregation == "30-day rolling mean":
            st.caption("Smoothing helps reveal longer-term trends.")
    if selected_series:
        plot_data = year_rows[["calendar_date", *selected_series]].copy()
        if aggregation == "30-day rolling mean":
            for series in selected_series:
                plot_data[series] = plot_data[series].rolling(
                    30, min_periods=1
                ).mean()
        long_data = plot_data.melt(
            id_vars="calendar_date",
            var_name="series",
            value_name="value",
        ).dropna(subset=["value"])
        if not long_data.empty:
            figure = px.line(
                long_data,
                x="calendar_date",
                y="value",
                color="series",
                markers=aggregation == "Daily",
                labels={
                    "calendar_date": "Date",
                    "value": group,
                    "series": "Measure",
                },
            )
            figure.update_layout(
                height=500,
                hovermode="x unified",
                legend_title_text="",
            )
            st.plotly_chart(figure, use_container_width=True)
    st.caption(
        f"Showing {len(year_rows):,} archived records from {start_year}–{end_year}."
    )


def _render_forecast(
    data: pd.DataFrame,
    feature_names: tuple[str, ...],
    dam_features: tuple[str, ...],
    models: dict[str, Any],
    best_model: str,
    test_end: str,
    rainfall_source_end: str,
) -> None:
    st.subheader("One-day-ahead inflow scenario")
    st.markdown(
        "Choose an archived observation date **t**. The forecast target is inflow "
        "on **t + 1 day**. The feature row uses information through date t, and "
        "rainfall features use observations only through t − 1 day."
    )
    dates = pd.to_datetime(data["calendar_date"])
    selected_date = st.date_input(
        "Feature date (observed dam date)",
        value=dates.max().date(),
        min_value=dates.min().date(),
        max_value=dates.max().date(),
        key="rain_dashboard_forecast_date",
    )
    selected_time = pd.Timestamp(selected_date)
    matching = data.loc[dates.eq(selected_time)]
    if matching.empty:
        st.error(
            "The selected date has no dam observation. Choose a date present in the archive."
        )
        return
    source = matching.iloc[0]
    forecast_date = selected_time + pd.Timedelta(days=1)
    st.caption(
        f"Forecast target date: **{forecast_date:%Y-%m-%d}** · "
        f"feature date: **{selected_time:%Y-%m-%d}**"
    )
    if selected_time > pd.Timestamp(test_end):
        st.warning(
            f"This feature date is after the experiment's test period ending "
            f"{test_end}. Its forecast is outside the saved validation/test evaluation."
        )

    try:
        scenario = feature_values_for_row(source, feature_names)
    except RainfallInferenceError as exc:
        st.error(f"Could not prepare forecast inputs: {exc}")
        return
    with st.expander("Edit dam and calendar feature inputs", expanded=False):
        scenario.update(_edit_dam_features(source, dam_features, selected_time))
        st.markdown("#### Calendar features (derived)")
        calendar_rows = [
            {
                "Feature": feature,
                "Value": scenario[feature],
                "How to modify": "Select another feature date above.",
            }
            for feature in feature_names
            if feature in CALENDAR_FEATURES
        ]
        st.dataframe(pd.DataFrame(calendar_rows), hide_index=True, use_container_width=True)
    with st.expander("Edit rainfall scenario inputs", expanded=True):
        rainfall_scenario, rainfall_inputs_valid = _edit_rainfall_features(
            source, selected_time, data
        )
        scenario.update(rainfall_scenario)
    if not rainfall_inputs_valid:
        st.error("Correct the rainfall input values before generating forecasts.")
        return

    missing_features = [name for name in feature_names if name not in scenario]
    if missing_features:
        st.error(f"Forecast feature values are missing: {missing_features}")
        return
    missing_dam_features = [
        name
        for name in dam_features
        if not np.isfinite(scenario[name])
    ]
    if missing_dam_features:
        st.error(
            "The selected archive row has missing dam/calendar inputs: "
            f"{missing_dam_features}. Choose another date or provide valid inputs."
        )
        return
    try:
        forecasts = predict_all_models(scenario, models)
    except RainfallInferenceError as exc:
        st.error(f"Could not generate the rainfall experiment forecasts: {exc}")
        return

    rain_inputs_missing = any(
        pd.isna(scenario[name]) for name in RAINFALL_FEATURE_COLUMNS
    )
    if rain_inputs_missing:
        st.warning(
            "One or more rainfall inputs are missing. No rainfall values were "
            "replaced with zero; learned pipelines handle missing features with "
            "their median imputer fitted on training data. The three rule-based "
            "forecasts do not use rainfall."
        )
    st.caption(
        f"Rainfall source coverage ends at **{rainfall_source_end}**. Rainfall values entered "
        "in this page are scenario assumptions, not measured observations."
    )
    if selected_time > pd.Timestamp(rainfall_source_end):
        st.caption(
            "Dam observations extend beyond the rainfall source. Those later rows "
            "therefore have missing rainfall unless you enter scenario values."
        )

    forecast_lookup = forecasts.set_index("model_name")
    chosen_model = st.selectbox(
        "Highlight a model forecast",
        EXPECTED_MODELS,
        index=EXPECTED_MODELS.index(best_model),
        key="rain_dashboard_selected_model",
        help=(
            "The initial selection is the model with the best average ranking on "
            "saved test metrics. It is not a guarantee of best future performance."
        ),
    )
    chosen_prediction = float(
        forecast_lookup.at[chosen_model, "forecast_cusecs"]
    )
    observed_inflow = float(source["inflow_cusecs"])
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Selected model",
        chosen_model,
    )
    metric_columns[1].metric(
        "Next-day inflow forecast",
        f"{chosen_prediction:,.0f} cusecs",
        f"{chosen_prediction - observed_inflow:+,.0f} vs feature-date inflow",
    )
    metric_columns[2].metric(
        "Feature-date observed inflow",
        f"{observed_inflow:,.0f} cusecs",
    )

    forecasts = forecasts.merge(
        metrics_test_counts(),
        on="model_name",
        how="left",
        validate="one_to_one",
    )
    st.dataframe(
        forecasts.rename(
            columns={
                "model_name": "Forecasting method",
                "forecast_cusecs": "Next-day forecast (cusecs)",
                "test_sample_count": "Held-out test rows",
            }
        ).style.format(
            {"Next-day forecast (cusecs)": "{:,.0f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    recent = data.loc[dates.le(selected_time)].tail(30)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent["calendar_date"],
            y=recent["inflow_cusecs"],
            name="Observed inflow",
            mode="lines",
        )
    )
    for _, row in forecasts.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[forecast_date],
                y=[row["forecast_cusecs"]],
                name=str(row["model_name"]),
                mode="markers",
                marker={"size": 10},
            )
        )
    figure.update_layout(
        title="Recent observed inflow and seven next-day model forecasts",
        xaxis_title="Date",
        yaxis_title="Inflow (cusecs)",
        height=420,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.info(
        "Persistence, 7-Day Moving Average, and Seasonal Naive use their original "
        "forecasting rules and ignore rainfall inputs. Only the four learned "
        "estimators use the 27-feature input row."
    )


def metrics_test_counts() -> pd.DataFrame:
    """Return saved test sample counts to label forecasts with their evaluation cohort."""
    metrics = load_metrics()
    test = metrics.loc[metrics["split"].eq("test"), ["model_name", "sample_count"]]
    return test.rename(columns={"sample_count": "test_sample_count"})


def _render_comparison(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    best_model: str,
) -> None:
    st.subheader("Model performance lab")
    st.success(f"Best average test-metric ranking: **{best_model}**")
    metric_labels = {
        "MAE": ("mae", "cusecs", True),
        "RMSE": ("rmse", "cusecs", True),
        "R-squared": ("r2", "R² units", False),
    }
    controls = st.columns([1, 1, 2])
    selected_metric = controls[0].selectbox(
        "Metric to explore",
        list(metric_labels),
        key="rain_comparison_metric",
    )
    split = controls[1].radio(
        "Evaluation period",
        ["test", "validation"],
        format_func=str.title,
        horizontal=True,
        key="rain_comparison_split",
    )
    model_types = sorted(metrics["model_type"].unique().tolist())
    selected_types = controls[2].multiselect(
        "Include model types",
        model_types,
        default=model_types,
        format_func=str.title,
        key="rain_comparison_types",
    )
    test = metrics.loc[
        metrics["split"].eq(split)
        & metrics["model_type"].isin(selected_types)
    ].copy()
    if test.empty:
        st.info("Select at least one model type to view results.")
        return
    metric_column, unit, lower_better = metric_labels[selected_metric]
    test = test.sort_values(
        metric_column,
        ascending=lower_better,
    )
    best_value = float(test.iloc[0][metric_column])
    best_label = str(test.iloc[0]["model_name"])
    card_cols = st.columns(3)
    card_cols[0].metric(
        f"Best {selected_metric}",
        f"{best_value:,.3f} {unit}",
        best_label,
    )
    card_cols[1].metric(
        "Models displayed",
        f"{len(test)} / {len(EXPECTED_MODELS)}",
    )
    card_cols[2].metric(
        f"{split.title()} sample count",
        f"{int(test.iloc[0]['sample_count']):,}",
    )
    figure = px.bar(
        test,
        x="model_name",
        y=metric_column,
        color="model_type",
        color_discrete_map={"learned": "#45d6b2", "rule-based": "#58a6ff"},
        text_auto=".3s",
        hover_data=["sample_count", "rainfall_features_used", "r2", "mae", "rmse"],
        labels={
            "model_name": "Forecasting method",
            metric_column: f"{selected_metric} ({unit})",
            "model_type": "Method type",
        },
    )
    figure.update_layout(
        height=440,
        xaxis_title=None,
        yaxis_title=f"{selected_metric} ({unit})",
        legend_title_text="",
        hovermode="x unified",
    )
    figure.update_xaxes(tickangle=-25)
    st.plotly_chart(figure, use_container_width=True)

    st.markdown("#### Error profile: MAE vs RMSE")
    selected_split = metrics.loc[
        metrics["split"].eq(split)
        & metrics["model_type"].isin(selected_types)
    ]
    scatter = px.scatter(
        selected_split,
        x="mae",
        y="rmse",
        size="sample_count",
        color="model_type",
        symbol="rainfall_features_used",
        text="model_name",
        color_discrete_map={"learned": "#45d6b2", "rule-based": "#58a6ff"},
        hover_data=["r2", "sample_count"],
        labels={
            "mae": "MAE (cusecs, lower is better)",
            "rmse": "RMSE (cusecs, lower is better)",
            "model_type": "Method type",
            "rainfall_features_used": "Uses rainfall features",
        },
    )
    scatter.update_traces(textposition="top center")
    scatter.update_layout(height=430, hovermode="closest", legend_title_text="")
    st.plotly_chart(scatter, use_container_width=True)
    with st.expander("Open the full validation/test scorecard"):
        display = comparison.copy()
        display.columns = [
            " / ".join(str(part) for part in column)
            if isinstance(column, tuple)
            else str(column)
            for column in display.columns
        ]
        st.dataframe(
            display.style.format(
                {
                    column: "{:,.2f}"
                    for column in display.columns
                    if column.endswith(("/mae", "/rmse"))
                }
                | {
                    column: "{:.4f}"
                    for column in display.columns
                    if column.endswith("/r2")
                }
            ),
            use_container_width=True,
        )
    st.caption(
        "Validation metrics are distinct from the final held-out test metrics. "
        "Do not use test results to tune model parameters. These results alone "
        "do not establish that rainfall improves forecasts."
    )
    with st.expander("Metric definitions"):
        st.markdown(
            "- **MAE:** average absolute prediction error in cusecs.\n"
            "- **RMSE:** prediction error that penalizes large errors more strongly, "
            "in cusecs.\n"
            "- **R²:** target variation explained relative to a mean-prediction "
            "baseline; it is not accuracy or a percentage of correct predictions."
        )


def _render_feature_guide(
    data: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> None:
    st.subheader("Feature explorer")
    st.caption(
        "Explore what enters the four learned models and how the selected feature "
        "varies across the archive. Values are observed/model inputs, not controls "
        "for historical records."
    )
    feature_groups = {
        "Dam / flow": [name for name in feature_names if name in DAM_FEATURE_LABELS],
        "Rainfall": [name for name in feature_names if name in RAINFALL_FEATURE_LABELS],
        "Calendar": [name for name in feature_names if name in CALENDAR_FEATURES],
        "Rainfall availability": [
            name
            for name in feature_names
            if name in RAINFALL_FEATURE_COLUMNS
            and name not in RAINFALL_FEATURE_LABELS
        ],
    }
    group = st.radio(
        "Feature family",
        list(feature_groups),
        horizontal=True,
        key="rain_feature_group",
    )
    options = feature_groups[group]
    if not options:
        st.info("There are no features in this group.")
        return
    feature = st.selectbox(
        "Feature",
        options,
        format_func=lambda value: FEATURE_DESCRIPTIONS.get(value, value),
        key="rain_feature_selected",
    )
    values = pd.to_numeric(data[feature], errors="coerce")
    observed = values.dropna()
    if observed.empty:
        st.warning("This feature has no available observations in the archive.")
        return
    latest_value = values.iloc[-1]
    summary = st.columns(4)
    summary[0].metric(
        "Latest archive value",
        f"{latest_value:,.3f}" if pd.notna(latest_value) else "Missing",
    )
    summary[1].metric("Available records", f"{len(observed):,}")
    summary[2].metric("Median", f"{observed.median():,.3f}")
    summary[3].metric(
        "Missing",
        f"{values.isna().mean():.1%}",
    )
    charts = st.columns(2)
    dated = data.loc[values.notna(), ["calendar_date"]].copy()
    dated["value"] = values.loc[values.notna()].to_numpy()
    dated["calendar_date"] = pd.to_datetime(dated["calendar_date"])
    with charts[0]:
        mode = st.radio(
            "Trend style",
            ["Daily", "30-day rolling mean"],
            horizontal=True,
            key="rain_feature_trend_mode",
        )
        trend_values = dated["value"]
        if mode == "30-day rolling mean":
            trend_values = trend_values.rolling(30, min_periods=1).mean()
        figure = px.line(
            x=dated["calendar_date"],
            y=trend_values,
            labels={"x": "Date", "y": FEATURE_DESCRIPTIONS.get(feature, feature)},
            title="Feature through time",
        )
        figure.update_layout(height=380)
        st.plotly_chart(figure, use_container_width=True)
    with charts[1]:
        figure = px.histogram(
            observed,
            nbins=35,
            labels={"value": FEATURE_DESCRIPTIONS.get(feature, feature)},
            title="Value distribution",
            color_discrete_sequence=["#45d6b2"],
        )
        figure.update_layout(height=380, xaxis_title=FEATURE_DESCRIPTIONS.get(feature, feature))
        st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": name,
                    "Group": label,
                    "Description": FEATURE_DESCRIPTIONS.get(name, name),
                    "Editable in forecast scenarios": name in (*DAM_EDITABLE_FEATURES, *RAINFALL_EDITABLE_FEATURES),
                }
                for label, feature_list in feature_groups.items()
                for name in feature_list
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "Changing one feature independently can make a scenario inconsistent "
        "(for example, changing a rolling mean without changing its lags). "
        "These controls are for educational what-if exploration, not operational "
        "dam decisions."
    )


def _render_experiment_details(
    metadata: dict[str, dict[str, Any]],
    test_start: str,
    test_end: str,
) -> None:
    reference = metadata[EXPECTED_MODELS[0]]
    st.subheader("Experiment at a glance")
    detail_columns = st.columns(4)
    detail_columns[0].metric(
        "Training rows",
        f"{reference['sample_counts']['train']:,}",
    )
    detail_columns[1].metric(
        "Validation rows",
        f"{reference['sample_counts']['validation']:,}",
    )
    detail_columns[2].metric(
        "Held-out test rows",
        f"{reference['sample_counts']['test']:,}",
    )
    detail_columns[3].metric(
        "Learned-model features",
        metadata["Ridge Regression"]["feature_count"],
    )
    st.markdown("#### What is included")
    st.write(
        f"- Target: `{reference['target']}`\n"
        f"- Dataset: `{reference['dataset_path']}`\n"
        f"- Rainfall observations through: {reference['rainfall_source_end_date']}\n"
        f"- Test period: {test_start} to {test_end}\n"
        f"- Metrics report: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`\n"
        f"- Model artifacts: `{MODEL_DIR.relative_to(PROJECT_ROOT)}`"
    )
    model_details = pd.DataFrame(
        [
            {
                "Forecasting method": model,
                "Type": metadata[model]["model_type"],
                "Algorithm": metadata[model]["algorithm"],
                "Feature count": metadata[model]["feature_count"],
                "Rainfall features used": metadata[model]["rainfall_features_used"],
            }
            for model in EXPECTED_MODELS
        ]
    )
    st.dataframe(model_details, hide_index=True, use_container_width=True)
    st.warning(
        "The three rule-based methods retain their original forecast formulas "
        "and do not use rainfall. This experiment does not establish that "
        "rainfall improves prediction performance."
    )
    with st.expander("Missing-data handling"):
        st.markdown(
            "Rainfall observations are not replaced with zeros. Incomplete rolling "
            "windows remain missing, and availability/completeness flags are "
            "preserved. The four learned model pipelines use median imputation "
            "fitted only on training data. Rule-based baselines do not use rainfall."
        )


def main() -> None:
    st.set_page_config(
        page_title="KRS Water Conservation Assistant",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dashboard_theme()
    try:
        metrics = load_metrics()
        metadata = load_model_metadata()
        validate_metrics_metadata(metrics, metadata)
        recommendation_model = "Tuned Random Forest"
    except (DashboardDataError, RainfallInferenceError) as exc:
        st.error(f"Could not load saved rainfall-model results: {exc}")
        st.stop()

    try:
        dam_history = _cached_dam_history()
        status = calculate_storage_status(dam_history, as_of=date.today())
    except (DashboardDataError, WaterAvailabilityError) as exc:
        st.error(f"Could not calculate reservoir status: {exc}")
        st.stop()

    rainfall_error = None
    try:
        rainfall_summary = summarize_rainfall(_cached_daily_rainfall())
    except (DashboardDataError, WaterAvailabilityError) as exc:
        rainfall_error = str(exc)
        rainfall_summary = summarize_rainfall(
            pd.DataFrame(
                columns=[
                    "Monitoring Date",
                    "Rainfall_MM",
                    "Stations_Reporting",
                ]
            )
        )

    forecast_error = None
    prediction = None
    missing_rainfall_features: tuple[str, ...] = ()
    latest_observed = pd.Timestamp(dam_history["calendar_date"].iloc[-1]).normalize()
    prediction_date = latest_observed + pd.Timedelta(days=1)
    forecast_data = None
    forecast_models = None
    try:
        forecast_data, feature_names, dam_features = _cached_forecast_data()
        forecast_models = _cached_forecast_models()
        (
            prediction,
            prediction_date,
            forecast_error,
            missing_rainfall_features,
        ) = _latest_saved_forecast(
            dam_history,
            forecast_data,
            feature_names,
            dam_features,
            forecast_models,
            recommendation_model,
        )
    except RainfallInferenceError as exc:
        forecast_error = str(exc)

    thresholds = dict(DEFAULT_STORAGE_THRESHOLDS)
    st.sidebar.markdown("## K.R.S. Water Assistant")
    st.sidebar.caption("Historical records · no live telemetry")
    pages = {
        "Water availability": "overview",
        "Next-day inflow": "forecast",
        "Reservoir history": "storage",
        "Rainfall analysis": "rainfall",
        "Alerts & recommendations": "alerts",
        "Conservation scenario": "scenario",
    }
    selected_page = st.sidebar.radio(
        "Navigate",
        list(pages),
        label_visibility="collapsed",
        key="rain_dashboard_navigation",
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Provisional alert controls")
    thresholds["moderate_pct"] = float(
        st.sidebar.slider(
            "Moderate watch threshold (%)",
            min_value=20,
            max_value=80,
            value=int(DEFAULT_STORAGE_THRESHOLDS["moderate_pct"]),
            step=5,
            key="water_moderate_threshold",
        )
    )
    thresholds["low_pct"] = float(
        st.sidebar.slider(
            "Low storage threshold (%)",
            min_value=10,
            max_value=int(thresholds["moderate_pct"] - 5),
            value=int(min(DEFAULT_STORAGE_THRESHOLDS["low_pct"], thresholds["moderate_pct"] - 5)),
            step=5,
            key="water_low_threshold",
        )
    )
    thresholds["critical_pct"] = float(
        st.sidebar.slider(
            "Critical threshold (%)",
            min_value=0,
            max_value=int(thresholds["low_pct"] - 5),
            value=int(min(DEFAULT_STORAGE_THRESHOLDS["critical_pct"], thresholds["low_pct"] - 5)),
            step=5,
            key="water_critical_threshold",
        )
    )
    dam_freshness_days = st.sidebar.slider(
        "Dam-data freshness (days)",
        min_value=1,
        max_value=30,
        value=3,
        key="water_dam_freshness",
    )
    rainfall_freshness_days = st.sidebar.slider(
        "Rainfall-data freshness (days)",
        min_value=1,
        max_value=30,
        value=7,
        key="water_rainfall_freshness",
    )

    inflow_references = derive_inflow_references(
        forecast_data if forecast_data is not None else dam_history
    )
    low_inflow_reference = inflow_references["low"]
    high_inflow_reference = inflow_references["high"]
    latest_dam_row = dam_history.iloc[-1]
    latest_outflow_value = pd.to_numeric(
        latest_dam_row.get("outflow_cusecs"), errors="coerce"
    )
    latest_outflow = (
        float(latest_outflow_value)
        if pd.notna(latest_outflow_value) and latest_outflow_value >= 0
        else None
    )
    latest_inflow_value = pd.to_numeric(
        latest_dam_row.get("inflow_cusecs"), errors="coerce"
    )
    latest_inflow = (
        float(latest_inflow_value)
        if pd.notna(latest_inflow_value) and latest_inflow_value >= 0
        else None
    )
    rainfall_reference = None
    heavy_rainfall_reference = None
    if not rainfall_summary["daily"].empty:
        measured_rainfall = pd.to_numeric(
            rainfall_summary["daily"]["Rainfall_MM"], errors="coerce"
        ).dropna()
        if not measured_rainfall.empty:
            rainfall_reference = float(measured_rainfall.quantile(0.10))
            heavy_rainfall_reference = float(measured_rainfall.quantile(0.90))
    try:
        alert = assess_water_alert(
            status,
            predicted_inflow_cusecs=prediction,
            prediction_date=prediction_date,
            low_inflow_reference_cusecs=low_inflow_reference,
            thresholds=thresholds,
        )
        recommendations = build_conservation_recommendations(
            status,
            alert,
            predicted_inflow_cusecs=prediction,
            prediction_date=prediction_date,
            low_inflow_reference_cusecs=low_inflow_reference,
            high_inflow_reference_cusecs=high_inflow_reference,
            latest_inflow_cusecs=latest_inflow,
            latest_inflow_date=status.observation_date,
            latest_outflow_cusecs=latest_outflow,
            latest_outflow_date=status.observation_date,
            latest_rainfall_date=rainfall_summary["latest_date"],
            latest_rainfall_mm=rainfall_summary["latest_mm"],
            low_rainfall_reference_mm=rainfall_reference,
            high_rainfall_reference_mm=heavy_rainfall_reference,
            reference_date=date.today(),
            max_observation_age_days=dam_freshness_days,
            max_rainfall_age_days=rainfall_freshness_days,
        )
    except WaterAvailabilityError as exc:
        st.error(f"Could not assess water availability: {exc}")
        st.stop()

    test_start, test_end = _common_test_period(metadata)
    st.markdown(
        "<div class='rain-hero'>"
        "<div class='rain-eyebrow'>K.R.S. Dam · Water Conservation Assistant</div>"
        "<h1>Water availability & conservation</h1>"
        "<p>Review recorded live storage, rainfall history, and a saved-model "
        "next-day inflow estimate. All guidance is provisional and non-operational.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Latest reservoir record: {status.observation_date:%Y-%m-%d} · "
        f"Forecast target: next-day inflow (cusecs) · Held-out test period: "
        f"{test_start} to {test_end}."
    )
    if pages[selected_page] == "overview":
        _render_conservation_overview(
            status,
            predicted_inflow=prediction,
            prediction_error=forecast_error,
            missing_rainfall_features=missing_rainfall_features,
            prediction_date=prediction_date,
            selected_model=recommendation_model,
            alert=alert,
            rainfall_summary=rainfall_summary,
            dam_freshness_days=dam_freshness_days,
        )
        if rainfall_error:
            st.error(f"Rainfall analysis is unavailable: {rainfall_error}")
    elif pages[selected_page] == "storage":
        _render_water_storage_trends(dam_history)
    elif pages[selected_page] == "rainfall":
        if rainfall_error:
            st.error(f"Rainfall analysis is unavailable: {rainfall_error}")
        else:
            _render_rainfall_conditions(rainfall_summary)
    elif pages[selected_page] == "forecast":
        if forecast_data is None or forecast_models is None:
            st.error(
                "Date-selectable predictions are unavailable because required "
                f"forecast inputs or saved models could not be loaded: "
                f"{forecast_error or 'unknown model/data loading error'}"
            )
        else:
            _render_forecast_and_model(
                model_data=forecast_data,
                dam_history=dam_history,
                feature_names=feature_names,
                dam_features=dam_features,
                models=forecast_models,
                selected_model=recommendation_model,
                metrics=metrics,
                metadata=metadata,
            )
        if rainfall_error:
            st.warning(
                f"Rainfall observations for conservation review are unavailable: "
                f"{rainfall_error}"
            )
    elif pages[selected_page] == "alerts":
        _render_water_alert_and_recommendations(
            status,
            alert,
            recommendations,
            prediction=prediction,
            prediction_date=prediction_date,
            prediction_error=forecast_error,
            missing_rainfall_features=missing_rainfall_features,
            inflow_references=inflow_references,
            latest_inflow=latest_inflow,
            latest_outflow=latest_outflow,
            rainfall_summary=rainfall_summary,
            dam_freshness_days=dam_freshness_days,
            rainfall_freshness_days=rainfall_freshness_days,
        )
    else:
        _render_conservation_scenario(dam_history.iloc[-1])
    st.markdown("---")
    st.caption(
        "Advisory demonstration only · historical source data · not live monitoring, "
        "dam-safety advice, or an instruction to operate the reservoir."
    )


if __name__ == "__main__":
    main()
