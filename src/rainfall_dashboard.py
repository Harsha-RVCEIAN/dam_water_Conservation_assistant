"""Separate Streamlit dashboard for the rainfall-enhanced model experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.rainfall_features import RAINFALL_FEATURE_COLUMNS
from src.models.rainfall_inference import (
    MODEL_DIR,
    RainfallInferenceError,
    feature_values_for_row,
    load_forecast_data,
    load_forecast_models,
    predict_all_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports/rainfall_enhanced_model_metrics.csv"

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
        page_title="KRS Rainfall Forecast Studio",
        page_icon="🌦️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dashboard_theme()
    try:
        metrics = load_metrics()
        metadata = load_model_metadata()
        validate_metrics_metadata(metrics, metadata)
        comparison, best_model = prepare_comparison(metrics)
        data, feature_names, dam_features = _cached_forecast_data()
        models = _cached_forecast_models()
    except (DashboardDataError, RainfallInferenceError) as exc:
        st.error(f"Could not load the rainfall experiment dashboard: {exc}")
        st.stop()

    test_start, test_end = _common_test_period(metadata)
    st.markdown(
        "<div class='rain-hero'>"
        "<div class='rain-eyebrow'>K.R.S. Dam · Student MLOps Project</div>"
        "<h1>Rainfall Forecast Studio</h1>"
        "<p>Explore reservoir history, test rainfall scenarios, and compare "
        "seven next-day inflow forecasting methods.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Target: next-day inflow in cusecs · Held-out test: {test_start} to "
        f"{test_end} · Dataset: `{metadata[EXPECTED_MODELS[0]]['dataset_path']}`"
    )
    pages = {
        "Reservoir overview": "overview",
        "Historical explorer": "history",
        "Forecast studio": "forecast",
        "Model performance lab": "comparison",
        "Feature explorer": "features",
        "Experiment details": "experiment",
    }
    st.sidebar.markdown("## 🌦️ Forecast Studio")
    st.sidebar.caption("K.R.S. Dam · rainfall-enhanced experiment")
    selected_page = st.sidebar.radio(
        "Navigate",
        list(pages),
        label_visibility="collapsed",
        key="rain_dashboard_navigation",
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Experiment snapshot")
    st.sidebar.metric("Methods", len(EXPECTED_MODELS))
    st.sidebar.metric("Learned inputs", metadata["Ridge Regression"]["feature_count"])
    st.sidebar.caption(f"Test cohort · {test_start} → {test_end}")
    st.sidebar.caption("Historical archive only · no live telemetry")

    if pages[selected_page] == "overview":
        _render_overview(data, test_end)
    elif pages[selected_page] == "history":
        _render_historical_trends(data)
    elif pages[selected_page] == "forecast":
        _render_forecast(
            data,
            feature_names,
            dam_features,
            models,
            best_model,
            test_end,
            str(metadata[EXPECTED_MODELS[0]]["rainfall_source_end_date"]),
        )
    elif pages[selected_page] == "comparison":
        _render_comparison(metrics, comparison, best_model)
    elif pages[selected_page] == "features":
        _render_feature_guide(data, feature_names)
    else:
        _render_experiment_details(metadata, test_start, test_end)
        st.caption(
            f"Test metrics cover {test_start} through {test_end}. The dashboard "
            "loads existing artifacts and does not retrain models."
        )
    st.markdown("---")
    st.caption(
        "Decision-support demonstration only · not live monitoring or operational "
        "dam-safety advice."
    )


if __name__ == "__main__":
    main()
