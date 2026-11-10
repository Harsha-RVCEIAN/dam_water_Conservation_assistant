"""Inference module for K.R.Sagara Dam inflow forecasting.

Loads trained production pipeline artifact and computes next-day inflow forecasts
with input validation, calibrated empirical prediction intervals, domain boundary checks,
baseline comparisons, and dynamic observation freshness verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
)

FRESHNESS_CONFIG_PATH = Path("config/freshness.yaml")


def load_freshness_policy(config_path: Path = FRESHNESS_CONFIG_PATH) -> Dict[str, Any]:
    """Load configurable freshness policy."""
    if not config_path.exists():
        return {
            "freshness": {
                "max_acceptable_delay_days": 1,
                "status_labels": {
                    "current": "CURRENT_OPERATIONAL",
                    "stale": "HISTORICAL_STALE",
                },
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class InflowForecaster:
    """Production wrapper for dam inflow forecasting with dynamic freshness checks."""

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load joblib model pipeline and JSON metadata with integrity checks."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path.resolve()}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {self.metadata_path.resolve()}. Mismatched artifact state.")

        self.pipeline = joblib.load(self.model_path)
        try:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse model metadata at {self.metadata_path}: {e}")

    def check_input_domain(self, feature_df: pd.DataFrame) -> List[str]:
        """Check whether input features exceed validated training domain boundaries."""
        warnings = []
        domain = self.metadata.get("domain_boundaries", {})
        hist_max_inflow = domain.get("historical_max_inflow_cusecs", 204200.0)
        frl_ft = domain.get("design_full_reservoir_level_ft", 2482.0)

        if "inflow_lag_1" in feature_df.columns:
            val = float(feature_df["inflow_lag_1"].iloc[0])
            if val > hist_max_inflow:
                warnings.append(
                    f"Input inflow ({val:,.0f} cfs) exceeds historical maximum ({hist_max_inflow:,.0f} cfs). "
                    f"Extrapolation uncertainty is elevated."
                )
            if val < 0.0:
                warnings.append("Input inflow is negative, which is physically impossible.")

        if "reservoir_level_ft" in feature_df.columns:
            lvl = float(feature_df["reservoir_level_ft"].iloc[0])
            if lvl > frl_ft + 5.0:
                warnings.append(
                    f"Reservoir level ({lvl:.1f} ft) exceeds Full Reservoir Level ({frl_ft:.1f} ft MSL). "
                    f"Dam is in surcharge or flood spillway routing."
                )

        return warnings

    def evaluate_freshness(
        self,
        latest_observation_date: Union[str, pd.Timestamp],
        current_date: Optional[Union[str, pd.Timestamp]] = None,
        data_source_mode: str = "HISTORICAL_ARCHIVE",
        telemetry_connection_status: str = "NOT_CONFIGURED",
    ) -> Dict[str, Any]:
        """Dynamically evaluate observation age, source mode, and forecast validity.

        Separates data age (FRESH vs STALE) from data source mode (HISTORICAL_ARCHIVE vs LIVE_TELEMETRY)
        and telemetry connection status to ensure historical archive rows are never mistaken for live telemetry.
        """
        policy = load_freshness_policy().get("freshness", {})
        max_delay = policy.get("max_acceptable_delay_days", 1)

        t_obs = pd.to_datetime(latest_observation_date).normalize()
        t_curr = pd.to_datetime(current_date).normalize() if current_date is not None else pd.Timestamp.now().normalize()

        data_age_days = int((t_curr - t_obs).days)
        is_stale = data_age_days > max_delay
        data_age_status = "STALE" if is_stale else "FRESH"

        # Maintain backward-compatible status while honoring source mode
        status = "HISTORICAL_STALE" if is_stale else ("LIVE_OPERATIONAL" if data_source_mode == "LIVE_TELEMETRY" and telemetry_connection_status == "CONNECTED" else "CURRENT_OPERATIONAL")

        origin_date = str(t_obs.date())
        target_date = str((t_obs + pd.Timedelta(days=1)).date())

        if is_stale:
            note = (
                f"Observation date ({origin_date}) is {data_age_days} day(s) older than current date ({t_curr.date()}). "
                "Forecast represents historical evaluation and must not be presented as a live operational prediction."
            )
        else:
            note = (
                f"Observation date ({origin_date}) meets freshness age threshold ({data_age_days} <= {max_delay} days). "
                f"Data source: {data_source_mode}; Live telemetry: {telemetry_connection_status} (advisory research prototype)."
            )

        return {
            "forecast_origin_date": origin_date,
            "target_date": target_date,
            "current_date": str(t_curr.date()),
            "data_age_days": data_age_days,
            "is_stale": is_stale,
            "data_age_status": data_age_status,
            "data_source_mode": data_source_mode,
            "telemetry_connection_status": telemetry_connection_status,
            "freshness_status": status,
            "freshness_note": note,
        }

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predict next-day inflow in cusecs from input DataFrame."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        df_eval = feature_df.copy()
        if "sin_day_of_year" not in df_eval.columns and "day_of_year" in df_eval.columns:
            df_eval["sin_day_of_year"] = np.sin(2.0 * np.pi * df_eval["day_of_year"] / 365.25)
        if "cos_day_of_year" not in df_eval.columns and "day_of_year" in df_eval.columns:
            df_eval["cos_day_of_year"] = np.cos(2.0 * np.pi * df_eval["day_of_year"] / 365.25)

        missing_cols = [c for c in FEATURE_COLUMNS if c not in df_eval.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required feature columns: {missing_cols}")

        X = df_eval[FEATURE_COLUMNS]
        preds = self.pipeline.predict(X)
        # Flows cannot be physically negative
        preds_non_neg = np.clip(preds, a_min=0.0, a_max=None)
        return preds_non_neg

    def predict_with_interval(
        self, feature_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next-day inflow along with empirical 80% prediction intervals (10th and 90th percentiles)."""
        point_preds = self.predict(feature_df)
        uncert = self.metadata.get("uncertainty_intervals_cusecs", {})
        q10_err = uncert.get("residual_quantile_10", -1500.0)
        q90_err = uncert.get("residual_quantile_90", 2500.0)

        lower_bounds = np.clip(point_preds + q10_err, a_min=0.0, a_max=None)
        upper_bounds = np.clip(point_preds + q90_err, a_min=0.0, a_max=None)

        return point_preds, lower_bounds, upper_bounds

    def predict_latest_from_processed(
        self,
        processed_path: Path = Path("data/processed/krs_dam_processed.csv"),
        current_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, Any]:
        """Predict inflow for the day following the most recent observation in processed dataset."""
        df = pd.read_csv(processed_path)
        latest_row = df.iloc[[-1]]
        latest_date = str(latest_row["calendar_date"].values[0])
        latest_inflow = float(latest_row["inflow_cusecs"].values[0])

        preds, lowers, uppers = self.predict_with_interval(latest_row)
        pred_val = float(preds[0])
        lower_val = float(lowers[0])
        upper_val = float(uppers[0])

        persistence_val = latest_inflow
        warnings = self.check_input_domain(latest_row)
        freshness_info = self.evaluate_freshness(latest_date, current_date=current_date)

        return {
            "forecast_horizon": "1 Day Ahead",
            "latest_observed_date": latest_date,
            "latest_observed_inflow_cusecs": latest_inflow,
            "forecast_origin_date": freshness_info["forecast_origin_date"],
            "forecast_target_date": freshness_info["target_date"],
            "forecast_inflow_cusecs": round(pred_val, 2),
            "interval_lower_cusecs": round(lower_val, 2),
            "interval_upper_cusecs": round(upper_val, 2),
            "confidence_level": "80% Empirical Residual Interval",
            "persistence_baseline_cusecs": round(persistence_val, 2),
            "difference_from_baseline": round(pred_val - persistence_val, 2),
            "model_type": self.metadata.get("model_type", "RandomForestRegressor"),
            "domain_warnings": warnings,
            "data_age_days": freshness_info["data_age_days"],
            "is_stale": freshness_info["is_stale"],
            "freshness_status": freshness_info["freshness_status"],
            "freshness_note": freshness_info["freshness_note"],
        }


def main() -> None:
    """CLI test for inference module."""
    forecaster = InflowForecaster()
    result = forecaster.predict_latest_from_processed()
    print("Latest Forecast Result with Dynamic Freshness & Uncertainty:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
