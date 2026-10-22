"""Inference module for K.R.Sagara Dam inflow forecasting.

Loads trained production pipeline artifact and computes next-day inflow forecasts
with input validation and confidence intervals / baseline comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.models.train import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
)


class InflowForecaster:
    """Production wrapper for dam inflow forecasting."""

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
        """Load joblib model pipeline and JSON metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path.resolve()}")
        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predict next-day inflow in cusecs from input DataFrame."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required feature columns: {missing_cols}")

        X = feature_df[FEATURE_COLUMNS]
        preds = self.pipeline.predict(X)
        # Ensure flows cannot be physically negative
        preds_non_neg = np.clip(preds, a_min=0.0, a_max=None)
        return preds_non_neg

    def predict_latest_from_processed(
        self, processed_path: Path = Path("data/processed/krs_dam_processed.csv")
    ) -> Dict[str, Any]:
        """Predict inflow for the day following the most recent observation in processed dataset."""
        df = pd.read_csv(processed_path)
        latest_row = df.iloc[[-1]]
        latest_date = str(latest_row["calendar_date"].values[0])
        latest_inflow = float(latest_row["inflow_cusecs"].values[0])

        pred_val = float(self.predict(latest_row)[0])
        persistence_val = latest_inflow

        return {
            "latest_observed_date": latest_date,
            "latest_observed_inflow_cusecs": latest_inflow,
            "forecast_date": str((pd.to_datetime(latest_date) + pd.Timedelta(days=1)).date()),
            "forecast_inflow_cusecs": round(pred_val, 2),
            "persistence_baseline_cusecs": round(persistence_val, 2),
            "difference_from_baseline": round(pred_val - persistence_val, 2),
            "model_type": self.metadata.get("model_type", "RandomForestRegressor"),
        }


def main() -> None:
    """CLI test for inference module."""
    forecaster = InflowForecaster()
    result = forecaster.predict_latest_from_processed()
    print("Latest Forecast Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
