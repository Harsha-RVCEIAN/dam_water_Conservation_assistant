"""Model Loader and In-Memory Artifact Holder for FastAPI Serving.

Loads the registered model (models:/KRS_Dam_Inflow_Forecaster/1) through MLflow
once during application startup and provides thread-safe inference access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.models.model_registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    get_registry_client,
)
from src.models.train import FEATURE_COLUMNS, METADATA_PATH, MLFLOW_TRACKING_URI, TARGET_COLUMN


@dataclass
class ModelArtifactHolder:
    """Holds the in-memory loaded MLflow model pipeline and its safe metadata."""

    model_name: str = REGISTERED_MODEL_NAME
    version: int = 1
    alias: str = CANDIDATE_ALIAS
    pipeline: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[str] = None
    load_error: Optional[str] = None

    def is_ready(self) -> bool:
        """Check if the model pipeline is loaded and ready to serve requests."""
        return self.pipeline is not None and self.load_error is None

    def predict(self, feature_df: pd.DataFrame) -> float:
        """Execute single-row prediction with non-negative physical flow clipping.

        Args:
            feature_df: DataFrame containing the 19 ordered features.

        Returns:
            Predicted next-day inflow in cusecs (float >= 0.0).
        """
        if not self.is_ready():
            raise RuntimeError(f"Model is not ready for inference. Error: {self.load_error}")

        # Ensure exact column ordering
        missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Feature vector is missing required columns: {missing}")

        X = feature_df[FEATURE_COLUMNS]
        raw_pred = self.pipeline.predict(X)
        pred_val = float(raw_pred[0])

        # Enforce physical non-negative flow constraint
        clipped_val = max(0.0, pred_val)
        return round(clipped_val, 2)


def load_registered_serving_model(
    model_name: str = REGISTERED_MODEL_NAME,
    version: int = 1,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    metadata_path: Path = METADATA_PATH,
) -> ModelArtifactHolder:
    """Load the registered model Version 1 from MLflow Model Registry.

    Does NOT silently fall back to local disk joblib files.
    Fails explicitly if the registered model cannot be retrieved from MLflow.

    Returns:
        ModelArtifactHolder populated with the loaded pipeline and safe metadata.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{version}"

    holder = ModelArtifactHolder(
        model_name=model_name,
        version=version,
        alias=CANDIDATE_ALIAS,
    )

    try:
        # Load through MLflow sklearn flavor
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)
        holder.pipeline = loaded_pipeline

        # Fetch version tags and metadata from client
        client = get_registry_client(tracking_uri)
        try:
            mv = client.get_model_version(model_name, version)
            version_tags = dict(mv.tags)
        except Exception:
            version_tags = {}

        # Load safe metadata if available
        base_meta = {}
        if metadata_path.exists():
            try:
                base_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        holder.metadata = {
            "model_name": model_name,
            "version": version,
            "alias": CANDIDATE_ALIAS,
            "target": TARGET_COLUMN,
            "target_unit": "cusecs",
            "quality_status": version_tags.get("quality_status", "CONDITIONAL"),
            "deployment_approved": (version_tags.get("deployment_approved", "false").lower() == "true"),
            "intended_use": version_tags.get("intended_use", "Historical inflow forecasting"),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "domain_boundaries": base_meta.get("domain_boundaries", {}),
        }
        return holder

    except Exception as exc:
        holder.pipeline = None
        holder.load_error = str(exc)
        raise RuntimeError(
            f"Failed to load registered model from MLflow registry at '{model_uri}'. "
            f"Error details: {exc}"
        ) from exc

