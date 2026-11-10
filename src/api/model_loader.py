"""Model Loader and In-Memory Artifact Holder for FastAPI Serving.

Loads the registered model (models:/KRS_Dam_Inflow_Forecaster/1) through MLflow
once during application startup and provides thread-safe inference access.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

logger = logging.getLogger("krs_dam_api.loader")


def _windows_artifact_root(artifact_uri: str) -> Optional[Path]:
    """Extract the Windows-style POSIX-translated root from an MLflow artifact URI.

    On Windows, MLflow records artifact_uri values such as:
        file:///C:/Users/.../mlruns/<experiment_id>/<run_id>/artifacts

    When this URI is later loaded inside a Linux container, MLflow's url2pathname
    translates it to the POSIX path:
        /C:/Users/.../mlruns/<experiment_id>/<run_id>/artifacts

    This function robustly parses that URI (regardless of experiment/run IDs) and
    returns the mlruns root portion:
        /C:/Users/.../mlruns   (as a Path)

    Returns None if the URI does not look like a Windows FileStore path.
    """
    # Match: file:// + /DriveLetter:/rest/of/path
    # The drive letter pattern distinguishes Windows paths from pure POSIX paths
    match = re.match(r"file://(/[A-Za-z]:/.*)", artifact_uri)
    if not match:
        return None

    posix_artifact_path = match.group(1)  # e.g. /C:/Users/.../mlruns/<exp>/<run>/artifacts
    posix_path = Path(posix_artifact_path)

    # Walk upward from the artifact path to find the directory named "mlruns"
    # (or whichever directory is the tracking store root).  Stop after at most
    # 6 levels so we do not accidentally traverse the entire filesystem.
    candidate = posix_path
    for _ in range(6):
        if candidate.name in ("mlruns", "mlflow", "mlruns_rainfall_enhanced"):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    # Fallback: return two levels above the artifact dir
    # (artifact_uri typically ends in .../mlruns/<exp>/<run>/artifacts)
    try:
        return posix_path.parents[3]
    except IndexError:
        return None


def _ensure_filestore_artifact_symlink(tracking_uri: str) -> None:
    """Ensure Windows-recorded FileStore artifact paths resolve inside Linux / Docker containers.

    When MLflow logs runs on Windows, artifact_uri in meta.yaml can be recorded as
    file:///C:/Users/.../mlruns/.... When running inside a Linux Docker container where
    the mlruns directory is mounted at /app/mlruns, MLflow url2pathname translates
    this URI to /C:/Users/.../mlruns. This helper automatically creates the required
    compatibility symlink by parsing the actual artifact_uri from each run's meta.yaml,
    so the resolution is generic across any experiment ID, run ID, or project path.
    """
    if os.name != "posix":
        return

    # Resolve the mounted local store path from the tracking URI
    clean_uri = tracking_uri
    if clean_uri.startswith("file:///"):
        local_store_path = Path(clean_uri[7:]).resolve()
    elif clean_uri.startswith("file:/"):
        local_store_path = Path(clean_uri[5:]).resolve()
    elif clean_uri.startswith("file:"):
        local_store_path = Path(clean_uri[5:]).resolve()
    else:
        local_store_path = Path(clean_uri).resolve()

    if not local_store_path.exists():
        return

    # Scan run meta.yaml files inside any experiment directory.
    # The glob pattern matches: <store>/<experiment_id>/<run_id>/meta.yaml
    for meta_file in local_store_path.glob("*/[0-9a-f]*/meta.yaml"):
        try:
            text = meta_file.read_text(encoding="utf-8")
            # Extract artifact_uri line; value may use file:/// or file://
            uri_match = re.search(r"artifact_uri:\s*(\S+)", text)
            if not uri_match:
                continue

            artifact_uri = uri_match.group(1)
            windows_root = _windows_artifact_root(artifact_uri)
            if windows_root is None:
                # Not a Windows-style path; no symlink needed on this system
                break

            if not windows_root.exists():
                windows_root.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(local_store_path), str(windows_root))
                logger.info(
                    "Created compatibility symlink for MLflow artifacts: %s -> %s",
                    windows_root,
                    local_store_path,
                )
            # One symlink covers all experiments/runs in this store; stop scanning.
            break
        except Exception as exc:
            logger.warning("Could not create artifact compatibility symlink: %s", exc)


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
    tracking_uri: Optional[str] = None,
    metadata_path: Path = METADATA_PATH,
) -> ModelArtifactHolder:
    """Load the registered model Version 1 from MLflow Model Registry.

    Does NOT silently fall back to local disk joblib files.
    Fails explicitly if the registered model cannot be retrieved from MLflow.

    Returns:
        ModelArtifactHolder populated with the loaded pipeline and safe metadata.
    """
    effective_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    _ensure_filestore_artifact_symlink(effective_uri)

    mlflow.set_tracking_uri(effective_uri)
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
        client = get_registry_client(effective_uri)
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

