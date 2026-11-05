"""Centralized hydrological unit conversion module for K.R.Sagara Dam project.

Converts flow rates (cusecs = cu ft/s) to volumetric amounts (TMC = thousand million cubic feet)
over specified durations (hours), and validates input parameters.

Formula:
  volume_tmc = flow_cusecs * 3600 * hours / 1_000_000_000
  For 24 hours: 1 cusec * 86,400 s / 1e9 = 0.0000864 TMC.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd

CUSEC_DAY_TO_TMC_FACTOR: float = 0.0000864
SECONDS_PER_HOUR: float = 3600.0
CUBIC_FEET_PER_TMC: float = 1_000_000_000.0


def cusecs_to_tmc(
    flow_cusecs: Union[float, int, np.ndarray, pd.Series],
    hours: float = 24.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Convert a mean flow rate in cusecs over given hours into volume in TMC.

    Args:
        flow_cusecs: Flow rate in cubic feet per second (cusecs).
        hours: Elapsed duration in hours (defaults to 24.0).

    Returns:
        Volume in TMC. Preserves NaN if input is NaN/None.

    Raises:
        ValueError: If hours is non-positive or flow_cusecs contains negative values.
    """
    if hours <= 0:
        raise ValueError(f"Duration hours must be positive, got {hours}")

    conversion_factor = (SECONDS_PER_HOUR * hours) / CUBIC_FEET_PER_TMC

    if isinstance(flow_cusecs, (pd.Series, np.ndarray)):
        arr = np.asarray(flow_cusecs, dtype=float)
        # Check for negative values excluding NaN
        if np.nanmin(arr) < 0:
            raise ValueError("Flow rate cannot be negative.")
        res = arr * conversion_factor
        if isinstance(flow_cusecs, pd.Series):
            return pd.Series(res, index=flow_cusecs.index, name=flow_cusecs.name)
        return res

    if flow_cusecs is None or (isinstance(flow_cusecs, float) and math.isnan(flow_cusecs)):
        return float("nan")

    if flow_cusecs < 0:
        raise ValueError(f"Flow rate cannot be negative: {flow_cusecs}")

    return float(flow_cusecs * conversion_factor)


def tmc_to_cusecs(
    volume_tmc: Union[float, int, np.ndarray, pd.Series],
    hours: float = 24.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Convert volume in TMC over given hours into mean flow rate in cusecs.

    Args:
        volume_tmc: Volume in TMC.
        hours: Elapsed duration in hours (defaults to 24.0).

    Returns:
        Equivalent flow rate in cusecs.

    Raises:
        ValueError: If hours is non-positive or volume_tmc contains negative values.
    """
    if hours <= 0:
        raise ValueError(f"Duration hours must be positive, got {hours}")

    conversion_factor = CUBIC_FEET_PER_TMC / (SECONDS_PER_HOUR * hours)

    if isinstance(volume_tmc, (pd.Series, np.ndarray)):
        arr = np.asarray(volume_tmc, dtype=float)
        if np.nanmin(arr) < 0:
            raise ValueError("Volume cannot be negative.")
        res = arr * conversion_factor
        if isinstance(volume_tmc, pd.Series):
            return pd.Series(res, index=volume_tmc.index, name=volume_tmc.name)
        return res

    if volume_tmc is None or (isinstance(volume_tmc, float) and math.isnan(volume_tmc)):
        return float("nan")

    if volume_tmc < 0:
        raise ValueError(f"Volume cannot be negative: {volume_tmc}")

    return float(volume_tmc * conversion_factor)

