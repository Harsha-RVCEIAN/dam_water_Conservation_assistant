"""Tests for centralized unit conversion module."""

import math
import numpy as np
import pandas as pd
import pytest

from src.utils.unit_conversions import (
    CUSEC_DAY_TO_TMC_FACTOR,
    cusecs_to_tmc,
    tmc_to_cusecs,
)


def test_one_cusec_over_24_hours():
    res = cusecs_to_tmc(1.0, hours=24.0)
    assert abs(res - CUSEC_DAY_TO_TMC_FACTOR) < 1e-12
    assert abs(res - 0.0000864) < 1e-12


def test_non_24_hour_duration():
    # 12 hours should be exactly half of 24 hours
    res_12 = cusecs_to_tmc(1000.0, hours=12.0)
    res_24 = cusecs_to_tmc(1000.0, hours=24.0)
    assert abs(res_12 - (res_24 / 2.0)) < 1e-10


def test_missing_flow_preserves_nan():
    res = cusecs_to_tmc(float("nan"))
    assert math.isnan(res)

    res_none = cusecs_to_tmc(None)
    assert math.isnan(res_none)


def test_negative_flow_raises_value_error():
    with pytest.raises(ValueError, match="Flow rate cannot be negative"):
        cusecs_to_tmc(-10.0)


def test_invalid_hours_raises_value_error():
    with pytest.raises(ValueError, match="Duration hours must be positive"):
        cusecs_to_tmc(100.0, hours=0.0)

    with pytest.raises(ValueError, match="Duration hours must be positive"):
        cusecs_to_tmc(100.0, hours=-5.0)


def test_tmc_to_cusecs_inverse():
    flow = 5432.1
    vol = cusecs_to_tmc(flow, hours=24.0)
    recovered = tmc_to_cusecs(vol, hours=24.0)
    assert abs(recovered - flow) < 1e-6


def test_vectorized_conversion():
    series = pd.Series([0.0, 1000.0, float("nan")])
    res = cusecs_to_tmc(series, hours=24.0)
    assert abs(res.iloc[0] - 0.0) < 1e-12
    assert abs(res.iloc[1] - 0.0864) < 1e-12
    assert math.isnan(res.iloc[2])

