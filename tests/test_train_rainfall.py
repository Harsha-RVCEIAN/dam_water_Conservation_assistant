import numpy as np
import pandas as pd
import pytest

from src.models import train_rainfall


def test_experiment_feature_contract_is_27_features_and_excludes_target():
    assert len(train_rainfall.FEATURE_COLUMNS) == 19
    assert len(train_rainfall.RAINFALL_FEATURE_COLUMNS) == 8
    assert len(train_rainfall.LEARNED_FEATURE_COLUMNS) == 27
    assert train_rainfall.TARGET_COLUMN not in train_rainfall.LEARNED_FEATURE_COLUMNS


def test_split_dates_are_inclusive_and_non_overlapping():
    dates = pd.date_range("2022-12-31", "2025-01-01", freq="D")
    frame = pd.DataFrame({"calendar_date": dates})

    splits = train_rainfall._split_by_date(frame)

    assert splits["train"]["calendar_date"].max() == pd.Timestamp("2022-12-31")
    assert splits["validation"]["calendar_date"].min() == pd.Timestamp("2023-01-01")
    assert splits["validation"]["calendar_date"].max() == pd.Timestamp("2024-12-31")
    assert splits["test"]["calendar_date"].min() == pd.Timestamp("2025-01-01")
    assert [len(splits[key]) for key in ("train", "validation", "test")] == [
        1,
        731,
        1,
    ]


def test_rule_baselines_keep_original_formulas_and_do_not_use_rainfall():
    train = pd.DataFrame(
        {
            "day_of_year": [1, 2, 1],
            train_rainfall.TARGET_COLUMN: [10.0, 20.0, 30.0],
        }
    )
    validation = pd.DataFrame(
        {
            "day_of_year": [1, 3],
            "inflow_lag_1": [8.0, 9.0],
            "inflow_roll_mean_7": [7.0, np.nan],
            train_rainfall.TARGET_COLUMN: [0.0, 0.0],
        }
    )
    splits = {"train": train, "validation": validation, "test": validation}

    predictions = train_rainfall._baseline_predictions(splits)

    np.testing.assert_array_equal(predictions["persistence"]["validation"], [8, 9])
    np.testing.assert_array_equal(
        predictions["moving_average"]["validation"], [7, 9]
    )
    np.testing.assert_array_equal(
        predictions["seasonal_naive"]["validation"], [20, 9]
    )


def test_output_conflicts_are_detected_before_writing(monkeypatch, tmp_path):
    existing_output = tmp_path / "existing.joblib"
    existing_output.touch()
    monkeypatch.setattr(
        train_rainfall,
        "_output_paths",
        lambda: [existing_output],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_rainfall._assert_output_paths_available()
