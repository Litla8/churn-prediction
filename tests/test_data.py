"""Tests for churn.data."""
from __future__ import annotations

import pandas as pd
import pytest

from churn.config import DataConfig
from churn.data import prepare_features, split_data, validate_raw


def test_validate_accepts_clean_frame(raw_df: pd.DataFrame) -> None:
    validate_raw(raw_df, DataConfig())  # must not raise


def test_validate_rejects_missing_column(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw(raw_df.drop(columns=["Age"]), DataConfig())


def test_validate_rejects_non_binary_target(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0, "Exited"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_raw(raw_df, DataConfig())


def test_validate_rejects_negative_balance(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0, "Balance"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate_raw(raw_df, DataConfig())


def test_prepare_drops_ids_and_leakage(raw_df: pd.DataFrame) -> None:
    X, y = prepare_features(raw_df, DataConfig())
    assert not {"RowNumber", "CustomerId", "Surname", "Complain", "Exited"} & set(X.columns)
    assert X.shape == (400, 13) and len(y) == 400


def test_prepare_guards_against_leakage_as_input(raw_df: pd.DataFrame) -> None:
    cfg = DataConfig(numeric_columns=DataConfig().numeric_columns + ("Complain",))
    with pytest.raises(ValueError, match="Leakage"):
        prepare_features(raw_df, cfg)


def test_prepare_removes_duplicate_customers(raw_df: pd.DataFrame) -> None:
    doubled = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
    X, _ = prepare_features(doubled, DataConfig())
    assert len(X) == 400


def test_split_is_stratified_and_disjoint(raw_df: pd.DataFrame) -> None:
    X, y = prepare_features(raw_df, DataConfig())
    s = split_data(X, y, DataConfig(), random_state=42)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == 400
    # 70/15/15 of 400 rows, allowing +-1 row of rounding inside train_test_split.
    assert abs(len(s.X_test) - 60) <= 1 and abs(len(s.X_val) - 60) <= 1
    assert abs(len(s.X_train) - 280) <= 2
    assert abs(s.y_train.mean() - y.mean()) < 0.02
    assert abs(s.y_test.mean() - y.mean()) < 0.03
    # Different splits must not share rows (checked via the reset index of X before splitting).
    assert not (set(s.X_train.index) & set(s.X_test.index))
    assert not (set(s.X_val.index) & set(s.X_test.index))


def test_split_rejects_bad_fractions(raw_df: pd.DataFrame) -> None:
    X, y = prepare_features(raw_df, DataConfig())
    with pytest.raises(ValueError, match="strictly between"):
        split_data(X, y, DataConfig(val_size=0.6, test_size=0.5), random_state=42)
