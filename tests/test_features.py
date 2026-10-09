"""Tests for churn.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.config import DataConfig
from churn.data import prepare_features
from churn.features import ENGINEERED_NUMERIC, ChurnFeatureEngineer, build_preprocessor


def test_engineered_values() -> None:
    X = pd.DataFrame({"Balance": [0.0, 5000.0], "EstimatedSalary": [1000.0, 0.0],
                      "NumOfProducts": [2, 1], "Tenure": [3, 0]})
    out = ChurnFeatureEngineer().fit_transform(X)
    assert out["zero_balance"].tolist() == [1, 0]
    assert out["products_per_tenure_year"].tolist() == [0.5, 1.0]
    # Zero salary is clipped to 1, so the ratio stays finite.
    assert np.isfinite(out["balance_salary_ratio"]).all()
    assert out["balance_salary_ratio"].iloc[1] == 5000.0


def test_engineer_requires_columns_and_dataframe() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        ChurnFeatureEngineer().fit(pd.DataFrame({"Balance": [1.0]}))
    with pytest.raises(TypeError):
        ChurnFeatureEngineer().fit(np.zeros((3, 3)))


def test_preprocessor_output_shape_and_no_nans(raw_df: pd.DataFrame) -> None:
    cfg = DataConfig()
    X, _ = prepare_features(raw_df, cfg)
    X.loc[0, "CreditScore"] = np.nan  # edge case: imputer must absorb missing values
    X.loc[1, "Geography"] = None
    features = ChurnFeatureEngineer().fit_transform(X)
    matrix = build_preprocessor(cfg).fit_transform(features)
    # numeric + engineered + one-hot (3 geography + 2 gender + 4 card types)
    expected_cols = len(cfg.numeric_columns) + len(ENGINEERED_NUMERIC) + 3 + 2 + 4
    assert matrix.shape == (400, expected_cols)
    assert not np.isnan(matrix).any()
