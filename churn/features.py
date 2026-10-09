"""Feature engineering and preprocessing, expressed as sklearn transformers.

Everything is stateless or fitted inside a Pipeline so train and inference treatment
stay symmetric and nothing leaks from validation/test data into the fit.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import DataConfig

ENGINEERED_NUMERIC: Tuple[str, ...] = (
    "balance_salary_ratio",
    "products_per_tenure_year",
    "zero_balance",
)
_REQUIRED_INPUTS: Tuple[str, ...] = ("Balance", "EstimatedSalary", "NumOfProducts", "Tenure")


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features: balance vs income, product depth per year, empty-account flag."""

    @staticmethod
    def _check(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer expects a pandas DataFrame")
        missing = [c for c in _REQUIRED_INPUTS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for feature engineering: {missing}")

    def fit(self, X: pd.DataFrame, y: object = None) -> "ChurnFeatureEngineer":
        """Validate the input schema; the transformer itself is stateless."""
        self._check(X)
        # Setting this attribute marks the estimator as fitted for sklearn's checks.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the engineered columns appended."""
        self._check(X)
        out = X.copy()
        # Clip salary at 1 so a zero salary cannot create inf and poison the scaler.
        out["balance_salary_ratio"] = out["Balance"] / out["EstimatedSalary"].clip(lower=1.0)
        out["products_per_tenure_year"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)
        out["zero_balance"] = (out["Balance"] == 0).astype(int)
        return out


def build_preprocessor(cfg: DataConfig) -> ColumnTransformer:
    """Impute + scale numerics and impute + one-hot encode categoricals.

    One-hot encoding is used (rather than native categorical support) so the same
    matrix works for linear models, forests and both boosting libraries.
    """
    numeric = list(cfg.numeric_columns) + list(ENGINEERED_NUMERIC)
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, list(cfg.categorical_columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: BaseEstimator, cfg: DataConfig) -> Pipeline:
    """Assemble features -> preprocess -> model into one leakage-safe Pipeline."""
    return Pipeline(
        [
            ("features", ChurnFeatureEngineer()),
            ("preprocess", build_preprocessor(cfg)),
            ("model", model),
        ]
    )
