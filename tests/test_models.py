"""Tests for churn.models."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pandas as pd
import pytest

from churn.models import ModelUnavailableError, build_model, imbalance_ratio


def test_builds_sklearn_models_with_seed() -> None:
    model = build_model("logreg", {"max_iter": 50}, random_state=42)
    assert model.get_params()["random_state"] == 42


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("svm", {}, random_state=42)


def test_missing_optional_library_is_reported() -> None:
    # Setting a module to None in sys.modules makes `import xgboost` raise ImportError.
    with patch.dict(sys.modules, {"xgboost": None}):
        with pytest.raises(ModelUnavailableError):
            build_model("xgboost", {}, random_state=42)


def test_imbalance_ratio() -> None:
    assert imbalance_ratio(pd.Series([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError):
        imbalance_ratio(pd.Series([0, 0]))
