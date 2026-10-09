"""Tests for churn.explain (pure helpers always run; the shap call needs the shap package)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import DataConfig
from churn.data import prepare_features
from churn.explain import compute_shap, drivers_markdown, importance_table
from churn.features import build_pipeline

NAMES = ["Age", "IsActiveMember", "Geography_Germany", "Constant"]


def _fake_arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 4))
    x[:, 2] = (x[:, 2] > 0).astype(float)
    x[:, 3] = 1.0  # constant column -> zero variance edge case
    shap_values = np.column_stack(
        [0.10 * x[:, 0], -0.05 * x[:, 1], 0.08 * x[:, 2], np.zeros(200)]
    )
    return shap_values, x


def test_importance_ranks_and_signs_directions() -> None:
    values, x = _fake_arrays()
    table = importance_table(values, x, NAMES)
    assert table["feature"].iloc[0] == "Age"
    by_name = table.set_index("feature")["direction_corr"]
    assert by_name["Age"] > 0.99 and by_name["IsActiveMember"] < -0.99
    assert by_name["Constant"] == 0.0 and not table.isna().any().any()


def test_importance_rejects_shape_mismatch() -> None:
    values, x = _fake_arrays()
    with pytest.raises(ValueError, match="same matrix"):
        importance_table(values[:, :3], x, NAMES)


def test_drivers_markdown_sentences() -> None:
    values, x = _fake_arrays()
    text = drivers_markdown(importance_table(values, x, NAMES), ("Geography", "Gender"), top_k=3)
    assert "Higher Age raises predicted churn risk" in text
    assert "Higher IsActiveMember lowers predicted churn risk" in text
    assert "Geography = Germany raises predicted churn risk" in text
    assert "not proven causes" in text


def test_compute_shap_shapes(raw_df: pd.DataFrame) -> None:
    pytest.importorskip("shap")
    X, y = prepare_features(raw_df, DataConfig())
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=42), DataConfig()).fit(X, y)
    result = compute_shap(pipe, X.head(30), X.tail(5), seed=42)
    assert result.values.shape == (5, len(result.feature_names))
    assert result.encoded.shape == result.values.shape
    # Attributions plus the baseline should reproduce each predicted probability.
    reconstructed = result.values.sum(axis=1) + result.base_value
    assert np.allclose(reconstructed, pipe.predict_proba(X.tail(5))[:, 1], atol=0.05)
