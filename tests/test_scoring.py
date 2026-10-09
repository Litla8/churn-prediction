"""Tests for churn.scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import DataConfig
from churn.data import prepare_features
from churn.features import build_pipeline
from churn.scoring import (
    assign_risk_tiers,
    build_customer_frame,
    model_input_columns,
    recommend_actions,
    score_customers,
    validate_batch,
)

CUSTOMER = {
    "Geography": "Germany", "Gender": "Female", "Age": 52, "CreditScore": 600, "Tenure": 4,
    "Balance": 90000.0, "NumOfProducts": 3, "EstimatedSalary": 50000.0, "Card Type": "GOLD",
    "Satisfaction Score": 3, "Point Earned": 500, "HasCrCard": 1, "IsActiveMember": 0,
}


@pytest.fixture
def fitted(raw_df: pd.DataFrame):
    X, y = prepare_features(raw_df, DataConfig())
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=42), DataConfig())
    return pipe.fit(X, y), X


def test_customer_frame_shape_and_order() -> None:
    frame = build_customer_frame(CUSTOMER, DataConfig())
    assert frame.shape == (1, 13)
    assert list(frame.columns) == model_input_columns(DataConfig())


def test_customer_frame_rejects_missing_and_negative() -> None:
    incomplete = {k: v for k, v in CUSTOMER.items() if k != "Age"}
    with pytest.raises(ValueError, match="Missing customer fields"):
        build_customer_frame(incomplete, DataConfig())
    with pytest.raises(ValueError, match="negative"):
        build_customer_frame({**CUSTOMER, "Balance": -1.0}, DataConfig())


def test_risk_tiers_boundaries() -> None:
    tiers = assign_risk_tiers(np.array([0.9, 0.7, 0.4, 0.34, 0.1]), threshold=0.7)
    assert tiers.tolist() == ["High", "High", "Medium", "Low", "Low"]
    with pytest.raises(ValueError):
        assign_risk_tiers(np.array([0.5]), threshold=1.0)


def test_score_customers_appends_columns_and_keeps_ids(fitted) -> None:
    pipe, X = fitted
    batch = X.head(5).assign(CustomerId=range(5))
    out = score_customers(pipe, batch, 0.5, DataConfig())
    assert {"churn_probability", "risk_tier", "flag_for_offer", "CustomerId"} <= set(out.columns)
    assert out["churn_probability"].between(0, 1).all()
    assert (out["flag_for_offer"] == (out["churn_probability"] >= 0.5)).all()


def test_validate_batch_lists_missing_columns(fitted) -> None:
    _, X = fitted
    with pytest.raises(ValueError, match="Age"):
        validate_batch(X.drop(columns=["Age"]), DataConfig())


def test_recommendations_by_tier() -> None:
    assert recommend_actions(CUSTOMER, "Low") == ["No retention offer needed; keep routine engagement."]
    high = " ".join(recommend_actions(CUSTOMER, "High"))
    assert "re-engagement" in high and "Germany" in high and "relationship-manager" in high
    medium = recommend_actions({**CUSTOMER, "IsActiveMember": 1, "NumOfProducts": 1,
                                "Geography": "France", "Age": 30}, "Medium")
    assert medium[0].startswith("Watch list") and "standard retention incentive" in medium[1]
