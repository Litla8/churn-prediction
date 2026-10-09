"""Customer-level scoring helpers shared by the Streamlit app and batch scripts.

Kept free of Streamlit imports so every function here is unit-testable.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import DataConfig

logger = logging.getLogger(__name__)

# Domain values offered in the app's drop-downs (not hyperparameters).
CATEGORY_LEVELS: Dict[str, Tuple[str, ...]] = {
    "Geography": ("France", "Germany", "Spain"),
    "Gender": ("Female", "Male"),
    "Card Type": ("DIAMOND", "GOLD", "PLATINUM", "SILVER"),
}
_NON_NEGATIVE = ("Age", "Tenure", "Balance", "EstimatedSalary")


def model_input_columns(cfg: DataConfig) -> List[str]:
    """Return the raw columns the saved pipeline expects, in training order."""
    return list(cfg.numeric_columns) + list(cfg.categorical_columns)


def build_customer_frame(inputs: Mapping[str, object], cfg: DataConfig) -> pd.DataFrame:
    """Turn one customer's form values into a validated single-row dataframe.

    Raises:
        ValueError: If a required field is missing, non-numeric, or negative where
            that is impossible.
    """
    columns = model_input_columns(cfg)
    missing = [c for c in columns if c not in inputs]
    if missing:
        raise ValueError(f"Missing customer fields: {missing}")
    frame = pd.DataFrame([{c: inputs[c] for c in columns}])
    for column in cfg.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in _NON_NEGATIVE:
        if column in frame.columns and (frame[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")
    return frame


def validate_batch(df: pd.DataFrame, cfg: DataConfig) -> None:
    """Fail loudly if an uploaded batch lacks columns the model needs.

    Raises:
        ValueError: Listing the missing columns.
    """
    missing = [c for c in model_input_columns(cfg) if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")


def assign_risk_tiers(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Map probabilities to High / Medium / Low.

    High = at or above the business threshold (gets an offer); Medium = at least half the
    threshold (worth watching); Low = everything else.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    proba = np.asarray(proba, dtype=float)
    return np.select([proba >= threshold, proba >= threshold / 2.0], ["High", "Medium"], "Low")


def score_customers(
    pipeline: Pipeline, frame: pd.DataFrame, threshold: float, cfg: DataConfig
) -> pd.DataFrame:
    """Score customers and append probability, risk tier and offer flag.

    Extra columns in ``frame`` (e.g. CustomerId) are passed through untouched.
    """
    validate_batch(frame, cfg)
    proba = pipeline.predict_proba(frame[model_input_columns(cfg)])[:, 1]
    out = frame.copy()
    out["churn_probability"] = proba
    out["risk_tier"] = assign_risk_tiers(proba, threshold)
    out["flag_for_offer"] = proba >= threshold
    return out


def recommend_actions(inputs: Mapping[str, object], tier: str) -> List[str]:
    """Rule-based retention suggestions for flagged or watch-list customers.

    These are heuristics drawn from the EDA segment rates (inactive members, Germany,
    3+ products and ages 41-60 churn far more often). They are correlations, not proven
    causes: validate any offer with an A/B test before rolling it out.
    """
    if tier == "Low":
        return ["No retention offer needed; keep routine engagement."]
    actions: List[str] = []
    if int(inputs.get("IsActiveMember", 1)) == 0:
        actions.append("Run a re-engagement campaign: inactive members churn roughly twice as often.")
    if int(inputs.get("NumOfProducts", 1)) >= 3:
        actions.append("Review product fit: customers holding 3+ products churned at very high rates.")
    if inputs.get("Geography") == "Germany":
        actions.append("Prioritise for a local-market retention review: Germany churns about twice as often.")
    if 41 <= float(inputs.get("Age", 0)) <= 60:
        actions.append("Offer a relationship-manager call: ages 41-60 are the highest-risk band.")
    if not actions:
        actions.append("Offer the standard retention incentive.")
    if tier == "Medium":
        actions.insert(0, "Watch list: below the offer threshold, so monitor before spending on an offer.")
    return actions
