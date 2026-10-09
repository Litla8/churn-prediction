"""Model factory. Optional libraries are imported lazily so a missing one only
disables that model instead of breaking the whole project."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.base import BaseEstimator

SUPPORTED_MODELS = ("logreg", "random_forest", "hist_gb", "xgboost", "lightgbm")


class ModelUnavailableError(RuntimeError):
    """Raised when a model's optional dependency is not installed."""


def imbalance_ratio(y: pd.Series) -> float:
    """Return negatives/positives, the value boosting libraries expect as scale_pos_weight."""
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("Cannot compute class ratio: no positive samples")
    return float((y == 0).sum()) / positives


def build_model(
    name: str,
    params: Dict[str, Any],
    random_state: int,
    scale_pos_weight: float = 1.0,
) -> BaseEstimator:
    """Instantiate a classifier by name.

    Class imbalance is handled with class weights (linear/tree models) or
    ``scale_pos_weight`` (boosters), so no resampling can ever leak into validation folds.

    Args:
        name: One of :data:`SUPPORTED_MODELS`.
        params: Hyperparameters from the config.
        random_state: Seed passed to the estimator.
        scale_pos_weight: Negative/positive ratio for XGBoost and LightGBM.

    Raises:
        ValueError: For an unknown model name.
        ModelUnavailableError: If the model's library is not installed.
    """
    merged = {**params, "random_state": random_state}
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**merged)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**merged)
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**merged)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelUnavailableError("xgboost is not installed") from exc
        return XGBClassifier(**merged, scale_pos_weight=scale_pos_weight)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ModelUnavailableError("lightgbm is not installed") from exc
        return LGBMClassifier(**merged, scale_pos_weight=scale_pos_weight)
    raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")
