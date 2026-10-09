"""Deterministic synthetic fixtures shared by the test suite (no real data needed)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import PipelineConfig


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """400-row synthetic frame with the same schema as Customer-Churn-Records.csv."""
    rng = np.random.default_rng(42)
    n = 400
    age = rng.integers(18, 80, n)
    active = rng.integers(0, 2, n)
    balance = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(1_000, 200_000, n))
    # Churn probability rises with age and falls for active members -> learnable signal.
    p = 1.0 / (1.0 + np.exp(-(-1.6 + 0.06 * (age - 38) - 0.8 * active)))
    exited = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": 15_000_000 + np.arange(n),
            "Surname": ["Smith"] * n,
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Female", "Male"], n),
            "Age": age,
            "Tenure": rng.integers(0, 11, n),
            "Balance": balance,
            "NumOfProducts": rng.integers(1, 4, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": active,
            "EstimatedSalary": rng.uniform(10_000, 190_000, n),
            "Exited": exited,
            "Complain": exited,  # deliberate leak, must be dropped by the pipeline
            "Satisfaction Score": rng.integers(1, 6, n),
            "Card Type": rng.choice(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], n),
            "Point Earned": rng.integers(100, 1000, n),
        }
    )


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    """Fast config writing every artifact under pytest's tmp_path."""
    base = PipelineConfig()
    return replace(
        base,
        paths=replace(base.paths, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports"),
        train=replace(base.train, cv_folds=3, models=("logreg", "random_forest")),
        model_params={**base.model_params,
                      "random_forest": {**base.model_params["random_forest"], "n_estimators": 20}},
    )
