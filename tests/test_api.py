"""Tests for churn.api (skipped automatically when fastapi/httpx are not installed)."""
from __future__ import annotations

import json
import typing
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression

from churn.config import DataConfig
from churn.data import prepare_features
from churn.features import build_pipeline

PAYLOAD = {
    "geography": "Germany", "gender": "Female", "age": 52, "credit_score": 650, "tenure": 5,
    "balance": 76000.0, "num_of_products": 1, "has_cr_card": 1, "is_active_member": 0,
    "estimated_salary": 100000.0, "card_type": "DIAMOND", "satisfaction_score": 3,
    "point_earned": 600,
}


@pytest.fixture
def artifacts(raw_df: pd.DataFrame, tmp_path: Path) -> tuple[Path, Path]:
    """Train a tiny model and save it with a metrics file, like the real pipeline does."""
    X, y = prepare_features(raw_df, DataConfig())
    pipeline = build_pipeline(LogisticRegression(max_iter=500, random_state=42), DataConfig())
    pipeline.fit(X, y)
    model_path, metrics_path = tmp_path / "model.joblib", tmp_path / "metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps({
        "selected_model": "logreg", "decision_threshold": 0.5,
        "environment": {"sklearn": sklearn.__version__},
    }))
    return model_path, metrics_path


@pytest.fixture
def client(artifacts: tuple[Path, Path]):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from churn.api import create_app

    return TestClient(create_app(model_path=artifacts[0], metrics_path=artifacts[1]))


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "logreg"
    assert body["sklearn_trained"] == body["sklearn_runtime"]


def test_predict_returns_valid_result(client) -> None:
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"High", "Medium", "Low"}
    assert body["flag_for_offer"] == (body["churn_probability"] >= body["threshold"])
    assert isinstance(body["suggested_actions"], list) and body["suggested_actions"]


def test_predict_rejects_invalid_input(client) -> None:
    assert client.post("/predict", json={**PAYLOAD, "age": 5}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "geography": "Mars"}).status_code == 422
    missing = {k: v for k, v in PAYLOAD.items() if k != "balance"}
    assert client.post("/predict", json=missing).status_code == 422


def test_batch_keeps_order_and_size(client) -> None:
    customers = [PAYLOAD, {**PAYLOAD, "age": 25, "is_active_member": 1}, {**PAYLOAD, "age": 60}]
    body = client.post("/predict/batch", json={"customers": customers}).json()
    assert len(body["predictions"]) == 3
    single = client.post("/predict", json=customers[0]).json()
    assert body["predictions"][0]["churn_probability"] == single["churn_probability"]


def test_batch_rejects_empty_list(client) -> None:
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_missing_model_fails_fast(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from churn.api import create_app

    with pytest.raises(FileNotFoundError):
        create_app(model_path=tmp_path / "nope.joblib", metrics_path=tmp_path / "nope.json")


def test_schema_levels_match_scoring_constants() -> None:
    pytest.importorskip("fastapi")
    from churn.api import CustomerIn
    from churn.scoring import CATEGORY_LEVELS

    for field, column in (("geography", "Geography"), ("gender", "Gender"),
                          ("card_type", "Card Type")):
        levels = typing.get_args(CustomerIn.model_fields[field].annotation)
        assert set(levels) == set(CATEGORY_LEVELS[column])
