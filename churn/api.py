"""FastAPI scoring service for the churn model.

Run locally:  uvicorn churn.api:create_app --factory --port 8000
Interactive docs are served at /docs. The model and threshold are loaded once at startup
and stored on ``app.state`` (no module-level globals).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from churn.config import DataConfig, PipelineConfig, load_config
from churn.scoring import recommend_actions, score_customers

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE: int = 1000

# JSON uses snake_case; the trained pipeline expects the original dataset column names.
FIELD_TO_COLUMN: Dict[str, str] = {
    "geography": "Geography",
    "gender": "Gender",
    "age": "Age",
    "credit_score": "CreditScore",
    "tenure": "Tenure",
    "balance": "Balance",
    "num_of_products": "NumOfProducts",
    "has_cr_card": "HasCrCard",
    "is_active_member": "IsActiveMember",
    "estimated_salary": "EstimatedSalary",
    "card_type": "Card Type",
    "satisfaction_score": "Satisfaction Score",
    "point_earned": "Point Earned",
}


class CustomerIn(BaseModel):
    """One customer's raw attributes. Out-of-range values are rejected with HTTP 422."""

    geography: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]
    age: int = Field(ge=18, le=100)
    credit_score: int = Field(ge=300, le=900)
    tenure: int = Field(ge=0, le=10)
    balance: float = Field(ge=0)
    num_of_products: int = Field(ge=1, le=4)
    has_cr_card: int = Field(ge=0, le=1)
    is_active_member: int = Field(ge=0, le=1)
    estimated_salary: float = Field(ge=0)
    card_type: Literal["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
    satisfaction_score: int = Field(ge=1, le=5)
    point_earned: int = Field(ge=0)


class PredictionOut(BaseModel):
    """Scoring result for one customer."""

    churn_probability: float
    risk_tier: Literal["High", "Medium", "Low"]
    flag_for_offer: bool
    threshold: float
    suggested_actions: List[str]


class BatchIn(BaseModel):
    """Several customers to score in one request."""

    customers: List[CustomerIn] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchOut(BaseModel):
    """Predictions in the same order as the request."""

    predictions: List[PredictionOut]


class HealthOut(BaseModel):
    """Service status and the library versions that matter for unpickling the model."""

    status: str
    model: str
    decision_threshold: float
    sklearn_trained: Optional[str]
    sklearn_runtime: str


def to_model_inputs(customer: CustomerIn) -> Dict[str, Any]:
    """Convert a request body into a dict keyed by the pipeline's column names."""
    return {FIELD_TO_COLUMN[key]: value for key, value in customer.model_dump().items()}


@dataclass(frozen=True)
class ScoringService:
    """Everything needed to score customers, bundled so it can live on ``app.state``."""

    pipeline: Pipeline
    threshold: float
    model_name: str
    data_cfg: DataConfig
    trained_sklearn: Optional[str]

    def score(self, customers: Sequence[CustomerIn]) -> List[PredictionOut]:
        """Score customers in order and attach rule-based retention suggestions."""
        rows = [to_model_inputs(c) for c in customers]
        scored = score_customers(self.pipeline, pd.DataFrame(rows), self.threshold, self.data_cfg)
        results = scored[["churn_probability", "risk_tier", "flag_for_offer"]]
        return [
            PredictionOut(
                churn_probability=float(row.churn_probability),
                risk_tier=str(row.risk_tier),
                flag_for_offer=bool(row.flag_for_offer),
                threshold=self.threshold,
                suggested_actions=recommend_actions(inputs, str(row.risk_tier)),
            )
            for inputs, row in zip(rows, results.itertuples(index=False))
        ]


def load_service(model_path: Path, metrics_path: Path, cfg: PipelineConfig) -> ScoringService:
    """Load the saved pipeline and its decision threshold.

    Raises:
        FileNotFoundError: If either artifact is missing (fail fast at startup).
    """
    for path in (model_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trained = metrics.get("environment", {}).get("sklearn")
    if trained and trained != sklearn.__version__:
        # Pickled sklearn models can misbehave across versions, so make the mismatch visible.
        logger.warning("Model trained with scikit-learn %s but running %s",
                       trained, sklearn.__version__)
    return ScoringService(
        pipeline=joblib.load(model_path),
        threshold=float(metrics["decision_threshold"]),
        model_name=str(metrics["selected_model"]),
        data_cfg=cfg.data,
        trained_sklearn=trained,
    )


def _service(request: Request) -> ScoringService:
    """Fetch the scoring service stored on the application."""
    return request.app.state.service


def create_app(
    model_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    cfg: Optional[PipelineConfig] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Paths default to the config values and can be overridden with the CHURN_MODEL_PATH and
    CHURN_METRICS_PATH environment variables (used by the Docker image).
    """
    cfg = cfg or load_config()
    model_path = Path(model_path or os.environ.get(
        "CHURN_MODEL_PATH", cfg.paths.models_dir / "churn_pipeline.joblib"))
    metrics_path = Path(metrics_path or os.environ.get(
        "CHURN_METRICS_PATH", cfg.paths.reports_dir / "metrics.json"))

    app = FastAPI(title="Bank Churn Scoring API", version="1.0.0",
                  description="Churn probability, risk tier and retention suggestions.")
    app.state.service = load_service(model_path, metrics_path, cfg)

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        service = _service(request)
        return HealthOut(status="ok", model=service.model_name,
                         decision_threshold=service.threshold,
                         sklearn_trained=service.trained_sklearn,
                         sklearn_runtime=sklearn.__version__)

    @app.post("/predict", response_model=PredictionOut)
    def predict(customer: CustomerIn, request: Request) -> PredictionOut:
        try:
            return _service(request).score([customer])[0]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn, request: Request) -> BatchOut:
        try:
            return BatchOut(predictions=_service(request).score(batch.customers))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
