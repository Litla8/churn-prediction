"""Cross-validated model comparison and final model fitting."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import PipelineConfig
from churn.data import DataSplits
from churn.features import build_pipeline
from churn.models import SUPPORTED_MODELS, ModelUnavailableError, build_model, imbalance_ratio

logger = logging.getLogger(__name__)

CV_SCORING: Dict[str, str] = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "f1": "f1",
}


def cross_validate_models(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: PipelineConfig
) -> pd.DataFrame:
    """Compare every configured model with stratified K-fold CV on the training set only.

    Preprocessing sits inside each Pipeline, so scalers/encoders are re-fitted per fold
    and nothing leaks across folds. Boosters run here without early stopping (there is
    no per-fold validation set); early stopping is applied in :func:`fit_final_model`.

    Returns:
        DataFrame indexed by model name, sorted by mean CV ROC-AUC (best first).

    Raises:
        ValueError: If ``train.models`` contains a name that is not supported.
        RuntimeError: If no configured model could be built.
    """
    unknown = [n for n in cfg.train.models if n not in SUPPORTED_MODELS]
    if unknown:  # a typo is a config error and must fail loudly, not be silently skipped
        raise ValueError(f"Unknown model(s) {unknown}; choose from {SUPPORTED_MODELS}")
    cv = StratifiedKFold(n_splits=cfg.train.cv_folds, shuffle=True,
                         random_state=cfg.train.random_state)
    spw = imbalance_ratio(y_train)
    rows: List[Dict[str, Any]] = []
    for name in cfg.train.models:
        try:
            model = build_model(name, cfg.model_params[name], cfg.train.random_state, spw)
        except ModelUnavailableError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue
        scores = cross_validate(
            build_pipeline(model, cfg.data), X_train, y_train,
            cv=cv, scoring=CV_SCORING, n_jobs=1, error_score="raise",
        )
        row: Dict[str, Any] = {"model": name}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std())
        rows.append(row)
        logger.info("CV %-14s ROC-AUC %.4f | PR-AUC %.4f | recall@0.5 %.4f",
                    name, row["cv_roc_auc_mean"], row["cv_pr_auc_mean"], row["cv_recall_mean"])
    if not rows:
        raise RuntimeError("No model could be trained; check train.models and installed packages")
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("cv_roc_auc_mean", ascending=False)


def fit_final_model(name: str, splits: DataSplits, cfg: PipelineConfig) -> Pipeline:
    """Fit the chosen model on the training set (early stopping on validation for boosters).

    Args:
        name: Model name selected by cross-validation.
        splits: Train/validation/test partitions.
        cfg: Pipeline configuration.

    Returns:
        A fitted end-to-end Pipeline (features -> preprocess -> model).
    """
    model = build_model(name, cfg.model_params[name], cfg.train.random_state,
                        imbalance_ratio(splits.y_train))
    pipeline = build_pipeline(model, cfg.data)
    if name not in {"xgboost", "lightgbm"}:
        return pipeline.fit(splits.X_train, splits.y_train)

    # Boosters need the validation set pre-transformed to enforce early stopping.
    prep = Pipeline(pipeline.steps[:-1])
    X_fit = prep.fit_transform(splits.X_train, splits.y_train)
    X_val = prep.transform(splits.X_val)
    rounds = cfg.train.early_stopping_rounds
    if name == "xgboost":
        model.set_params(early_stopping_rounds=rounds)
        model.fit(X_fit, splits.y_train, eval_set=[(X_val, splits.y_val)], verbose=False)
    else:
        import lightgbm as lgb

        model.fit(X_fit, splits.y_train, eval_set=[(X_val, splits.y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(rounds, verbose=False)])
    return Pipeline([*prep.steps, ("model", model)])
