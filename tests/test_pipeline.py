"""End-to-end test of churn.pipeline with the dataset load mocked out."""
from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import joblib
import pandas as pd
import pytest

from churn.config import PipelineConfig, load_config
from churn.pipeline import run_pipeline


def test_pipeline_writes_all_artifacts(raw_df: pd.DataFrame, cfg: PipelineConfig) -> None:
    with patch("churn.pipeline.load_raw", return_value=raw_df) as mocked:
        payload = run_pipeline(cfg)
    mocked.assert_called_once()
    assert 0.5 < payload["test_metrics_at_business_threshold"]["roc_auc"] <= 1.0
    for name in ("roc_pr_test.png", "confusion_matrix_test.png", "threshold_vs_net_benefit.png"):
        assert (cfg.paths.figures_dir / name).is_file()
    saved = json.loads((cfg.paths.reports_dir / "metrics.json").read_text())
    assert saved["random_state"] == 42 and "hyperparameters" in saved
    model = joblib.load(cfg.paths.models_dir / "churn_pipeline.joblib")
    # The saved pipeline accepts raw feature columns and returns one probability pair per row.
    features = raw_df[list(cfg.data.numeric_columns) + list(cfg.data.categorical_columns)]
    assert model.predict_proba(features.head(7)).shape == (7, 2)


def test_pipeline_is_idempotent(raw_df: pd.DataFrame, cfg: PipelineConfig) -> None:
    with patch("churn.pipeline.load_raw", return_value=raw_df):
        first = run_pipeline(cfg)
        second = run_pipeline(cfg)  # re-run into the same directories, no cleanup in between
    assert first["test_metrics_at_business_threshold"] == second["test_metrics_at_business_threshold"]
    assert first["decision_threshold"] == second["decision_threshold"]


def test_pipeline_rejects_unknown_model_names(raw_df: pd.DataFrame, cfg: PipelineConfig) -> None:
    cfg = replace(cfg, train=replace(cfg.train, models=("logreg", "definitely_missing_ok")))
    # An unknown name is a configuration error and must fail loudly, not be skipped.
    with patch("churn.pipeline.load_raw", return_value=raw_df):
        with pytest.raises(ValueError, match="Unknown model"):
            run_pipeline(cfg)


def test_load_config_merges_yaml(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  cv_folds: 3\nmodel_params:\n  logreg:\n    C: 0.5\n")
    cfg = load_config(path)
    assert cfg.train.cv_folds == 3
    assert cfg.model_params["logreg"]["C"] == 0.5
    assert cfg.model_params["logreg"]["max_iter"] == 2000  # untouched default survives
