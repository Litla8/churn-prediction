"""End-to-end orchestration: load -> validate -> split -> compare -> tune threshold -> report."""
from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn.config import PipelineConfig, load_config
from churn.data import load_raw, prepare_features, split_data, validate_raw
from churn.evaluate import (
    business_summary,
    compute_metrics,
    plot_confusion,
    plot_roc_pr,
    plot_threshold_curve,
    select_threshold,
)
from churn.train import cross_validate_models, fit_final_model

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Serialise numpy scalars/arrays and other non-JSON types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    """Write ``payload`` as pretty JSON, overwriting any previous run (idempotent)."""
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def configure_logging(level: str) -> None:
    """Configure root logging once, with a dynamic level (INFO / DEBUG / ...)."""
    logging.basicConfig(level=getattr(logging, level.upper()), force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """Run the full churn workflow and persist all artifacts.

    Args:
        cfg: Fully resolved configuration.

    Returns:
        The metrics payload that was also written to ``reports/metrics.json``.
    """
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg.paths.raw_data)
    validate_raw(df, cfg.data)
    X, y = prepare_features(df, cfg.data)
    splits = split_data(X, y, cfg.data, cfg.train.random_state)

    cv_table = cross_validate_models(splits.X_train, splits.y_train, cfg)
    best_name = str(cv_table.index[0])
    logger.info("Selected model: %s", best_name)
    pipeline = fit_final_model(best_name, splits, cfg)

    # Threshold is tuned on validation data; the test set is used exactly once below.
    val_proba = pipeline.predict_proba(splits.X_val)[:, 1]
    threshold, sweep = select_threshold(splits.y_val.to_numpy(), val_proba, cfg.cost)

    y_test = splits.y_test.to_numpy()
    test_proba = pipeline.predict_proba(splits.X_test)[:, 1]
    at_business = compute_metrics(y_test, test_proba, threshold)
    at_default = compute_metrics(y_test, test_proba, 0.5)

    figures = cfg.paths.figures_dir
    plot_roc_pr(y_test, test_proba, figures / "roc_pr_test.png")
    plot_confusion(at_business, figures / "confusion_matrix_test.png")
    plot_threshold_curve(sweep, threshold, figures / "threshold_vs_net_benefit.png")

    joblib.dump(pipeline, cfg.paths.models_dir / "churn_pipeline.joblib")
    sweep.to_csv(cfg.paths.reports_dir / "threshold_sweep_validation.csv", index=False)

    payload: Dict[str, Any] = {
        "selected_model": best_name,
        "random_state": cfg.train.random_state,
        "decision_threshold": threshold,
        "cv_results": cv_table.reset_index().to_dict(orient="records"),
        "test_metrics_at_business_threshold": at_business,
        "test_metrics_at_0.5": at_default,
        "business_impact_test_set": business_summary(y_test, test_proba, threshold, cfg.cost),
        "meets_pdf_targets": {
            f"roc_auc>{cfg.train.target_roc_auc}": at_business["roc_auc"] > cfg.train.target_roc_auc,
            f"recall>{cfg.train.target_recall}_at_business_threshold":
                at_business["recall"] > cfg.train.target_recall,
        },
        "hyperparameters": pipeline.named_steps["model"].get_params(),
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(payload, cfg.paths.reports_dir / "metrics.json")
    logger.info("Test ROC-AUC %.4f | recall %.4f | precision %.4f @ threshold %.2f",
                at_business["roc_auc"], at_business["recall"], at_business["precision"], threshold)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the bank churn model.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML overrides")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING ...")
    parser.add_argument("--include-complain", action="store_true",
                        help="Demonstrate leakage: add the Complain column as a feature")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = cfg.paths
    if args.data_path:
        paths = replace(paths, raw_data=args.data_path)
    if args.models_dir:
        paths = replace(paths, models_dir=args.models_dir)
    if args.reports_dir:
        paths = replace(paths, reports_dir=args.reports_dir)
    data = cfg.data
    if args.include_complain:
        data = replace(data, leakage_columns=(),
                       numeric_columns=data.numeric_columns + ("Complain",))
    train = replace(cfg.train, log_level=args.log_level) if args.log_level else cfg.train
    cfg = replace(cfg, paths=paths, data=data, train=train)
    configure_logging(cfg.train.log_level)
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
