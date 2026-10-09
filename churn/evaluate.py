"""Metrics, business-cost threshold selection and standard evaluation plots."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend: pipelines must run without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

from churn.config import CostConfig  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_GRID: np.ndarray = np.round(np.linspace(0.05, 0.95, 91), 2)


def _validate_scores(y_true: np.ndarray, proba: np.ndarray) -> None:
    """Raise ValueError unless labels and probabilities are aligned and well-formed."""
    if y_true.shape != proba.shape or y_true.ndim != 1:
        raise ValueError(f"y_true and proba must be 1-D and equal length, got "
                         f"{y_true.shape} and {proba.shape}")
    if np.isnan(proba).any() or proba.min() < 0.0 or proba.max() > 1.0:
        raise ValueError("proba must be finite and within [0, 1]")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must be binary 0/1")


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute ranking and thresholded classification metrics.

    Args:
        y_true: Binary labels.
        proba: Predicted churn probabilities.
        threshold: Probability at or above which a customer is flagged.

    Returns:
        Dict with roc_auc, pr_auc, precision, recall, f1, accuracy and the confusion counts.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    _validate_scores(y_true, proba)

    flagged = proba >= threshold
    positives = y_true == 1
    tp = int(np.sum(flagged & positives))
    fp = int(np.sum(flagged & ~positives))
    fn = int(np.sum(~flagged & positives))
    tn = int(np.sum(~flagged & ~positives))
    # Zero-division is defined as 0.0: "flagged nobody" must not look like perfect precision.
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def sweep_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost: CostConfig,
    grid: np.ndarray = DEFAULT_THRESHOLD_GRID,
) -> pd.DataFrame:
    """Evaluate expected net benefit of the retention campaign at every threshold.

    Net benefit = TP * success_rate * customer_value - flagged * offer_cost.
    Fully vectorised: one (n_samples, n_thresholds) boolean matrix, no Python loop.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    _validate_scores(y_true, proba)

    flagged = proba[:, None] >= grid[None, :]
    positives = (y_true == 1)[:, None]
    tp = (flagged & positives).sum(axis=0)
    n_flagged = flagged.sum(axis=0)
    fp = n_flagged - tp
    total_pos = int(positives.sum())
    benefit = tp * cost.retention_success_rate * cost.customer_value - n_flagged * cost.offer_cost
    return pd.DataFrame(
        {
            "threshold": grid,
            "flagged": n_flagged,
            "tp": tp,
            "fp": fp,
            "recall": tp / total_pos,
            "precision": np.divide(tp, n_flagged, out=np.zeros(len(grid)), where=n_flagged > 0),
            "net_benefit": benefit,
        }
    )


def select_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost: CostConfig,
    grid: np.ndarray = DEFAULT_THRESHOLD_GRID,
) -> Tuple[float, pd.DataFrame]:
    """Pick the threshold that maximises expected net benefit.

    Returns:
        Tuple of the best threshold and the full sweep table (for plotting/auditing).
    """
    sweep = sweep_thresholds(y_true, proba, cost, grid)
    best = sweep.loc[sweep["net_benefit"].idxmax()]
    return float(best["threshold"]), sweep


def business_summary(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, cost: CostConfig
) -> Dict[str, float]:
    """Compare the model's campaign against 'do nothing' and 'offer everyone'."""
    y_true = np.asarray(y_true)
    row = sweep_thresholds(y_true, proba, cost, np.array([threshold])).iloc[0]
    offer_all = (
        int(y_true.sum()) * cost.retention_success_rate * cost.customer_value
        - len(y_true) * cost.offer_cost
    )
    return {
        "net_benefit_model": float(row["net_benefit"]),
        "net_benefit_offer_everyone": float(offer_all),
        "net_benefit_do_nothing": 0.0,
        "customers_flagged": int(row["flagged"]),
        "customers_in_test_set": int(len(y_true)),
    }


def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    """Save side-by-side ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1])
    axes[1].axhline(float(np.mean(y_true)), color="k", linestyle="--", linewidth=0.8)
    axes[1].set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(metrics: Dict[str, float], path: Path) -> None:
    """Save an annotated confusion matrix from the counts in ``metrics``."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {metrics['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(sweep: pd.DataFrame, best_threshold: float, path: Path) -> None:
    """Save net benefit vs threshold, marking the chosen operating point."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["net_benefit"], color="tab:green")
    ax.axvline(best_threshold, color="k", linestyle="--", linewidth=0.8,
               label=f"chosen = {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected net benefit (validation set)")
    ax.set_title("Business-cost threshold selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
