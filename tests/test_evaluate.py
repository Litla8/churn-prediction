"""Tests for churn.evaluate."""
from __future__ import annotations

import numpy as np
import pytest

from churn.config import CostConfig
from churn.evaluate import business_summary, compute_metrics, select_threshold

Y = np.array([1, 1, 0, 0])
P = np.array([0.9, 0.8, 0.3, 0.1])


def test_perfect_ranking_metrics() -> None:
    m = compute_metrics(Y, P, threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)


def test_flagging_nobody_gives_zero_precision() -> None:
    m = compute_metrics(Y, P, threshold=0.99)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        compute_metrics(Y, P[:3], 0.5)
    with pytest.raises(ValueError):
        compute_metrics(Y, np.array([1.2, 0.8, 0.3, 0.1]), 0.5)


def test_threshold_maximises_net_benefit() -> None:
    cost = CostConfig(customer_value=1000.0, offer_cost=100.0, retention_success_rate=0.3)
    threshold, sweep = select_threshold(Y, P, cost)
    # Flagging exactly the two churners: 2 * 300 - 2 * 100 = 400 (the best achievable here).
    assert sweep["net_benefit"].max() == 400.0
    assert 0.3 < threshold <= 0.8


def test_business_summary_vs_baselines() -> None:
    cost = CostConfig()
    s = business_summary(Y, P, 0.5, cost)
    assert s["net_benefit_model"] == 400.0
    assert s["net_benefit_offer_everyone"] == 2 * 300 - 4 * 100
    assert s["customers_flagged"] == 2
