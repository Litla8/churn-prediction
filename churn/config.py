"""Central configuration for the churn project.

Every tunable value (paths, column roles, hyperparameters, business costs) lives here
so no other module hardcodes configuration. Defaults can be overridden from a YAML
file via :func:`load_config` without touching the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

RANDOM_STATE: int = 42


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    raw_data: Path = Path("data/raw/Customer-Churn-Records.csv")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    @property
    def figures_dir(self) -> Path:
        """Directory that holds every generated plot."""
        return self.reports_dir / "figures"


@dataclass(frozen=True)
class DataConfig:
    """Column roles and split sizes for the Bank Customer Churn dataset."""

    target: str = "Exited"
    id_columns: Tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")
    dedupe_key: Optional[str] = "CustomerId"
    # `Complain` is ~99.8% identical to the target (it is recorded after the customer
    # has already decided to leave), so it must never be a model input.
    leakage_columns: Tuple[str, ...] = ("Complain",)
    categorical_columns: Tuple[str, ...] = ("Geography", "Gender", "Card Type")
    numeric_columns: Tuple[str, ...] = (
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Satisfaction Score",
        "Point Earned",
    )
    val_size: float = 0.15
    test_size: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    """Cross-validation, model selection and reproducibility settings."""

    random_state: int = RANDOM_STATE
    cv_folds: int = 5
    models: Tuple[str, ...] = ("logreg", "random_forest", "hist_gb", "xgboost", "lightgbm")
    early_stopping_rounds: int = 50
    log_level: str = "INFO"
    target_roc_auc: float = 0.84
    target_recall: float = 0.75


@dataclass(frozen=True)
class CostConfig:
    """Business assumptions used to pick the decision threshold.

    These are placeholders: replace them with the bank's real numbers.
    A flagged customer receives a retention offer (costs `offer_cost`). If the customer
    really was a churner, the offer works with probability `retention_success_rate`
    and saves `customer_value` of lifetime value.
    """

    customer_value: float = 1000.0
    offer_cost: float = 100.0
    retention_success_rate: float = 0.3


def _default_model_params() -> Dict[str, Dict[str, Any]]:
    """Baseline hyperparameters per model family."""
    return {
        "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
        "random_forest": {
            "n_estimators": 400,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        "hist_gb": {
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "class_weight": "balanced",
        },
        "xgboost": {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_jobs": -1,
        },
        "lightgbm": {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "n_jobs": -1,
            "verbose": -1,
        },
    }


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration object passed explicitly through the pipeline."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=_default_model_params)


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """Build a :class:`PipelineConfig`, applying overrides from a YAML file if given.

    Args:
        path: Optional YAML file. Top-level keys are config sections (``paths``,
            ``data``, ``train``, ``cost``, ``model_params``).

    Returns:
        The resolved configuration.

    Raises:
        KeyError: If the YAML refers to an unknown section or field.
        FileNotFoundError: If ``path`` is given but does not exist.
    """
    cfg = PipelineConfig()
    if path is None:
        return cfg
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    updates: Dict[str, Any] = {}
    for section, values in raw.items():
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown config section: {section!r}")
        if section == "model_params":
            # Merge per model so overriding one key keeps the other defaults.
            merged = {name: dict(params) for name, params in cfg.model_params.items()}
            for name, params in values.items():
                merged.setdefault(name, {}).update(params)
            updates[section] = merged
            continue
        current = getattr(cfg, section)
        coerced: Dict[str, Any] = {}
        for key, value in values.items():
            if not hasattr(current, key):
                raise KeyError(f"Unknown field {key!r} in section {section!r}")
            default = getattr(current, key)
            if isinstance(default, Path):
                value = Path(value)
            elif isinstance(default, tuple):
                value = tuple(value)
            coerced[key] = value
        updates[section] = replace(current, **coerced)
    return replace(cfg, **updates)
