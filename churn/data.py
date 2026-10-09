"""Loading, validating, cleaning and splitting the churn dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from churn.config import DataConfig

logger = logging.getLogger(__name__)

# Columns with more missing values than this are flagged loudly in the logs.
MISSING_WARN_FRACTION: float = 0.05


class DataSplits(NamedTuple):
    """Stratified train / validation / test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw churn CSV.

    Args:
        path: Location of the CSV file.

    Returns:
        The raw dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data not found at {path.resolve()}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def validate_raw(df: pd.DataFrame, cfg: DataConfig) -> None:
    """Fail loudly if the raw data does not match the expected schema.

    Args:
        df: Raw dataframe.
        cfg: Column-role configuration.

    Raises:
        ValueError: On missing columns, non-numeric dtypes, a non-binary or
            single-class target, or impossible (negative) values.
    """
    required = set(cfg.numeric_columns) | set(cfg.categorical_columns) | {cfg.target}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    non_numeric = [c for c in cfg.numeric_columns if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Columns expected to be numeric but are not: {non_numeric}")

    if df[cfg.target].isna().any():
        raise ValueError(f"Target column {cfg.target!r} contains missing values")
    classes = set(df[cfg.target].unique())
    if not classes <= {0, 1}:
        raise ValueError(f"Target must be binary 0/1, found values {sorted(classes)}")
    if len(classes) < 2:
        raise ValueError("Target contains a single class; cannot train a classifier")

    for column in ("Balance", "Tenure", "EstimatedSalary", "Age"):
        if column in df.columns and (df[column] < 0).any():
            raise ValueError(f"Column {column!r} contains negative values")

    missing_frac = df[sorted(required)].isna().mean()
    for column, fraction in missing_frac[missing_frac > MISSING_WARN_FRACTION].items():
        logger.warning("Column %r is %.1f%% missing (imputed inside the pipeline)",
                       column, 100 * fraction)


def prepare_features(df: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Deduplicate, drop identifiers/leakage columns and separate X from y.

    Args:
        df: Validated raw dataframe.
        cfg: Column-role configuration.

    Returns:
        Tuple ``(X, y)`` with only modelling columns in ``X``.

    Raises:
        ValueError: If a configured leakage column is also a model input.
    """
    feature_cols = list(cfg.numeric_columns) + list(cfg.categorical_columns)
    leaked = sorted(set(cfg.leakage_columns) & set(feature_cols))
    if leaked:
        raise ValueError(f"Leakage columns present among model inputs: {leaked}")

    before = len(df)
    if cfg.dedupe_key and cfg.dedupe_key in df.columns:
        df = df.drop_duplicates(subset=cfg.dedupe_key)
    else:
        df = df.drop_duplicates()
    if len(df) != before:
        logger.warning("Dropped %d duplicate rows", before - len(df))

    X = df[feature_cols].reset_index(drop=True)
    y = df[cfg.target].astype(int).reset_index(drop=True)
    logger.info("Prepared %d rows, %d features, churn rate %.2f%%",
                len(X), X.shape[1], 100 * y.mean())
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, cfg: DataConfig, random_state: int) -> DataSplits:
    """Create stratified train/validation/test splits (default 70/15/15).

    The test set is carved off first and never touched again until final evaluation.

    Args:
        X: Feature dataframe.
        y: Binary target.
        cfg: Provides ``val_size`` and ``test_size`` as fractions of the full data.
        random_state: Seed for reproducibility.

    Returns:
        A :class:`DataSplits` tuple.

    Raises:
        ValueError: If the fractions are invalid or inputs are misaligned.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    if not 0 < cfg.val_size + cfg.test_size < 1:
        raise ValueError("val_size + test_size must be strictly between 0 and 1")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=random_state
    )
    # Rescale so the validation share is relative to the full dataset, not the remainder.
    val_fraction = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=random_state
    )
    logger.info("Split sizes -> train %d, val %d, test %d", len(X_train), len(X_val), len(X_test))
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
