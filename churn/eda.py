"""Exploratory data analysis: segment churn tables, plots and automatic leakage detection."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churn.config import PipelineConfig, load_config  # noqa: E402
from churn.data import load_raw, validate_raw  # noqa: E402

logger = logging.getLogger(__name__)

LEAKAGE_CORR_THRESHOLD: float = 0.9


def churn_rate_by(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """Return customer count and churn rate per level of ``column``."""
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}")
    out = df.groupby(column, observed=True)[target].agg(customers="size", churn_rate="mean")
    return out.reset_index()


def detect_leakage_candidates(
    df: pd.DataFrame, target: str, threshold: float = LEAKAGE_CORR_THRESHOLD
) -> Dict[str, float]:
    """Flag numeric columns whose |correlation| with the target is suspiciously high."""
    corr = df.select_dtypes("number").corr()[target].drop(target)
    return corr[corr.abs() >= threshold].round(4).to_dict()


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable segment columns used only for EDA."""
    out = df.copy()
    out["AgeBand"] = pd.cut(out["Age"], bins=[17, 30, 40, 50, 60, 120],
                            labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    out["BalanceStatus"] = np.where(out["Balance"] == 0, "zero balance", "has balance")
    return out


def _plot_segments(df: pd.DataFrame, columns: Sequence[str], target: str, path: Path) -> None:
    """Grid of churn-rate bar charts with the overall churn rate as a reference line."""
    overall = float(df[target].mean())
    cols = 4
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    for ax, column in zip(np.ravel(axes), columns):
        table = churn_rate_by(df, column, target)
        ax.bar(table[column].astype(str), table["churn_rate"] * 100, color="tab:blue")
        ax.axhline(overall * 100, color="tab:red", linestyle="--", linewidth=0.9)
        ax.set_title(f"Churn % by {column}")
        ax.tick_params(axis="x", rotation=30)
    for ax in np.ravel(axes)[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, path: Path) -> None:
    """Annotated correlation heatmap of numeric columns."""
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for (i, j), value in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(cfg: PipelineConfig) -> Dict[str, Any]:
    """Run EDA, save figures + JSON summary, and return the summary."""
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw(cfg.paths.raw_data)
    validate_raw(df, cfg.data)
    df = add_segments(df)
    target = cfg.data.target

    segments = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "AgeBand",
                "Card Type", "BalanceStatus", "Satisfaction Score"]
    tables = {c: churn_rate_by(df, c, target).to_dict(orient="records") for c in segments}
    leakage = detect_leakage_candidates(df.drop(columns=["AgeBand", "BalanceStatus"]), target)
    for column, corr in leakage.items():
        logger.warning("LEAKAGE CANDIDATE: %r has correlation %.3f with %r", column, corr, target)

    _plot_segments(df, segments, target, cfg.paths.figures_dir / "eda_churn_by_segment.png")
    _plot_correlation(df.drop(columns=["AgeBand", "BalanceStatus"]),
                      cfg.paths.figures_dir / "eda_correlation_heatmap.png")

    summary = {
        "rows": int(len(df)),
        "overall_churn_rate": float(df[target].mean()),
        "leakage_candidates": leakage,
        "segment_churn_tables": tables,
        "missing_values": int(df.isna().sum().sum()),
    }
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run churn EDA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = cfg.paths
    if args.data_path:
        paths = paths.__class__(**{**paths.__dict__, "raw_data": args.data_path})
    if args.reports_dir:
        paths = paths.__class__(**{**paths.__dict__, "reports_dir": args.reports_dir})
    from dataclasses import replace

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", force=True)
    run_eda(replace(cfg, paths=paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
