"""SHAP explainability: global importance, plain-English drivers and per-customer reasons.

A model-agnostic permutation explainer is used on the *encoded* feature matrix so the same
code works for HistGradientBoosting (not supported by TreeExplainer), XGBoost, LightGBM,
random forest and logistic regression. Attributions are in probability points.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.config import PipelineConfig, load_config  # noqa: E402
from churn.data import load_raw, prepare_features, split_data, validate_raw  # noqa: E402

logger = logging.getLogger(__name__)

# |correlation| below this means the feature has no consistent up/down effect.
MIXED_EFFECT_CORR: float = 0.1


class ShapResult(NamedTuple):
    """SHAP output plus the encoded matrix it was computed on."""

    values: np.ndarray  # [n_samples, n_features], probability points
    encoded: np.ndarray  # [n_samples, n_features]
    feature_names: List[str]
    base_value: float
    explanation: Any  # shap.Explanation, kept for the shap plotting API


def _encode(pipeline: Pipeline, raw: pd.DataFrame) -> np.ndarray:
    """Apply every pipeline step except the final model."""
    return np.asarray(pipeline[:-1].transform(raw), dtype=float)


def compute_shap(
    pipeline: Pipeline,
    background_raw: pd.DataFrame,
    explain_raw: pd.DataFrame,
    seed: int = 42,
) -> ShapResult:
    """Compute SHAP values for ``explain_raw`` against a background sample.

    Args:
        pipeline: Fitted features -> preprocess -> model pipeline.
        background_raw: Raw rows (training data) defining the "typical customer" baseline.
        explain_raw: Raw rows to explain.
        seed: Seed for the permutation sampler.

    Returns:
        A :class:`ShapResult`.
    """
    import shap  # lazy: only needed when explaining

    model = pipeline.named_steps["model"]
    names = [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    background = _encode(pipeline, background_raw)
    encoded = _encode(pipeline, explain_raw)
    if encoded.shape[1] != len(names):
        raise ValueError(f"Encoded width {encoded.shape[1]} != {len(names)} feature names")

    def predict_churn(matrix: np.ndarray) -> np.ndarray:
        """Churn probability for an encoded matrix."""
        return model.predict_proba(matrix)[:, 1]

    explainer = shap.Explainer(
        predict_churn,
        shap.maskers.Independent(background, max_samples=100),
        feature_names=names,
        algorithm="permutation",
        seed=seed,
    )
    # Permutation SHAP needs at least 2 * n_features + 1 model evaluations per row.
    explanation = explainer(encoded, max_evals=max(500, 2 * len(names) + 1), silent=True)
    return ShapResult(np.asarray(explanation.values), encoded, names,
                      float(np.mean(explanation.base_values)), explanation)


def importance_table(
    values: np.ndarray, encoded: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Rank features by mean |SHAP| and estimate each one's overall direction.

    ``direction_corr`` is the correlation between a feature's value and its SHAP value:
    positive means higher values push churn risk up, near zero means no consistent effect.

    Raises:
        ValueError: If the array shapes disagree.
    """
    if values.shape != encoded.shape or values.shape[1] != len(feature_names):
        raise ValueError("values, encoded and feature_names must describe the same matrix")
    v_c = values - values.mean(axis=0)
    x_c = encoded - encoded.mean(axis=0)
    denom = np.sqrt((v_c ** 2).sum(axis=0) * (x_c ** 2).sum(axis=0))
    # Constant columns give denom == 0: report 0.0 instead of NaN.
    corr = np.divide((v_c * x_c).sum(axis=0), denom, out=np.zeros(values.shape[1]), where=denom > 0)
    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "direction_corr": corr,
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _describe(name: str, corr: float, categorical: Sequence[str]) -> str:
    """One plain-English sentence about a feature's effect."""
    verb = "raises" if corr > 0 else "lowers"
    for column in categorical:
        if name.startswith(f"{column}_"):
            level = name[len(column) + 1:]
            return f"{column} = {level} {verb} predicted churn risk"
    if name == "zero_balance":
        return f"Having a zero balance {verb} predicted churn risk"
    return f"Higher {name} {verb} predicted churn risk"


def drivers_markdown(
    table: pd.DataFrame, categorical_columns: Sequence[str], top_k: int = 5
) -> str:
    """Write the top drivers as a short stakeholder-friendly Markdown note."""
    lines = ["# Top drivers of churn (SHAP)", ""]
    for rank, row in enumerate(table.head(top_k).itertuples(index=False), start=1):
        if abs(row.direction_corr) < MIXED_EFFECT_CORR:
            text = f"{row.feature}: effect is mixed or non-monotonic (see the beeswarm plot)"
        else:
            text = _describe(row.feature, row.direction_corr, categorical_columns)
        lines.append(f"{rank}. **{text}** (average impact {row.mean_abs_shap * 100:.1f} "
                     f"probability points)")
    lines += ["", "Direction is the overall tendency; some relationships are not monotonic "
              "(for example, risk rises with age and then falls in the oldest group). "
              "These are associations in the data, not proven causes."]
    return "\n".join(lines) + "\n"


def _save_shap_plots(explanation: Any, figures_dir: Path) -> None:
    """Save the standard SHAP beeswarm and mean-|SHAP| bar plots."""
    import shap

    for filename, plotter in (
        ("shap_summary_beeswarm.png", shap.plots.beeswarm),
        ("shap_importance_bar.png", shap.plots.bar),
    ):
        plt.figure()
        plotter(explanation, max_display=15, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_explain(
    cfg: PipelineConfig, n_background: int = 100, n_explain: int = 500
) -> Dict[str, Any]:
    """Explain the saved model on held-out test rows and write all SHAP artifacts.

    The train/test split is re-created from the same seed, so the test rows are exactly
    those the model never saw during training.
    """
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(cfg.paths.models_dir / "churn_pipeline.joblib")
    df = load_raw(cfg.paths.raw_data)
    validate_raw(df, cfg.data)
    X, y = prepare_features(df, cfg.data)
    splits = split_data(X, y, cfg.data, cfg.train.random_state)

    seed = cfg.train.random_state
    background = splits.X_train.sample(min(n_background, len(splits.X_train)), random_state=seed)
    sample = splits.X_test.sample(min(n_explain, len(splits.X_test)), random_state=seed)
    result = compute_shap(pipeline, background, sample, seed=seed)

    table = importance_table(result.values, result.encoded, result.feature_names)
    table.to_csv(cfg.paths.reports_dir / "shap_importance.csv", index=False)
    (cfg.paths.reports_dir / "churn_drivers.md").write_text(
        drivers_markdown(table, cfg.data.categorical_columns), encoding="utf-8")
    _save_shap_plots(result.explanation, cfg.paths.figures_dir)
    # Saved so the Streamlit app can explain single customers without re-reading raw data.
    background.to_csv(cfg.paths.models_dir / "shap_background.csv", index=False)
    summary = {
        "rows_explained": int(len(sample)),
        "base_value": result.base_value,
        "top_features": table.head(10)["feature"].tolist(),
    }
    (cfg.paths.reports_dir / "shap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Top SHAP drivers: %s", ", ".join(summary["top_features"][:5]))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--n-explain", type=int, default=500)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from dataclasses import replace

    cfg = load_config(args.config)
    paths = cfg.paths
    if args.data_path:
        paths = replace(paths, raw_data=args.data_path)
    if args.models_dir:
        paths = replace(paths, models_dir=args.models_dir)
    if args.reports_dir:
        paths = replace(paths, reports_dir=args.reports_dir)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    run_explain(replace(cfg, paths=paths), n_explain=args.n_explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
