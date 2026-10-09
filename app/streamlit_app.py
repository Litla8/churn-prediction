"""Streamlit app: score one customer or a whole CSV and explain the result.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# Streamlit puts app/ on sys.path, not the project root, so add the root to import `churn`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn.config import load_config  # noqa: E402
from churn.scoring import (  # noqa: E402
    CATEGORY_LEVELS,
    build_customer_frame,
    model_input_columns,
    recommend_actions,
    score_customers,
    validate_batch,
)

CFG = load_config()
MODEL_PATH = ROOT / CFG.paths.models_dir / "churn_pipeline.joblib"
METRICS_PATH = ROOT / CFG.paths.reports_dir / "metrics.json"
BACKGROUND_PATH = ROOT / CFG.paths.models_dir / "shap_background.csv"
FIGURES_DIR = ROOT / CFG.paths.figures_dir

st.set_page_config(page_title="Bank Churn Risk", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, metrics_path: str) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Load the trained pipeline, its decision threshold and the saved metrics once."""
    pipeline = joblib.load(model_path)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return pipeline, float(metrics["decision_threshold"]), metrics


def explain_customer(pipeline: Pipeline, frame: pd.DataFrame) -> pd.Series:
    """Per-feature SHAP contributions (probability points) for one customer."""
    from churn.explain import compute_shap

    background = pd.read_csv(BACKGROUND_PATH)
    result = compute_shap(pipeline, background, frame)
    contributions = pd.Series(result.values[0], index=result.feature_names)
    return contributions.reindex(contributions.abs().sort_values(ascending=False).index)


if not (MODEL_PATH.is_file() and METRICS_PATH.is_file()):
    st.error("Trained model not found. Run `python -m churn.pipeline` first, then reload this page.")
    st.stop()

pipeline, threshold, metrics = load_artifacts(str(MODEL_PATH), str(METRICS_PATH))

st.title("🏦 Bank Customer Churn Risk")
st.caption(f"Model: {metrics['selected_model']} | offer threshold: {threshold:.2f}")
tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Batch scoring", "About the model"])

with tab_single:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        geography = c1.selectbox("Geography", CATEGORY_LEVELS["Geography"])
        gender = c1.selectbox("Gender", CATEGORY_LEVELS["Gender"])
        age = c1.number_input("Age", min_value=18, max_value=100, value=40)
        credit_score = c1.number_input("Credit score", min_value=300, max_value=900, value=650)
        tenure = c2.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
        balance = c2.number_input("Balance", min_value=0.0, max_value=300000.0, value=76000.0, step=1000.0)
        products = c2.number_input("Number of products", min_value=1, max_value=4, value=1)
        salary = c2.number_input("Estimated salary", min_value=0.0, max_value=250000.0,
                                 value=100000.0, step=1000.0)
        card_type = c3.selectbox("Card type", CATEGORY_LEVELS["Card Type"])
        satisfaction = c3.number_input("Satisfaction score", min_value=1, max_value=5, value=3)
        points = c3.number_input("Points earned", min_value=0, max_value=1500, value=600)
        has_card = c3.checkbox("Has credit card", value=True)
        is_active = c3.checkbox("Is active member", value=True)
        submitted = st.form_submit_button("Score customer")

    if submitted:
        inputs: Dict[str, Any] = {
            "Geography": geography, "Gender": gender, "Age": age, "CreditScore": credit_score,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
            "EstimatedSalary": salary, "Card Type": card_type, "Satisfaction Score": satisfaction,
            "Point Earned": points, "HasCrCard": int(has_card), "IsActiveMember": int(is_active),
        }
        frame = build_customer_frame(inputs, CFG.data)
        scored = score_customers(pipeline, frame, threshold, CFG.data).iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{scored['churn_probability']:.1%}")
        m2.metric("Risk tier", scored["risk_tier"])
        m3.metric("Offer threshold", f"{threshold:.0%}")
        if scored["flag_for_offer"]:
            st.warning("Above the threshold: eligible for a retention offer.")
        else:
            st.success("Below the threshold: no offer needed.")
        st.subheader("Suggested actions")
        for action in recommend_actions(inputs, scored["risk_tier"]):
            st.write(f"- {action}")
        st.caption("Rule-based ideas from EDA patterns, not proven causes. Test offers before rollout.")

        with st.expander("Why this score? (SHAP)"):
            if not BACKGROUND_PATH.is_file():
                st.info("Run `python -m churn.explain` once to enable per-customer explanations.")
            else:
                try:
                    contributions = explain_customer(pipeline, frame)
                    st.bar_chart(contributions.head(8))
                    st.caption("Positive bars push churn risk up; negative bars push it down "
                               "(probability points vs a typical customer).")
                except ImportError:
                    st.info("Install shap (`pip install shap`) to see explanations.")

with tab_batch:
    st.write("Upload a CSV with the model columns: " + ", ".join(model_input_columns(CFG.data)))
    uploaded = st.file_uploader("Customer CSV", type="csv")
    if uploaded is not None:
        customers = pd.read_csv(uploaded)
        try:
            validate_batch(customers, CFG.data)
        except ValueError as exc:
            st.error(str(exc))
        else:
            results = score_customers(pipeline, customers, threshold, CFG.data)
            results = results.sort_values("churn_probability", ascending=False)
            st.write(f"{int(results['flag_for_offer'].sum())} of {len(results)} customers "
                     "are above the offer threshold.")
            st.dataframe(results, use_container_width=True)
            st.download_button("Download scored CSV", results.to_csv(index=False).encode("utf-8"),
                               file_name="scored_customers.csv", mime="text/csv")

with tab_about:
    test = metrics["test_metrics_at_business_threshold"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    a2.metric("Recall at threshold", f"{test['recall']:.2f}")
    a3.metric("Precision at threshold", f"{test['precision']:.2f}")
    st.write("`Complain` is excluded from the model because it is almost identical to the churn "
             "label (target leakage). The threshold maximises expected net benefit under the "
             "cost assumptions in `config.yaml`.")
    for figure in ("roc_pr_test.png", "shap_summary_beeswarm.png", "threshold_vs_net_benefit.png"):
        if (FIGURES_DIR / figure).is_file():
            st.image(str(FIGURES_DIR / figure))
