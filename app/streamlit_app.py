"""LoanShield multi-page Streamlit application entry point."""
from pathlib import Path
import json
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="LoanShield | Credit Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ LoanShield")
st.subheader("Transparent Financial Credit Scoring & Risk Analytics")
st.markdown(
    "LoanShield is an academic decision-support prototype that combines a trained "
    "credit-risk classifier, local and global model explanations, customer segmentation, "
    "and model evaluation."
)

model_meta_path = ROOT / "models/credit/model_metadata.json"
metrics_path = ROOT / "reports/results/final_test_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    meta = json.loads(model_meta_path.read_text()) if model_meta_path.exists() else {}
    st.caption(
        f"Model: {meta.get('model', 'Random Forest')} · "
        f"Training rows: {meta.get('training_samples', '—')} · "
        f"Held-out test rows: {meta.get('held_out_test_samples', '—')}"
    )
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
    b.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    c.metric("Recall (Bad Credit)", f"{metrics.get('recall', 0):.1%}")
    d.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
else:
    st.warning("Credit model artifacts have not been generated yet.")
    st.code("python -m src.models.train_credit_model\npython -m src.clustering.kmeans")

st.markdown("### Explore the workspace")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Customer Assessment**\n\nEstimate the trained model's Bad Credit probability and inspect a local SHAP explanation.")
    st.markdown("**SHAP Audit**\n\nReview global model drivers and SHAP contribution patterns.")
    st.markdown("**Customer Segmentation**\n\nExplore three descriptive K-Means customer groups.")
with col2:
    st.markdown("**Model Evaluation**\n\nInspect held-out metrics, confusion matrix, ROC and precision-recall curves, plus cross-validation comparisons.")

st.divider()
st.info(
    "Important: LoanShield is an educational prototype built on the UCI German Credit "
    "dataset. The dataset is small and historical. Predictions and SHAP values are not "
    "causal, calibrated lending decisions, or a substitute for fair-lending review. "
    "Do not use this prototype to approve or reject real loan applications."
)
