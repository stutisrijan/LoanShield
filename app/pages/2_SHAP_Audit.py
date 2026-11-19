"""Global SHAP audit page."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.credit_preprocessor import load_german_credit
from src.explainability.local_shap import global_shap_values

st.title("🔍 SHAP Audit — Global Explainability")
st.markdown("Understand which input fields influence the trained Random Forest across a sample of customers.")
st.caption("SHAP describes model behaviour. It does not establish causation or prove that a feature should determine lending eligibility.")

@st.cache_resource(show_spinner="Loading trained model and preprocessor...")
def load_artifacts():
    model = joblib.load(ROOT / "models/credit/random_forest.joblib")
    preprocessor = joblib.load(ROOT / "models/credit/preprocessor.joblib")
    return model, preprocessor

@st.cache_data(show_spinner="Computing global SHAP values...")
def compute_global(sample_size: int):
    model, preprocessor = load_artifacts()
    X, y = load_german_credit(ROOT / "data/raw/german.data")
    # Keep the audit display responsive and deterministic.
    sample = X.sample(n=min(sample_size, len(X)), random_state=42).sort_index()
    Xp = preprocessor.transform(sample)
    values = global_shap_values(model, Xp, class_index=1)
    names = list(preprocessor.get_feature_names_out())
    dense = np.asarray(Xp)
    return sample, dense, values, names

try:
    sample_size = st.slider("Number of customers sampled for global explanation", 100, 500, 250, 50)
    with st.spinner("Preparing global explanation..."):
        raw_sample, X_values, shap_values, feature_names = compute_global(sample_size)
except Exception as exc:
    st.error(f"Unable to compute global SHAP values: {exc}")
    st.info("Generate the trained artifacts first with `python -m src.models.train_credit_model`.")
    st.stop()

importance = np.mean(np.abs(shap_values), axis=0)
mean_signed = np.mean(shap_values, axis=0)
summary = pd.DataFrame({
    "Feature": feature_names,
    "Mean absolute SHAP": importance,
    "Mean signed SHAP": mean_signed,
}).sort_values("Mean absolute SHAP", ascending=False)

m1, m2, m3 = st.columns(3)
m1.metric("Customers explained", len(raw_sample))
m2.metric("Processed features", len(feature_names))
m3.metric("Most influential feature", summary.iloc[0]["Feature"].replace("cat__", "").replace("num__", ""))

st.markdown("### Global feature importance")
top = summary.head(15).sort_values("Mean absolute SHAP", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top["Feature"], top["Mean absolute SHAP"], color="#3572A5")
ax.set_xlabel("Mean absolute SHAP value")
ax.set_title("Average magnitude of contribution to Bad Credit output")
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("### SHAP contribution distribution")
st.caption("Each dot is one sampled customer. Horizontal position is that customer's SHAP contribution; positive values push toward Bad Credit.")
top_indices = np.argsort(importance)[-12:]
fig, ax = plt.subplots(figsize=(11, 7))
for row, idx in enumerate(top_indices):
    vals = shap_values[:, idx]
    jitter = np.random.default_rng(42 + int(idx)).uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(vals, np.full(len(vals), row) + jitter, s=12, alpha=0.55, label=None)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(range(len(top_indices)))
ax.set_yticklabels([feature_names[i] for i in top_indices])
ax.set_xlabel("SHAP contribution (model output scale)")
ax.set_title("Distribution for the 12 strongest global features")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("### Feature audit table")
st.dataframe(summary.head(20), hide_index=True, use_container_width=True)
st.download_button("Download global SHAP summary CSV", summary.to_csv(index=False), "global_shap_summary.csv", "text/csv")
