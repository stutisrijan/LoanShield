"""Model evaluation and comparison dashboard."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.title("📈 Model Performance & Evaluation")
st.markdown("Held-out test performance for the frozen Random Forest model, alongside five-fold cross-validation comparisons.")
st.caption("Cross-validation is performed on the training partition. The held-out test partition is kept separate for final evaluation.")

metrics_path = ROOT / "reports/results/final_test_metrics.json"
comparison_path = ROOT / "reports/results/model_comparison_cv.csv"
if not metrics_path.exists():
    st.warning("Evaluation reports are missing. Train the credit model first:")
    st.code("python -m src.models.train_credit_model")
    st.stop()

metrics = json.loads(metrics_path.read_text())
metric_labels = [
    ("Accuracy", "accuracy"), ("Precision", "precision"),
    ("Recall", "recall"), ("Specificity", "specificity"),
    ("F1 score", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
]
row1 = st.columns(4)
for col, (label, key) in zip(row1, metric_labels[:4]):
    col.metric(label, f"{metrics.get(key, 0):.3f}")
row2 = st.columns(3)
for col, (label, key) in zip(row2, metric_labels[4:]):
    col.metric(label, f"{metrics.get(key, 0):.3f}")

st.markdown("### Confusion matrix counts")
cm = pd.DataFrame(
    [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
    index=["Actual Good", "Actual Bad"], columns=["Predicted Good", "Predicted Bad"]
)
st.dataframe(cm, use_container_width=True)

figure_map = [
    ("Confusion matrix", "final_confusion_matrix.png"),
    ("ROC curve", "final_roc_curve.png"),
    ("Precision–Recall curve", "final_pr_curve.png"),
]
cols = st.columns(3)
for col, (caption, filename) in zip(cols, figure_map):
    path = ROOT / "reports/figures" / filename
    with col:
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Missing: {filename}")

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.markdown("### Five-fold cross-validation — training data only")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    metric_choice = st.selectbox("Metric to compare", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"], index=4)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = comparison[f"{metric_choice}_mean"]
    err = comparison[f"{metric_choice}_std"]
    ax.bar(comparison["model"], x, yerr=err, capsize=5)
    ax.set_ylabel(metric_choice.replace("_", " ").upper())
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-validation comparison: {metric_choice}")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.download_button("Download CV comparison CSV", comparison.to_csv(index=False), "model_comparison_cv.csv", "text/csv")
else:
    st.info("Cross-validation comparison not generated yet. Re-run the training command.")

st.caption(f"Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}. Metrics are specific to this dataset split and are not a guarantee of real-world performance.")
