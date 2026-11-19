"""K-Means customer segmentation dashboard."""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
st.title("🧩 Customer Segmentation")
st.markdown("Explore descriptive customer groups generated with K-Means using the seven numeric German Credit attributes.")
st.caption("Clusters are descriptive, unsupervised groups. They are not credit-risk grades and do not imply causation.")

assignments_path = ROOT / "reports/results/cluster_assignments.csv"
profiles_path = ROOT / "reports/results/cluster_profiles.csv"
diag_path = ROOT / "reports/results/kmeans_diagnostics.csv"
if not assignments_path.exists() or not profiles_path.exists():
    st.warning("Segmentation artifacts are not available yet.")
    st.code("python -m src.clustering.kmeans")
    st.stop()

assignments = pd.read_csv(assignments_path)
profiles = pd.read_csv(profiles_path)
counts = assignments["cluster"].value_counts().sort_index()
cols = st.columns(len(counts))
for col, (cluster_id, count) in zip(cols, counts.items()):
    col.metric(f"Cluster {cluster_id} customers", int(count), f"{count/len(assignments):.1%} of dataset")

st.markdown("### PCA projection of customer segments")
fig, ax = plt.subplots(figsize=(10, 6))
for cid, group in assignments.groupby("cluster"):
    ax.scatter(group["pca_1"], group["pca_2"], s=24, alpha=0.65, label=f"Cluster {cid}")
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.set_title("K-Means clusters projected into two dimensions")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("### Cluster profiles")
show_profiles = profiles.copy()
show_profiles["bad_credit_rate"] = show_profiles["bad_credit_rate"].map(lambda v: f"{v:.1%}")
show_profiles["average_credit_amount"] = show_profiles["average_credit_amount"].map(lambda v: f"{v:,.0f}")
show_profiles["average_age"] = show_profiles["average_age"].map(lambda v: f"{v:.1f}")
show_profiles["average_duration"] = show_profiles["average_duration"].map(lambda v: f"{v:.1f} months")
st.dataframe(show_profiles, use_container_width=True, hide_index=True)

st.markdown("### Segment-level Bad Credit prevalence")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(profiles["cluster"].astype(str), profiles["bad_credit_rate"] * 100)
ax.set_xlabel("Cluster")
ax.set_ylabel("Bad Credit records (%)")
ax.set_title("Observed target prevalence in each unsupervised cluster")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if diag_path.exists():
    st.markdown("### K-Means diagnostics (k = 2 to 8)")
    st.dataframe(pd.read_csv(diag_path), use_container_width=True, hide_index=True)

st.download_button("Download customer cluster assignments", assignments.to_csv(index=False), "cluster_assignments.csv", "text/csv")
