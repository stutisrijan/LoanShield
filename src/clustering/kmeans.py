"""K-Means customer segmentation using the seven numeric German Credit fields."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.credit_preprocessor import load_german_credit, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def train_clustering():
    X, y = load_german_credit(ROOT / "data/raw/german.data")
    numeric = X[NUMERIC_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    scores = []
    for k in range(2, 9):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = candidate.fit_predict(X_scaled)
        scores.append({
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            "inertia": float(candidate.inertia_),
        })
    # Use k=3 to match the exploratory project analysis.
    model = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    out = X.copy()
    out["credit_risk"] = y.to_numpy()
    out["cluster"] = labels
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    out.to_csv(ROOT / "reports/results/cluster_assignments.csv", index=False)

    profile = out.groupby("cluster").agg(
        customers=("credit_risk", "size"),
        bad_credit_rate=("credit_risk", "mean"),
        average_age=("age", "mean"),
        average_credit_amount=("credit_amount", "mean"),
        average_duration=("duration", "mean"),
        average_installment_rate=("installment_rate", "mean"),
    ).reset_index()
    profile.to_csv(ROOT / "reports/results/cluster_profiles.csv", index=False)
    pd.DataFrame(scores).to_csv(ROOT / "reports/results/kmeans_diagnostics.csv", index=False)

    joblib.dump(scaler, ROOT / "models/credit/cluster_scaler.joblib")
    joblib.dump(model, ROOT / "models/credit/kmeans.joblib")
    joblib.dump(pca, ROOT / "models/credit/cluster_pca.joblib")
    metadata = {
        "algorithm": "KMeans", "k": 3, "random_state": 42,
        "features": NUMERIC_FEATURES,
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "warning": "Clusters are descriptive groups, not lending risk grades or causal categories.",
    }
    (ROOT / "models/credit/clustering_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("K-Means segmentation saved. Cluster profiles:")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    train_clustering()
