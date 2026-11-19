"""Train the reproducible LoanShield credit model and evaluation artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.preprocessing.credit_preprocessor import (
    load_german_credit, build_preprocessor, FEATURE_COLUMNS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/german.data"
MODEL_DIR = ROOT / "models/credit"
RESULTS_DIR = ROOT / "reports/results"
FIGURES_DIR = ROOT / "reports/figures"


def _metrics(y_true, y_pred, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train():
    for directory in (MODEL_DIR, RESULTS_DIR, FIGURES_DIR, ROOT / "data/processed"):
        directory.mkdir(parents=True, exist_ok=True)

    X, y = load_german_credit(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        max_features="sqrt",
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_processed, y_train)

    pred = model.predict(X_test_processed)
    prob = model.predict_proba(X_test_processed)[:, list(model.classes_).index(1)]
    metrics = _metrics(y_test, pred, prob)

    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(list(preprocessor.get_feature_names_out()), MODEL_DIR / "feature_names.joblib")
    metadata = {
        "model": "RandomForestClassifier",
        "target_mapping": {"0": "Good Credit", "1": "Bad Credit"},
        "training_samples": int(len(X_train)),
        "held_out_test_samples": int(len(X_test)),
        "raw_feature_count": int(X.shape[1]),
        "processed_feature_count": int(X_train_processed.shape[1]),
        "random_state": 42,
        "threshold": 0.5,
        "parameters": model.get_params(deep=False),
        "test_metrics": metrics,
        "note": "Academic prototype; not a lending decision system.",
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (RESULTS_DIR / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2))

    predictions = X_test.reset_index(drop=True).copy()
    predictions["actual_credit_risk"] = y_test.reset_index(drop=True)
    predictions["predicted_credit_risk"] = pred
    predictions["bad_credit_probability"] = prob
    predictions.to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)
    X_test.to_csv(ROOT / "data/processed/X_test.csv", index=False)
    y_test.to_csv(ROOT / "data/processed/y_test.csv", index=False, header=["credit_risk"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1],
        display_labels=["Good", "Bad"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Random Forest — Held-out Test Confusion Matrix")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "final_confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax, name="Random Forest")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("Held-out ROC Curve")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "final_roc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, prob, ax=ax, name="Random Forest")
    ax.set_title("Held-out Precision–Recall Curve")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "final_pr_curve.png", dpi=160); plt.close(fig)

    # Five-fold CV is performed only on the 800-row training partition.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=5000, class_weight="balanced"), True),
        "Random Forest": (RandomForestClassifier(
            n_estimators=300, max_depth=15, max_features="sqrt",
            min_samples_split=10, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ), False),
        "SVM (RBF)": (SVC(C=1.0, kernel="rbf", probability=True,
                         class_weight="balanced", random_state=42), True),
    }
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision",
    }
    comparison_rows = []
    for name, (estimator, scale_numeric) in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(scale_numeric=scale_numeric)), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                                n_jobs=1, return_train_score=False)
        row = {"model": name}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1))
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "model_comparison_cv.csv", index=False)

    print("LoanShield credit model trained successfully")
    print(f"Training rows: {len(X_train)} | held-out rows: {len(X_test)}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
