"""Loading and preprocessing utilities for the UCI German Credit dataset."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "credit_risk_raw",
]

CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]
NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]
FEATURE_COLUMNS = [c for c in COLUMNS if c != "credit_risk_raw"]


def load_german_credit(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the whitespace-delimited original data; label 1=good, 2=bad."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"German Credit data not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    if frame.shape[1] != len(COLUMNS) or len(frame) != 1000:
        raise ValueError(f"Unexpected German Credit shape: {frame.shape}; expected 1000 x 21")
    y_raw = pd.to_numeric(frame.pop("credit_risk_raw"), errors="raise")
    if not set(y_raw.unique()).issubset({1, 2}):
        raise ValueError(f"Unexpected target labels: {sorted(y_raw.unique())}")
    y = y_raw.map({1: 0, 2: 1}).astype("int64")
    X = frame[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise")
    return X, y


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot encode categorical fields and pass numeric fields unchanged.

    The resulting 61-column representation is consistent with the project's
    saved-feature layout: 54 categorical indicator columns + 7 numeric fields.
    """
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
