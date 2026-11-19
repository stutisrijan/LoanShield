from pathlib import Path
from src.preprocessing.credit_preprocessor import load_german_credit, build_preprocessor, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_german_credit_loader_shape_and_target():
    X, y = load_german_credit(ROOT / "data/raw/german.data")
    assert X.shape == (1000, 20)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert y.value_counts().to_dict() == {0: 700, 1: 300}


def test_preprocessor_expected_width():
    X, _ = load_german_credit(ROOT / "data/raw/german.data")
    prep = build_preprocessor()
    transformed = prep.fit_transform(X.iloc[:800])
    assert transformed.shape == (800, 61)
