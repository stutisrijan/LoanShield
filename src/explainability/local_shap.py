"""SHAP helpers. Class 1 corresponds to Bad Credit."""
import numpy as np
import shap


def _class_shap_values(raw_values, class_index: int = 1):
    if isinstance(raw_values, list):
        if len(raw_values) <= class_index:
            raise ValueError("SHAP output does not contain the requested class")
        return np.asarray(raw_values[class_index])

    arr = np.asarray(raw_values)
    if arr.ndim == 3:
        # SHAP versions have used both (samples, features, classes) and
        # (samples, classes, features) for multi-output tree explanations.
        if arr.shape[-1] > class_index:
            return arr[:, :, class_index]
        if arr.shape[1] > class_index:
            return arr[:, class_index, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")


def explain_customer(model, preprocessor, customer):
    """Return a SHAP Explanation for one row and the Bad Credit class."""
    X_processed = preprocessor.transform(customer)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    try:
        raw_values = explainer.shap_values(X_processed, check_additivity=False)
    except TypeError:
        raw_values = explainer.shap_values(X_processed)
    values_all = _class_shap_values(raw_values, class_index=1)
    values = np.asarray(values_all[0], dtype=float).reshape(-1)

    expected = np.asarray(explainer.expected_value)
    if expected.ndim == 0:
        base_value = float(expected)
    else:
        flat = expected.reshape(-1)
        base_value = float(flat[1] if len(flat) > 1 else flat[0])

    if len(values) != len(feature_names):
        raise ValueError(
            f"SHAP values ({len(values)}) do not match feature names ({len(feature_names)})"
        )
    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=np.asarray(X_processed[0]).reshape(-1),
        feature_names=list(feature_names),
    )


def global_shap_values(model, X_processed, class_index: int = 1):
    """Compute matrix of SHAP values for class 1."""
    explainer = shap.TreeExplainer(model)
    try:
        raw_values = explainer.shap_values(X_processed, check_additivity=False)
    except TypeError:
        raw_values = explainer.shap_values(X_processed)
    values = _class_shap_values(raw_values, class_index=class_index)
    return np.asarray(values, dtype=float)
