"""LoanShield — Individual customer credit-risk assessment page.

This page loads the already-trained Random Forest model and fitted
preprocessor. It does not retrain either artifact.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explainability.local_shap import explain_customer


# -----------------------------------------------------------------------------
# Project paths and persisted model artifacts
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "credit"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"


# The column names and ordering below match the original German Credit dataset.
FEATURE_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
]


# -----------------------------------------------------------------------------
# German Credit category codes and display labels
# -----------------------------------------------------------------------------
CATEGORY_OPTIONS = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 to < 200 DM",
        "A13": ">= 200 DM / salary assignment",
        "A14": "No checking account",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 to < 500 DM",
        "A63": "500 to < 1,000 DM",
        "A64": ">= 1,000 DM",
        "A65": "Unknown / no savings account",
    },
    "credit_history": {
        "A30": "No previous credits / all paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits",
    },
    "purpose": {
        "A40": "New car",
        "A41": "Used car",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Other purpose",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 to < 4 years",
        "A74": "4 to < 7 years",
        "A75": ">= 7 years",
    },
    "personal_status": {
        "A91": "Male: divorced / separated",
        "A92": "Female: divorced / separated / married",
        "A93": "Male: single",
        "A94": "Male: married / widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings agreement / life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "Free",
    },
    "job": {
        "A171": "Unskilled / non-resident",
        "A172": "Unskilled / resident",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "None",
        "A192": "Yes, registered under customer's name",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

FIELD_LABELS = {
    "checking_status": "Checking account",
    "savings_status": "Savings account",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "employment": "Employment duration",
    "personal_status": "Personal status",
    "other_debtors": "Other debtors / guarantors",
    "property": "Property",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job category",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

NUMERIC_FEATURE_LABELS = {
    "duration": "Loan duration (months)",
    "credit_amount": "Credit amount",
    "installment_rate": "Installment rate (%)",
    "residence_since": "Years at current residence",
    "age": "Age",
    "existing_credits": "Existing credits",
    "num_dependents": "Number of dependents",
}


# -----------------------------------------------------------------------------
# Cached artifact loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LoanShield credit model...")
def load_credit_artifacts():
    """Load persisted model and fitted preprocessing pipeline once per session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Fitted preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------
def code_selectbox(label, field_name, default_code, key):
    """Render a category selectbox and return the underlying dataset code."""
    options = CATEGORY_OPTIONS[field_name]
    codes = list(options.keys())
    labels = list(options.values())
    default_index = codes.index(default_code)
    selected_label = st.selectbox(
        label,
        labels,
        index=default_index,
        key=key,
    )
    return codes[labels.index(selected_label)]


def make_human_readable_feature(feature_name):
    """Convert ColumnTransformer/OneHotEncoder feature names into readable text."""
    name = str(feature_name)

    if name.startswith("num__"):
        raw_name = name.split("__", 1)[1]
        return NUMERIC_FEATURE_LABELS.get(raw_name, raw_name.replace("_", " ").title())

    if name.startswith("cat__"):
        encoded_name = name.split("__", 1)[1]
        for field_name, options in CATEGORY_OPTIONS.items():
            prefix = f"{field_name}_"
            if encoded_name.startswith(prefix):
                code = encoded_name[len(prefix):]
                category_label = options.get(code, code)
                field_label = FIELD_LABELS.get(field_name, field_name)
                return f"{field_label}: {category_label}"
        return encoded_name.replace("_", " ").title()

    return name.replace("_", " ").title()


def render_shap_explanation(model, preprocessor, customer_df, bad_probability):
    """Render a local contribution chart and ranked SHAP feature table."""
    st.divider()
    st.subheader("🔍 Explainable AI — SHAP Analysis")
    st.caption(
        "The explanation describes how the trained model uses this customer's "
        "attributes. It does not establish causation or determine actual creditworthiness."
    )

    with st.spinner("Generating individual customer explanation..."):
        explanation = explain_customer(model, preprocessor, customer_df)

    shap_values = np.asarray(explanation.values, dtype=float).reshape(-1)
    feature_names = list(explanation.feature_names)

    if len(feature_names) != len(shap_values):
        st.warning(
            "The SHAP feature names and contribution values have different lengths. "
            "The local explanation cannot be displayed reliably."
        )
        return

    shap_df = pd.DataFrame(
        {
            "Feature": [make_human_readable_feature(name) for name in feature_names],
            "SHAP Value": shap_values,
        }
    )
    shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
    shap_df["Direction"] = np.where(
        shap_df["SHAP Value"] >= 0,
        "Towards Bad Credit",
        "Away from Bad Credit",
    )
    shap_df = shap_df.sort_values("Absolute Impact", ascending=False)

    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    reconstructed_output = base_value + float(shap_values.sum())

    st.markdown("#### Individual Customer Explanation")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Baseline model output", f"{base_value:.1%}")
    metric_col2.metric("Predicted Bad Credit probability", f"{bad_probability:.1%}")
    metric_col3.metric("Baseline + SHAP contributions", f"{reconstructed_output:.1%}")

    st.caption(
        "Positive SHAP values push the model output towards the Bad Credit class; "
        "negative values push it away. Values are shown in the model output scale."
    )

    # A standard Matplotlib horizontal bar chart is used instead of SHAP's
    # built-in plotting renderer to keep the page rendering predictable.
    plot_df = shap_df.head(12).sort_values("SHAP Value", ascending=True)
    bar_colors = [
        "#d95f5f" if value > 0 else "#3985c6"
        for value in plot_df["SHAP Value"]
    ]

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.48 * len(plot_df))))
    ax.barh(plot_df["Feature"], plot_df["SHAP Value"], color=bar_colors)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_xlabel("SHAP contribution")
    ax.set_title("Features with the strongest influence on this prediction")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### Top Contributing Features")
    display_df = shap_df[
        ["Feature", "SHAP Value", "Direction"]
    ].head(10).copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SHAP Value": st.column_config.NumberColumn(
                "SHAP Value",
                format="%.5f",
            ),
        },
    )

    st.caption(
        "SHAP explains the behaviour of this Random Forest model for this input. "
        "It is not a causal explanation and should not be used as the sole basis "
        "for an actual lending decision."
    )


# -----------------------------------------------------------------------------
# Page heading and model loading
# -----------------------------------------------------------------------------
st.title("🏦 Customer Credit Assessment")
st.markdown(
    "Enter the customer's financial information below. LoanShield will estimate "
    "the probability of the **Bad Credit** class using the trained Random Forest model."
)

try:
    model, preprocessor = load_credit_artifacts()
except Exception as exc:
    st.error(f"Unable to load LoanShield model artifacts: {exc}")
    st.info(
        "Check that `random_forest.joblib` and `preprocessor.joblib` are present "
        "inside the project's `models/credit/` directory."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Customer input form
# -----------------------------------------------------------------------------
with st.form("customer_assessment_form", clear_on_submit=False):
    st.subheader("Customer Information")

    numeric_col1, numeric_col2, numeric_col3 = st.columns(3)

    with numeric_col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35, step=1
        )
        duration = st.number_input(
            "Loan Duration (months)", min_value=4, max_value=72, value=24, step=1
        )
        credit_amount = st.number_input(
            "Credit Amount", min_value=250, max_value=20000, value=3000, step=50
        )

    with numeric_col2:
        installment_rate = st.number_input(
            "Installment Rate (%)", min_value=1, max_value=4, value=3, step=1
        )
        residence_since = st.number_input(
            "Years at Current Residence", min_value=1, max_value=4, value=3, step=1
        )
        existing_credits = st.number_input(
            "Number of Existing Credits", min_value=1, max_value=4, value=1, step=1
        )

    with numeric_col3:
        num_dependents = st.number_input(
            "Number of Dependents", min_value=1, max_value=2, value=1, step=1
        )

    st.subheader("Account and Credit History")
    account_col1, account_col2, account_col3 = st.columns(3)

    with account_col1:
        checking_status = code_selectbox(
            "Checking Account Status", "checking_status", "A11", "checking_status"
        )
        savings_status = code_selectbox(
            "Savings Account", "savings_status", "A61", "savings_status"
        )

    with account_col2:
        credit_history = code_selectbox(
            "Credit History", "credit_history", "A30", "credit_history"
        )
        purpose = code_selectbox(
            "Loan Purpose", "purpose", "A40", "purpose"
        )

    with account_col3:
        employment = code_selectbox(
            "Employment Duration", "employment", "A71", "employment"
        )
        personal_status = code_selectbox(
            "Personal Status", "personal_status", "A91", "personal_status"
        )

    st.subheader("Personal and Employment Information")
    personal_col1, personal_col2, personal_col3 = st.columns(3)

    with personal_col1:
        other_debtors = code_selectbox(
            "Other Debtors / Guarantors", "other_debtors", "A101", "other_debtors"
        )
        property_status = code_selectbox(
            "Property", "property", "A121", "property"
        )

    with personal_col2:
        housing = code_selectbox(
            "Housing", "housing", "A151", "housing"
        )
        job = code_selectbox(
            "Job", "job", "A171", "job"
        )

    with personal_col3:
        other_installment_plans = code_selectbox(
            "Other Installment Plans",
            "other_installment_plans",
            "A141",
            "other_installment_plans",
        )
        telephone = code_selectbox(
            "Telephone", "telephone", "A191", "telephone"
        )
        foreign_worker = code_selectbox(
            "Foreign Worker", "foreign_worker", "A201", "foreign_worker"
        )

    submitted = st.form_submit_button(
        "🔍 Assess Customer",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Prediction and individual explanation
# -----------------------------------------------------------------------------
if submitted:
    customer_data = {
        "checking_status": checking_status,
        "duration": int(duration),
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": int(credit_amount),
        "savings_status": savings_status,
        "employment": employment,
        "installment_rate": int(installment_rate),
        "personal_status": personal_status,
        "other_debtors": other_debtors,
        "residence_since": int(residence_since),
        "property": property_status,
        "age": int(age),
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": int(existing_credits),
        "job": job,
        "num_dependents": int(num_dependents),
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }

    customer_df = pd.DataFrame([customer_data], columns=FEATURE_COLUMNS)

    try:
        # Use the fitted preprocessing pipeline saved during model training.
        # Never fit or retrain it on a single customer record.
        X_processed = preprocessor.transform(customer_df)
        predicted_class = int(model.predict(X_processed)[0])
        probabilities = model.predict_proba(X_processed)[0]

        model_classes = list(model.classes_)
        if 1 not in model_classes:
            raise ValueError(
                f"The trained model does not contain class 1. Classes: {model_classes}"
            )
        bad_class_index = model_classes.index(1)
        bad_probability = float(probabilities[bad_class_index])

    except Exception as exc:
        st.error(f"Customer assessment failed: {exc}")
        st.stop()

    st.divider()
    st.subheader("Assessment Result")

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric(
            "Predicted Bad-Credit Probability",
            f"{bad_probability:.1%}",
        )
    with result_col2:
        classification = "Bad Credit" if predicted_class == 1 else "Good Credit"
        st.metric("Model Classification", classification)

    if predicted_class == 1:
        st.error(f"Model classification: **{classification}**")
    else:
        st.success(f"Model classification: **{classification}**")

    st.caption(
        "This is the trained model's estimated probability of the Bad Credit class. "
        "It is not a probability of default calibrated for real-world lending, "
        "nor an automatic lending decision."
    )

    try:
        render_shap_explanation(
            model=model,
            preprocessor=preprocessor,
            customer_df=customer_df,
            bad_probability=bad_probability,
        )
    except Exception as exc:
        st.warning(
            "The credit prediction completed, but the SHAP explanation could not "
            "be rendered. The prediction itself is unaffected."
        )
        st.exception(exc)


# -----------------------------------------------------------------------------
# Prototype disclaimer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "LoanShield is an academic decision-support prototype based on the German "
    "Credit dataset. It should not be used to approve or reject real loan applications."
)
