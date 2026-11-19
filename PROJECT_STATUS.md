# Build and artifact status

Generated from the included `data/raw/german.data` on 2026-11-15.

## Available and generated

- Credit dataset: 1,000 records; 20 predictor fields; 700 Good / 300 Bad target labels.
- Credit split: stratified 800 training / 200 untouched held-out test records (`random_state=42`).
- Preprocessor output width: 61 features.
- Random Forest model and fitted preprocessor saved under `models/credit/`.
- Local SHAP utility: verified 61 contributions; `base value + SHAP sum` matched the class-1 model probability to numerical tolerance on a smoke-test row.
- Global SHAP utility: verified a 20 × 61 contribution matrix.
- K-Means artifacts and cluster profiles for k=3 generated.
- Evaluation CSV/JSON and confusion matrix, ROC and PR plots generated.
- Two preprocessing tests passed; source files compile.

## Current held-out result for this packaged rebuild

| Metric | Value |
|---|---:|
| Accuracy | 0.7400 |
| Precision | 0.5645 |
| Recall | 0.5833 |
| Specificity | 0.8071 |
| F1 | 0.5738 |
| ROC-AUC | 0.7939 |
| PR-AUC | 0.6459 |

Confusion matrix: TN=113, FP=27, FN=25, TP=35.

These values are for the newly generated model in this package. They may differ slightly from model metrics reported during earlier project iterations because this package rebuilds the model using the explicit preprocessing/training implementation included here.

## Not yet trained

The optional signature CNN has a model definition, training script, and Streamlit inference page. It has **no trained weights** in this package because no labelled genuine/forged signature dataset was provided. The page clearly reports this rather than generating a fake verification score.

## UI test status

Streamlit is listed in `requirements.txt`, but it was not installed in the packaging environment. The Streamlit source modules passed Python compilation; launch and browser interaction should be checked after installing requirements on the target machine.
