# Bank Customer Churn Prediction (Portfolio Project 1)

Predict which bank customers will leave (`Exited`), choose the decision threshold from a
business cost model, and produce a reproducible, tested pipeline.

## Key data finding: `Complain` is target leakage
`Complain` has a 0.996 correlation with `Exited` (99.8% of churners complained). It is
almost certainly recorded after the customer decided to leave, so it is excluded from the
model. Including it gives a meaningless ROC-AUC of ~1.0 (reproduce with `--include-complain`).

## Project layout
```
churn/            config.py data.py features.py models.py train.py evaluate.py pipeline.py
                  eda.py explain.py scoring.py
app/              streamlit_app.py (single-customer + batch scoring, SHAP reasons)
tests/            pytest suite (synthetic fixtures, mocked data loading)
config.yaml       optional overrides (paths, CV folds, business costs, hyperparameters)
data/raw/         put Customer-Churn-Records.csv here (git-ignored)
models/ reports/  generated artifacts (git-ignored)
```

## Run it (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m churn.eda          # figures + eda_summary.json in reports/
python -m churn.pipeline     # CV, final model, threshold, plots, metrics.json
python -m churn.explain      # SHAP plots, shap_importance.csv, churn_drivers.md
streamlit run app/streamlit_app.py   # interactive scoring app
pytest -q                    # run the tests
```
Useful flags: `--config config.yaml`, `--data-path`, `--log-level DEBUG`, `--include-complain`.

## What the pipeline does
1. Validates the schema (columns, dtypes, binary target, negative values) and fails loudly.
2. Drops IDs and leakage columns, de-duplicates on `CustomerId`.
3. Stratified 70/15/15 split (`random_state=42`); the test set is used once at the end.
4. Compares logistic regression, random forest, HistGradientBoosting, XGBoost and LightGBM
   with 5-fold stratified CV (preprocessing re-fitted inside every fold, class weights for imbalance).
5. Trains the best model (early stopping on validation for XGBoost/LightGBM).
6. Picks the threshold that maximises expected net benefit on the validation set.
7. Saves `models/churn_pipeline.joblib`, `reports/metrics.json` (metrics, hyperparameters,
   config, versions), the threshold sweep CSV and ROC/PR, confusion-matrix and
   threshold-vs-benefit plots.

## Business cost model (placeholders: edit `cost` in `config.yaml`)
A flagged customer gets a 100 offer; a real churner is saved with probability 0.3, worth 1000.
Net benefit = TP x 0.3 x 1000 - flagged x 100.

## Explainability and app
`python -m churn.explain` explains the saved model on held-out test rows with a
model-agnostic SHAP permutation explainer (works for every model in the comparison). It writes
the beeswarm and importance plots, `reports/shap_importance.csv` and a plain-English
`reports/churn_drivers.md`. The Streamlit app scores one customer or an uploaded CSV, shows the
risk tier against the cost-optimal threshold, rule-based retention ideas and a per-customer SHAP chart.

## Next steps for this project
Optuna tuning, Docker/FastAPI, 5-slide summary.
