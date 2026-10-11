# Methodology

## 1. Data Generation / Ingestion
A synthetic demonstration dataset is generated with a fixed random seed
(`config.RANDOM_SEED = 42`), covering a configurable spatial grid over the
study region at weekly time steps for 5 years. Weather, vegetation, and
fire-ignition signals are built from seasonal (dry-season / monsoon)
sinusoidal patterns plus per-cell terrain-driven fuel factors, so the data
contains learnable, realistic relationships without being real observed
data. See `01_Research/SCOPE_AND_LIMITATIONS.md` and `generate_demo_data.py`.

## 2. Preprocessing
Duplicate rows are dropped, physically-impossible values are clipped
(e.g. humidity to [0,100]), and missing sensor readings (present by design
in ~1% of rows to demonstrate robust handling) are imputed via per-cell
forward/backward fill with a global median fallback (`preprocessing.py`).

## 3. Feature Engineering & Temporal-Leakage Prevention
All rolling/history features (`temperature_7day_mean`, `rainfall_7day_sum`,
`fire_count_previous_7_days`, `fire_count_previous_30_days`,
`days_since_rain`, `temperature_anomaly`) are computed using `.shift(1)`
before any rolling/aggregation operation, so a row's own current-week
values never leak into its own "recent history" features. The target,
`fire_target`, is the ignition flag for the **following** week, and the
raw same-week ignition flag is dropped from the feature set entirely once
the target is created (`feature_engineering.py`).

## 4. Chronological Train/Validation/Test Split
Dates are sorted; the earliest 70% form the training set, the next 15%
the validation set, and the final 15% the test set (`config.TRAIN_FRACTION`
etc., `train_models.time_based_split()`). A random split is deliberately
avoided because it would let the model see future weather patterns during
training -- an invalid setup for a forecasting task.

## 5. Model Training
Three classifiers are trained: Logistic Regression (scaled features, class-
weighted, interpretable baseline), Random Forest (class-weighted), and
XGBoost if installed, else scikit-learn's `GradientBoostingClassifier` as
an automatic, documented fallback (`train_models.py`). Random seeds are
fixed throughout for reproducibility.

## 6. Evaluation
Because wildfire occurrence is a rare-event, imbalanced target, accuracy
is reported but is explicitly not the primary metric. Precision, recall,
F1-score, ROC-AUC, PR-AUC, and the full confusion matrix are computed on
the held-out chronological test split for every model
(`evaluate_models.py`). The best model is selected by PR-AUC.

## 7. Explainability
SHAP is used when installed (`TreeExplainer` for tree models,
`LinearExplainer` for logistic regression); if SHAP is unavailable or
raises an error, the system falls back automatically to built-in feature
importance (tree models), coefficient magnitude (logistic regression), or
permutation importance (`explainability.py`). Results are surfaced both
globally (top features) and per-cell (top contributing factors) in the
dashboard.

## 8. Risk Mapping
Predicted probabilities are converted into four risk categories using
configurable thresholds (`config.RISK_THRESHOLDS`) and rendered as an
interactive, colour-coded Folium map with a legend (`risk_mapping.py`).

## 9. Fire-Spread Simulation
A probabilistic cellular-automata simulation propagates fire from an
ignition cell to its 8 neighbours at each discrete time step, with
ignition probability shaped by wind alignment/speed, upslope alignment,
land-cover fuel factor, and fuel moisture (`fire_spread.py`). This is
explicitly a simplified, explainable approximation, not a physically exact
model.

## 10. Dashboard Integration
All of the above are unified in a single local Streamlit dashboard
(`07_Dashboard/app.py`) with no external dependencies beyond the Python
packages in `requirements.txt`.
