# Objectives

## Primary Objectives
1. Design and implement a reproducible data pipeline that converts raw
   environmental, vegetation, terrain, and historical-fire data into a
   model-ready, leak-free, grid-cell-level training dataset.
2. Train and compare multiple machine-learning classifiers (Logistic
   Regression, Random Forest, XGBoost/Gradient Boosting) to estimate
   wildfire occurrence probability, using an imbalance-aware, time-based
   evaluation methodology.
3. Convert model output into an interpretable spatial risk map with
   configurable risk categories (Low / Moderate / High / Extreme).
4. Provide model explainability (SHAP or a documented fallback) so
   predictions can be understood by non-ML stakeholders.

## Secondary Objectives
5. Implement a simplified, explainable fire-spread simulation
   (cellular-automata) that estimates which nearby cells may be affected
   given an ignition point and weather/terrain/fuel conditions.
6. Build an interactive Streamlit dashboard that unifies risk mapping,
   model evaluation, explainability, and fire-spread simulation into a
   single, locally runnable tool requiring no API key or cloud service.
7. Document the system thoroughly (architecture, methodology, data
   pipeline, model card, user guide) to a standard suitable for capstone
   evaluation and for a smooth Phase-II transition to real data.

## Explicitly Out of Scope for Phase-I
- Real-time operational deployment.
- Integration with live NASA FIRMS / weather / satellite APIs (documented
  as adapters, not executed by default).
- Physically exact fire-behaviour modelling (Rothermel-equivalent).
- Deep-learning / spatio-temporal neural architectures (documented as
  future work; classical ML is used for interpretability and to fit a
  short capstone timeline).
