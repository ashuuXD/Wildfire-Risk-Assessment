# 07_Dashboard

Streamlit dashboard for the Wildfire Risk Assessment & Fire-Spread
Decision Support prototype.

## Run
From the project root:
```
streamlit run 07_Dashboard/app.py
```
The first run will generate the demo dataset and train all three models
(~30-60 seconds); subsequent tab interactions reuse the cached results
within the session.

## Files
- `app.py` -- main Streamlit application (4 tabs: Risk Map, Model
  Performance & Explainability, Fire-Spread Simulation, About &
  Limitations).
- `dashboard_utils.py` -- data/model loading helpers used by `app.py`.

## Requirements
No database, cloud deployment, or API key required. Requires the packages
listed in the project root `requirements.txt` (streamlit, pandas, numpy,
scikit-learn, folium, matplotlib; xgboost and shap are optional with
automatic fallback).
