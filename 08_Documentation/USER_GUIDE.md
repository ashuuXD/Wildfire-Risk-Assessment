# User Guide

## 1. Setup
### Windows
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the Full Demo Pipeline
```
python run_demo.py
```
This generates synthetic data, cleans it, engineers features, trains all
three models, evaluates them, produces an explainability report, an
interactive risk map, a sample fire-spread simulation, and EDA plots --
all in under about a minute on a typical laptop. Console output shows
progress for each of the 8 pipeline stages.

## 3. Launch the Dashboard
```
streamlit run 07_Dashboard/app.py
```
Open the local URL Streamlit prints (typically http://localhost:8501).

### Dashboard Tabs
1. **Wildfire Risk Map** -- interactive map + per-cell details (weather,
   NDVI, risk probability/category, top contributing factors).
2. **Model Performance & Explainability** -- comparison table, confusion
   matrix, feature importance chart.
3. **Fire-Spread Simulation** -- pick an ignition cell, set wind
   speed/direction, duration, and fuel moisture, then run the simulation
   and view the affected-area map.
4. **About & Limitations** -- demo-mode notice, simplifications, and
   planned real data sources.

## 4. Explore Individual Pipeline Stages
Each module in `04_Source_Code/` can be run standalone, e.g.:
```
python 04_Source_Code/generate_demo_data.py
python 04_Source_Code/preprocessing.py
python 04_Source_Code/feature_engineering.py
python 04_Source_Code/train_models.py
python 04_Source_Code/evaluate_models.py
python 04_Source_Code/explainability.py
python 04_Source_Code/risk_mapping.py
python 04_Source_Code/fire_spread.py
python 04_Source_Code/visualization.py
```

## 5. Explore the Notebooks
Open any notebook in `03_Notebooks/` with Jupyter/JupyterLab or VS Code:
```
jupyter lab 03_Notebooks/
```
Each notebook imports the same source modules used by `run_demo.py`, so
results stay consistent between the scripts, dashboard, and notebooks.

## 6. Change the Study Region or Parameters
Edit `04_Source_Code/config.py`:
- `STUDY_REGION` -- bounding box (min/max lat/lon).
- `GRID_RESOLUTION_DEG` -- grid cell size in degrees.
- `DEMO_START_DATE` / `DEMO_END_DATE` -- synthetic data time range.
- `RISK_THRESHOLDS` -- Low/Moderate/High/Extreme probability cut-offs.
- `RF_PARAMS` / `LOGREG_PARAMS` / `XGB_PARAMS` -- model hyperparameters.

Then re-run `python run_demo.py` (delete `02_Data/processed/*.csv` first,
or set `force_rebuild=True` in `data_loader.load_dataset()`, to force
regeneration with the new settings).

## 7. Troubleshooting
- **"No module named X"** -- run `pip install -r requirements.txt` inside
  your activated virtual environment.
- **XGBoost / SHAP not installed** -- the pipeline automatically falls
  back to scikit-learn GradientBoosting / permutation importance; no
  action needed unless you specifically want XGBoost/SHAP results.
- **Folium map not rendering in the dashboard** -- ensure `folium` is
  installed; otherwise a CSV table fallback is shown instead.
- **Slow training** -- reduce `GRID_RESOLUTION_DEG` (coarser grid, fewer
  cells) or the demo date range in `config.py`.
