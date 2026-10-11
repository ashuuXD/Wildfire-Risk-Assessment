# Data Pipeline

## Pipeline Stages
```
Data sources -> data ingestion -> cleaning -> missing-value handling ->
spatial grid creation -> spatial/temporal joining -> feature engineering ->
target generation -> exploratory analysis -> model training ->
temporal validation -> risk prediction -> GIS map -> explainability ->
fire-spread simulation -> dashboard
```

## Demo Mode (implemented, runs without any API key or internet access)
| Stage | Script | Output |
|---|---|---|
| Spatial grid + raw synthetic data | `generate_demo_data.py` | `02_Data/raw/synthetic_raw_weekly.csv` |
| Cleaning / missing values | `preprocessing.py` | `02_Data/processed/clean_weekly.csv`, `missing_value_report.csv` |
| Feature engineering + target | `feature_engineering.py` | `02_Data/processed/demo_wildfire_dataset.csv` |
| Training | `train_models.py` | `05_Models/*.joblib`, `training_metadata.json` |
| Evaluation | `evaluate_models.py` | `02_Data/processed/model_comparison.csv`, `05_Models/evaluation_detailed.json` |
| Explainability | `explainability.py` | `02_Data/processed/explainability_<model>.csv` |
| Risk mapping | `risk_mapping.py` | `06_GIS/risk_maps/wildfire_risk_map.html` |
| Fire spread | `fire_spread.py` | `06_GIS/spread_maps/fire_spread_map.html` |
| EDA plots | `visualization.py` | `02_Data/processed/eda_plots/*.png` |
| Dashboard | `07_Dashboard/app.py` | interactive local web app |

## Real-Data Adapters (documented pseudocode, NOT executed by default)
Kept in `data_loader.py`, deliberately separate from the demo path so the
demo never breaks due to a missing API key or network access:

- `load_firms_csv(csv_path)` -- parse a NASA FIRMS archive-download CSV
  (https://firms.modaps.eosdis.nasa.gov/download/), snap detections to the
  project grid, and aggregate into a weekly `fire_occurred_flag`.
- `load_weather_csv(csv_path)` -- parse gridded/station weather data (e.g.
  NASA POWER https://power.larc.nasa.gov/, or IMD), resample to weekly.
- `load_gee_export(export_path)` -- parse a Google Earth Engine CSV/GeoTIFF
  export (MODIS NDVI/NDMI, LST, land cover) reduced to the project grid.
- `load_dem_raster(raster_path)` -- parse a Copernicus DEM GLO-30 raster
  (https://registry.opendata.aws/copernicus-dem/) via `rasterio`, deriving
  elevation/slope/aspect per grid cell.

To switch the whole project to real data in Phase-II: implement the four
`TODO` blocks in these functions, then change `data_loader.load_dataset(source=...)`
from `"demo"` to `"real"`.

## Spatial Grid
Grid cells are built by `generate_demo_data.build_spatial_grid()` from
`config.STUDY_REGION` and `config.GRID_RESOLUTION_DEG`. Changing either
value in `config.py` regenerates the whole pipeline for a new region or
resolution automatically -- no other code changes required.
