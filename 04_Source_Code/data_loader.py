"""
data_loader.py
===============
Central data-loading entry point.

`load_dataset()` is what train_models.py / notebooks / dashboard should call.
It transparently builds the full demo pipeline (generate -> clean -> engineer)
if cached files are missing, or loads the cached processed CSV if present.

Real-data adapters below are DOCUMENTED PSEUDOCODE / thin wrappers only.
They are kept separate from the working demo path so the demo NEVER breaks
because of a missing API key or network access. To use them for Phase-II,
implement the marked TODOs and swap `load_dataset(source="demo")` for
`source="real"`.
"""

from __future__ import annotations
import pandas as pd

import config
from utils import get_logger

logger = get_logger(__name__)


def load_dataset(source: str = "demo", force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the model-ready wildfire dataset.

    Parameters
    ----------
    source : "demo" (default) uses the synthetic pipeline. "real" raises
        NotImplementedError with guidance, since no real data source is
        configured in this prototype.
    force_rebuild : if True, regenerate the demo pipeline even if a cached
        processed CSV already exists.
    """
    if source == "real":
        raise NotImplementedError(
            "Real-data mode is not implemented in this prototype. See the "
            "adapter functions in this file (load_firms_csv, load_weather_csv, "
            "load_gee_export, load_dem_raster) and 08_Documentation/DATA_PIPELINE.md "
            "for the documented plan to wire in NASA FIRMS / weather / satellite data."
        )

    out_path = config.PROCESSED_DATA_DIR / "demo_wildfire_dataset.csv"
    if out_path.exists() and not force_rebuild:
        logger.info(f"Loading cached demo dataset -> {out_path}")
        return pd.read_csv(out_path, parse_dates=["date"])

    logger.info("No cached demo dataset found (or rebuild forced) -- running full demo pipeline")
    from generate_demo_data import generate_raw_demo_data
    from preprocessing import clean_dataset, save_missing_value_report
    from feature_engineering import build_model_ready_dataset

    raw = generate_raw_demo_data(save=True)
    save_missing_value_report(raw, config.PROCESSED_DATA_DIR / "missing_value_report.csv")
    clean = clean_dataset(raw)
    clean.to_csv(config.PROCESSED_DATA_DIR / "clean_weekly.csv", index=False)
    df = build_model_ready_dataset(clean)
    df.to_csv(out_path, index=False)
    df.sample(min(2000, len(df)), random_state=config.RANDOM_SEED).to_csv(
        config.SAMPLE_DATA_DIR / "demo_wildfire_dataset_sample.csv", index=False)
    return pd.read_csv(out_path, parse_dates=["date"])


# ---------------------------------------------------------------------------
# REAL-DATA ADAPTERS (documented pseudocode / thin stubs -- NOT executed by
# the demo pipeline). Implement in Phase-II once API access is available.
# ---------------------------------------------------------------------------

def load_firms_csv(csv_path: str) -> pd.DataFrame:
    """
    Adapter for a NASA FIRMS archive-download CSV
    (https://firms.modaps.eosdis.nasa.gov/download/).
    Expected columns include: latitude, longitude, acq_date, acq_time,
    brightness, confidence, frp, satellite, instrument, daynight.

    TODO (Phase-II):
      1. pd.read_csv(csv_path)
      2. Filter to STUDY_REGION bounding box (config.STUDY_REGION)
      3. Snap latitude/longitude to the project grid (see
         generate_demo_data.build_spatial_grid) to get grid_id
      4. Aggregate multiple detections per (grid_id, date) into a single
         fire_occurred_flag = 1 row
    """
    raise NotImplementedError("Implement FIRMS CSV parsing for Phase-II; see docstring for the plan.")


def load_weather_csv(csv_path: str) -> pd.DataFrame:
    """
    Adapter for a gridded/station weather CSV (e.g. IMD, NASA POWER, or
    ERA5 export) with columns: date, latitude, longitude, temperature,
    humidity, wind_speed, wind_direction, rainfall, pressure.

    TODO (Phase-II): read, reproject/snap to grid, resample to weekly.
    NASA POWER API (no key required): https://power.larc.nasa.gov/
    """
    raise NotImplementedError("Implement weather CSV/API parsing for Phase-II.")


def load_gee_export(export_path: str) -> pd.DataFrame:
    """
    Adapter for a Google Earth Engine CSV/GeoTIFF export containing NDVI,
    NDMI, land-surface temperature, and land-cover classification for the
    study region grid.

    TODO (Phase-II): use the `earthengine-api` / `geemap` Python packages to
    export MOD13Q1 (NDVI/EVI), MOD11A2 (LST), and ESA WorldCover, reduced to
    the project grid via ee.Image.reduceRegions.
    """
    raise NotImplementedError("Implement Earth Engine export parsing for Phase-II.")


def load_dem_raster(raster_path: str) -> pd.DataFrame:
    """
    Adapter for a DEM raster (e.g. Copernicus GLO-30,
    https://registry.opendata.aws/copernicus-dem/) using rasterio to derive
    elevation, slope, and aspect per grid cell.

    TODO (Phase-II):
      import rasterio
      from rasterio.warp import transform_bounds
      with rasterio.open(raster_path) as src:
          ... reduce to grid cells, compute slope/aspect via numpy.gradient ...
    """
    raise NotImplementedError("Implement DEM raster parsing for Phase-II.")


if __name__ == "__main__":
    df = load_dataset(source="demo")
    logger.info(f"Loaded dataset: {df.shape}")
    print(df.head())
