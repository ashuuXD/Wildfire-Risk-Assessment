"""
generate_demo_data.py
======================
Generates a fully SYNTHETIC, clearly-labelled RAW demonstration dataset that
mimics the structure of real wildfire-risk source data, without using any
real NASA FIRMS / weather / satellite data.

Why synthetic data?
--------------------
Real data (NASA FIRMS active-fire detections, gridded weather, MODIS NDVI,
Copernicus DEM) normally requires internet access, API keys, and multi-GB
raster downloads that are not feasible inside a capstone demo session. This
generator builds RAW weekly grid-cell records with the SAME schema and
REALISTIC statistical relationships (dry + hot + windy + low-NDVI +
forested cells during the Feb-May fire season have a higher chance of
fire ignition) so the rest of the pipeline (preprocessing, feature
engineering, modelling, evaluation, risk mapping, explainability, fire
spread) can run end-to-end and be demonstrated live.

Output of this script = the "raw" layer. It intentionally does NOT contain
engineered features (rolling means, fire-history counts, target labels) --
those are added downstream by preprocessing.py and feature_engineering.py,
mirroring a realistic data-pipeline separation of concerns.

This is clearly a DEMONSTRATION DATASET for software validation only.
Real-world validation with NASA FIRMS / IMD weather / MODIS data is future work.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config
from utils import get_logger, timed

logger = get_logger(__name__)


def build_spatial_grid() -> pd.DataFrame:
    """
    Create the spatial grid of cells covering the configured study region
    (config.STUDY_REGION), at config.GRID_RESOLUTION_DEG resolution.
    Reused by preprocessing.py and the dashboard for consistent grid_ids.
    """
    lats = np.arange(config.STUDY_REGION["min_lat"], config.STUDY_REGION["max_lat"],
                      config.GRID_RESOLUTION_DEG)
    lons = np.arange(config.STUDY_REGION["min_lon"], config.STUDY_REGION["max_lon"],
                      config.GRID_RESOLUTION_DEG)

    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append({"row": i, "col": j, "latitude": round(float(lat), 4),
                         "longitude": round(float(lon), 4)})
    grid = pd.DataFrame(rows)
    grid["grid_id"] = grid.apply(lambda r: f"R{int(r['row']):03d}C{int(r['col']):03d}", axis=1)
    return grid


def _assign_static_terrain_and_landcover(grid: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Assign per-cell static attributes (elevation, slope, aspect, land cover)
    using smooth, spatially-correlated synthetic surfaces so neighbouring
    cells share similar terrain (realistic spatial autocorrelation), rather
    than pure random noise per cell.
    """
    n_rows = grid["row"].max() + 1
    n_cols = grid["col"].max() + 1

    elevation_field = np.zeros((n_rows, n_cols))
    for _ in range(3):
        freq_r = rng.uniform(0.5, 2.0)
        freq_c = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(80, 250)
        rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        elevation_field += amp * np.sin(freq_r * rr / n_rows * np.pi + phase) * \
            np.cos(freq_c * cc / n_cols * np.pi + phase)
    elevation_field = 250 + (elevation_field - elevation_field.min()) / \
        (elevation_field.max() - elevation_field.min()) * 700  # 250-950 m, plausible for central India

    grad_r, grad_c = np.gradient(elevation_field)
    slope_field = np.sqrt(grad_r ** 2 + grad_c ** 2)
    slope_field = (slope_field / slope_field.max()) * 30  # 0-30 degrees
    aspect_field = (np.degrees(np.arctan2(grad_c, grad_r)) + 360) % 360

    land_cover_field = np.empty((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            elev = elevation_field[i, j]
            noise = rng.random()
            if elev > 750 and noise > 0.25:
                land_cover_field[i, j] = 5 if noise > 0.55 else 4
            elif elev > 500:
                land_cover_field[i, j] = rng.choice([2, 3, 4], p=[0.4, 0.35, 0.25])
            else:
                land_cover_field[i, j] = rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30])
            if noise < 0.02:
                land_cover_field[i, j] = 0  # small chance of water body

    grid = grid.copy()
    grid["elevation"] = grid.apply(lambda r: round(float(elevation_field[int(r["row"]), int(r["col"])]), 1), axis=1)
    grid["slope"] = grid.apply(lambda r: round(float(slope_field[int(r["row"]), int(r["col"])]), 2), axis=1)
    grid["aspect"] = grid.apply(lambda r: round(float(aspect_field[int(r["row"]), int(r["col"])]), 1), axis=1)
    grid["land_cover_code"] = grid.apply(lambda r: int(land_cover_field[int(r["row"]), int(r["col"])]), axis=1)
    grid["land_cover"] = grid["land_cover_code"].map(config.LAND_COVER_CLASSES)
    return grid


@timed
def generate_raw_demo_data(save: bool = True) -> pd.DataFrame:
    """
    Build the RAW synthetic grid x date dataset: weather, vegetation,
    terrain, and a raw weekly fire-ignition flag. No engineered / rolling
    features and no leak-free target label yet -- those come from
    preprocessing.py and feature_engineering.py. Uses a fixed random seed
    for full reproducibility.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    grid = build_spatial_grid()
    grid = _assign_static_terrain_and_landcover(grid, rng)
    logger.info(f"Spatial grid built: {len(grid)} cells over {config.STUDY_REGION['name']}")

    dates = pd.date_range(config.DEMO_START_DATE, config.DEMO_END_DATE, freq="7D")
    logger.info(f"Generating {len(dates)} weekly time steps x {len(grid)} cells "
                f"= {len(dates) * len(grid)} raw rows")

    ndvi_base = grid["land_cover_code"].map({0: 0.05, 1: 0.15, 2: 0.45, 3: 0.35, 4: 0.55, 5: 0.7}).values
    fuel_factor = grid["land_cover_code"].map(config.LAND_COVER_FUEL_FACTOR).values

    all_rows = []
    for date in dates:
        day_of_year = date.dayofyear
        season_signal = np.exp(-0.5 * ((day_of_year - 90) / 45) ** 2)  # fire season peaks ~day 90 (late March)
        monsoon_signal = np.exp(-0.5 * ((day_of_year - 210) / 40) ** 2)  # monsoon peaks ~day 210 (late July)

        temperature = 22 + 14 * season_signal + rng.normal(0, 2.5, len(grid))
        humidity = np.clip(70 - 45 * season_signal + rng.normal(0, 8, len(grid)), 5, 100)
        wind_speed = np.clip(8 + 10 * season_signal + rng.normal(0, 3, len(grid)), 0, None)
        wind_direction = rng.uniform(0, 360, len(grid))
        rainfall = np.clip(rng.exponential(1.5, len(grid)) * (1 + 25 * monsoon_signal)
                            - 3 * season_signal, 0, None)
        pressure = 1005 + rng.normal(0, 4, len(grid))
        ndvi = np.clip(ndvi_base + 0.25 * monsoon_signal - 0.15 * season_signal
                        + rng.normal(0, 0.05, len(grid)), 0.02, 0.95)
        ndmi = np.clip(ndvi - 0.1 - 0.2 * season_signal + rng.normal(0, 0.05, len(grid)), -0.3, 0.9)

        rows = pd.DataFrame({
            "date": date, "grid_id": grid["grid_id"].values,
            "latitude": grid["latitude"].values, "longitude": grid["longitude"].values,
            "elevation": grid["elevation"].values, "slope": grid["slope"].values,
            "aspect": grid["aspect"].values, "land_cover_code": grid["land_cover_code"].values,
            "land_cover": grid["land_cover"].values,
            "temperature": np.round(temperature, 1), "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1), "wind_direction": np.round(wind_direction, 1),
            "rainfall": np.round(rainfall, 2), "pressure": np.round(pressure, 1),
            "ndvi": np.round(ndvi, 3), "ndmi": np.round(ndmi, 3),
        })

        # Latent fire-danger score drives the synthetic ignition probability this week
        danger = (
            0.035 * (rows["temperature"] - 25).clip(lower=0)
            + 0.02 * (55 - rows["humidity"]).clip(lower=0)
            + 0.05 * rows["wind_speed"]
            - 0.9 * rows["rainfall"].clip(upper=5)
            - 3.0 * rows["ndvi"]
            + 1.6 * fuel_factor
            + rng.normal(0, 0.6, len(grid))
        )
        fire_prob = 1 / (1 + np.exp(-(danger - 3.2)))
        rows["fire_occurred_flag"] = (rng.random(len(grid)) < fire_prob).astype(int)

        # Simulate a small amount of real-world messiness for the preprocessing
        # step to demonstrably handle (missing sensor readings, rare bad rows).
        missing_mask = rng.random(len(grid)) < 0.01
        rows.loc[missing_mask, "ndvi"] = np.nan
        missing_mask2 = rng.random(len(grid)) < 0.005
        rows.loc[missing_mask2, "humidity"] = np.nan

        all_rows.append(rows)

    df = pd.concat(all_rows, ignore_index=True).sort_values(["grid_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(f"Raw synthetic dataset generated: {len(df):,} rows | "
                f"raw ignition rate = {df['fire_occurred_flag'].mean():.3%} | "
                f"missing values injected for realism")

    if save:
        out_path = config.RAW_DATA_DIR / "synthetic_raw_weekly.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved RAW demo dataset -> {out_path}")

    return df


if __name__ == "__main__":
    generate_raw_demo_data(save=True)
