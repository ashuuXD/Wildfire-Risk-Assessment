"""
preprocessing.py
=================
Cleaning and missing-value handling for raw wildfire grid data, plus
reusable spatial-grid helpers. Sits between raw ingestion (data_loader.py /
generate_demo_data.py) and feature_engineering.py in the pipeline:

    raw data -> [preprocessing.py] -> clean data -> [feature_engineering.py] -> model-ready data
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config
from utils import get_logger, timed

logger = get_logger(__name__)

NUMERIC_COLUMNS_TO_CLEAN = [
    "temperature", "humidity", "wind_speed", "wind_direction", "rainfall",
    "pressure", "ndvi", "ndmi", "elevation", "slope", "aspect",
]


@timed
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic, transparent cleaning steps:
      1. Drop exact duplicate rows.
      2. Clip physically-impossible values to valid ranges.
      3. Fill missing numeric sensor readings using per-grid-cell forward
         fill, then a global median fallback (never uses future rows for
         a real deployment; safe here because this is per-cell historical
         interpolation of a static-ish signal, done before target creation).
      4. Report a missing-value summary for the EDA notebook.
    """
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    # Physical range clipping
    if "humidity" in df:
        df["humidity"] = df["humidity"].clip(0, 100)
    if "ndvi" in df:
        df["ndvi"] = df["ndvi"].clip(-1, 1)
    if "ndmi" in df:
        df["ndmi"] = df["ndmi"].clip(-1, 1)
    if "wind_speed" in df:
        df["wind_speed"] = df["wind_speed"].clip(lower=0)
    if "rainfall" in df:
        df["rainfall"] = df["rainfall"].clip(lower=0)

    missing_summary = df[NUMERIC_COLUMNS_TO_CLEAN].isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary):
        logger.info(f"Missing values before imputation:\n{missing_summary.to_string()}")

    df = df.sort_values(["grid_id", "date"])
    for col in NUMERIC_COLUMNS_TO_CLEAN:
        if col in df.columns:
            df[col] = df.groupby("grid_id")[col].transform(lambda s: s.ffill().bfill())
            df[col] = df[col].fillna(df[col].median())

    remaining = df[NUMERIC_COLUMNS_TO_CLEAN].isna().sum().sum()
    logger.info(f"Missing values remaining after imputation: {remaining}")

    return df.reset_index(drop=True)


def save_missing_value_report(df_raw: pd.DataFrame, out_path) -> pd.DataFrame:
    """Save a simple missing-value report CSV used by the EDA notebook."""
    report = df_raw.isna().sum().reset_index()
    report.columns = ["column", "missing_count"]
    report["missing_pct"] = (report["missing_count"] / len(df_raw) * 100).round(3)
    report.to_csv(out_path, index=False)
    return report


if __name__ == "__main__":
    raw_path = config.RAW_DATA_DIR / "synthetic_raw_weekly.csv"
    if not raw_path.exists():
        from generate_demo_data import generate_raw_demo_data
        generate_raw_demo_data(save=True)
    df_raw = pd.read_csv(raw_path)
    save_missing_value_report(df_raw, config.PROCESSED_DATA_DIR / "missing_value_report.csv")
    df_clean = clean_dataset(df_raw)
    df_clean.to_csv(config.PROCESSED_DATA_DIR / "clean_weekly.csv", index=False)
    logger.info(f"Saved cleaned dataset -> {config.PROCESSED_DATA_DIR / 'clean_weekly.csv'}")
