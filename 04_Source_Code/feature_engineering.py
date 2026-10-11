"""
feature_engineering.py
=======================
Builds all model-ready predictor features and the leak-free binary target
from a CLEANED weekly grid dataset (output of preprocessing.py).

Temporal-leakage prevention
----------------------------
Every engineered feature for a given (grid_id, date) row uses ONLY
information available strictly BEFORE that date:
  - rolling means / sums use `.shift(1)` before rolling, so the current
    week's own reading is excluded from its own "recent history" features.
  - fire_count_previous_7_days / 30_days count PAST ignition events only.
  - the target `fire_target` is the ignition flag for the FOLLOWING week,
    i.e. it is deliberately shifted from the future into the current row's
    label column, then the model is trained to predict it from
    present-and-past features only. The raw future flag itself is dropped
    from the feature set so it can never leak into X.
"""

from __future__ import annotations
import pandas as pd

import config
from utils import get_logger, timed

logger = get_logger(__name__)


def _days_since_rain(rainfall_series: pd.Series) -> pd.Series:
    """Consecutive prior weeks with negligible rainfall, in days (uses only past values)."""
    out, counter = [], 0
    for val in rainfall_series.shift(1).fillna(0):
        counter = counter + 7 if val < 0.5 else 0
        out.append(counter)
    return pd.Series(out, index=rainfall_series.index)


@timed
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling weather stats, fire-history counts, vegetation change, and anomaly features."""
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)

    g = df.groupby("grid_id")

    df["fire_count_previous_7_days"] = g["fire_occurred_flag"].shift(1).fillna(0)
    df["fire_count_previous_30_days"] = (
        g["fire_occurred_flag"].shift(1).fillna(0)
        .groupby(df["grid_id"]).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    df["temperature_7day_mean"] = g["temperature"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(df["temperature"])
    df["humidity_7day_mean"] = g["humidity"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(df["humidity"])
    df["rainfall_7day_sum"] = g["rainfall"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).sum()).fillna(0)

    df["ndvi_change"] = g["ndvi"].diff().fillna(0)
    df["temperature_anomaly"] = df["temperature"] - g["temperature"].transform(
        lambda s: s.shift(1).expanding().mean()).fillna(df["temperature"])
    df["days_since_rain"] = g["rainfall"].transform(_days_since_rain)

    logger.info("Engineered features added: rolling weather means, fire history counts, "
                "NDVI change, temperature anomaly, days since rain")
    return df


@timed
def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the leak-free binary target: does a fire occur in the cell during
    the NEXT prediction window (config.PREDICTION_WINDOW_DAYS)? Drops the
    raw future-ignition column afterwards and drops the final timestep per
    cell (no future label available for it, so it cannot be evaluated honestly).
    """
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    df[config.TARGET_COLUMN] = (
        df.groupby("grid_id")["fire_occurred_flag"].shift(-1).fillna(0).astype(int)
    )
    df = df[df.groupby("grid_id")["date"].transform("max") != df["date"]]
    df = df.drop(columns=["fire_occurred_flag"]).reset_index(drop=True)
    logger.info(f"Target generated: fire_target positive rate = {df[config.TARGET_COLUMN].mean():.3%}")
    return df


@timed
def build_model_ready_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline: raw/clean -> engineered features -> leak-free target."""
    df = add_engineered_features(df_clean)
    df = generate_target(df)
    ordered = ["date", "latitude", "longitude", "grid_id"] + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    ordered = [c for c in ordered if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


if __name__ == "__main__":
    clean_path = config.PROCESSED_DATA_DIR / "clean_weekly.csv"
    if not clean_path.exists():
        from generate_demo_data import generate_raw_demo_data
        from preprocessing import clean_dataset
        raw = generate_raw_demo_data(save=True)
        clean = clean_dataset(raw)
        clean.to_csv(clean_path, index=False)
    df_clean = pd.read_csv(clean_path)
    df_model = build_model_ready_dataset(df_clean)
    out_path = config.PROCESSED_DATA_DIR / "demo_wildfire_dataset.csv"
    df_model.to_csv(out_path, index=False)
    df_model.sample(min(2000, len(df_model)), random_state=config.RANDOM_SEED).to_csv(
        config.SAMPLE_DATA_DIR / "demo_wildfire_dataset_sample.csv", index=False)
    logger.info(f"Saved model-ready dataset -> {out_path}")
