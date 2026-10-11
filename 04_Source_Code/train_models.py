"""
train_models.py
================
Trains and compares three wildfire-occurrence classifiers using a
CHRONOLOGICAL (time-based) train/validation/test split -- never a random
split, because randomly shuffling dates would leak future weather patterns
into training and produce an unrealistically optimistic (and scientifically
invalid) evaluation for a forecasting task.

Models:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (non-linear baseline, handles interactions well)
  3. XGBoost if installed, otherwise sklearn GradientBoostingClassifier
     (documented, automatic fallback -- see utils.try_import)
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import config
from utils import get_logger, timed, try_import

logger = get_logger(__name__)

xgboost = try_import("xgboost")


@timed
def time_based_split(df: pd.DataFrame):
    """
    Chronological split: earliest TRAIN_FRACTION of unique dates -> train,
    next VAL_FRACTION -> validation, final TEST_FRACTION -> test.
    Returns (train_df, val_df, test_df).
    """
    dates = sorted(df["date"].unique())
    n = len(dates)
    n_train = int(n * config.TRAIN_FRACTION)
    n_val = int(n * config.VAL_FRACTION)

    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train:n_train + n_val])
    test_dates = set(dates[n_train + n_val:])

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    logger.info(f"Time-based split -> train: {len(train_df)} rows ({min(train_dates)} to {max(train_dates)}), "
                f"val: {len(val_df)} rows, test: {len(test_df)} rows")
    return train_df, val_df, test_df


def _prepare_xy(df: pd.DataFrame):
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


@timed
def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train Logistic Regression, Random Forest, and XGBoost/GradientBoosting
    on the chronological training split. Returns a dict of fitted model
    bundles (model, scaler) plus the train/val/test splits for downstream
    evaluation, keyed by model name.
    """
    train_df, val_df, test_df = time_based_split(df)

    X_train, y_train = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)
    X_test, y_test = _prepare_xy(test_df)

    logger.info(f"Class balance -- train positive rate: {y_train.mean():.3%}, "
                f"val: {y_val.mean():.3%}, test: {y_test.mean():.3%}")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    models = {}

    logger.info("Training Logistic Regression baseline...")
    logreg = LogisticRegression(**config.LOGREG_PARAMS)
    logreg.fit(X_train_scaled, y_train)
    models["logistic_regression"] = {"model": logreg, "scaler": scaler, "uses_scaler": True}

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    models["random_forest"] = {"model": rf, "scaler": None, "uses_scaler": False}

    if xgboost is not None:
        logger.info("Training XGBoost (installed) ...")
        pos = max(y_train.sum(), 1)
        neg = max(len(y_train) - y_train.sum(), 1)
        gb = xgboost.XGBClassifier(scale_pos_weight=neg / pos, **config.XGB_PARAMS)
        gb.fit(X_train, y_train)
        gb_name = "xgboost"
    else:
        logger.info("XGBoost not installed -- falling back to sklearn GradientBoostingClassifier")
        # GradientBoostingClassifier has no class_weight / scale_pos_weight parameter (unlike
        # RandomForest/LogisticRegression/XGBoost above), so without explicit handling it
        # collapses to predicting the majority class under this level of imbalance. sample_weight
        # is sklearn's native equivalent -- same pos/neg ratio idea as XGBoost's scale_pos_weight.
        pos = max(y_train.sum(), 1)
        neg = max(len(y_train) - y_train.sum(), 1)
        sample_weight = np.where(y_train == 1, neg / pos, 1.0)
        gb = GradientBoostingClassifier(**config.GB_FALLBACK_PARAMS)
        gb.fit(X_train, y_train, sample_weight=sample_weight)
        gb_name = "gradient_boosting_fallback"
    models["gradient_boosting"] = {"model": gb, "scaler": None, "uses_scaler": False, "backend": gb_name}

    for name, bundle in models.items():
        model_path = config.MODELS_DIR / f"{name}.joblib"
        joblib.dump(bundle["model"], model_path)
        if bundle["scaler"] is not None:
            joblib.dump(bundle["scaler"], config.MODELS_DIR / f"{name}_scaler.joblib")
        logger.info(f"Saved trained model -> {model_path}")

    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    meta = {"gb_backend": models["gradient_boosting"].get("backend", "xgboost"),
            "feature_columns": config.FEATURE_COLUMNS,
            "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
            "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean())}
    with open(config.MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"models": models, "splits": splits, "meta": meta}


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset(source="demo")
    result = train_all_models(df)
    logger.info(f"Training complete. Models saved to {config.MODELS_DIR}")
