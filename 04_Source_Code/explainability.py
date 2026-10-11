"""
explainability.py
==================
Model explanation utilities. Uses SHAP if installed for the tree-based
models; falls back gracefully to permutation importance / built-in feature
importance / logistic-regression coefficients if SHAP is unavailable or
fails (e.g. version mismatch, memory limits).

IMPORTANT: feature importance / SHAP values show ASSOCIATION learned by the
model, not proven causal drivers of real wildfire ignition. Do not present
these as causal claims.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config
from utils import get_logger, try_import

logger = get_logger(__name__)

shap = try_import("shap")


def explain_model(bundle: dict, model_name: str, X_sample: pd.DataFrame, y_sample: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame of feature -> importance_score for the given model,
    using SHAP mean(|value|) if available, otherwise a documented fallback.
    Also saves the result to 02_Data/processed/explainability_<model>.csv.
    """
    model = bundle["model"]
    method = "shap"

    try:
        if shap is None:
            raise RuntimeError("shap not installed")
        if model_name == "logistic_regression":
            X_input = bundle["scaler"].transform(X_sample)
            explainer = shap.LinearExplainer(model, X_input)
            shap_values = explainer.shap_values(X_input)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        importance = np.abs(shap_values).mean(axis=0)
        result = pd.DataFrame({"feature": config.FEATURE_COLUMNS, "importance": importance})
    except Exception as e:  # noqa: BLE001 -- intentional broad catch for graceful fallback
        logger.warning(f"SHAP unavailable/failed for {model_name} ({e}); using fallback explainability")
        method = "fallback"
        if model_name == "logistic_regression":
            coefs = np.abs(model.coef_[0])
            result = pd.DataFrame({"feature": config.FEATURE_COLUMNS, "importance": coefs})
            method = "logistic_regression_coefficients"
        elif hasattr(model, "feature_importances_"):
            result = pd.DataFrame({"feature": config.FEATURE_COLUMNS,
                                    "importance": model.feature_importances_})
            method = "builtin_feature_importance"
        else:
            X_input = X_sample
            perm = permutation_importance(model, X_input, y_sample, n_repeats=5,
                                           random_state=config.RANDOM_SEED, n_jobs=1)
            result = pd.DataFrame({"feature": config.FEATURE_COLUMNS,
                                    "importance": perm.importances_mean})
            method = "permutation_importance"

    result = result.sort_values("importance", ascending=False).reset_index(drop=True)
    result["method"] = method
    out_path = config.PROCESSED_DATA_DIR / f"explainability_{model_name}.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"[{model_name}] explainability method = {method}; top feature = {result.iloc[0]['feature']}")
    return result


def top_reasons_for_cell(explainability_df: pd.DataFrame, cell_row: pd.Series, top_n: int = 5) -> list[str]:
    """
    Human-readable "why is this cell risky" bullet list, combining global
    feature importance ranking with the cell's own feature values.
    Used by the dashboard's per-cell explanation panel.
    """
    reasons = []
    label_map = {
        "temperature": "high temperature", "humidity": "low humidity",
        "wind_speed": "strong wind", "rainfall_7day_sum": "low recent rainfall",
        "days_since_rain": "prolonged dry spell", "ndvi": "low vegetation moisture (NDVI)",
        "ndvi_change": "vegetation drying trend", "fire_count_previous_7_days": "recent fire activity nearby",
        "fire_count_previous_30_days": "historical fire activity",
        "slope": "steep terrain", "elevation": "terrain elevation",
        "temperature_anomaly": "unusually hot conditions",
    }
    for feat in explainability_df["feature"].head(top_n):
        if feat in cell_row.index:
            reasons.append(label_map.get(feat, feat.replace("_", " ")))
    return reasons if reasons else ["insufficient data for explanation"]


if __name__ == "__main__":
    from data_loader import load_dataset
    from train_models import train_all_models

    df = load_dataset(source="demo")
    result = train_all_models(df)
    X_test, y_test = result["splits"]["test"]
    sample_idx = X_test.sample(min(1000, len(X_test)), random_state=config.RANDOM_SEED).index
    for name, bundle in result["models"].items():
        explain_model(bundle, name, X_test.loc[sample_idx], y_test.loc[sample_idx])
