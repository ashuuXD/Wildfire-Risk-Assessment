"""
dashboard_utils.py
===================
Helper functions used by 07_Dashboard/app.py. Kept separate from app.py so
caching / data-loading logic is easy to test and reuse.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "04_Source_Code"
sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from data_loader import load_dataset  # noqa: E402
from train_models import train_all_models  # noqa: E402
from evaluate_models import evaluate_all_models, best_model_name  # noqa: E402
from explainability import explain_model, top_reasons_for_cell  # noqa: E402
from risk_mapping import build_risk_table  # noqa: E402
from fire_spread import run_fire_spread_simulation  # noqa: E402
from generate_demo_data import build_spatial_grid, _assign_static_terrain_and_landcover  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_pipeline_bundle():
    """
    Load (or train, if not cached) the full demo pipeline: dataset, trained
    models, evaluation comparison table, and the spatial grid used for the
    fire-spread module. This is the single entry point the dashboard calls.

    Wrapped in @st.cache_resource so training only happens ONCE per app
    process (not on every widget interaction/rerun) -- this is what was
    causing repeated ~60s retrains and multiple parallel training runs in
    the terminal log.
    """
    df = load_dataset(source="demo")
    result = train_all_models(df)
    comparison = evaluate_all_models(result)
    best = best_model_name(comparison)

    rng = np.random.default_rng(config.RANDOM_SEED)
    grid = build_spatial_grid()
    grid = _assign_static_terrain_and_landcover(grid, rng)

    return {"df": df, "result": result, "comparison": comparison, "best_model": best, "grid": grid}


def latest_week_risk_table(bundle: dict) -> pd.DataFrame:
    """Predicted risk table for the most recent week in the demo dataset, using the best model."""
    df = bundle["df"]
    best_bundle = bundle["result"]["models"][bundle["best_model"]]
    latest_date = sorted(df["date"].unique())[-1]
    latest_cells = df[df["date"] == latest_date].reset_index(drop=True)

    if best_bundle.get("uses_scaler"):
        X = best_bundle["scaler"].transform(latest_cells[config.FEATURE_COLUMNS])
    else:
        X = latest_cells[config.FEATURE_COLUMNS]
    proba = best_bundle["model"].predict_proba(X)[:, 1]
    return build_risk_table(latest_cells, proba)


@st.cache_data(show_spinner=False)
def get_explainability(_bundle: dict) -> pd.DataFrame:
    """Leading underscore on _bundle tells st.cache_data to skip hashing it
    (it contains unhashable model objects) while still caching the result
    for the lifetime of the app process, avoiding repeated SHAP/fallback runs."""
    best_bundle = _bundle["result"]["models"][_bundle["best_model"]]
    X_test, y_test = _bundle["result"]["splits"]["test"]
    sample_idx = X_test.sample(min(800, len(X_test)), random_state=config.RANDOM_SEED).index
    return explain_model(best_bundle, _bundle["best_model"], X_test.loc[sample_idx], y_test.loc[sample_idx])


def run_spread_simulation(bundle: dict, ignition_id: str, wind_speed: float,
                           wind_direction: float, duration_minutes: int, ndmi_value: float = 0.15):
    grid = bundle["grid"].copy()
    grid["ndmi"] = ndmi_value
    return run_fire_spread_simulation(grid, ignition_grid_id=ignition_id, wind_speed=wind_speed,
                                       wind_direction=wind_direction, duration_minutes=duration_minutes)
