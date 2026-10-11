"""
run_demo.py
============
ONE-COMMAND DEMO RUNNER.

Executes the full pipeline end-to-end using the synthetic demonstration
dataset (no internet / API key required):

  generate raw data -> clean -> engineer features -> generate target
  -> train 3 models (chronological split) -> evaluate -> explainability
  -> risk map -> fire-spread simulation -> EDA plots

Run from the project root:
    python run_demo.py

Then launch the dashboard separately:
    streamlit run 07_Dashboard/app.py

DEMO MODE NOTICE: all data used here is synthetically generated for
software validation only. See 01_Research/SCOPE_AND_LIMITATIONS.md.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "04_Source_Code"))

import config  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger("run_demo")


def main():
    t0 = time.time()
    print("=" * 78)
    print(" AI-BASED WILDFIRE RISK ASSESSMENT & FIRE-SPREAD DECISION SUPPORT")
    print(" DEMO MODE -- synthetic demonstration dataset, software validation only")
    print("=" * 78)

    from generate_demo_data import generate_raw_demo_data, build_spatial_grid, _assign_static_terrain_and_landcover
    from preprocessing import clean_dataset, save_missing_value_report
    from feature_engineering import build_model_ready_dataset
    from train_models import train_all_models
    from evaluate_models import evaluate_all_models, best_model_name
    from explainability import explain_model
    from risk_mapping import build_risk_table, generate_risk_map_html
    from fire_spread import run_fire_spread_simulation, generate_spread_map_html
    from visualization import run_all_eda_plots
    import pandas as pd
    import numpy as np

    print("\n[1/8] Generating synthetic raw demo data ...")
    raw = generate_raw_demo_data(save=True)

    print("\n[2/8] Cleaning & handling missing values ...")
    missing_report = save_missing_value_report(raw, config.PROCESSED_DATA_DIR / "missing_value_report.csv")
    clean = clean_dataset(raw)
    clean.to_csv(config.PROCESSED_DATA_DIR / "clean_weekly.csv", index=False)

    print("\n[3/8] Engineering features & generating leak-free target ...")
    df = build_model_ready_dataset(clean)
    df.to_csv(config.PROCESSED_DATA_DIR / "demo_wildfire_dataset.csv", index=False)
    df.sample(min(2000, len(df)), random_state=config.RANDOM_SEED).to_csv(
        config.SAMPLE_DATA_DIR / "demo_wildfire_dataset_sample.csv", index=False)

    print("\n[4/8] Training models (Logistic Regression, Random Forest, XGBoost/GB) "
          "with a chronological train/val/test split ...")
    result = train_all_models(df)

    print("\n[5/8] Evaluating models (precision/recall/F1/ROC-AUC/PR-AUC, confusion matrix) ...")
    comparison = evaluate_all_models(result)
    best = best_model_name(comparison)
    print(comparison.to_string(index=False))
    print(f"  -> Best model by PR-AUC: {best}")

    print("\n[6/8] Generating explainability report for the best model ...")
    X_test, y_test = result["splits"]["test"]
    sample_idx = X_test.sample(min(1000, len(X_test)), random_state=config.RANDOM_SEED).index
    explain_df = explain_model(result["models"][best], best, X_test.loc[sample_idx], y_test.loc[sample_idx])
    print(f"  Top 5 features: {list(explain_df['feature'].head(5))}")

    print("\n[7/8] Generating GIS risk map for the most recent available week ...")
    latest_date = sorted(df["date"].unique())[-1]
    latest_cells = df[df["date"] == latest_date].reset_index(drop=True)
    latest_model = result["models"][best]["model"]
    if result["models"][best].get("uses_scaler"):
        X_latest = result["models"][best]["scaler"].transform(latest_cells[config.FEATURE_COLUMNS])
    else:
        X_latest = latest_cells[config.FEATURE_COLUMNS]
    latest_proba = latest_model.predict_proba(X_latest)[:, 1]
    risk_df = build_risk_table(latest_cells, latest_proba)
    risk_df.to_csv(config.PROCESSED_DATA_DIR / "latest_risk_predictions.csv", index=False)
    generate_risk_map_html(risk_df)
    print(f"  Risk category counts:\n{risk_df['risk_category'].value_counts().to_string()}")

    print("\n[8/8] Running a sample fire-spread simulation & EDA plots ...")
    rng = np.random.default_rng(config.RANDOM_SEED)
    spatial_grid = build_spatial_grid()
    spatial_grid = _assign_static_terrain_and_landcover(spatial_grid, rng)
    spatial_grid = spatial_grid.merge(
        latest_cells[["grid_id", "ndmi"]], on="grid_id", how="left")
    spatial_grid["ndmi"] = spatial_grid["ndmi"].fillna(0.15)
    ignition_id = risk_df.sort_values("risk_probability", ascending=False).iloc[0]["grid_id"]
    sim = run_fire_spread_simulation(spatial_grid, ignition_grid_id=ignition_id,
                                      wind_speed=20, wind_direction=90,
                                      duration_minutes=config.DEFAULT_SPREAD_DURATION_MIN)
    generate_spread_map_html(sim)
    print(f"  Ignition cell (highest risk): {ignition_id} | "
          f"Affected cells: {len(sim['affected_cells'])} | Area: ~{sim['affected_area_km2']} km2")

    run_all_eda_plots(df, missing_report)

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f" DEMO PIPELINE COMPLETE in {elapsed:.1f}s")
    print(" Outputs:")
    print(f"   - Processed data:     {config.PROCESSED_DATA_DIR}")
    print(f"   - Trained models:     {config.MODELS_DIR}")
    print(f"   - Risk map (HTML):    {config.RISK_MAPS_DIR / 'wildfire_risk_map.html'}")
    print(f"   - Fire spread map:    {config.SPREAD_MAPS_DIR / 'fire_spread_map.html'}")
    print(f"   - EDA plots:          {config.PLOTS_DIR}")
    print("\n Next: launch the interactive dashboard with:")
    print("   streamlit run 07_Dashboard/app.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
