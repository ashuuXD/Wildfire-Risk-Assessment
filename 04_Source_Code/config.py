"""
config.py
=========
Central configuration for the AI-Based Wildfire Risk Assessment and
Fire-Spread Decision Support System.

Every tunable parameter used across the project (paths, study region,
grid resolution, date ranges, model settings, risk thresholds, fire-spread
physics knobs, and dashboard defaults) lives here so the whole system can be
reconfigured from a single file.

IMPORTANT (Academic Honesty Notice):
This is a PROTOTYPE / DEMONSTRATION configuration. Default values (bounding
box, dates, thresholds) are reasonable starting points for a B.Tech capstone
prototype, not scientifically calibrated operational parameters.
"""

from __future__ import annotations
from pathlib import Path
import random
import numpy as np

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# PROJECT PATHS (relative, no hardcoded absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "02_Data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PLOTS_DIR = PROCESSED_DATA_DIR / "eda_plots"

MODELS_DIR = PROJECT_ROOT / "05_Models"

GIS_DIR = PROJECT_ROOT / "06_GIS"
RISK_MAPS_DIR = GIS_DIR / "risk_maps"
SPREAD_MAPS_DIR = GIS_DIR / "spread_maps"

DASHBOARD_DIR = PROJECT_ROOT / "07_Dashboard"
DOCS_DIR = PROJECT_ROOT / "08_Documentation"

# Create any missing directories automatically (never assume they exist)
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, PLOTS_DIR,
           MODELS_DIR, RISK_MAPS_DIR, SPREAD_MAPS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# STUDY REGION  (Default: Central India -- Madhya Pradesh & Chhattisgarh)
# ---------------------------------------------------------------------------
# NOTE (simplification): Real state boundaries are irregular polygons.
# For this capstone prototype we use a RECTANGULAR bounding box that loosely
# covers Madhya Pradesh + Chhattisgarh. This is a deliberate, documented
# simplification -- see 01_Research/SCOPE_AND_LIMITATIONS.md.
# Change this dict to point the whole system at a different region.
STUDY_REGION = {
    "name": "Central India (Madhya Pradesh & Chhattisgarh) - Demo Region",
    "min_lat": 17.8,
    "max_lat": 26.9,
    "min_lon": 74.0,
    "max_lon": 84.5,
}

# Grid resolution in decimal degrees (~ this many degrees per cell).
# 0.4 deg ~ 44 km at this latitude -- coarse on purpose, to keep the demo
# dataset small enough to train in seconds on a single-core laptop/VM.
GRID_RESOLUTION_DEG = 0.4

# ---------------------------------------------------------------------------
# DEMO DATE RANGE
# ---------------------------------------------------------------------------
# 5 years so that EVERY chronological split (train/val/test) contains at
# least one full seasonal cycle (fire season + monsoon) -- a single- or
# two-year range makes the val/test splits land entirely inside the
# monsoon and produces a near-zero positive rate there, which is
# statistically meaningless for evaluation.
DEMO_START_DATE = "2019-01-01"
DEMO_END_DATE = "2023-12-31"    # 5 fire seasons of synthetic weekly data
PREDICTION_WINDOW_DAYS = 7      # "will a fire occur in the next 7 days?"

# ---------------------------------------------------------------------------
# TIME-BASED TRAIN / VALIDATION / TEST SPLIT (chronological, not random)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# TARGET / FEATURE DEFINITIONS
# ---------------------------------------------------------------------------
TARGET_COLUMN = "fire_target"

FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "wind_direction", "rainfall",
    "pressure", "ndvi", "ndmi", "elevation", "slope", "aspect",
    "land_cover_code",
    "fire_count_previous_7_days", "fire_count_previous_30_days",
    "days_since_rain", "temperature_7day_mean", "humidity_7day_mean",
    "rainfall_7day_sum", "ndvi_change", "temperature_anomaly",
]

# ---------------------------------------------------------------------------
# RISK CLASSIFICATION THRESHOLDS  (PROTOTYPE VALUES -- require calibration)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": (0.00, 0.25),
    "Moderate": (0.25, 0.50),
    "High": (0.50, 0.75),
    "Extreme": (0.75, 1.00),
}

RISK_COLORS = {
    "Low": "#2ecc71",       # green
    "Moderate": "#f1c40f",  # yellow
    "High": "#e67e22",      # orange
    "Extreme": "#e74c3c",   # red
}


def classify_risk(probability: float) -> str:
    """Map a predicted probability [0,1] to a prototype risk category."""
    for label, (low, high) in RISK_THRESHOLDS.items():
        if label == "Extreme":
            if low <= probability <= high:
                return label
        else:
            if low <= probability < high:
                return label
    return "Low"


# ---------------------------------------------------------------------------
# MODEL TRAINING SETTINGS
# ---------------------------------------------------------------------------
MODEL_LIST = ["logistic_regression", "random_forest", "gradient_boosting"]
# "gradient_boosting" becomes XGBoost automatically if xgboost is installed,
# otherwise it falls back to sklearn's GradientBoostingClassifier.

RF_PARAMS = dict(n_estimators=150, max_depth=8, min_samples_leaf=5,
                  random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)
LOGREG_PARAMS = dict(max_iter=2000, class_weight="balanced",
                      random_state=RANDOM_SEED)
XGB_PARAMS = dict(n_estimators=150, max_depth=5, learning_rate=0.08,
                   subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", random_state=RANDOM_SEED)
GB_FALLBACK_PARAMS = dict(n_estimators=120, max_depth=3, learning_rate=0.08,
                           random_state=RANDOM_SEED)

# ---------------------------------------------------------------------------
# FIRE-SPREAD SIMULATION SETTINGS
# ---------------------------------------------------------------------------
SPREAD_TIME_STEPS_MINUTES = [10, 30, 60, 120, 180]
DEFAULT_SPREAD_DURATION_MIN = 60
SPREAD_CELL_STEP_MINUTES = 10          # simulation resolution
BASE_SPREAD_PROBABILITY = 0.18         # baseline chance a burning cell ignites a calm, neutral neighbour per step

# ---------------------------------------------------------------------------
# LAND COVER CODES (simplified demo classes)
# ---------------------------------------------------------------------------
LAND_COVER_CLASSES = {
    0: "Water",
    1: "Barren/Urban",
    2: "Cropland",
    3: "Grassland/Shrubland",
    4: "Open Forest",
    5: "Dense Forest",
}
# Relative fuel-load multiplier used by the fire-spread module
LAND_COVER_FUEL_FACTOR = {
    0: 0.0, 1: 0.05, 2: 0.35, 3: 0.6, 4: 0.85, 5: 1.0,
}

# ---------------------------------------------------------------------------
# DASHBOARD DEFAULTS
# ---------------------------------------------------------------------------
DASHBOARD_TITLE = "AI-Based Wildfire Risk Assessment & Fire-Spread Decision Support (Prototype)"
DEMO_MODE_NOTICE = (
    "DEMO MODE: This dashboard runs on a synthetically generated demonstration "
    "dataset for software validation only. It does NOT use live NASA FIRMS, "
    "weather, or satellite data, and must NOT be used for real wildfire "
    "detection, monitoring, or emergency response."
)
