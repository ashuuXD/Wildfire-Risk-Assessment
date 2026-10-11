"""
visualization.py
=================
Exploratory Data Analysis (EDA) plotting functions. All plots are saved as
PNG files to 02_Data/processed/eda_plots/ for reuse in notebooks, the
dashboard, and the presentation. Uses matplotlib only (no seaborn
dependency) to keep the environment minimal.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import get_logger, ensure_dir

logger = get_logger(__name__)

PLOT_STYLE = {"figsize": (8, 5), "dpi": 110}


def _save(fig, name: str):
    ensure_dir(config.PLOTS_DIR)
    path = config.PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    logger.info(f"Saved plot -> {path}")
    return path


def plot_feature_distributions(df: pd.DataFrame, cols=None):
    cols = cols or ["temperature", "humidity", "wind_speed", "rainfall", "ndvi", "elevation"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col].dropna(), bins=40, color="#3b6e8f")
        ax.set_title(col)
    fig.suptitle("Feature Distributions (Demo Dataset)")
    fig.tight_layout()
    return _save(fig, "feature_distributions.png")


def plot_missing_values(missing_report: pd.DataFrame):
    report = missing_report[missing_report["missing_count"] > 0].sort_values("missing_count", ascending=True)
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    if len(report):
        ax.barh(report["column"], report["missing_count"], color="#c0392b")
    ax.set_title("Missing Values by Column (Raw Demo Data)")
    ax.set_xlabel("Missing count")
    fig.tight_layout()
    return _save(fig, "missing_values.png")


def plot_correlation_matrix(df: pd.DataFrame):
    numeric_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns] + [config.TARGET_COLUMN]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return _save(fig, "correlation_matrix.png")


def plot_monthly_fire_counts(df: pd.DataFrame):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month
    monthly = d.groupby("month")[config.TARGET_COLUMN].sum()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.bar(monthly.index, monthly.values, color="#d35400")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Fire-positive grid-weeks")
    ax.set_title("Seasonal Fire Occurrence Pattern (Demo Data)\nPeak expected Feb-May (dry season)")
    fig.tight_layout()
    return _save(fig, "monthly_fire_counts.png")


def plot_feature_vs_fire(df: pd.DataFrame, feature: str):
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.boxplot([df[df[config.TARGET_COLUMN] == 0][feature].dropna(),
                df[df[config.TARGET_COLUMN] == 1][feature].dropna()],
               tick_labels=["No Fire", "Fire"])
    ax.set_title(f"{feature} vs Fire Occurrence")
    ax.set_ylabel(feature)
    fig.tight_layout()
    return _save(fig, f"{feature}_vs_fire.png")


def plot_spatial_fire_density(df: pd.DataFrame):
    fires = df[df[config.TARGET_COLUMN] == 1]
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.hexbin(df["longitude"], df["latitude"], gridsize=25, cmap="Greys", mincnt=1, alpha=0.4)
    if len(fires):
        ax.scatter(fires["longitude"], fires["latitude"], s=4, c="red", alpha=0.4, label="Fire event")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of Synthetic Fire Events")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "spatial_fire_density.png")


def plot_model_comparison(comparison_df: pd.DataFrame):
    metrics = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    width = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[m], width, label=m)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df["model"], rotation=15)
    ax.set_title("Model Comparison (Test Split, Chronological)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_confusion_matrix(cm, model_name: str, labels=("No Fire", "Fire")):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, f"confusion_matrix_{model_name}.png")


def plot_feature_importance(explain_df: pd.DataFrame, model_name: str, top_n: int = 12):
    d = explain_df.sort_values("importance", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["feature"], d["importance"], color="#27ae60")
    ax.set_title(f"Feature Importance - {model_name} ({d['method'].iloc[0] if 'method' in d else ''})")
    fig.tight_layout()
    return _save(fig, f"feature_importance_{model_name}.png")


def run_all_eda_plots(df: pd.DataFrame, missing_report: pd.DataFrame | None = None):
    """Convenience function: generate the full standard EDA plot set in one call."""
    plot_feature_distributions(df)
    if missing_report is not None:
        plot_missing_values(missing_report)
    plot_correlation_matrix(df)
    plot_monthly_fire_counts(df)
    for feat in ["temperature", "humidity", "rainfall_7day_sum", "ndvi"]:
        plot_feature_vs_fire(df, feat)
    plot_spatial_fire_density(df)
    logger.info("All standard EDA plots generated.")


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset(source="demo")
    missing_path = config.PROCESSED_DATA_DIR / "missing_value_report.csv"
    missing_report = pd.read_csv(missing_path) if missing_path.exists() else None
    run_all_eda_plots(df, missing_report)
