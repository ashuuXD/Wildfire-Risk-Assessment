"""
evaluate_models.py
===================
Evaluates trained models on the held-out chronological TEST split using
imbalance-aware metrics. Accuracy is reported but explicitly NOT used as
the primary metric, because wildfire occurrence is a rare-event / highly
imbalanced target -- a model that always predicts "no fire" would score
>95% accuracy while being useless.

Reports: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and the
full sklearn classification report, for every trained model, saved as JSON
and CSV for the notebooks / dashboard / documentation to reuse.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score, confusion_matrix,
                              classification_report)

import config
from utils import get_logger, timed

logger = get_logger(__name__)


def _predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    if bundle.get("uses_scaler") and bundle.get("scaler") is not None:
        X = bundle["scaler"].transform(X)
    return model.predict_proba(X)[:, 1]


@timed
def evaluate_all_models(train_result: dict) -> pd.DataFrame:
    """Evaluate every trained model on the test split; returns a comparison table."""
    models = train_result["models"]
    X_test, y_test = train_result["splits"]["test"]

    rows = []
    detailed = {}
    for name, bundle in models.items():
        proba = _predict_proba(bundle, X_test)
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4) if y_test.nunique() > 1 else float("nan"),
            "pr_auc": round(average_precision_score(y_test, proba), 4) if y_test.nunique() > 1 else float("nan"),
        }
        rows.append(metrics)

        cm = confusion_matrix(y_test, pred).tolist()
        # Row-normalized (percentage of each TRUE class) -- the standard way to present a
        # confusion matrix under class imbalance: raw counts are dominated by the majority
        # class's cell purely because there are far more negative weeks than fire weeks in
        # the test set, which can visually look "unbalanced" even when the model performs
        # reasonably on BOTH classes. Normalizing by row puts both classes on the same
        # 0-100% scale so per-class performance is directly comparable at a glance.
        cm_normalized = np.round(confusion_matrix(y_test, pred, normalize="true") * 100, 1).tolist()
        report = classification_report(y_test, pred, zero_division=0, output_dict=True)
        detailed[name] = {"confusion_matrix": cm, "confusion_matrix_normalized_pct": cm_normalized,
                           "classification_report": report, **metrics}

        logger.info(f"[{name}] accuracy={metrics['accuracy']} balanced_accuracy={metrics['balanced_accuracy']} "
                    f"precision={metrics['precision']} recall={metrics['recall']} "
                    f"f1={metrics['f1_score']} ROC-AUC={metrics['roc_auc']} PR-AUC={metrics['pr_auc']}")

    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(config.PROCESSED_DATA_DIR / "model_comparison.csv", index=False)
    with open(config.MODELS_DIR / "evaluation_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    logger.info(f"Model comparison saved -> {config.PROCESSED_DATA_DIR / 'model_comparison.csv'}")
    logger.info("NOTE: class distribution is imbalanced (wildfire is a rare event); "
                "PR-AUC and recall matter more here than raw accuracy.")
    return comparison


def best_model_name(comparison: pd.DataFrame) -> str:
    """Select the best model by PR-AUC (most informative metric under class imbalance)."""
    return comparison.sort_values("pr_auc", ascending=False).iloc[0]["model"]


if __name__ == "__main__":
    from data_loader import load_dataset
    from train_models import train_all_models

    df = load_dataset(source="demo")
    result = train_all_models(df)
    comparison = evaluate_all_models(result)
    print(comparison.to_string(index=False))
    logger.info(f"Best model by PR-AUC: {best_model_name(comparison)}")
