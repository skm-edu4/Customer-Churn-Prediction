"""Classification metrics beyond plain accuracy.

Churn is a classic *imbalanced* problem, so we report ROC-AUC **and**
PR-AUC (average precision), plus a calibration measure (ECE) and a
business-oriented threshold analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy across probability bins."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Standard binary-classification metrics for a churn model."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "ece": expected_calibration_error(y_true, y_proba),
    }


def threshold_table(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]
) -> pd.DataFrame:
    """Precision / recall / F1 at several decision thresholds.

    The right threshold is a business decision: catching every churner
    (low threshold) means more false alarms, and vice versa.
    """
    rows = []
    y_true = np.asarray(y_true).astype(int)
    for t in thresholds:
        m = compute_metrics(y_true, y_proba, threshold=t)
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "predicted_churners": int((y_proba >= t).sum()),
            }
        )
    return pd.DataFrame(rows).round(4)


def decision_cost(y_true: np.ndarray, y_pred: np.ndarray, fp_cost: float, fn_cost: float) -> float:
    """Total business cost of a hard classification.

    FP = offering retention to someone who was never going to leave (wasted
    budget). FN = losing a customer who wanted to stay (lost lifetime value).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return fp * fp_cost + fn * fn_cost


def cost_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fp_cost: float,
    fn_cost: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Expected cost at every point on a threshold grid.

    Returns one row per threshold with the confusion counts split into their
    cost components, so the caller can both plot the curve and pick the
    cost-minimising threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "fp": fp,
                "fn": fn,
                "fp_cost_total": fp * fp_cost,
                "fn_cost_total": fn * fn_cost,
                "expected_cost": fp * fp_cost + fn * fn_cost,
            }
        )
    table = pd.DataFrame(rows)
    table["cost_per_customer"] = table["expected_cost"] / len(y_true)
    return table
