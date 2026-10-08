"""Metric implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.evaluation.metrics import (
    compute_metrics,
    expected_calibration_error,
    threshold_table,
)


def test_perfect_ranking_scores_high():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["ece"] <= 0.2
    assert set(m) >= {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier", "ece"}


def test_inverted_predictions_are_worst_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_metrics(y, p)["roc_auc"] == 0.0


def test_ece_bounds_and_known_case():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert ece == 0.0  # perfectly calibrated at 0.5


def test_threshold_table_recall_monotone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    table = threshold_table(y, p, [0.2, 0.4, 0.6, 0.8])
    assert list(table["threshold"]) == [0.2, 0.4, 0.6, 0.8]
    assert table["recall"].is_monotonic_decreasing
    assert ((table["f1"] >= 0) & (table["f1"] <= 1)).all()
    assert isinstance(table, pd.DataFrame)
