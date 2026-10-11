"""Metric implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.evaluation.metrics import (
    compute_metrics,
    cost_table,
    decision_cost,
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


def test_decision_cost_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([1, 0, 0, 1])  # one FP, one FN
    assert decision_cost(y, p, fp_cost=10.0, fn_cost=100.0) == 110.0
    assert decision_cost(y, y, fp_cost=10.0, fn_cost=100.0) == 0.0


def test_cost_table_shape_and_minimum():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    table = cost_table(y, p, fp_cost=1.0, fn_cost=5.0)
    assert len(table) == 91  # thresholds 0.05 … 0.95 step 0.01
    assert (table["fp"].diff().dropna() <= 0).all(), "raising threshold can only add FNs"
    assert (table["fn"].diff().dropna() >= 0).all()
    # expected_cost is exactly the component sum
    assert np.allclose(table["expected_cost"], table["fp_cost_total"] + table["fn_cost_total"])
    # the argmin row is genuinely the minimum
    opt = table.loc[table["expected_cost"].idxmin()]
    assert opt["expected_cost"] <= table["expected_cost"].min() + 1e-9


def test_expensive_false_negatives_pull_threshold_down():
    """When losing a customer is costly, the optimum can only move left.

    cost(t) = fp(t) + c·fn(t): fp is non-increasing and fn non-decreasing in
    t, so raising c makes every lower threshold relatively better — the argmin
    is provably non-increasing in c (no randomness needed for the direction).
    """
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    cheap = cost_table(y, p, fp_cost=1.0, fn_cost=1.0)
    expensive = cost_table(y, p, fp_cost=1.0, fn_cost=50.0)
    t_cheap = float(cheap.loc[cheap["expected_cost"].idxmin(), "threshold"])
    t_exp = float(expensive.loc[expensive["expected_cost"].idxmin(), "threshold"])
    assert t_exp <= t_cheap
    # and with expensive churn, the recall-heavy end is far cheaper overall
    assert expensive.iloc[0]["expected_cost"] < expensive.iloc[-1]["expected_cost"]
