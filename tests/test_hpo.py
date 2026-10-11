"""Optuna hyperparameter search: usable, reproducible, fast."""

from __future__ import annotations

import pytest

from churn.config import HPOConfig
from churn.data import split_features_target
from churn.training.baselines import build_candidate, build_candidates
from churn.training.hpo import tune_families, tune_family

FAST_CFG = HPOConfig(
    enabled=True, families=["logistic_regression"], n_trials=3, cv_folds=2, timeout_seconds=60
)


def test_tune_family_returns_search_space_params(tiny_df):
    X, y = split_features_target(tiny_df)
    params = tune_family("logistic_regression", X, y, FAST_CFG, seed=0)
    assert set(params) == {"C", "class_weight"}
    assert 1e-3 <= params["C"] <= 1e2
    assert params["class_weight"] in ("balanced", None)


def test_tuned_params_build_and_fit_a_model(tiny_df):
    X, y = split_features_target(tiny_df)
    params = tune_family("logistic_regression", X, y, FAST_CFG, seed=0)
    pipe = build_candidate("logistic_regression", seed=0, overrides=params)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tune_is_reproducible_under_seed(tiny_df):
    X, y = split_features_target(tiny_df)
    first = tune_families(X, y, FAST_CFG, seed=5)
    second = tune_families(X, y, FAST_CFG, seed=5)
    assert first == second, "TPESampler(seed=...) must make studies deterministic"


def test_unknown_family_raises(tiny_df):
    X, y = split_features_target(tiny_df)
    with pytest.raises(KeyError, match="no_such_family"):
        tune_family("no_such_family", X, y, FAST_CFG, seed=0)


def test_build_candidates_applies_overrides_without_touching_others():
    cands = build_candidates(seed=1, overrides={"logistic_regression": {"C": 0.007}})
    tuned = cands["logistic_regression"].named_steps["model"]
    assert tuned.C == 0.007
    # untouched families keep their project defaults
    assert cands["xgboost"].named_steps["model"].n_estimators == 400
    assert cands["random_forest"].named_steps["model"].n_estimators == 300
    assert set(cands) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "gradient_boosting_smote",
    }


def test_default_config_keeps_hpo_disabled():
    from churn.config import load_config

    cfg = load_config()
    assert cfg.training.hpo.enabled is False, "published numbers must stay reproducible"
