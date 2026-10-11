"""Optuna-driven hyperparameter search for the classical model families.

Why Optuna instead of the hand-picked grids:

* **TPE sampling** adapts to what it has already seen — later trials focus on
  promising regions instead of exhaustively covering a fixed grid.
* **Pruning-ready**: the same study object can early-stop unpromising trials
  (kept simple here: each trial is a full CV run, the objective is CV ROC-AUC).
* **Deterministic**: the sampler is seeded, so a study on the same data with
  the same seed returns the same best params — HPO stays reproducible.

The search is *opt-in* (``training.hpo.enabled`` or ``train.py --hpo``) so
the default pipeline keeps reproducing the published baseline numbers.
"""

from __future__ import annotations

from collections.abc import Callable

import optuna
import pandas as pd
from sklearn.model_selection import cross_validate

from churn.config import HPOConfig
from churn.training.baselines import build_candidate

# Search space per family: trial → dict of constructor kwargs.
# log-uniform ranges where the parameter spans orders of magnitude.
SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "logistic_regression": lambda t: {
        "C": t.suggest_float("C", 1e-3, 1e2, log=True),
        "class_weight": t.suggest_categorical("class_weight", ["balanced", None]),
    },
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 600),
        "max_depth": t.suggest_int("max_depth", 4, 24),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    },
    "gradient_boosting": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 400),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "subsample": t.suggest_float("subsample", 0.7, 1.0),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 800),
        "max_depth": t.suggest_int("max_depth", 2, 10),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
}


def tune_family(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: HPOConfig,
    seed: int,
) -> dict:
    """Run one study for ``family``; returns the best hyperparameter dict."""
    if family not in SEARCH_SPACES:
        raise KeyError(f"No search space for family {family!r}. Known: {sorted(SEARCH_SPACES)}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SEARCH_SPACES[family]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        pipe = build_candidate(family, seed, overrides=params)
        cv = cross_validate(
            pipe, X, y, cv=cfg.cv_folds, scoring="roc_auc", n_jobs=-1, error_score="raise"
        )
        return float(cv["test_score"].mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"churn-{family}",
    )
    study.optimize(
        objective, n_trials=cfg.n_trials, timeout=cfg.timeout_seconds, show_progress_bar=False
    )
    return dict(study.best_params)


def tune_families(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: HPOConfig,
    seed: int,
    *,
    tracker=None,
) -> dict[str, dict]:
    """Tune every configured family; returns {family: best_params}.

    Each study gets a *different* seed (derived from the global one) so the
    TPE initial random exploration is decorrelated across families while the
    run as a whole stays reproducible.
    """
    best: dict[str, dict] = {}
    for i, family in enumerate(cfg.families):
        family_seed = seed + i
        params = tune_family(family, X, y, cfg, family_seed)
        best[family] = params
        if tracker is not None:
            tracker.log_metrics(
                {
                    f"hpo/{family}_{k}": float(v)
                    for k, v in params.items()
                    if isinstance(v, (int, float))
                }
            )
    return best
