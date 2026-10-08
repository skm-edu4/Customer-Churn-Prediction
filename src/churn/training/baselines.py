"""Classical ML candidates: sklearn pipelines, XGBoost and an imblearn+SMOTE variant."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.features import build_sklearn_preprocessor


def _preprocessor(sparse: bool = True):
    pre = build_sklearn_preprocessor()
    if not sparse:
        pre.set_params(cat__encode__sparse_output=False)
    return pre


def make_pipeline(estimator, *, sparse: bool = True) -> Pipeline:
    return Pipeline([("pre", _preprocessor(sparse)), ("model", estimator)])


def build_candidates(seed: int = 42) -> dict[str, Pipeline]:
    """All sklearn-API model candidates, each a self-contained pipeline."""
    candidates: dict[str, Pipeline] = {
        "logistic_regression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
            )
        ),
        "gradient_boosting": make_pipeline(GradientBoostingClassifier(random_state=seed)),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        # imbalanced-learn: SMOTE inside the CV fold, then a model that has no
        # class_weight option — the classic reason to reach for resampling.
        "gradient_boosting_smote": ImbPipeline(
            [
                ("pre", _preprocessor(sparse=False)),
                ("smote", SMOTE(random_state=seed)),
                ("model", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }
    return candidates


def cross_validate_candidates(
    candidates: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """5-fold CV for every candidate; returns a results table sorted by ROC-AUC."""
    rows = []
    for name, pipe in candidates.items():
        start = time.perf_counter()
        cv = cross_validate(
            pipe,
            X,
            y,
            cv=cv_folds,
            scoring=["accuracy", "f1", "roc_auc", "average_precision"],
            return_train_score=True,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "train_roc_auc": cv["train_roc_auc"].mean(),
                "val_accuracy": cv["test_accuracy"].mean(),
                "val_f1": cv["test_f1"].mean(),
                "val_roc_auc": cv["test_roc_auc"].mean(),
                "val_pr_auc": cv["test_average_precision"].mean(),
                "cv_seconds": time.perf_counter() - start,
            }
        )
    table = pd.DataFrame(rows).set_index("model").sort_values("val_roc_auc", ascending=False)
    return table.round(4)


def fit_candidates(
    candidates: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> dict[str, Pipeline]:
    """Fit every candidate on the full training set, keeping fit times."""
    fitted: dict[str, Pipeline] = {}
    for name, pipe in candidates.items():
        start = time.perf_counter()
        pipe.fit(X, y)
        fitted[name] = pipe
        pipe.fit_seconds = time.perf_counter() - start  # type: ignore[attr-defined]
    return fitted


def positive_rate(y: pd.Series | np.ndarray) -> float:
    return float(np.mean(y))
