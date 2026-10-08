"""Shared helpers for the CLI scripts (kept importable from any script)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data import split_features_target


def train_test_frames(df: pd.DataFrame, cfg):
    """Deterministic stratified train/test split.

    Both ``train.py`` and ``evaluate.py`` call this with the same seed so the
    test set is identical — evaluate.py never needs the model-training code.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seed
    )
    return X_train, X_test, y_train, y_test
