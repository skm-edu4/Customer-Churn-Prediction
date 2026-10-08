"""Dataset generation, caching and schema tests."""

from __future__ import annotations

import pandas as pd

from churn.config import Config
from churn.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    generate_churn_dataset,
    load_dataset,
    split_features_target,
)


def test_generator_is_deterministic():
    a = generate_churn_dataset(500, seed=7)
    b = generate_churn_dataset(500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a = generate_churn_dataset(500, seed=7)
    b = generate_churn_dataset(500, seed=8)
    assert not a["churned"].equals(b["churned"])


def test_schema_and_missing_values():
    df = generate_churn_dataset(1000, seed=1)
    assert list(df.columns) == [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET_COLUMN]
    assert df[TARGET_COLUMN].isin([0, 1]).all()
    assert df["total_charges"].isna().any(), "expected injected missing values"
    assert df["num_tickets"].isna().any()
    assert 0.15 < df[TARGET_COLUMN].mean() < 0.65


def test_split_features_target(tiny_df):
    X, y = split_features_target(tiny_df)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == len(tiny_df)


def test_load_dataset_caches_to_parquet(tmp_path):
    cfg = Config(project_root=tmp_path, seed=42)
    first = load_dataset(cfg)
    assert cfg.processed_data_path.exists()
    second = load_dataset(cfg)  # served from cache
    pd.testing.assert_frame_equal(first, second)
