"""Dataset generation, caching, schema and split-strategy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import Config
from churn.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    generate_churn_dataset,
    load_dataset,
    split_features_target,
    train_test_frames,
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
    assert list(df.columns) == [
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
        TIME_COLUMN,
        TARGET_COLUMN,
    ]
    assert df[TARGET_COLUMN].isin([0, 1]).all()
    assert df[TIME_COLUMN].between(1, 12).all()
    assert df["total_charges"].isna().any(), "expected injected missing values"
    assert df["num_tickets"].isna().any()
    assert 0.15 < df[TARGET_COLUMN].mean() < 0.65


def test_split_features_target(tiny_df):
    X, y = split_features_target(tiny_df)
    assert TARGET_COLUMN not in X.columns
    assert TIME_COLUMN not in X.columns, "time metadata must never reach a model"
    assert len(X) == len(y) == len(tiny_df)


def test_zero_drift_does_not_touch_features_or_month():
    """Drift is the ONLY thing that may change when temporal_drift toggles."""
    plain = generate_churn_dataset(1000, seed=7)
    drifted = generate_churn_dataset(1000, seed=7, temporal_drift=0.9)
    # features + cohorts identical (independent RNG stream for the month)
    pd.testing.assert_frame_equal(
        plain.drop(columns=[TARGET_COLUMN]), drifted.drop(columns=[TARGET_COLUMN])
    )
    # but the labels respond to the drift
    assert not plain[TARGET_COLUMN].equals(drifted[TARGET_COLUMN])


def test_drift_makes_late_cohorts_churn_more():
    df = generate_churn_dataset(6000, seed=3, temporal_drift=0.9)
    early = df.loc[df[TIME_COLUMN] <= 4, TARGET_COLUMN].mean()
    late = df.loc[df[TIME_COLUMN] >= 9, TARGET_COLUMN].mean()
    assert late > early + 0.05, f"expected visible drift, got {early:.3f} → {late:.3f}"


def test_temporal_split_orders_by_cohort():
    df = generate_churn_dataset(500, seed=42)
    cfg = Config(seed=42, data={"split_strategy": "temporal", "test_size": 0.2})
    X_train, X_test, y_train, y_test = train_test_frames(df, cfg)

    assert len(X_train) == 400 and len(X_test) == 100
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    months_train = df.loc[X_train.index, TIME_COLUMN]
    months_test = df.loc[X_test.index, TIME_COLUMN]
    assert months_test.min() >= months_train.max(), "test cohort must be newer than train"
    # no leakage: neither frame carries target or time columns
    assert TARGET_COLUMN not in X_train.columns and TIME_COLUMN not in X_test.columns


def test_temporal_split_falls_back_without_time_column():
    """Telco-shaped data has no cohort column → falls back to the random split."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, n),
            "monthly_charges": rng.normal(65, 20, n),
            "total_charges": rng.normal(2000, 800, n),
            "num_tickets": rng.poisson(1.2, n).astype(float),
            "contract_type": "Month-to-month",
            "internet_service": "DSL",
            "tech_support": "No",
            "payment_method": "Mailed check",
            TARGET_COLUMN: rng.integers(0, 2, n),
        }
    )
    cfg = Config(seed=42, data={"split_strategy": "temporal", "test_size": 0.2})
    X_train, X_test, y_train, y_test = train_test_frames(df, cfg)
    assert len(X_train) == 240 and len(X_test) == 60
    assert sorted(y_test.unique()) == [0, 1], "random fallback keeps stratification"


def test_load_dataset_caches_to_parquet(tmp_path):
    cfg = Config(project_root=tmp_path, seed=42)
    first = load_dataset(cfg)
    assert cfg.processed_data_path.exists()
    second = load_dataset(cfg)  # served from cache
    pd.testing.assert_frame_equal(first, second)


def test_drift_is_part_of_the_cache_key(tmp_path):
    """Two configs with different drift must not share a stale cache."""
    cfg_plain = Config(project_root=tmp_path, seed=42)
    cfg_drifted = Config(project_root=tmp_path, seed=42, data={"temporal_drift": 0.9})
    plain = load_dataset(cfg_plain)
    drifted = load_dataset(cfg_drifted)
    assert not plain[TARGET_COLUMN].equals(drifted[TARGET_COLUMN])
