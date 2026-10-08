"""Shared pytest fixtures.

The OMP guard must be in place before any ML library loads — see the
explanation in ``churn/__init__.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import Config
from churn.data import generate_churn_dataset

SEED = 42


@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    """A small dataset shared by fast tests."""
    return generate_churn_dataset(n_customers=400, seed=SEED)


@pytest.fixture()
def tmp_cfg(tmp_path: Path) -> Config:
    """A config whose artefacts land in an isolated temp directory."""
    return Config(project_root=tmp_path, seed=SEED, device="cpu")


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
