"""Feature engineering: shared preprocessing for every model family.

Two encoders are provided:

* :func:`build_sklearn_preprocessor` — the classic ``ColumnTransformer``
  (median impute + standard scale, mode impute + one-hot) used by every
  sklearn/XGBoost pipeline.

* :class:`TorchFeatureEncoder` — a fitted encoder that produces the two
  tensor streams a tabular deep model needs: integer categorical indices
  (for embeddings) and scaled continuous features. It is picklable, so it
  ships together with a trained torch checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_sklearn_preprocessor() -> ColumnTransformer:
    """Build the shared sklearn preprocessor (fit inside CV folds: no leakage)."""
    numeric_transform = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_transform = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transform, list(NUMERIC_COLUMNS)),
            ("cat", categorical_transform, list(CATEGORICAL_COLUMNS)),
        ]
    )


class TorchFeatureEncoder:
    """Encode a raw DataFrame into (categorical indices, continuous features).

    Categorical values are mapped to ``0..k-1`` with index ``k`` reserved for
    unseen categories, which keeps every embedding-table lookup safe at
    inference time.
    """

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.numeric_columns = list(numeric_columns or NUMERIC_COLUMNS)
        self.categorical_columns = list(categorical_columns or CATEGORICAL_COLUMNS)
        self.vocab: dict[str, dict[str, int]] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> TorchFeatureEncoder:
        for col in self.categorical_columns:
            values = sorted(df[col].astype(str).dropna().unique().tolist())
            self.vocab[col] = {v: i for i, v in enumerate(values)}

        for col in self.numeric_columns:
            series = df[col].astype(float)
            self.means[col] = float(series.mean())
            std = float(series.std())
            self.stds[col] = std if std > 1e-9 else 1.0
        return self

    @property
    def vocab_sizes(self) -> list[int]:
        """Embedding table sizes (k known categories + 1 unknown bucket)."""
        return [len(self.vocab[col]) + 1 for col in self.categorical_columns]

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_columns)

    @property
    def n_continuous(self) -> int:
        return len(self.numeric_columns)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cat_idx [n, n_cat] int64, cont [n, n_cont] float32)``."""
        n = len(df)
        cat = np.zeros((n, self.n_categorical), dtype=np.int64)
        for j, col in enumerate(self.categorical_columns):
            mapping = self.vocab[col]
            unknown = len(mapping)
            cat[:, j] = [
                mapping.get(str(v), unknown) if pd.notna(v) else unknown for v in df[col].tolist()
            ]

        cont = np.zeros((n, self.n_continuous), dtype=np.float32)
        for j, col in enumerate(self.numeric_columns):
            values = df[col].astype(float).to_numpy()
            values = np.nan_to_num(values, nan=self.means[col], posinf=self.means[col])
            cont[:, j] = ((values - self.means[col]) / self.stds[col]).astype(np.float32)
        return cat, cont

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__
