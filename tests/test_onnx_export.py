"""ONNX export: eager ↔ ONNX Runtime parity for both torch models."""

from __future__ import annotations

import numpy as np

from churn.data import split_features_target
from churn.export.onnx_export import (
    OnnxMLPPredictor,
    OnnxTabPredictor,
    export_mlp_onnx,
    export_tabtransformer_onnx,
    parity_max_diff,
)
from churn.features import TorchFeatureEncoder, build_sklearn_preprocessor
from churn.models_io import TabTransformerPredictor, TorchMLPPredictor
from churn.training.tab_transformer import TabTransformer
from churn.training.torch_mlp import ChurnMLP

# fp32 graph execution vs eager: tiny numeric drift is expected, big drift is not.
MAX_DIFF = 1e-3


def _dense(Z) -> np.ndarray:
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    return np.asarray(Z, dtype=np.float32)


def test_mlp_onnx_matches_eager(tmp_path, tiny_df):
    X, _ = split_features_target(tiny_df)
    pre = build_sklearn_preprocessor().fit(X)
    n_features = _dense(pre.transform(X)).shape[1]
    model = ChurnMLP(n_features=n_features, hidden_dims=[16], dropout=0.0)

    eager = TorchMLPPredictor("mlp", pre, model)
    path = export_mlp_onnx(model, n_features, tmp_path / "mlp.onnx")
    assert path.exists() and path.stat().st_size > 0

    onnx_pred = OnnxMLPPredictor("mlp", pre, path)
    diff = parity_max_diff(eager.predict_proba(X), onnx_pred.predict_proba(X))
    assert diff < MAX_DIFF, f"eager vs ONNX diverged by {diff:.2e}"

    # batch independence: a subset must equal the corresponding slice
    sub = X.head(7)
    sub_diff = parity_max_diff(eager.predict_proba(sub), onnx_pred.predict_proba(sub))
    assert sub_diff < MAX_DIFF


def test_tabtransformer_onnx_matches_eager(tmp_path, tiny_df):
    X, _ = split_features_target(tiny_df)
    enc = TorchFeatureEncoder().fit(X)
    model = TabTransformer(
        vocab_sizes=enc.vocab_sizes,
        n_cont=enc.n_continuous,
        embed_dim=8,
        n_heads=2,
        n_layers=1,
        mlp_hidden=16,
        dropout=0.0,
    )

    eager = TabTransformerPredictor("tab", enc, model)
    path = export_tabtransformer_onnx(
        model, enc.vocab_sizes, enc.n_continuous, tmp_path / "tab.onnx"
    )
    assert path.exists() and path.stat().st_size > 0

    onnx_pred = OnnxTabPredictor("tab", enc, path)
    diff = parity_max_diff(eager.predict_proba(X), onnx_pred.predict_proba(X))
    assert diff < MAX_DIFF, f"eager vs ONNX diverged by {diff:.2e}"


def test_parity_helper_reports_worst_case():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.1, 0.52, 0.87])
    assert abs(parity_max_diff(a, b) - 0.03) < 1e-12
