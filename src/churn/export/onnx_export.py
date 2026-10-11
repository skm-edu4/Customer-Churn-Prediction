"""ONNX export and inference for the trained torch models.

Why ONNX:

* **Portability** — the graph is framework-neutral: serve it with ONNX
  Runtime, a mobile runtime, or a C++ host without shipping PyTorch.
* **Latency** — ONNX Runtime fuses ops and runs a static graph, which is
  typically faster than eager mode for inference-sized batches.
* **Parity by construction** — the exported graph contains the sigmoid, so
  ``predict_proba`` outputs are directly comparable (and are checked against
  eager mode before anything is reported).

Both wrappers implement the :class:`~churn.models_io.Predictor` protocol, so
the benchmark script and (potentially) the API can treat eager and ONNX
inference interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from churn.features import TorchFeatureEncoder
from churn.training.tab_transformer import TabTransformer
from churn.training.torch_mlp import ChurnMLP

OPSET = 17
# Torch 2.14 defaults to the dynamo-based exporter. It is the future, but for
# the two-input TabTransformer it currently bakes a static batch dim into the
# graph (inputs traced as [1, k], breaking batch>1 inference). The legacy
# exporter produces correct dynamic-batch graphs for both models today, so we
# pin it explicitly until dynamo parity is verified in a newer torch.
_USE_DYNAMO = False


class _SigmoidMLP(nn.Module):
    """Turns the MLP's raw logits into P(churn) inside the ONNX graph."""

    def __init__(self, inner: ChurnMLP) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(features))


class _SigmoidTabTransformer(nn.Module):
    """Sigmoid wrapper for the TabTransformer (two inputs, one output)."""

    def __init__(self, inner: TabTransformer) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(cat, cont))


def export_mlp_onnx(model: ChurnMLP, n_features: int, path: str | Path) -> Path:
    """Export the MLP (+ sigmoid) to ONNX with a dynamic batch axis.

    NOTE: do not wrap this in ``torch.no_grad()`` — with grad mode off the
    legacy tracer takes TransformerEncoder's fused fast-path op, which has no
    ONNX symbolic (breaks the TabTransformer export in the same module).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _SigmoidMLP(model).eval()
    dummy = torch.zeros(1, n_features, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["features"],
        output_names=["churn_probability"],
        dynamic_axes={"features": {0: "batch"}, "churn_probability": {0: "batch"}},
        opset_version=OPSET,
        dynamo=_USE_DYNAMO,
    )
    return path


def export_tabtransformer_onnx(
    model: TabTransformer, vocab_sizes: list[int], n_cont: int, path: str | Path
) -> Path:
    """Export the TabTransformer (+ sigmoid) to ONNX with dynamic batch axes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _SigmoidTabTransformer(model).eval()
    dummy_cat = torch.zeros(1, len(vocab_sizes), dtype=torch.long)
    dummy_cont = torch.zeros(1, n_cont, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy_cat, dummy_cont),
        str(path),
        input_names=["cat", "cont"],
        output_names=["churn_probability"],
        dynamic_axes={
            "cat": {0: "batch"},
            "cont": {0: "batch"},
            "churn_probability": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=_USE_DYNAMO,
    )
    return path


def parity_max_diff(eager: np.ndarray, onnx: np.ndarray) -> float:
    """Max absolute difference between eager and ONNX predictions."""
    a = np.asarray(eager, dtype=np.float64).ravel()
    b = np.asarray(onnx, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: eager {a.shape} vs onnx {b.shape}")
    return float(np.max(np.abs(a - b)))


class OnnxMLPPredictor:
    """Predictor-compatible ONNX Runtime inference for the MLP."""

    kind = "onnx_torch_mlp"

    def __init__(self, name: str, preprocessor, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self.name = name
        self.preprocessor = preprocessor
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.preprocessor.transform(X)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        feats = np.ascontiguousarray(np.asarray(Z), dtype=np.float32)
        (proba,) = self.session.run(None, {"features": feats})
        return np.asarray(proba, dtype=np.float64).ravel()


class OnnxTabPredictor:
    """Predictor-compatible ONNX Runtime inference for the TabTransformer."""

    kind = "onnx_torch_tabtransformer"

    def __init__(self, name: str, encoder: TorchFeatureEncoder, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self.name = name
        self.encoder = encoder
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cat, cont = self.encoder.transform(X)
        inputs = {
            "cat": np.ascontiguousarray(cat, dtype=np.int64),
            "cont": np.ascontiguousarray(cont, dtype=np.float32),
        }
        (proba,) = self.session.run(None, inputs)
        return np.asarray(proba, dtype=np.float64).ravel()
