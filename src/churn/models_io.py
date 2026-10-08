"""Save / load every model family behind one uniform predictor interface.

``bundle.json`` at the root of ``models/`` describes all trained models and
marks the best one — it is the single source of truth used by the evaluation
script, the FastAPI service and the Streamlit dashboard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
import torch

from churn.features import TorchFeatureEncoder
from churn.training.tab_transformer import TabTransformer
from churn.training.torch_mlp import ChurnMLP


@runtime_checkable
class Predictor(Protocol):
    name: str
    kind: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(churn) for every row of a raw feature DataFrame."""
        ...


# ---------------------------------------------------------------------------
# Concrete predictors
# ---------------------------------------------------------------------------
class SklearnPredictor:
    """Wraps a fitted sklearn/imblearn/XGBoost pipeline."""

    kind = "sklearn"

    def __init__(self, name: str, pipeline) -> None:
        self.name = name
        self.pipeline = pipeline

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


class TorchMLPPredictor:
    """Wraps a fitted sklearn preprocessor + a trained ChurnMLP."""

    kind = "torch_mlp"

    def __init__(self, name: str, preprocessor, model: ChurnMLP, device: str = "cpu") -> None:
        self.name = name
        self.preprocessor = preprocessor
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def _features(self, X: pd.DataFrame) -> torch.Tensor:
        Z = self.preprocessor.transform(X)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        return torch.as_tensor(np.asarray(Z), dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return torch.sigmoid(self.model(self._features(X))).cpu().numpy()


class TabTransformerPredictor:
    """Wraps a fitted TorchFeatureEncoder + a trained TabTransformer."""

    kind = "torch_tabtransformer"

    def __init__(
        self, name: str, encoder: TorchFeatureEncoder, model: TabTransformer, device: str = "cpu"
    ) -> None:
        self.name = name
        self.encoder = encoder
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cat, cont = self.encoder.transform(X)
        cat_t = torch.as_tensor(cat, dtype=torch.long).to(self.device)
        cont_t = torch.as_tensor(cont, dtype=torch.float32).to(self.device)
        return torch.sigmoid(self.model(cat_t, cont_t)).cpu().numpy()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_sklearn_predictor(name: str, pipeline, models_dir: Path) -> dict:
    path = models_dir / "sklearn" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return {"kind": "sklearn", "path": str(path.relative_to(models_dir))}


def save_torch_mlp(name: str, preprocessor, model: ChurnMLP, arch: dict, models_dir: Path) -> dict:
    path = models_dir / "torch_mlp" / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    joblib.dump(preprocessor, path / "preprocessor.joblib")
    (path / "arch.json").write_text(json.dumps(arch, indent=2))
    return {"kind": "torch_mlp", "path": str(path.relative_to(models_dir))}


def save_tabtransformer(
    name: str,
    encoder: TorchFeatureEncoder,
    model: TabTransformer,
    arch: dict,
    models_dir: Path,
) -> dict:
    path = models_dir / "torch_tabtransformer" / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    joblib.dump(encoder, path / "encoder.joblib")
    (path / "arch.json").write_text(json.dumps(arch, indent=2))
    return {"kind": "torch_tabtransformer", "path": str(path.relative_to(models_dir))}


def write_bundle(models_dir: Path, best: str, models: dict[str, dict]) -> Path:
    bundle = {"best": best, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": models}
    (models_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    return models_dir / "bundle.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_one(models_dir: Path, name: str, meta: dict) -> Predictor:
    kind = meta["kind"]
    path = models_dir / meta["path"]

    if kind == "sklearn":
        return SklearnPredictor(name, joblib.load(path))

    if kind == "torch_mlp":
        arch = json.loads((path / "arch.json").read_text())
        model = ChurnMLP(
            n_features=arch["n_features"],
            hidden_dims=arch["hidden_dims"],
            dropout=arch["dropout"],
        )
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        preprocessor = joblib.load(path / "preprocessor.joblib")
        return TorchMLPPredictor(name, preprocessor, model)

    if kind == "torch_tabtransformer":
        arch = json.loads((path / "arch.json").read_text())
        tab_model = TabTransformer(
            vocab_sizes=arch["vocab_sizes"],
            n_cont=arch["n_cont"],
            embed_dim=arch["embed_dim"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
            mlp_hidden=arch["mlp_hidden"],
            dropout=arch["dropout"],
        )
        tab_model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        encoder = joblib.load(path / "encoder.joblib")
        return TabTransformerPredictor(name, encoder, tab_model)

    raise ValueError(f"Unknown model kind: {kind}")


def load_bundle(models_dir: Path) -> dict:
    return json.loads((models_dir / "bundle.json").read_text())


def load_predictor(models_dir: Path, name: str | None = None) -> Predictor:
    """Load one model (default: the best one recorded in bundle.json)."""
    bundle = load_bundle(models_dir)
    target = name or bundle["best"]
    if target not in bundle["models"]:
        raise KeyError(f"Model '{target}' not in bundle: {sorted(bundle['models'])}")
    return _load_one(models_dir, target, bundle["models"][target])


def load_all_predictors(models_dir: Path) -> dict[str, Predictor]:
    bundle = load_bundle(models_dir)
    return {name: _load_one(models_dir, name, meta) for name, meta in bundle["models"].items()}
