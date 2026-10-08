"""PyTorch models: architecture, training loop, determinism."""

from __future__ import annotations

import numpy as np
import torch

from churn.config import TabTransformerConfig, TorchMLPConfig
from churn.features import TorchFeatureEncoder
from churn.training.tab_transformer import TabTransformer, train_tabtransformer
from churn.training.torch_common import build_criterion, resolve_device, seed_everything
from churn.training.torch_mlp import ChurnMLP, train_mlp

MLP_CFG = TorchMLPConfig(
    hidden_dims=[16],
    dropout=0.0,
    lr=0.01,
    max_epochs=40,
    patience=40,
    batch_size=32,
)
TAB_CFG = TabTransformerConfig(
    embed_dim=8, n_heads=2, n_layers=1, mlp_hidden=16, max_epochs=12, patience=12, batch_size=32
)


def _separable_data(n: int = 320, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 0.2, n) > 0).astype(np.float32)
    return X, y


def test_mlp_forward_shape():
    model = ChurnMLP(n_features=6, hidden_dims=[16], dropout=0.0)
    out = model(torch.randn(5, 6))
    assert out.shape == (5,)


def test_tabtransformer_forward_shape():
    model = TabTransformer(vocab_sizes=[4, 3], n_cont=3, embed_dim=8, n_heads=2, n_layers=1)
    out = model(torch.randint(0, 3, (5, 2)), torch.randn(5, 3))
    assert out.shape == (5,)


def test_mlp_learns_separable_data():
    X, y = _separable_data()
    model, history, trainer = train_mlp(X, y, MLP_CFG, seed=0, device="cpu")
    assert history.epochs_run >= 2
    assert history.train_loss[-1] < history.train_loss[0], "training loss must decrease"
    preds = trainer.predict_proba_array([X])
    acc = ((preds >= 0.5) == (y >= 0.5)).mean()
    assert acc > 0.8, f"model should learn an easy pattern (acc={acc:.2f})"


def test_training_is_deterministic_under_seed():
    X, y = _separable_data()
    _, h1, _ = train_mlp(X, y, MLP_CFG, seed=123, device="cpu")
    _, h2, _ = train_mlp(X, y, MLP_CFG, seed=123, device="cpu")
    assert h1.train_loss == h2.train_loss


def test_focal_loss_downweights_easy_examples():
    focal = build_criterion("focal", np.array([0, 1, 0, 1]), focal_gamma=2.0)
    weighted = build_criterion("weighted_bce", np.array([0, 1, 0, 1]))
    easy = torch.tensor([8.0, -8.0])
    easy_t = torch.tensor([1.0, 0.0])
    assert float(focal(easy, easy_t)) < float(weighted(easy, easy_t))


def test_tabtransformer_trains_on_encoded_frame(tiny_df):
    X = tiny_df.drop(columns=["churned"])
    y = tiny_df["churned"].to_numpy().astype(np.float32)
    enc = TorchFeatureEncoder().fit(X)
    cat, cont = enc.transform(X)
    model, history, trainer = train_tabtransformer(
        cat, cont, y, enc.vocab_sizes, TAB_CFG, seed=0, device="cpu"
    )
    assert history.epochs_run >= 1
    probs = trainer.predict_proba_array([cat, cont])
    assert probs.shape == (len(y),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_seed_everything_reproducible():
    seed_everything(7)
    a = torch.randn(3)
    seed_everything(7)
    b = torch.randn(3)
    assert torch.equal(a, b)
