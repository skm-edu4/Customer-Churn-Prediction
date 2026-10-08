"""PyTorch feed-forward network for tabular churn prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from churn.config import TorchMLPConfig
from churn.training.torch_common import (
    History,
    Trainer,
    build_criterion,
    make_loaders,
    resolve_device,
    seed_everything,
)


class ChurnMLP(nn.Module):
    """Multi-layer perceptron over preprocessed (scaled + one-hot) features."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers: list[nn.Module] = []
        prev = n_features
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TorchMLPConfig,
    *,
    seed: int = 42,
    device: str = "auto",
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[ChurnMLP, History, Trainer]:
    """Train the MLP with early stopping. ``X`` is the dense feature matrix."""
    seed_everything(seed)
    dev = resolve_device(device)
    model = ChurnMLP(
        n_features=X.shape[1],
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    train_loader, val_loader, _ = make_loaders(
        [X],
        y,
        batch_size=cfg.batch_size,
        seed=seed,
        val_inputs=[X_val] if X_val is not None else None,
        y_val=y_val,
    )
    trainer = Trainer(
        model,
        build_criterion(cfg.loss, y, cfg.focal_gamma),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=dev,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
    )
    history = trainer.fit(train_loader, val_loader)
    return model, history, trainer
