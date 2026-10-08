"""TabTransformer: entity embeddings + self-attention for tabular data.

Architecture (adapted from Gorishniy et al., 2021):

    categorical columns ──► embeddings ──► [CLS] token ──► TransformerEncoder
    continuous columns  ──► linear projection ─────────────► concat ──► MLP head

Categorical embeddings replace one-hot encoding: each category gets a
learned dense representation, and attention lets the model mix them
contextually instead of treating them independently.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from churn.config import TabTransformerConfig
from churn.training.torch_common import (
    History,
    Trainer,
    build_criterion,
    make_loaders,
    resolve_device,
    seed_everything,
)


class TabTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        n_cont: int,
        *,
        embed_dim: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim) for v in vocab_sizes])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, len(vocab_sizes) + 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cont_projection = nn.Linear(n_cont, embed_dim)
        self.dropout = nn.Dropout(dropout)

        head_in = embed_dim * 2  # [CLS] output ⊕ continuous projection
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        emb = torch.stack([e(cat[:, j]) for j, e in enumerate(self.embeddings)], dim=1)
        cls = self.cls_token.expand(cat.shape[0], -1, -1)
        x = torch.cat([cls, emb], dim=1) + self.pos_embedding
        x = self.dropout(self.transformer(x))
        cls_out = x[:, 0]
        cont_out = self.cont_projection(cont)
        return self.head(torch.cat([cls_out, cont_out], dim=1)).squeeze(-1)


def train_tabtransformer(
    cat: np.ndarray,
    cont: np.ndarray,
    y: np.ndarray,
    vocab_sizes: list[int],
    cfg: TabTransformerConfig,
    *,
    seed: int = 42,
    device: str = "auto",
    cat_val: np.ndarray | None = None,
    cont_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[TabTransformer, History, Trainer]:
    seed_everything(seed)
    dev = resolve_device(device)
    model = TabTransformer(
        vocab_sizes=vocab_sizes,
        n_cont=cont.shape[1],
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    )
    val_inputs = None
    if cat_val is not None and cont_val is not None:
        val_inputs = [cat_val, cont_val]
    train_loader, val_loader, _ = make_loaders(
        [cat, cont],
        y,
        batch_size=cfg.batch_size,
        seed=seed,
        val_inputs=val_inputs,
        y_val=y_val,
    )
    trainer = Trainer(
        model,
        build_criterion("weighted_bce", y),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=dev,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
    )
    history = trainer.fit(train_loader, val_loader)
    return model, history, trainer
