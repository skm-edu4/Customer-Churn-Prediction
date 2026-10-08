"""Shared PyTorch training machinery: seeding, loaders, losses, early-stopping trainer.

The same :class:`Trainer` drives both the MLP and the TabTransformer — the
models only differ in their inputs and architecture, never in how they are
trained, evaluated or checkpointed.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.evaluation.metrics import compute_metrics


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for binary classification (Lin et al., 2017).

    Down-weights easy examples so the model concentrates on the hard,
    usually-minority, churners.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.pos_weight is not None:
            w = torch.where(
                targets > 0.5, torch.as_tensor(self.pos_weight, device=logits.device), 1.0
            )
            bce = bce * w
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1 - probs)
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def build_criterion(
    kind: str, y_train: np.ndarray, focal_gamma: float = 2.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = n_neg / max(n_pos, 1.0)

    if kind == "focal":
        criterion: FocalLoss = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        return criterion
    # default: weighted binary cross entropy
    weight = torch.as_tensor([pos_weight], dtype=torch.float32)

    def weighted_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weight.to(logits.device)
        )

    return weighted_bce


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _to_tensors(inputs: Sequence[np.ndarray]) -> list[torch.Tensor]:
    out = []
    for a in inputs:
        a = np.asarray(a)
        if a.dtype.kind == "f":
            out.append(torch.as_tensor(a.astype(np.float32)))
        else:
            out.append(torch.as_tensor(a.astype(np.int64)))
    return out


def make_loader(
    inputs: Sequence[np.ndarray],
    y: np.ndarray,
    batch_size: int = 64,
    *,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader from numpy arrays.

    ``inputs`` may be one array (MLP: dense features) or two (TabTransformer:
    categorical indices + continuous features).
    """
    tensors = _to_tensors(inputs)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    dataset = TensorDataset(*tensors, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_loaders(
    inputs: Sequence[np.ndarray],
    y: np.ndarray,
    batch_size: int = 64,
    val_frac: float = 0.2,
    seed: int = 42,
    val_inputs: Sequence[np.ndarray] | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders.

    If ``val_inputs``/``y_val`` are given they are used verbatim (the caller
    controls the split, so preprocessing can be fit on the train part only).
    Otherwise an internal seeded split is taken.
    """
    train_loader = make_loader(inputs, y, batch_size, shuffle=True)

    if val_inputs is not None and y_val is not None:
        val_loader = make_loader(val_inputs, y_val, batch_size, shuffle=False)
        return train_loader, val_loader, len(y_val)

    tensors = _to_tensors(inputs)
    y_t = np.asarray(y, dtype=np.float32)
    dataset = TensorDataset(*tensors, torch.as_tensor(y_t))
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    _train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_val


def loader_to_numpy(loader: DataLoader, n_inputs: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Materialise a loader back into numpy arrays (used for metrics/SHAP)."""
    xs: list[list[np.ndarray]] = [[] for _ in range(n_inputs)]
    ys: list[np.ndarray] = []
    for batch in loader:
        *inputs, yb = batch
        for i, t in enumerate(inputs):
            xs[i].append(t.numpy())
        ys.append(yb.numpy())
    return [np.concatenate(x) for x in xs], np.concatenate(ys)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    best_val_roc_auc: float = 0.0
    fit_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_roc_auc": self.val_roc_auc,
            "lr": self.lr,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_roc_auc": round(self.best_val_roc_auc, 4),
            "fit_seconds": round(self.fit_seconds, 2),
        }


class Trainer:
    """Early-stopping trainer with cosine LR schedule and best-checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device | str = "cpu",
        max_epochs: int = 60,
        patience: int = 8,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.device = torch.device(device)
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

    # -- internals ---------------------------------------------------------
    def _logits(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return self.model(*inputs)  # type: ignore[arg-type]

    def _step(self, batch, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        *inputs, yb = batch
        inputs = [t.to(self.device) for t in inputs]
        yb = yb.to(self.device)
        if train:
            self.optimizer.zero_grad()
        logits = self._logits(inputs).view(-1)
        loss = self.criterion(logits, yb)
        if train:
            loss.backward()
            self.optimizer.step()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        return float(loss.detach().cpu()), probs, yb.detach().cpu().numpy()

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        losses, all_probs, all_y = [], [], []
        for batch in loader:
            loss, probs, yb = self._step(batch, train=False)
            losses.append(loss)
            all_probs.append(probs)
            all_y.append(yb)
        y = np.concatenate(all_y)
        p = np.concatenate(all_probs)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        return float(np.mean(losses)), auc, y, p

    # -- public ------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> History:
        import time

        history = History()
        best_state: dict[str, Any] | None = None
        best_auc = -1.0
        wait = 0
        start = time.perf_counter()

        for _epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                loss, _, _ = self._step(batch, train=True)
                train_losses.append(loss)

            val_loss, val_auc, _, _ = self._evaluate(val_loader)
            history.train_loss.append(float(np.mean(train_losses)))
            history.val_loss.append(val_loss)
            history.val_roc_auc.append(val_auc)
            history.lr.append(self.optimizer.param_groups[0]["lr"])
            history.epochs_run += 1

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(self.model.state_dict())
                history.best_epoch = history.epochs_run
                wait = 0
            else:
                wait += 1

            self.scheduler.step()
            if wait >= self.patience:
                break

        history.best_val_roc_auc = max(best_auc, 0.0)
        history.fit_seconds = time.perf_counter() - start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        probs: list[np.ndarray] = []
        for batch in loader:
            *inputs, _yb = batch
            inputs = [t.to(self.device) for t in inputs]
            logits = self._logits(inputs).view(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    @torch.no_grad()
    def predict_proba_array(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Score raw numpy inputs without a DataLoader (used at inference)."""
        self.model.eval()
        tensors = []
        for a in inputs:
            a = np.asarray(a)
            t = torch.as_tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int64))
            tensors.append(t.to(self.device))
        logits = self.model(*tensors).view(-1)  # type: ignore[arg-type]
        return torch.sigmoid(logits).cpu().numpy()

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        _, _, y, p = self._evaluate(loader)
        return compute_metrics(y, p)
