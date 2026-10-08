"""Experiment tracking with MLflow (local file store, zero server required).

Every call is best-effort: if MLflow is unavailable or misbehaves the
training pipeline keeps running — tracking must never break an experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import warnings  # noqa: E402

import mlflow  # noqa: E402

from churn.config import Config  # noqa: E402


class Tracker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._active = False
        try:
            cfg.tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(cfg.tracking_dir.as_uri())
            self._active = True
        except Exception as exc:  # pragma: no cover - tracking is best effort
            warnings.warn(f"MLflow disabled: {exc}", stacklevel=2)

    def start(self, run_name: str) -> None:
        if not self._active:
            return
        try:
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({"seed": self.cfg.seed, "dataset": self.cfg.data.source})
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.start failed: {exc}", stacklevel=2)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            flat = {k: str(v) for k, v in params.items()}
            mlflow.log_params(flat)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.log_params failed: {exc}", stacklevel=2)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        try:
            clean = {k: float(v) for k, v in metrics.items()}
            mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.log_metrics failed: {exc}", stacklevel=2)

    def log_figure(self, figure: Any, artifact_path: str) -> None:
        if not self._active:
            return
        try:
            mlflow.log_figure(figure, artifact_path)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.log_figure failed: {exc}", stacklevel=2)

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        try:
            mlflow.log_artifact(str(path))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.log_artifact failed: {exc}", stacklevel=2)

    def end(self) -> None:
        if not self._active:
            return
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"mlflow.end failed: {exc}", stacklevel=2)
