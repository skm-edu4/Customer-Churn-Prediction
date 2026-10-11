"""Typed configuration loaded from YAML.

Everything the pipeline does is driven by ``configs/default.yaml`` so that
experiments are reproducible and reviewable: change the config, not the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class DataConfig(BaseModel):
    source: Literal["synthetic", "telco"] = "synthetic"
    n_customers: int = 5000
    telco_csv: str = "data/raw/telco_customer_churn.csv"
    test_size: float = 0.20
    # How the held-out set is carved out of the dataset.
    #   random  — stratified shuffle split (i.i.d. assumption)
    #   temporal — sort by ``observation_month``, train on the earlier cohorts,
    #              test on the latest ones (no i.i.d. assumption)
    split_strategy: Literal["random", "temporal"] = "random"
    # Synthetic only: concept drift injected into the churn log-odds. A value
    # of ``0.9`` makes late cohorts visibly churn more than early ones, which
    # is what a temporal split is designed to expose. ``0.0`` keeps the
    # original (drift-free) dataset bit-for-bit identical.
    temporal_drift: float = 0.0


class TorchMLPConfig(BaseModel):
    hidden_dims: list[int] = Field(default=[64, 32])
    dropout: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8
    loss: Literal["weighted_bce", "focal"] = "weighted_bce"
    focal_gamma: float = 2.0


class TabTransformerConfig(BaseModel):
    embed_dim: int = 16
    n_heads: int = 4
    n_layers: int = 2
    mlp_hidden: int = 32
    dropout: float = 0.10
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8


class HPOConfig(BaseModel):
    """Optuna-driven hyperparameter search for the classical model families.

    Disabled by default so ``make pipeline`` keeps reproducing the published
    numbers bit-for-bit; enable it (``training.hpo.enabled`` or ``--hpo``) to
    replace the hand-picked defaults with TPE-suggested ones.
    """

    enabled: bool = False
    families: list[str] = Field(
        default=["logistic_regression", "xgboost", "random_forest", "gradient_boosting"]
    )
    n_trials: int = 20
    timeout_seconds: int = 600
    cv_folds: int = 3  # lighter than the main CV: search speed beats precision


class TrainingConfig(BaseModel):
    cv_folds: int = 5
    smote: bool = True
    torch: TorchMLPConfig = Field(default_factory=TorchMLPConfig)
    tabtransformer: TabTransformerConfig = Field(default_factory=TabTransformerConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)


class PathsConfig(BaseModel):
    models: str = "models"
    reports: str = "reports"
    figures: str = "reports/figures"
    tracking: str = "mlruns"


class CostConfig(BaseModel):
    """Business cost of a misclassified customer (in currency units).

    A false positive wastes a retention offer on someone who was never going
    to leave; a false negative loses an entire customer. The ratio between
    the two decides where the cost-optimal threshold sits.
    """

    false_positive: float = 1.0
    false_negative: float = 5.0


class EvaluationConfig(BaseModel):
    thresholds: list[float] = Field(default=[0.30, 0.40, 0.50, 0.60])
    calibration_bins: int = 10
    shap_max_samples: int = 500
    costs: CostConfig = Field(default_factory=CostConfig)


class BenchmarkConfig(BaseModel):
    """Latency benchmark settings (scripts/benchmark.py)."""

    warmup: int = 10
    n_iter: int = 200
    batch_sizes: list[int] = Field(default=[1, 64])


class Config(BaseModel):
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # resolved at load time
    project_root: Path = PROJECT_ROOT

    @property
    def models_dir(self) -> Path:
        return self.project_root / self.paths.models

    @property
    def reports_dir(self) -> Path:
        return self.project_root / self.paths.reports

    @property
    def figures_dir(self) -> Path:
        return self.project_root / self.paths.figures

    @property
    def tracking_dir(self) -> Path:
        return self.project_root / self.paths.tracking

    @property
    def processed_data_path(self) -> Path:
        return self.project_root / "data" / "processed" / "churn.parquet"


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config file into a validated :class:`Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    cfg = Config.model_validate(raw)
    cfg.project_root = cfg_path.resolve().parents[1]
    return cfg
