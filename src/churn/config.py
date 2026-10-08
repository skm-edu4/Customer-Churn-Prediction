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


class TrainingConfig(BaseModel):
    cv_folds: int = 5
    smote: bool = True
    torch: TorchMLPConfig = Field(default_factory=TorchMLPConfig)
    tabtransformer: TabTransformerConfig = Field(default_factory=TabTransformerConfig)


class PathsConfig(BaseModel):
    models: str = "models"
    reports: str = "reports"
    figures: str = "reports/figures"
    tracking: str = "mlruns"


class EvaluationConfig(BaseModel):
    thresholds: list[float] = Field(default=[0.30, 0.40, 0.50, 0.60])
    calibration_bins: int = 10
    shap_max_samples: int = 500


class Config(BaseModel):
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

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
