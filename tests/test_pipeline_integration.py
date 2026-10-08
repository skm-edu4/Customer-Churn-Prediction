"""End-to-end pipeline test: train → evaluate → report in a temp directory.

Runs the real CLI scripts as subprocesses (the same entry points users call)
against a reduced config, then asserts every artefact exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMALL_CONFIG = """
seed: 42
device: cpu
data:
  source: synthetic
  n_customers: 300
  test_size: 0.20
training:
  cv_folds: 2
  torch:
    hidden_dims: [16]
    dropout: 0.1
    batch_size: 64
    max_epochs: 3
    patience: 2
  tabtransformer:
    embed_dim: 8
    n_heads: 2
    n_layers: 1
    mlp_hidden: 16
    batch_size: 64
    max_epochs: 3
    patience: 2
paths:
  models: models
  reports: reports
  figures: reports/figures
  tracking: mlruns
evaluation:
  thresholds: [0.50]
  calibration_bins: 5
  shap_max_samples: 30
"""


def _run(script: str, config_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline")
    (root / "configs").mkdir()
    config_path = root / "configs" / "default.yaml"
    config_path.write_text(SMALL_CONFIG)

    for script in ("train.py", "evaluate.py", "generate_report.py"):
        result = _run(script, config_path)
        assert result.returncode == 0, (
            f"{script} failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
    return root


def test_bundle_registers_all_model_families(pipeline: Path):
    bundle = json.loads((pipeline / "models" / "bundle.json").read_text())
    kinds = {meta["kind"] for meta in bundle["models"].values()}
    assert bundle["best"] in bundle["models"]
    assert kinds == {"sklearn", "torch_mlp", "torch_tabtransformer"}
    assert len(bundle["models"]) >= 6  # 5 sklearn candidates + 2 torch models


def test_results_contain_cv_and_test_metrics(pipeline: Path):
    results = json.loads((pipeline / "models" / "results.json").read_text())
    assert results["best"] in results["test"]
    for name, metrics in results["test"].items():
        assert 0.0 <= metrics["roc_auc"] <= 1.0, name
        assert "pr_auc" in metrics and "ece" in metrics
    assert results["torch"]["torch_mlp"]["epochs_run"] >= 1


def test_all_figures_generated(pipeline: Path):
    results = json.loads((pipeline / "models" / "results.json").read_text())
    for key, rel in results["figures"].items():
        assert (pipeline / rel).exists(), f"missing figure for {key}"


def test_report_markdown(pipeline: Path):
    report = pipeline / "reports" / "REPORT.md"
    content = report.read_text()
    assert "Best model" in content
    assert "Held-out test results" in content
    assert "SHAP" in content


def test_saved_predictor_scores_new_rows(pipeline: Path, tiny_df):
    from churn.models_io import load_all_predictors, load_predictor

    models_dir = pipeline / "models"
    best = load_predictor(models_dir)
    X = tiny_df.drop(columns=["churned"]).head(10)
    probs = best.predict_proba(X)
    assert probs.shape == (10,)
    assert ((probs >= 0) & (probs <= 1)).all()

    all_preds = load_all_predictors(models_dir)
    for name, predictor in all_preds.items():
        out = predictor.predict_proba(X)
        assert out.shape == (10,), name
