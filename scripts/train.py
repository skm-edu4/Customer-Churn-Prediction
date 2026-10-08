"""Train every model family, compare them with cross-validation, save the best.

Usage::

    python scripts/train.py [--config configs/default.yaml]

Outputs (under ``models/``):
    cv_results.csv   — 5-fold CV comparison of the classical candidates
    bundle.json      — registry of all saved models + the selected best one
    results.json     — CV scores, torch training histories, timings
    sklearn/*.joblib, torch_mlp/*, torch_tabtransformer/* — artefacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402
from churn.data import load_dataset  # noqa: E402
from churn.features import TorchFeatureEncoder, build_sklearn_preprocessor  # noqa: E402
from churn.models_io import (  # noqa: E402
    save_sklearn_predictor,
    save_tabtransformer,
    save_torch_mlp,
    write_bundle,
)
from churn.tracking import Tracker  # noqa: E402
from churn.training.baselines import (  # noqa: E402
    build_candidates,
    cross_validate_candidates,
    fit_candidates,
)
from churn.training.tab_transformer import train_tabtransformer  # noqa: E402
from churn.training.torch_common import seed_everything  # noqa: E402
from churn.training.torch_mlp import train_mlp  # noqa: E402
from scripts_common import train_test_frames  # noqa: E402


def _dense(Z) -> np.ndarray:
    return Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_frames(df, cfg)
    print(
        f"[data] {len(df)} rows | train {len(X_train)} | test {len(X_test)} | "
        f"churn rate {df['churned'].mean():.1%}"
    )

    tracker = Tracker(cfg)
    tracker.start("churn-v2-baselines")

    # ------------------------------------------- classical CV + final fits
    candidates = build_candidates(cfg.seed)
    cv = cross_validate_candidates(candidates, X_train, y_train, cfg.training.cv_folds, cfg.seed)
    print("\n[cv] 5-fold cross-validation:")
    print(cv.to_string())
    cv.to_csv(models_dir / "cv_results.csv")

    fitted = fit_candidates(candidates, X_train, y_train)
    entries: dict[str, dict] = {}
    scores: dict[str, float] = {}
    fit_seconds: dict[str, float] = {}

    for name, pipe in fitted.items():
        entries[name] = save_sklearn_predictor(name, pipe, models_dir)
        scores[name] = float(cv.loc[name, "val_roc_auc"])
        fit_seconds[name] = round(float(getattr(pipe, "fit_seconds", 0.0)), 2)
        tracker.log_metrics(
            {
                f"{name}/cv_roc_auc": float(cv.loc[name, "val_roc_auc"]),
                f"{name}/cv_pr_auc": float(cv.loc[name, "val_pr_auc"]),
                f"{name}/cv_f1": float(cv.loc[name, "val_f1"]),
            }
        )

    # ------------------------------------------------------ torch: sub-split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=cfg.seed
    )

    # MLP over preprocessed dense features
    mlp_cfg = cfg.training.torch
    pre_mlp = build_sklearn_preprocessor().fit(X_fit)
    Z_fit, Z_val = _dense(pre_mlp.transform(X_fit)), _dense(pre_mlp.transform(X_val))
    mlp_model, mlp_hist, mlp_trainer = train_mlp(
        Z_fit,
        y_fit.to_numpy(),
        mlp_cfg,
        seed=cfg.seed,
        device=cfg.device,
        X_val=Z_val,
        y_val=y_val.to_numpy(),
    )
    entries["torch_mlp"] = save_torch_mlp(
        "torch_mlp",
        pre_mlp,
        mlp_model,
        {
            "n_features": int(Z_fit.shape[1]),
            "hidden_dims": mlp_cfg.hidden_dims,
            "dropout": mlp_cfg.dropout,
        },
        models_dir,
    )
    scores["torch_mlp"] = mlp_hist.best_val_roc_auc
    fit_seconds["torch_mlp"] = round(mlp_hist.fit_seconds, 2)
    print(
        f"[torch] MLP: {mlp_hist.epochs_run} epochs, "
        f"best val ROC-AUC {mlp_hist.best_val_roc_auc:.3f}"
    )
    tracker.log_metrics({"torch_mlp/val_roc_auc": mlp_hist.best_val_roc_auc})

    # TabTransformer over entity embeddings
    tab_cfg = cfg.training.tabtransformer
    encoder = TorchFeatureEncoder().fit(X_fit)
    cat_fit, cont_fit = encoder.transform(X_fit)
    cat_val, cont_val = encoder.transform(X_val)
    tab_model, tab_hist, _tab_trainer = train_tabtransformer(
        cat_fit,
        cont_fit,
        y_fit.to_numpy(),
        encoder.vocab_sizes,
        tab_cfg,
        seed=cfg.seed,
        device=cfg.device,
        cat_val=cat_val,
        cont_val=cont_val,
        y_val=y_val.to_numpy(),
    )
    entries["tabtransformer"] = save_tabtransformer(
        "tabtransformer",
        encoder,
        tab_model,
        {
            "vocab_sizes": encoder.vocab_sizes,
            "n_cont": encoder.n_continuous,
            "embed_dim": tab_cfg.embed_dim,
            "n_heads": tab_cfg.n_heads,
            "n_layers": tab_cfg.n_layers,
            "mlp_hidden": tab_cfg.mlp_hidden,
            "dropout": tab_cfg.dropout,
        },
        models_dir,
    )
    scores["tabtransformer"] = tab_hist.best_val_roc_auc
    fit_seconds["tabtransformer"] = round(tab_hist.fit_seconds, 2)
    print(
        f"[torch] TabTransformer: {tab_hist.epochs_run} epochs, "
        f"best val ROC-AUC {tab_hist.best_val_roc_auc:.3f}"
    )
    tracker.log_metrics({"tabtransformer/val_roc_auc": tab_hist.best_val_roc_auc})

    # ------------------------------------------------------------- selection
    best = max(scores, key=lambda k: scores[k])
    print(f"\n[best] {best} (held-out score {scores[best]:.3f})")

    results = {
        "seed": cfg.seed,
        "dataset": {
            "source": cfg.data.source,
            "rows": int(len(df)),
            "churn_rate": round(float(df["churned"].mean()), 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cv": cv.reset_index().to_dict(orient="records"),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "best": best,
        "fit_seconds": fit_seconds,
        "torch": {
            "torch_mlp": mlp_hist.as_dict(),
            "tabtransformer": tab_hist.as_dict(),
        },
    }
    (models_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_bundle(models_dir, best, entries)

    tracker.log_metrics({"best/score": scores[best]})
    tracker.log_artifact(models_dir / "results.json")
    tracker.log_artifact(models_dir / "cv_results.csv")
    tracker.end()

    print(f"\n[saved] {len(entries)} models + bundle.json + results.json in {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
