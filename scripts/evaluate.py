"""Held-out evaluation: score every saved model, produce all figures.

Usage::

    python scripts/evaluate.py [--config configs/default.yaml]

Reads ``models/results.json`` + ``models/bundle.json`` (produced by train.py),
evaluates on the untouched test set and writes ``reports/figures/*.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402
from churn.data import load_dataset  # noqa: E402
from churn.evaluation import plots  # noqa: E402
from churn.evaluation.metrics import compute_metrics, threshold_table  # noqa: E402
from churn.explain.shap_analysis import (  # noqa: E402
    permutation_importance,
    shap_deep_summary,
    shap_tree_summary,
)
from churn.models_io import (  # noqa: E402
    SklearnPredictor,
    TorchMLPPredictor,
    load_all_predictors,
)
from churn.training.torch_common import seed_everything  # noqa: E402
from scripts_common import train_test_frames  # noqa: E402


def _is_tree_pipeline(predictor) -> bool:
    if not isinstance(predictor, SklearnPredictor):
        return False
    model = predictor.pipeline.named_steps["model"]
    return hasattr(model, "feature_importances_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    models_dir = cfg.models_dir
    figures_dir = cfg.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((models_dir / "results.json").read_text())
    predictors = load_all_predictors(models_dir)
    best = results["best"]

    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_frames(df, cfg)
    y_true = y_test.to_numpy()

    # ------------------------------------------------- score every model
    test_metrics: dict[str, dict] = {}
    score_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, predictor in predictors.items():
        proba = predictor.predict_proba(X_test)
        test_metrics[name] = compute_metrics(y_true, proba)
        score_pairs[name] = (y_true, proba)

    metrics_df = pd.DataFrame(test_metrics).T.sort_values("roc_auc", ascending=False)
    print("[test] held-out metrics:")
    print(metrics_df.round(4).to_string())

    # -------------------------------------------------------- figures
    figs: dict[str, str] = {}

    def _track(key: str, path: Path) -> None:
        figs[key] = str(path.relative_to(cfg.project_root))

    _track("eda", plots.eda_overview(df, figures_dir / "01_eda.png"))
    cv_df = pd.DataFrame(results["cv"]).set_index("model")
    _track("cv_comparison", plots.cv_comparison(cv_df, figures_dir / "02_cv_comparison.png"))
    _track("roc", plots.roc_overlay(score_pairs, figures_dir / "03_roc_curves.png"))
    _track("pr", plots.pr_overlay(score_pairs, figures_dir / "04_pr_curves.png"))
    _track(
        "calibration",
        plots.calibration_overlay(
            score_pairs, figures_dir / "05_calibration.png", cfg.evaluation.calibration_bins
        ),
    )

    best_proba = score_pairs[best][1]
    _track("confusion", plots.confusion_plot(y_true, best_proba, figures_dir / "06_confusion.png"))
    _track(
        "threshold",
        plots.threshold_curve(
            y_true, best_proba, figures_dir / "07_threshold_curve.png", cfg.evaluation.thresholds
        ),
    )

    # torch training dynamics
    for key, fig_name in [
        ("torch_mlp", "08_history_mlp.png"),
        ("tabtransformer", "09_history_tabtransformer.png"),
    ]:
        if key in results.get("torch", {}):
            _track(
                f"history_{key}",
                plots.training_history(results["torch"][key], figures_dir / fig_name),
            )

    # learning curve + clusters (diagnostics on the training split)
    tree_name = best if _is_tree_pipeline(predictors[best]) else "xgboost"
    if isinstance(predictors.get(tree_name), SklearnPredictor):
        _track(
            "learning_curve",
            plots.learning_curve_plot(
                predictors[tree_name].pipeline,
                X_train,
                y_train,
                figures_dir / "10_learning_curve.png",
                cfg.seed,
            ),
        )

    pre = (
        predictors[tree_name].pipeline.named_steps["pre"]
        if isinstance(predictors.get(tree_name), SklearnPredictor)
        else None
    )
    if pre is not None:
        Z = pre.transform(X_train)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        Z_pca = PCA(n_components=2, random_state=cfg.seed).fit_transform(Z)
        clusters = KMeans(n_clusters=3, n_init=10, random_state=cfg.seed).fit_predict(Z)
        _track("clusters", plots.clusters_plot(Z_pca, clusters, figures_dir / "11_clusters.png"))

    # ------------------------------------------------ explainability
    pi = permutation_importance(
        predictors[best],
        X_test,
        y_test,
        n_repeats=5,
        seed=cfg.seed,
    )
    _track(
        "permutation_importance",
        plots.feature_importance_plot(
            pi,
            figures_dir / "12_permutation_importance.png",
            title=f"Permutation importance — {best} (ROC-AUC drop when shuffled)",
        ),
    )
    print("\n[explain] permutation importance (test-set ROC-AUC drop):")
    print(pi.round(4).to_string())

    shap_notes: list[str] = []
    sample = X_test.sample(min(cfg.evaluation.shap_max_samples, len(X_test)), random_state=cfg.seed)

    # Tree SHAP: explain the best *tree* model even when a torch model won.
    tree_candidates = [n for n in predictors if _is_tree_pipeline(predictors[n])]
    if tree_candidates:
        tree_name = (
            best
            if best in tree_candidates
            else max(tree_candidates, key=lambda n: results.get("scores", {}).get(n, 0.0))
        )
        try:
            path = shap_tree_summary(
                predictors[tree_name].pipeline,
                sample,
                figures_dir / "13_shap_tree.png",
                max_samples=cfg.evaluation.shap_max_samples,
            )
            _track("shap_tree", path)
            shap_notes.append(f"SHAP (TreeExplainer) for {tree_name}")
        except Exception as exc:
            shap_notes.append(f"tree SHAP skipped: {exc}")

    try:
        mlp = predictors.get("torch_mlp")
        if isinstance(mlp, TorchMLPPredictor):
            Z_sample = mlp.preprocessor.transform(sample)
            Z_sample = Z_sample.toarray() if hasattr(Z_sample, "toarray") else np.asarray(Z_sample)
            path = shap_deep_summary(
                mlp.model,
                Z_sample,
                figures_dir / "14_shap_deep_mlp.png",
                max_samples=cfg.evaluation.shap_max_samples,
            )
            _track("shap_deep", path)
            shap_notes.append("SHAP (DeepExplainer) for torch_mlp")
    except Exception as exc:
        shap_notes.append(f"deep SHAP skipped: {exc}")

    # ------------------------------------------------ persist everything
    results["test"] = {k: {m: round(v, 4) for m, v in d.items()} for k, d in test_metrics.items()}
    results["thresholds"] = threshold_table(y_true, best_proba, cfg.evaluation.thresholds).to_dict(
        orient="records"
    )
    results["permutation_importance"] = {k: round(float(v), 4) for k, v in pi.items()}
    results["figures"] = figs
    results["shap_notes"] = shap_notes
    (models_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n[done] {len(figs)} figures -> {figures_dir}")
    print(f"[done] results.json updated with test metrics (best = {best})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
