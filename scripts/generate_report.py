"""Generate ``reports/REPORT.md`` from results.json + the produced figures.

Usage::

    python scripts/generate_report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join([str(df.index.name or "model")] + [str(c) for c in cols]) + " |"
    sep = "|" + "---|" * (len(cols) + 1)

    def _fmt(v) -> str:
        if isinstance(v, str):
            return v
        fv = float(v)
        if float(fv).is_integer() and abs(fv) >= 1:
            return str(int(fv))
        return floatfmt.format(fv)

    rows = [
        "| " + " | ".join([str(idx), *(_fmt(v) for v in row)]) + " |" for idx, row in df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = json.loads((cfg.models_dir / "results.json").read_text())
    figs = results.get("figures", {})
    best = results["best"]

    cv_df = pd.DataFrame(results["cv"]).set_index("model")
    cv_df = cv_df[["val_accuracy", "val_f1", "val_roc_auc", "val_pr_auc", "cv_seconds"]]
    cv_df = cv_df.sort_values("val_roc_auc", ascending=False)

    test_df = pd.DataFrame(results["test"]).T
    test_df = test_df.sort_values("roc_auc", ascending=False)
    test_df.insert(0, "model", test_df.index)

    thresholds = pd.DataFrame(results.get("thresholds", []))
    pi = pd.Series(results.get("permutation_importance", {}))
    torch_info = results.get("torch", {})
    dataset = results.get("dataset", {})

    def fig(key: str, caption: str) -> str:
        path = figs.get(key)
        if not path:
            return ""
        return f"![{caption}]({path.replace('reports/', '')})\n\n*{caption}*\n"

    # ------------------------------------------------------------------
    # Sections are assembled conditionally and numbered sequentially, so
    # optional analyses (HPO, benchmark, drift) never leave numbering gaps.
    # ------------------------------------------------------------------
    sections: list[tuple[str, list[str]]] = []

    # --- 1. model selection
    cv_body: list[str] = ["", _md_table(cv_df), ""]
    if results.get("hpo"):
        hpo_df = pd.DataFrame(results["hpo"]).T
        hpo_df.index.name = "model"
        cv_body += [
            "### Optuna HPO — best parameters for this run",
            "",
            _md_table(hpo_df, floatfmt="{:.4g}"),
            "",
        ]
    sections.append(("Model selection (5-fold cross-validation on the training set)", cv_body))

    # --- 2. test results
    sections.append(
        (
            "Held-out test results",
            [
                "",
                _md_table(
                    test_df.set_index("model")[
                        ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "ece"]
                    ]
                ),
                "",
                f"**Best model: `{best}`** "
                f"(ROC-AUC {test_df.loc[best, 'roc_auc']:.4f}, "
                f"PR-AUC {test_df.loc[best, 'pr_auc']:.4f})",
                "",
                fig("roc", "ROC curves — every model on the untouched test set"),
                fig("pr", "Precision–Recall curves"),
                fig("calibration", "Calibration — are predicted probabilities trustworthy?"),
                fig("confusion", "Confusion matrix of the best model"),
                fig("threshold", "Precision / recall / F1 across decision thresholds"),
                "",
            ],
        )
    )

    # --- 3. threshold + cost analysis
    threshold_body: list[str] = (
        ["", _md_table(thresholds.set_index("threshold"))] if len(thresholds) else [""]
    )
    ca = results.get("cost_analysis")
    if ca:
        threshold_body += [
            "### Cost-sensitive threshold",
            "",
            f"With a false-positive cost of **{ca['fp_cost']:g}** (wasted retention "
            f"offer) and a false-negative cost of **{ca['fn_cost']:g}** (lost customer):",
            "",
            f"- cost-optimal threshold: **{ca['optimal_threshold']:.2f}** → expected cost "
            f"**{ca['cost_at_optimum']:.0f}** ({ca['cost_per_customer_at_optimum']:.3f}/customer)",
            f"- at the default 0.50 threshold: expected cost **{ca['cost_at_0_5']:.0f}**",
            f"- saving by moving to the cost-optimal threshold: **{ca['saving_vs_0_5']:.0f}**",
            "",
            fig("cost_curve", "Expected cost across thresholds (FP/FN breakdown)"),
        ]
    sections.append(("Threshold analysis (business decision)", threshold_body + [""]))

    # --- 4. torch training dynamics
    torch_body: list[str] = [""]
    for name, hist in torch_info.items():
        torch_body += [
            f"### `{name}`",
            "",
            f"- epochs run: **{hist['epochs_run']}** (best epoch {hist['best_epoch']})",
            f"- best val ROC-AUC: **{hist['best_val_roc_auc']:.4f}**",
            f"- fit time: {hist['fit_seconds']:.1f}s",
            "",
            fig(f"history_{name}", f"Training curves — {name}"),
        ]
    sections.append(("PyTorch training dynamics", torch_body))

    # --- 5. latency benchmark (optional)
    bench = results.get("benchmark")
    if bench:
        bench_df = pd.DataFrame(bench["table"]).set_index("model")
        parity_lines = [
            f"- `{k}`: max |Δ| = {v:.2e}" for k, v in bench.get("parity_max_diff", {}).items()
        ]
        sections.append(
            (
                "Inference latency (eager PyTorch vs ONNX Runtime)",
                [
                    "",
                    f"End-to-end `predict_proba` (preprocessing included), "
                    f"{bench['n_iter']} timed calls after {bench['warmup']} warmup calls:",
                    "",
                    _md_table(bench_df),
                    "",
                    "**ONNX export parity** (eager vs ONNX on the same rows):",
                    "",
                    *parity_lines,
                    "",
                    fig("latency", "Inference latency by model and batch size"),
                ],
            )
        )

    # --- 6. concept drift (optional)
    if results.get("drift"):
        drift_df = pd.DataFrame(results["drift"])
        sections.append(
            (
                "Concept drift (temporal cohorts)",
                [
                    "",
                    "Churn rate by observation month — recent cohorts churn more, "
                    "which a random split would silently average away:",
                    "",
                    _md_table(drift_df.set_index("month")),
                    "",
                    fig("drift", "Churn rate by observation month (concept drift)"),
                ],
            )
        )

    # --- EDA
    sections.append(
        (
            "Exploratory data analysis",
            [
                "",
                fig("eda", "EDA overview"),
                fig("cv_comparison", "Cross-validated model comparison"),
                fig("learning_curve", "Learning curve (bias–variance diagnostic)"),
                fig("clusters", "Unsupervised baseline: K-Means segments on PCA projection"),
                "",
            ],
        )
    )

    # --- explainability
    sections.append(
        (
            "Explainability",
            [
                "",
                "Permutation importance — mean ROC-AUC drop when a column is shuffled:",
                "",
                _md_table(
                    pi.rename("auc_drop").to_frame().sort_values("auc_drop", ascending=False)
                ),
                "",
                fig("permutation_importance", "Permutation importance (test set)"),
                fig("shap_tree", "SHAP beeswarm — tree model"),
                fig("shap_deep", "SHAP beeswarm — PyTorch MLP (DeepExplainer)"),
                "\n".join(f"- {n}" for n in results.get("shap_notes", [])),
                "",
            ],
        )
    )

    parts: list[str] = [
        "# Customer Churn — Model Evaluation Report",
        "",
        f"> Generated by `scripts/generate_report.py` · seed `{results['seed']}` · "
        f"dataset `{dataset.get('source', '?')}` "
        f"({dataset.get('rows', '?')} rows, churn rate "
        f"{dataset.get('churn_rate', 0.0):.1%}, "
        f"split `{dataset.get('split_strategy', 'random')}`)",
        "",
    ]
    for i, (title, body) in enumerate(sections, start=1):
        parts += [f"## {i}. {title}", *body]

    parts += [
        "---",
        "",
        "## Environment",
        "",
        f"- Python {platform.python_version()} on {platform.system()}",
        f"- project root: `{cfg.project_root}`",
        "",
        "Reproduce: `make pipeline`",
        "",
    ]

    out = cfg.reports_dir / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p for p in parts if p != ""))
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
