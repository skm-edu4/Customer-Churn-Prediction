"""Export torch models to ONNX, verify parity, and benchmark inference latency.

Usage::

    python scripts/benchmark.py [--config configs/default.yaml]

What it does:

1. Exports every torch model in ``models/bundle.json`` to ONNX
   (``models/onnx/*.onnx``) with the sigmoid baked into the graph.
2. Verifies eager ↔ ONNX parity on real rows (max |Δ| must stay tiny).
3. Times every predictor — sklearn, torch eager, torch ONNX Runtime — at
   batch sizes from ``benchmark.batch_sizes`` and reports p50/p95 latency.

Outputs: ``reports/benchmark.md``, ``reports/figures/17_latency.png``, and a
``benchmark`` key in ``models/results.json`` (consumed by generate_report.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402
from churn.data import load_dataset, train_test_frames  # noqa: E402
from churn.evaluation import plots  # noqa: E402
from churn.export.onnx_export import (  # noqa: E402
    OnnxMLPPredictor,
    OnnxTabPredictor,
    export_mlp_onnx,
    export_tabtransformer_onnx,
    parity_max_diff,
)
from churn.models_io import (  # noqa: E402
    TabTransformerPredictor,
    TorchMLPPredictor,
    load_all_predictors,
)


def _time_predict(fn, X_batch: pd.DataFrame, warmup: int, n_iter: int) -> tuple[float, float]:
    """Return (p50_ms, p95_ms) for end-to-end predict_proba calls."""
    for _ in range(warmup):
        fn(X_batch)
    times: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(X_batch)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional ``tabulate`` dependency)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def _export_torch_models(models_dir: Path, predictors: dict) -> dict:
    """Export every torch predictor to ONNX; returns the ONNX predictors."""
    onnx_dir = models_dir / "onnx"
    onnx_predictors: dict = {}

    for name, pred in predictors.items():
        if isinstance(pred, TorchMLPPredictor):
            n_features = int(pred.model.net[0].in_features)  # first Linear layer
            path = export_mlp_onnx(pred.model, n_features, onnx_dir / f"{name}.onnx")
            onnx_predictors[f"{name} (onnx)"] = OnnxMLPPredictor(
                f"{name} (onnx)", pred.preprocessor, path
            )
        elif isinstance(pred, TabTransformerPredictor):
            path = export_tabtransformer_onnx(
                pred.model,
                list(pred.encoder.vocab_sizes),
                int(pred.encoder.n_continuous),
                onnx_dir / f"{name}.onnx",
            )
            onnx_predictors[f"{name} (onnx)"] = OnnxTabPredictor(
                f"{name} (onnx)", pred.encoder, path
            )
    return onnx_predictors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    models_dir = cfg.models_dir
    reports_dir = cfg.reports_dir
    figures_dir = cfg.figures_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictors = load_all_predictors(models_dir)
    df = load_dataset(cfg)
    X_train, *_ = train_test_frames(df, cfg)

    # ------------------------------------------------------------- export
    print("[onnx] exporting torch models …")
    onnx_predictors = _export_torch_models(models_dir, predictors)
    for key in onnx_predictors:
        print(f"[onnx] wrote models/onnx/{key.split(' ')[0]}.onnx")

    # ------------------------------------------------------------ parity
    X_par = X_train.iloc[: min(512, len(X_train))]
    parity: dict[str, float] = {}
    for name, eager in predictors.items():
        onnx_key = f"{name} (onnx)"
        if onnx_key not in onnx_predictors:
            continue
        diff = parity_max_diff(
            eager.predict_proba(X_par), onnx_predictors[onnx_key].predict_proba(X_par)
        )
        parity[name] = diff
        status = "OK" if diff < 5e-3 else "MISMATCH"
        print(f"[onnx] parity {name}: max |Δ| = {diff:.2e} [{status}]")

    # ---------------------------------------------------------- latency
    all_predictors = {**predictors, **onnx_predictors}
    rows: list[dict] = []
    print(f"\n[bench] {cfg.benchmark.n_iter} timed calls after {cfg.benchmark.warmup} warmup calls")
    for name, pred in all_predictors.items():
        row: dict = {"model": name}
        for b in cfg.benchmark.batch_sizes:
            Xb = X_train.iloc[: min(b, len(X_train))]
            p50, p95 = _time_predict(
                pred.predict_proba, Xb, cfg.benchmark.warmup, cfg.benchmark.n_iter
            )
            row[f"p50_{b}"] = round(p50, 3)
            row[f"p95_{b}"] = round(p95, 3)
            row[f"rows_s_{b}"] = round(len(Xb) / (p50 / 1000.0))
            print(
                f"[bench] {name:34s} batch {b:>3d}  p50 {p50:7.3f} ms  "
                f"p95 {p95:7.3f} ms  {row[f'rows_s_{b}']:,} rows/s"
            )
        rows.append(row)

    bench_df = pd.DataFrame(rows)

    # ----------------------------------------------------------- outputs
    fig_path = plots.latency_benchmark_plot(bench_df, figures_dir / "17_latency.png")

    parity_df = pd.DataFrame([{"model": k, "max_abs_diff": f"{v:.3e}"} for k, v in parity.items()])
    md = "\n".join(
        [
            "# Inference latency benchmark",
            "",
            f"- warmup: {cfg.benchmark.warmup} calls, timed: {cfg.benchmark.n_iter} calls",
            "- preprocessing is **included** (end-to-end `predict_proba`)",
            "- latency measured per call on this machine's CPU; absolute numbers "
            "will differ on other hardware",
            "",
            _md_table(bench_df),
            "",
            "## ONNX parity vs eager (max |Δ| on test rows)",
            "",
            _md_table(parity_df),
            "",
        ]
    )
    (reports_dir / "benchmark.md").write_text(md)

    results_path = models_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["benchmark"] = {
        "warmup": cfg.benchmark.warmup,
        "n_iter": cfg.benchmark.n_iter,
        "batch_sizes": cfg.benchmark.batch_sizes,
        "parity_max_diff": parity,
        "table": rows,
        "figure": fig_path.name,
    }
    # Register with the other figures so report + dashboard can find it.
    results.setdefault("figures", {})["latency"] = str(fig_path.relative_to(cfg.project_root))
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n[done] benchmark.md + {fig_path.name} + results.json updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
