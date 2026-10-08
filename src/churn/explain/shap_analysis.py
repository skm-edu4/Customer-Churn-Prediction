"""Explainability: SHAP for tree ensembles and deep models, plus a
model-agnostic permutation-importance fallback that works for every predictor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from churn.models_io import Predictor  # noqa: E402


def _as_array(shap_values) -> np.ndarray:
    """Normalise SHAP output across shap versions (list → class-1 array)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    return np.asarray(shap_values)


def shap_tree_summary(
    pipeline,
    X: pd.DataFrame,
    path: str | Path,
    max_samples: int = 500,
) -> Path:
    """SHAP beeswarm for a sklearn/XGBoost pipeline (tree models)."""
    import shap

    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["pre"]
    Z = pre.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    Z = np.asarray(Z)[:max_samples]
    feature_names = [str(f) for f in pre.get_feature_names_out()]

    explainer = shap.TreeExplainer(model)
    values = _as_array(explainer.shap_values(Z))

    plt.figure()
    shap.summary_plot(values, Z, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP feature impact on churn prediction (tree model)")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return path


def shap_deep_summary(
    model,
    Z: np.ndarray,
    path: str | Path,
    max_samples: int = 500,
) -> Path:
    """SHAP DeepExplainer for a torch model over a dense feature matrix."""
    import shap
    import torch

    class _OutputWrapper(torch.nn.Module):
        """DeepExplainer requires a [batch, n_outputs] tensor."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x).unsqueeze(-1)

    model = model.to("cpu").eval()
    Z = np.asarray(Z)[:max_samples].astype(np.float32)
    tensor = torch.as_tensor(Z)

    explainer = shap.DeepExplainer(_OutputWrapper(model), tensor)
    values = _as_array(explainer.shap_values(tensor))

    plt.figure()
    shap.summary_plot(values, Z, show=False, max_display=15)
    plt.title("SHAP feature impact on churn prediction (deep model)")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return path


def permutation_importance(
    predictor: Predictor,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Model-agnostic permutation importance (mean ROC-AUC drop per column).

    Works for *any* object exposing ``predict_proba(DataFrame)`` — sklearn
    pipelines, torch bundles, or a live service — which is exactly why it is
    the fallback whenever SHAP does not apply.
    """
    y = np.asarray(y)
    baseline = roc_auc_score(y, predictor.predict_proba(X))
    rng = np.random.default_rng(seed)
    drops: dict[str, float] = {}

    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].to_numpy())
            score = roc_auc_score(y, predictor.predict_proba(X_shuffled))
            deltas.append(baseline - score)
        drops[col] = float(np.mean(deltas))

    return pd.Series(drops, name="auc_drop").sort_values(ascending=False)
