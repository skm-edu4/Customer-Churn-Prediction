"""All matplotlib/seaborn figures used by the report and the dashboard.

Every function takes plain arrays plus an output path and returns the path,
so scripts, tests and the Streamlit app can reuse the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")

ScoreDict = dict[str, tuple[np.ndarray, np.ndarray]]


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def eda_overview(df: pd.DataFrame, path: str | Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    df["churned"].value_counts().sort_index().plot.bar(
        ax=axes[0, 0], color=["steelblue", "tomato"], rot=0
    )
    axes[0, 0].set_title("Churn distribution (0 = stayed, 1 = left)")
    axes[0, 0].set_xlabel("")

    for label, color in [(0, "steelblue"), (1, "tomato")]:
        df.loc[df["churned"] == label, "tenure"].hist(
            ax=axes[0, 1],
            alpha=0.6,
            bins=18,
            color=color,
            label="stayed" if label == 0 else "churned",
        )
    axes[0, 1].set_title("Tenure distribution by outcome")
    axes[0, 1].legend()

    (
        df.groupby("contract_type")["churned"]
        .mean()
        .sort_values()
        .plot.bar(ax=axes[1, 0], color="seagreen", rot=15)
    )
    axes[1, 0].set_title("Churn rate by contract type")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].scatter(
        df["tenure"],
        df["monthly_charges"],
        c=df["churned"],
        cmap="coolwarm",
        alpha=0.35,
        s=12,
    )
    axes[1, 1].set_title("Tenure vs monthly charges (coloured by churn)")
    axes[1, 1].set_xlabel("tenure")
    axes[1, 1].set_ylabel("monthly charges")

    fig.suptitle("Exploratory data analysis", y=1.01, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def cv_comparison(cv_df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(cv_df) + 2))
    plot_df = cv_df[["val_roc_auc", "val_pr_auc"]].sort_values("val_roc_auc")
    plot_df.rename(columns={"val_roc_auc": "ROC-AUC", "val_pr_auc": "PR-AUC"}).plot.barh(
        ax=ax, color=["mediumpurple", "goldenrod"]
    )
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Cross-validated model comparison")
    ax.set_xlabel("score (5-fold CV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def roc_overlay(scores: ScoreDict, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, (y, p) in sorted(scores.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — test set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def pr_overlay(scores: ScoreDict, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, (y, p) in sorted(scores.items()):
        precision, recall, _ = precision_recall_curve(y, p)
        ap = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.3f})")
    baseline = np.mean(np.asarray(next(iter(scores.values()))[0]))
    ax.axhline(baseline, color="k", ls="--", lw=1, label=f"no-skill ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves — test set")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def calibration_overlay(scores: ScoreDict, path: str | Path, n_bins: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, (y, p) in sorted(scores.items()):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration curves — are the probabilities trustworthy?")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def confusion_plot(
    y_true: np.ndarray, y_proba: np.ndarray, path: str | Path, threshold: float = 0.5
) -> Path:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["stayed", "churned"],
        yticklabels=["stayed", "churned"],
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    return _save(fig, path)


def threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    path: str | Path,
    thresholds: list[float] | None = None,
) -> Path:
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = thresholds or list(np.arange(0.05, 0.95, 0.05))
    y_true = np.asarray(y_true).astype(int)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precs, label="precision", color="steelblue")
    ax.plot(thresholds, recs, label="recall", color="tomato")
    ax.plot(thresholds, f1s, label="F1", color="seagreen", lw=2)
    best = thresholds[int(np.argmax(f1s))]
    ax.axvline(best, color="gray", ls=":", lw=1, label=f"best F1 @ {best:.2f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_title("Precision / recall / F1 vs threshold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------------
def training_history(history: dict[str, Any], path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], label="val loss", color="tomato")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training curves")
    axes[0].legend()

    axes[1].plot(epochs, history["val_roc_auc"], color="seagreen", lw=2)
    best_epoch = history.get("best_epoch") or int(np.argmax(history["val_roc_auc"]) + 1)
    axes[1].axvline(best_epoch, color="gray", ls=":", lw=1, label=f"best epoch {best_epoch}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val ROC-AUC")
    axes[1].set_title("Validation ROC-AUC (early stopping picks the best)")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, path)


def learning_curve_plot(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    path: str | Path,
    seed: int = 42,
) -> Path:
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=3,
        scoring="roc_auc",
        train_sizes=np.linspace(0.3, 1.0, 5),
        n_jobs=-1,
        random_state=seed,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="train", color="steelblue")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="validation", color="tomato")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
        color="tomato",
    )
    ax.set_xlabel("training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning curve (bias–variance diagnostic)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Unsupervised baseline
# ---------------------------------------------------------------------------
def clusters_plot(X_pca: np.ndarray, clusters: np.ndarray, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.35, s=10)
    ax.legend(*scatter.legend_elements(), title="cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means segments (PCA projection)")
    fig.tight_layout()
    return _save(fig, path)


def feature_importance_plot(
    importance: pd.Series, path: str | Path, title: str = "Top features"
) -> Path:
    top = importance.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 2))
    top.plot.barh(ax=ax, color="mediumpurple")
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)
