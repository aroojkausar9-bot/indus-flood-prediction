"""
evaluation.py
=============
Evaluation utilities: metrics, confusion matrices, ROC curves,
temporal split, and cross-basin validation wrappers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba, model_name: str = "",
                    split_name: str = "") -> pd.DataFrame:
    """Return a one-row DataFrame with all evaluation metrics."""
    return pd.DataFrame([{
        "model":     model_name,
        "split":     split_name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc":   (roc_auc_score(y_true, y_proba)
                      if len(np.unique(y_true)) > 1 else np.nan),
    }])


def print_report(y_true, y_pred, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(classification_report(y_true, y_pred,
                                  target_names=["No Flood", "Flood"]))


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix",
                           ax=None, save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, _ax = (None, ax) if ax else plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=_ax,
                xticklabels=["No Flood", "Flood"],
                yticklabels=["No Flood", "Flood"])
    _ax.set_xlabel("Predicted")
    _ax.set_ylabel("True")
    _ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_roc_curves(results: dict, title: str = "ROC Curves",
                    save_path: str = None) -> None:
    """
    Parameters
    ----------
    results : {label: (y_true, y_proba)} mapping
    """
    plt.figure(figsize=(5, 4))
    for label, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_bar(metrics_df: pd.DataFrame,
                     title: str = "Model Comparison",
                     save_path: str = None) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = metrics_df["model"].unique()
    x = np.arange(len(metric_cols))
    width = 0.8 / len(models)

    plt.figure(figsize=(7, 4))
    for i, model in enumerate(models):
        vals = metrics_df[metrics_df["model"] == model][metric_cols].values.flatten()
        plt.bar(x + i * width, vals, width, label=model, alpha=0.85)
    plt.xticks(x + width * (len(models) - 1) / 2, metric_cols, rotation=30)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Temporal Split ─────────────────────────────────────────────────────────────

def temporal_train_test_split(df: pd.DataFrame,
                               X: pd.DataFrame,
                               y: pd.Series,
                               train_end_year: int = 2020):
    """
    Split on time: train ≤ train_end_year, test > train_end_year.
    Returns scaled arrays and the fitted scaler.
    """
    tr_mask = df["year"] <= train_end_year
    te_mask = df["year"] > train_end_year

    X_train = X.loc[tr_mask].reset_index(drop=True)
    y_train = y.loc[tr_mask].reset_index(drop=True)
    X_test  = X.loc[te_mask].reset_index(drop=True)
    y_test  = y.loc[te_mask].reset_index(drop=True)

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    tr_yrs = df.loc[tr_mask, "year"]
    te_yrs = df.loc[te_mask, "year"]
    print(f"[split] Train: {tr_yrs.min()}–{tr_yrs.max()} ({len(X_train):,} samples, "
          f"{y_train.mean()*100:.1f}% floods)")
    print(f"[split] Test : {te_yrs.min()}–{te_yrs.max()} ({len(X_test):,} samples, "
          f"{y_test.mean()*100:.1f}% floods)")

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler


# ── Cross-Basin Validation ─────────────────────────────────────────────────────

def _fit_and_evaluate(model_fn, X_train, y_train, X_test, y_test,
                       threshold: float = 0.10):
    """Train a model (via callable) and return a metrics dict."""
    scaler    = RobustScaler()
    X_tr_s    = scaler.fit_transform(X_train)
    X_te_s    = scaler.transform(X_test)
    model, proba = model_fn(X_tr_s, y_train, X_te_s)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "recall":    recall_score(y_test, pred, pos_label=1, zero_division=0),
        "f1":        f1_score(y_test, pred, pos_label=1, zero_division=0),
        "roc_auc":   (roc_auc_score(y_test, proba)
                      if len(np.unique(y_test)) > 1 else np.nan),
    }


def lobo_validation(df: pd.DataFrame,
                    X: pd.DataFrame,
                    y: pd.Series,
                    model_fn,
                    threshold: float = 0.10) -> pd.DataFrame:
    """
    Leave-One-Basin-Out cross-validation.

    Parameters
    ----------
    model_fn : callable(X_train_scaled, y_train, X_test_scaled) → (model, proba)
    threshold : decision threshold for the positive class

    Returns
    -------
    DataFrame with per-basin metrics
    """
    basins  = df["HYBAS_ID"].unique()
    records = []

    for bid in basins:
        te_mask = df["HYBAS_ID"] == bid
        tr_mask = ~te_mask
        X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
        X_te, y_te = X.loc[te_mask], y.loc[te_mask]

        if y_te.sum() == 0:          # skip basins with no flood events
            continue

        metrics = _fit_and_evaluate(model_fn, X_tr, y_tr, X_te, y_te, threshold)
        metrics["basin"] = bid
        records.append(metrics)
        print(f"  [LOBO] Basin {bid}: recall={metrics['recall']:.3f}, "
              f"f1={metrics['f1']:.3f}")

    result_df = pd.DataFrame(records)
    print("\n[LOBO] Mean metrics:")
    print(result_df[["accuracy","precision","recall","f1","roc_auc"]].mean())
    return result_df


def basin_grouped_kfold(df: pd.DataFrame,
                         X: pd.DataFrame,
                         y: pd.Series,
                         model_fn,
                         n_splits: int = 5,
                         threshold: float = 0.10) -> pd.DataFrame:
    """
    Basin-grouped k-fold cross-validation.

    Parameters
    ----------
    model_fn  : callable(X_train_scaled, y_train, X_test_scaled) → (model, proba)
    n_splits  : number of folds (default 5)
    threshold : decision threshold

    Returns
    -------
    DataFrame with per-fold metrics
    """
    groups  = df["HYBAS_ID"].values
    gkf     = GroupKFold(n_splits=n_splits)
    records = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        if y_te.sum() == 0:
            print(f"  [KFold] Fold {fold}: skipped (no flood events in test set)")
            continue

        metrics = _fit_and_evaluate(model_fn, X_tr, y_tr, X_te, y_te, threshold)
        metrics["fold"] = fold
        records.append(metrics)
        print(f"  [KFold] Fold {fold}: recall={metrics['recall']:.3f}, "
              f"f1={metrics['f1']:.3f}")

    result_df = pd.DataFrame(records)
    print("\n[KFold] Mean metrics:")
    print(result_df[["accuracy","precision","recall","f1","roc_auc"]].mean())
    return result_df
