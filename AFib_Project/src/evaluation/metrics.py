"""
=============================================================================
FILE: src/evaluation/metrics.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 13: EVALUATION — METRICS AND VISUALISATION
=============================================================================

PURPOSE:
    Compute all performance metrics for a trained binary classifier and
    generate publication-quality figures for the thesis.

METRICS COMPUTED:
    - Accuracy          : overall correct fraction
    - F1-Score (AF)     : harmonic mean of Precision and Recall for AF class
    - AUC-ROC           : area under the Receiver Operating Characteristic curve
    - Sensitivity       : True Positive Rate = TP/(TP+FN) — AF correctly detected
    - Specificity       : True Negative Rate = TN/(TN+FP) — NSR correctly identified
    - PPV / NPV         : Positive/Negative Predictive Values (clinical relevance)

WHY SENSITIVITY AND SPECIFICITY INSTEAD OF JUST F1?
    For a medical screening device the costs of the two error types are
    asymmetric. A missed AF episode (False Negative) means a patient does
    not get anticoagulant therapy and may have a stroke. A False Positive
    means unnecessary further testing — much less severe. Sensitivity
    captures the first risk; Specificity captures the second. F1 collapses
    them into one number and loses this distinction.

FIGURES GENERATED:
    - ROC curves (one per model, overlay for comparison)
    - Confusion matrix (normalised and absolute)
    - Feature importance bar chart (XGBoost)
    - Per-fold metrics box plots

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix,
    precision_score, recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    FIGURES_DIR,
    FEATURES_DIR,
    FIGURE_DPI,
    COLOR_AFIB,
    COLOR_NORMAL,
    COLOR_ECG,
    CLASS_AFIB,
    CLASS_NORMAL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use a clean, professional style
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("seaborn-darkgrid")

sns.set_palette("muted")


# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full set of binary classification metrics.

    PARAMETERS
    ----------
    y_true : array of int (0 or 1)
    y_pred : array of int (0 or 1) — thresholded predictions
    y_prob : array of float, optional — probability of positive class (for AUC)

    RETURNS
    -------
    dict of metric_name → float
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(1, tp + fn)   # Recall / True Positive Rate
    specificity = tn / max(1, tn + fp)   # True Negative Rate
    ppv = tp / max(1, tp + fp)           # Precision / Positive Predictive Value
    npv = tn / max(1, tn + fn)           # Negative Predictive Value

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, pos_label=CLASS_AFIB,
                                      zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv":         float(ppv),
        "npv":         float(npv),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")

    return metrics


def aggregate_cv_metrics(
    y_true_folds: List[np.ndarray],
    y_pred_folds: List[np.ndarray],
    y_prob_folds: Optional[List[np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Compute per-fold metrics and return a summary DataFrame.
    """
    rows = []
    for i, (y_t, y_p) in enumerate(zip(y_true_folds, y_pred_folds)):
        y_pr = y_prob_folds[i] if y_prob_folds else None
        m = compute_metrics(y_t, y_p, y_pr)
        m["fold"] = i + 1
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FIGURE 1: ROC CURVES
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves — AF Detection",
    save_name: str = "roc_curves.png",
) -> str:
    """
    Plot ROC curves for one or more models on the same axis.

    PARAMETERS
    ----------
    models : dict of model_name → (y_true, y_prob)
    title  : plot title
    save_name : filename in FIGURES_DIR

    RETURNS
    -------
    Path to saved figure.
    """
    colours = [COLOR_AFIB, COLOR_ECG, "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50)")

    for (name, (y_true, y_prob)), colour in zip(models.items(), colours):
        if y_prob is None or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=CLASS_AFIB)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=colour,
                label=f"{name}  (AUC = {auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color=colour)

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC figure saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# FIGURE 2: CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_name: str = "confusion_matrix.png",
) -> str:
    """
    Plot a normalised confusion matrix with absolute counts annotated.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    class_labels = ["NSR (Non-AF)", "AF"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised rate")

    for i in range(2):
        for j in range(2):
            colour = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", fontsize=12,
                    color=colour, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# FIGURE 3: FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "XGBoost",
    top_n: int = 15,
    save_name: str = "feature_importance.png",
) -> str:
    """
    Horizontal bar chart of the top-N feature importances.
    """
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    colours = [COLOR_AFIB if v > vals[len(vals)//2] else COLOR_ECG for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(range(len(names)), vals[::-1], color=colours[::-1], edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=12, fontweight="bold")

    # Annotate values
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance figure saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# FIGURE 4: CV METRICS BOX PLOT
# ---------------------------------------------------------------------------

def plot_cv_metrics(
    cv_results_svm: pd.DataFrame,
    cv_results_xgb: pd.DataFrame,
    save_name: str = "cv_metrics_comparison.png",
) -> str:
    """
    Side-by-side box plots comparing SVM and XGBoost across CV folds.
    """
    metrics = ["accuracy", "f1", "auc", "sensitivity", "specificity"]
    available = [m for m in metrics
                 if m in cv_results_svm.columns and m in cv_results_xgb.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(3 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        data = {
            "SVM":     cv_results_svm[metric].dropna().values,
            "XGBoost": cv_results_xgb[metric].dropna().values,
        }
        positions = [1, 2]
        bp = ax.boxplot(
            [data["SVM"], data["XGBoost"]],
            positions=positions, widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor(COLOR_ECG)
        bp["boxes"][1].set_facecolor(COLOR_AFIB)

        for model_pos, model_data in zip(positions, data.values()):
            ax.scatter(
                np.full(len(model_data), model_pos) +
                np.random.uniform(-0.1, 0.1, len(model_data)),
                model_data, color="black", s=20, zorder=3, alpha=0.7
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(["SVM", "XGB"], fontsize=9)
        ax.set_title(metric.capitalize(), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score" if metric == available[0] else "")

    fig.suptitle("Cross-Validation Performance Comparison", fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("CV comparison figure saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# PRINT REPORT
# ---------------------------------------------------------------------------

def print_results_table(
    cv_svm: pd.DataFrame,
    cv_xgb: pd.DataFrame,
) -> None:
    """Print the final results table for the report (Table 4 in the paper)."""
    print("\n" + "=" * 72)
    print("RESULTS TABLE — Mean ± Std over GroupKFold CV")
    print("=" * 72)
    print(f"{'Metric':<18} {'SVM':>18} {'XGBoost':>18}")
    print("-" * 56)
    for metric in ["accuracy", "f1", "auc", "sensitivity", "specificity"]:
        if metric not in cv_svm.columns:
            continue
        svm_vals = cv_svm[metric].dropna()
        xgb_vals = cv_xgb[metric].dropna()
        print(f"{metric:<18} "
              f"{svm_vals.mean():.3f} ± {svm_vals.std():.3f}     "
              f"{xgb_vals.mean():.3f} ± {xgb_vals.std():.3f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# ENTRY POINT — runs evaluation on saved CV results
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    svm_cv_path = os.path.join(FEATURES_DIR, "svm_cv_results.csv")
    xgb_cv_path = os.path.join(FEATURES_DIR, "xgb_cv_results.csv")
    imp_path    = os.path.join(FEATURES_DIR, "xgb_feature_importance.csv")

    if not os.path.isfile(svm_cv_path) or not os.path.isfile(xgb_cv_path):
        raise SystemExit(
            "CV result CSVs not found. Run train_svm.py and train_xgboost.py first."
        )

    cv_svm = pd.read_csv(svm_cv_path)
    cv_xgb = pd.read_csv(xgb_cv_path)

    print_results_table(cv_svm, cv_xgb)

    plot_cv_metrics(cv_svm, cv_xgb)

    if os.path.isfile(imp_path):
        imp_df = pd.read_csv(imp_path)
        plot_feature_importance(
            imp_df["feature"].tolist(),
            imp_df["importance"].values,
        )

    print(f"\nAll figures saved to: {FIGURES_DIR}")
