"""
=============================================================================
FILE: src/evaluation/pr_curves.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PRECISION-RECALL CURVES FOR ALL CLASSIFIERS
=============================================================================

WHY ADD PR CURVES WHEN WE ALREADY HAVE ROC?
    They answer different questions, and for a screening device the PR
    question is the one that matters.

    An ROC curve plots sensitivity against the FALSE POSITIVE RATE, where
    that rate is measured as a fraction of all the healthy segments. When
    healthy segments greatly outnumber sick ones, a small false-positive
    *rate* can still mean a large false-positive *count* - and ROC will
    look reassuring anyway.

    A PR curve plots precision against recall. Precision asks the question
    a patient actually cares about:

        "When this device tells me I have AF, how often is it right?"

    Because precision has the number of false alarms in its denominator
    rather than the number of healthy segments, it does not get flattered
    by an abundance of easy negatives.

WHY THIS MATTERS ESPECIALLY FOR THIS PROJECT:
    Our headline finding is that the detector false-alarms on 78-84% of
    motion-corrupted segments. That is a precision problem. Reporting only
    ROC would understate it. Hasan & Motin (SPICSCON 2025) include PR
    curves alongside ROC for the same reason, and it is good practice worth
    following.

HOW TO READ THE FIGURE:
    - The horizontal dashed line is the no-skill baseline: the result you
      would get by declaring everything AF. It sits at the AF prevalence,
      about 0.54 here. A curve must clearly beat that line to mean anything.
    - Average Precision (AP) is the area under the PR curve. Like AUC,
      higher is better, but unlike AUC its floor is the prevalence rather
      than 0.5.
    - The right-hand end of the curve (high recall) is the operating region
      for screening, where missing an AF case is worse than a false alarm.
      Look at how fast precision collapses there.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (  # noqa: E402
    FEATURES_DIR, FIGURES_DIR, FIGURE_DPI,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MODEL_STYLES = {
    "svm":    ("SVM (RBF)",             "#c0392b", "-"),
    "xgb":    ("XGBoost",               "#2980b9", "-"),
    "logreg": ("Logistic Regression",   "#16a085", "--"),
}


def load_oof(model_key: str) -> Optional[pd.DataFrame]:
    """Load out-of-fold predictions for one model, or None if absent."""
    path = os.path.join(FEATURES_DIR, f"oof_predictions_{model_key}.csv")
    if not os.path.isfile(path):
        logger.warning("No out-of-fold file for '%s' (%s)", model_key, path)
        return None
    return pd.read_csv(path)


def plot_pr_curves(model_keys: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Draw PR curves for every available model on one set of axes.

    We use POOLED out-of-fold predictions rather than averaging per-fold
    curves. Folds here contain very different mixes of patients, so a mean
    of five curves would describe no real operating point. The pooled curve
    describes what the system would actually have done across every
    subject, each held out exactly once.

    Returns
    -------
    dict mapping model key -> average precision
    """
    if model_keys is None:
        model_keys = ["svm", "xgb", "logreg"]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    scores: Dict[str, float] = {}
    prevalence = None

    for key in model_keys:
        oof = load_oof(key)
        if oof is None:
            continue

        y_true = oof["y_true"].to_numpy()
        y_prob = oof["y_prob"].to_numpy()
        label, colour, style = MODEL_STYLES.get(key, (key, "#555", "-"))

        if prevalence is None:
            prevalence = float((y_true == 1).mean())

        # --- PR curve -----------------------------------------------------
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        scores[key] = float(ap)

        axes[0].plot(recall, precision, color=colour, linestyle=style,
                     linewidth=2, label=f"{label}  (AP = {ap:.3f})")

        # --- precision as a function of the decision threshold ------------
        # Shows directly how much precision must be surrendered to reach a
        # given recall - the trade every screening device has to make.
        thresholds = np.linspace(0.05, 0.95, 60)
        prec_at, rec_at = [], []
        for t in thresholds:
            predicted = (y_prob >= t).astype(int)
            tp = int(((predicted == 1) & (y_true == 1)).sum())
            fp = int(((predicted == 1) & (y_true == 0)).sum())
            fn = int(((predicted == 0) & (y_true == 1)).sum())
            prec_at.append(tp / max(1, tp + fp))
            rec_at.append(tp / max(1, tp + fn))
        axes[1].plot(thresholds, prec_at, color=colour, linestyle=style,
                     linewidth=2, label=f"{label} precision")
        axes[1].plot(thresholds, rec_at, color=colour, linestyle=":",
                     linewidth=1.4, alpha=0.75, label=f"{label} recall")

        logger.info("%s: AP=%.4f  AUC=%.4f", label, ap,
                    roc_auc_score(y_true, y_prob))

    # --- baseline and cosmetics -------------------------------------------
    if prevalence is not None:
        axes[0].axhline(prevalence, color="black", linestyle="--", linewidth=1,
                        label=f"No-skill baseline ({prevalence:.3f})")

    axes[0].set_xlabel("Recall  (share of real AF that was caught)")
    axes[0].set_ylabel("Precision  (share of AF alarms that were correct)")
    axes[0].set_title("Precision-Recall curves\npooled out-of-fold predictions",
                      fontweight="bold")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1,
                    label="default threshold (0.5)")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision and recall against threshold\n"
                      "solid = precision, dotted = recall", fontweight="bold")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.suptitle("How often is an AF alarm correct?", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "pr_curves.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("PR curve figure saved -> %s", out)
    return scores


def comparison_table(model_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pooled out-of-fold metrics for every model, in one table.

    These are the numbers to put in the thesis. They are computed across all
    predictions at once rather than averaged over folds, for the reason
    given in plot_pr_curves.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score,
    )

    if model_keys is None:
        model_keys = ["svm", "xgb", "logreg"]

    rows = []
    for key in model_keys:
        oof = load_oof(key)
        if oof is None:
            continue
        y_true, y_pred = oof["y_true"], oof["y_pred"]
        y_prob = oof["y_prob"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        rows.append({
            "model": MODEL_STYLES.get(key, (key,))[0],
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / max(1, tn + fp),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob),
            "avg_precision": average_precision_score(y_true, y_prob),
            "false_alarms": int(fp),
            "missed_af": int(fn),
        })

    table = pd.DataFrame(rows)
    out = os.path.join(FEATURES_DIR, "model_comparison_pooled.csv")
    table.to_csv(out, index=False)
    logger.info("Pooled comparison saved -> %s", out)
    return table


if __name__ == "__main__":
    print("=" * 78)
    print("PRECISION-RECALL ANALYSIS  (pooled out-of-fold)")
    print("=" * 78)

    scores = plot_pr_curves()
    if not scores:
        raise SystemExit(
            "No out-of-fold prediction files found. Run phases 5 and 6 first."
        )

    table = comparison_table()
    print("\n--- Pooled out-of-fold comparison ---")
    print(table.round(4).to_string(index=False))

    prevalence_note = (
        "\nNOTE: Average Precision has a floor equal to the AF prevalence "
        "(~0.54 here),\nnot 0.5. Compare AP against that baseline, not "
        "against zero."
    )
    print(prevalence_note)
    print("=" * 78)
