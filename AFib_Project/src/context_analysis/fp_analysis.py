"""
=============================================================================
FILE: src/context_analysis/fp_analysis.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 14: CONTEXT-AWARE FALSE POSITIVE ANALYSIS
=============================================================================

THIS IS THE PRIMARY RESEARCH CONTRIBUTION OF THE PROJECT.

PURPOSE:
    A False Positive (FP) is a segment the model predicted as AF but was
    annotated as NSR by the cardiologist. Understanding WHEN and WHY the
    model fires incorrectly is the clinically actionable question that
    separates this project from a vanilla ML classification paper.

HYPOTHESIS:
    Many false positives occur during physical activity. Motion artifacts
    from the chest-mounted sensor corrupt the ECG signal, making the RR
    series look irregular — resembling AF. This is the main failure mode
    of ambulatory AF detectors in real-world use.

VALIDATION STRATEGY:
    1. CHI-SQUARE TEST (ActivityClass)
       H₀: FP rate is independent of activity class.
       If we reject H₀ (p < 0.05), activity is a significant FP predictor.

    2. MOVEMENT ACCELERATION INDEX (MAI) ANALYSIS
       Split NSR segments by MovementAcceleration [g] threshold.
       Compare FP rate in "at rest" vs "in motion" groups.
       Use Mann-Whitney U test (non-parametric; MAI is not normally distributed).

    3. VISUALISATION
       - FP rate by ActivityClass bar chart
       - FP rate vs MovementAcceleration scatter + regression
       - Box plot of MAI for correct NSR vs false-positive NSR segments

CONTEXT FILE STRUCTURE:
    Each part folder has context.xlsx — 32 columns on a 10-second grid.
    The dataset_index.py assigned each segment a `context_row` = t_start_sec // 10,
    which is the matching row index in context.xlsx.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    FIGURES_DIR,
    FEATURES_DIR,
    PROCESSED_DIR,
    FIGURE_DPI,
    COLOR_AFIB,
    COLOR_NORMAL,
    COLOR_ECG,
    CONTEXT_COLUMNS,
    CONTEXT_TIME_COLUMN,
    CONTEXT_GRID_SEC,
    MAI_MOTION_THRESHOLD,
    CLASS_NORMAL,
    CLASS_AFIB,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
os.makedirs(FIGURES_DIR, exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("seaborn-darkgrid")


# ---------------------------------------------------------------------------
# CONTEXT DATA LOADING
# ---------------------------------------------------------------------------

_context_cache: Dict[str, pd.DataFrame] = {}


def _load_context(context_path: str) -> Optional[pd.DataFrame]:
    """
    Load and cache context.xlsx from disk.

    Returns None if the file is missing or unreadable (expected for some
    parts that have no context file).
    """
    if context_path in _context_cache:
        return _context_cache[context_path]

    if not context_path or not os.path.isfile(context_path):
        return None

    try:
        ctx = pd.read_excel(context_path, engine="openpyxl")
        _context_cache[context_path] = ctx
        return ctx
    except Exception as exc:
        logger.debug("Could not read context %s: %s", context_path, exc)
        return None


def attach_context(
    segment_index: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join context variables from context.xlsx to each NSR segment's prediction.

    PARAMETERS
    ----------
    segment_index : the full segment index (with context_path and context_row)
    predictions   : DataFrame with columns [segment_id, y_true, y_pred, y_prob]
                    Must contain only NSR (label == 0) segments for FP analysis.

    RETURNS
    -------
    DataFrame with all prediction columns + context columns.
    """
    # Join predictions with index metadata
    merged = predictions.merge(
        segment_index[["segment_id", "context_path", "context_row",
                        "subject", "session", "part"]],
        on="segment_id", how="left"
    )

    context_rows: List[Dict] = []

    for _, row in merged.iterrows():
        ctx_dict: Dict = {"segment_id": row["segment_id"]}

        ctx = _load_context(str(row["context_path"]))
        if ctx is None:
            for col in CONTEXT_COLUMNS:
                ctx_dict[col] = float("nan")
        else:
            ctx_row_idx = int(row["context_row"])
            if ctx_row_idx < len(ctx):
                ctx_row = ctx.iloc[ctx_row_idx]
                for col in CONTEXT_COLUMNS:
                    ctx_dict[col] = ctx_row.get(col, float("nan"))
            else:
                for col in CONTEXT_COLUMNS:
                    ctx_dict[col] = float("nan")

        context_rows.append(ctx_dict)

    ctx_df = pd.DataFrame(context_rows)
    return merged.merge(ctx_df, on="segment_id", how="left")


# ---------------------------------------------------------------------------
# FALSE POSITIVE DETECTION
# ---------------------------------------------------------------------------

def identify_false_positives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Build a prediction table tagging each segment as TP, TN, FP, or FN.

    For FP analysis we focus on NSR segments (y_true == 0) and ask:
    which ones did the model call AF (y_pred == 1)?
    """
    outcome_map = {
        (CLASS_AFIB, CLASS_AFIB):     "TP",
        (CLASS_NORMAL, CLASS_NORMAL): "TN",
        (CLASS_NORMAL, CLASS_AFIB):   "FP",   # <-- FALSE POSITIVE: NSR → predicted AF
        (CLASS_AFIB, CLASS_NORMAL):   "FN",
    }

    outcomes = [outcome_map.get((int(t), int(p)), "?")
                for t, p in zip(y_true, y_pred)]

    return pd.DataFrame({
        "segment_id": segment_ids,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "outcome":    outcomes,
        "is_fp":      [o == "FP" for o in outcomes],
    })


# ---------------------------------------------------------------------------
# STATISTICAL TESTS
# ---------------------------------------------------------------------------

def chi_square_activity_test(
    fp_context: pd.DataFrame,
    activity_col: str = "ActivityClass []",
) -> Dict:
    """
    Chi-square test: are FPs uniformly distributed across activity classes?

    H₀: FP occurrence is independent of activity class.
    H₁: Some activity classes have significantly higher FP rates.

    Returns
    -------
    dict with p_value, chi2_statistic, degrees_of_freedom, per_class_fp_rates
    """
    col = activity_col
    if col not in fp_context.columns:
        # Try a fuzzy match
        matches = [c for c in fp_context.columns if "activity" in c.lower()]
        if matches:
            col = matches[0]
        else:
            return {"error": f"Column '{activity_col}' not found"}

    valid = fp_context.dropna(subset=[col])
    if len(valid) < 5:
        return {"error": "Too few segments with context data"}

    # Build contingency table: activity class × (FP / not-FP)
    contingency = pd.crosstab(valid[col], valid["is_fp"])
    if contingency.shape[1] < 2:
        return {"error": "Only one outcome class — cannot run chi-square"}

    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    # Per-class FP rates
    per_class = valid.groupby(col)["is_fp"].agg(["sum", "count"])
    per_class["fp_rate"] = per_class["sum"] / per_class["count"]

    return {
        "chi2":        float(chi2),
        "p_value":     float(p),
        "dof":         int(dof),
        "significant": bool(p < 0.05),
        "per_class_fp_rate": per_class["fp_rate"].to_dict(),
    }


def mann_whitney_mai_test(
    fp_context: pd.DataFrame,
    mai_col: str = "MovementAcceleration [g]",
) -> Dict:
    """
    Mann-Whitney U test: is MAI higher in FP segments than TN segments?

    H₀: MAI distribution is the same for FP and TN segments.
    H₁: FP segments have higher MAI (more motion → more FPs).

    Mann-Whitney is used instead of t-test because MAI is not normally
    distributed (it is right-skewed; most segments are at rest).
    """
    col = mai_col
    if col not in fp_context.columns:
        matches = [c for c in fp_context.columns if "movement" in c.lower()
                   or "acceleration" in c.lower()]
        if matches:
            col = matches[0]
        else:
            return {"error": f"Column '{mai_col}' not found"}

    valid = fp_context.dropna(subset=[col])
    fp_mai  = valid[valid["is_fp"]][col].values
    tn_mai  = valid[~valid["is_fp"]][col].values

    if len(fp_mai) < 3 or len(tn_mai) < 3:
        return {"error": "Too few segments for Mann-Whitney test"}

    stat, p = stats.mannwhitneyu(fp_mai, tn_mai, alternative="greater")

    return {
        "u_statistic":   float(stat),
        "p_value":       float(p),
        "significant":   bool(p < 0.05),
        "fp_mai_median": float(np.median(fp_mai)),
        "tn_mai_median": float(np.median(tn_mai)),
        "fp_n":          int(len(fp_mai)),
        "tn_n":          int(len(tn_mai)),
    }


def mai_threshold_analysis(
    fp_context: pd.DataFrame,
    mai_col: str = "MovementAcceleration [g]",
    threshold: float = MAI_MOTION_THRESHOLD,
) -> Dict:
    """
    Compare FP rate between 'at rest' (MAI < threshold) and 'in motion' (≥).
    """
    col = mai_col
    valid = fp_context.dropna(subset=[col]) if col in fp_context.columns else fp_context

    if col not in valid.columns or len(valid) < 5:
        return {"error": "Insufficient data for MAI threshold analysis"}

    at_rest  = valid[valid[col] < threshold]
    in_motion = valid[valid[col] >= threshold]

    fp_rate_rest   = float(at_rest["is_fp"].mean()) if len(at_rest) else float("nan")
    fp_rate_motion = float(in_motion["is_fp"].mean()) if len(in_motion) else float("nan")

    return {
        "threshold":       threshold,
        "at_rest_n":       int(len(at_rest)),
        "in_motion_n":     int(len(in_motion)),
        "fp_rate_at_rest": fp_rate_rest,
        "fp_rate_in_motion": fp_rate_motion,
        "motion_fp_ratio": (fp_rate_motion / fp_rate_rest
                            if fp_rate_rest > 0 else float("nan")),
    }


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def plot_fp_by_activity(
    fp_context: pd.DataFrame,
    activity_col: str = "ActivityClass []",
    save_name: str = "fp_by_activity.png",
) -> str:
    """Bar chart: FP rate for each activity class."""
    col = activity_col
    if col not in fp_context.columns:
        matches = [c for c in fp_context.columns if "activity" in c.lower()]
        col = matches[0] if matches else None

    if col is None or col not in fp_context.columns:
        logger.warning("Activity column not found for FP plot")
        return ""

    valid = fp_context.dropna(subset=[col])
    rates = valid.groupby(col)["is_fp"].agg(["mean", "count"]).reset_index()
    rates.columns = ["activity", "fp_rate", "n"]
    rates = rates.sort_values("fp_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = [COLOR_AFIB if r > rates["fp_rate"].median() else COLOR_ECG
               for r in rates["fp_rate"]]
    bars = ax.bar(range(len(rates)), rates["fp_rate"], color=colours, edgecolor="none")

    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={int(row['n'])}", ha="center", fontsize=8)

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates["activity"].astype(str), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.set_title("False Positive Rate by Activity Class\n"
                 "(NSR segments incorrectly predicted as AF)", fontsize=12,
                 fontweight="bold")
    ax.axhline(valid["is_fp"].mean(), color="black", lw=1.2, ls="--",
               label=f"Overall FP rate = {valid['is_fp'].mean():.2%}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("FP-by-activity figure saved → %s", out)
    return out


def plot_mai_vs_fp(
    fp_context: pd.DataFrame,
    mai_col: str = "MovementAcceleration [g]",
    save_name: str = "mai_vs_fp.png",
) -> str:
    """
    Two-panel figure:
      Left : box plot — MAI for TN vs FP segments
      Right: scatter — individual segment MAI coloured by outcome
    """
    col = mai_col
    if col not in fp_context.columns:
        matches = [c for c in fp_context.columns if "movement" in c.lower()]
        col = matches[0] if matches else None
    if col is None:
        return ""

    valid = fp_context.dropna(subset=[col])

    fig, (ax_box, ax_scat) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Box plot ---
    tn_mai = valid[~valid["is_fp"]][col]
    fp_mai = valid[valid["is_fp"]][col]

    bp = ax_box.boxplot([tn_mai, fp_mai],
                         tick_labels=["True Negative\n(NSR)", "False Positive"],
                         patch_artist=True, widths=0.5,
                         medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor(COLOR_NORMAL)
    bp["boxes"][1].set_facecolor(COLOR_AFIB)
    ax_box.set_ylabel("Movement Acceleration (g)", fontsize=11)
    ax_box.set_title("MAI: TN vs FP Segments", fontsize=11, fontweight="bold")
    ax_box.axhline(MAI_MOTION_THRESHOLD, color="grey", lw=1.2, ls="--",
                   label=f"Threshold = {MAI_MOTION_THRESHOLD} g")
    ax_box.legend(fontsize=8)

    # --- Scatter ---
    colours_scat = [COLOR_AFIB if fp else COLOR_NORMAL for fp in valid["is_fp"]]
    ax_scat.scatter(range(len(valid)), valid[col], c=colours_scat, s=10, alpha=0.6)
    ax_scat.axhline(MAI_MOTION_THRESHOLD, color="grey", lw=1.2, ls="--",
                    label=f"Motion threshold ({MAI_MOTION_THRESHOLD} g)")
    ax_scat.set_xlabel("NSR Segment Index", fontsize=11)
    ax_scat.set_ylabel("Movement Acceleration (g)", fontsize=11)
    ax_scat.set_title("Motion Level per NSR Segment\n"
                       "(Red = False Positive, Blue = Correct)",
                       fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    ax_scat.legend(handles=[
        Patch(color=COLOR_AFIB, label="False Positive"),
        Patch(color=COLOR_NORMAL, label="True Negative"),
    ] + [plt.Line2D([0], [0], color="grey", lw=1.2, ls="--",
                    label=f"Threshold ({MAI_MOTION_THRESHOLD} g)")],
        fontsize=8, loc="upper right"
    )

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("MAI vs FP figure saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# FULL ANALYSIS RUNNER
# ---------------------------------------------------------------------------

def run_fp_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_ids: np.ndarray,
    segment_index: pd.DataFrame,
    model_name: str = "Model",
) -> Dict:
    """
    Run the complete false positive analysis pipeline.

    PARAMETERS
    ----------
    y_true, y_pred, segment_ids — from the cross-validation on NSR segments.
    segment_index — the full segment index (has context_path, context_row).
    model_name — for plot titles.

    RETURNS
    -------
    dict of all statistical test results.
    """
    # 1. Identify FPs (on NSR segments only)
    nsr_mask = y_true == CLASS_NORMAL
    pred_df = identify_false_positives(
        y_true[nsr_mask], y_pred[nsr_mask], segment_ids[nsr_mask]
    )

    n_nsr = nsr_mask.sum()
    n_fp  = pred_df["is_fp"].sum()
    logger.info("FP analysis: %d NSR segments, %d FPs (%.1f%%)",
                n_nsr, n_fp, n_fp / max(1, n_nsr) * 100)

    # 2. Attach context data
    fp_context = attach_context(segment_index, pred_df)

    # 3. Statistical tests
    chi2_result = chi_square_activity_test(fp_context)
    mw_result   = mann_whitney_mai_test(fp_context)
    mai_result  = mai_threshold_analysis(fp_context)

    # 4. Visualisations
    plot_fp_by_activity(fp_context, save_name=f"fp_by_activity_{model_name.lower()}.png")
    plot_mai_vs_fp(fp_context, save_name=f"mai_vs_fp_{model_name.lower()}.png")

    # 5. Print findings
    print("\n" + "=" * 72)
    print(f"CONTEXT-AWARE FALSE POSITIVE ANALYSIS — {model_name}")
    print("=" * 72)
    print(f"NSR segments evaluated : {n_nsr}")
    print(f"False Positives        : {n_fp} ({n_fp/max(1,n_nsr):.1%})")
    print(f"\nChi-square (Activity vs FP): {chi2_result}")
    print(f"\nMann-Whitney (MAI FP vs TN): {mw_result}")
    print(f"\nMAI threshold ({MAI_MOTION_THRESHOLD} g):   {mai_result}")
    print("=" * 72)

    return {
        "n_nsr":        int(n_nsr),
        "n_fp":         int(n_fp),
        "fp_rate":      float(n_fp / max(1, n_nsr)),
        "chi_square":   chi2_result,
        "mann_whitney": mw_result,
        "mai_threshold": mai_result,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Requires that train_svm.py or train_xgboost.py has saved OOF predictions.
    oof_path   = os.path.join(FEATURES_DIR, "oof_predictions_xgb.csv")
    index_path = os.path.join(PROCESSED_DIR, "segment_index.csv")

    if not os.path.isfile(oof_path):
        print("OOF predictions not found. Run train_xgboost.py first.")
        print(f"Expected: {oof_path}")
        raise SystemExit(1)

    print("=" * 72)
    print("PHASE 14: CONTEXT-AWARE FALSE POSITIVE ANALYSIS")
    print("=" * 72)

    oof_df = pd.read_csv(oof_path)
    index  = pd.read_csv(index_path)

    results = run_fp_analysis(
        y_true=oof_df["y_true"].values,
        y_pred=oof_df["y_pred"].values,
        segment_ids=oof_df["segment_id"].values,
        segment_index=index,
        model_name="XGBoost",
    )

    print(f"\nFigures saved to: {FIGURES_DIR}")
