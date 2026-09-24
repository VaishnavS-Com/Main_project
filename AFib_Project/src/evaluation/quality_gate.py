"""
=============================================================================
FILE: src/evaluation/quality_gate.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

SECTION 6: SIGNAL-QUALITY GATE EXPERIMENT
=============================================================================

WHAT THIS MODULE DOES
---------------------
Before the AF classifier is allowed to make a call it runs a quick check:
"Is this segment's ECG readable enough to be worth judging?"

If the answer is "no", the segment is rejected — no prediction is issued.
The experiment sweeps a threshold from lenient to strict, measuring the
trade-off at each step:

    GAIN : fewer false alarms on corrupted (Noise) segments
    COST : some real AF segments are also rejected (lost sensitivity)

The output is a curve — one operating point per threshold — that a device
manufacturer can use to choose how aggressive the gate should be.

WHY A GATE IS NEEDED (from the data)
-------------------------------------
The AF classifier fires on 78-84% of "Noise" segments because motion artifact
breaks R-peak detection and the resulting erratic RR intervals are
indistinguishable from fibrillation on every HRV feature. No amount of
classifier tuning fixes this — the features themselves are poisoned.

The fix must happen before feature extraction, at the signal level.
This gate approximates a signal-quality check using features that are ALREADY
computed in hrv_features.csv, so no new signal processing is required.

THE QUALITY SCORE
-----------------
We combine three signals that discriminate Noise from clean signal:

1. BEAT-RATE PLAUSIBILITY  (weight alpha)
   Implied heart rate = 60,000 / MeanNN_ms.
   A wearable ECG recorded on an ambulatory subject at rest should sit
   roughly in 40–120 bpm. Artifact triggers spurious R-peaks, inflating the
   apparent rate. We penalise segments whose implied rate exceeds a ceiling
   (default 110 bpm — chosen as the point where Noise and clean signal begin
   to separate in the empirical distribution).

   Component = clip( (HR - ceiling) / range, 0, 1 )
   → 0 for all segments at or below the ceiling (no penalty)
   → 1 for segments at ceiling + range bpm or above (maximum penalty)

2. RR COEFFICIENT OF VARIATION  (weight beta)
   CVNN = SDNN / MeanNN. High values mean the beat-to-beat gaps vary wildly
   relative to the mean — the AF signature. BUT Noise has even higher CVNN
   than AF (0.32 vs 0.25), so CVNN alone cannot separate them.
   Combined with the heart-rate signal it adds discriminating power.

   Component = clip( CVNN / cv_ceiling, 0, 1 )

3. BEAT COUNT EXCESS  (weight gamma)
   In 10 s at a believable heart rate, we expect roughly 7–20 beats.
   Artifact produces extra detections; n_beats_detected > 20 is suspicious.
   We do not penalise low counts (that is AF, not noise).

   Component = clip( (beats - count_ceil) / count_range, 0, 1 )

QUALITY SCORE:
    noise_score = alpha * hr_component + beta * cv_component + gamma * count_component
    quality_score = 1 - noise_score          (1 = pristine, 0 = almost certainly noise)

A threshold tau in [0, 1]: reject if quality_score < tau.

DEFAULT WEIGHTS (alpha=0.4, beta=0.4, gamma=0.2)
    Heart rate and CVNN are the strongest signals.
    Beat count provides a tiebreaker. Equal weight between HR and CVNN
    because neither alone is sufficient.

IMPORTANT INVARIANTS (from CONTINUE_HERE.md Section 3)
--------------------------------------------------------
- Do not use ectopic_fraction as a quality signal — the filter is OFF (Section
  3.4) so ectopic_fraction == 0 for every row.
- Class 1 = AF, Class 2 = NSR. Do not swap.
- Noise segments are IS_NOISE = True. They are excluded from classifier
  training but retained here for the quality analysis.
- Frequency-domain HRV is excluded throughout.

OUTPUTS
-------
- reports/figures/quality_gate_curve.png
    Operating curve: x = AF sensitivity retained, y = Noise rejection rate.
    Each point is one threshold value.

- reports/figures/quality_gate_scores.png
    Histogram of quality scores per class. Verifies Step 2 (score separation).

- data/features/quality_gate_results.csv
    One row per threshold: tau, n_accepted, n_af_accepted, n_noise_accepted,
    af_sensitivity, noise_rejection_rate, nsr_fp_rate.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import FEATURES_DIR, FIGURES_DIR, FIGURE_DPI, COLOR_AFIB, COLOR_NORMAL, COLOR_ECG
from src.utils.logger import get_logger

logger = get_logger(__name__)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# STEP 1 — BUILD THE QUALITY SCORE
# ============================================================================

def compute_quality_scores(
    feat: pd.DataFrame,
    hr_ceil: float   = 90.0,    # bpm: Noise 25th-pctile is 93; set ceiling below it
    hr_range: float  = 50.0,    # span to reach full penalty (90->140 bpm)
    cv_ceil: float   = 0.22,    # CVNN: NSR 90th-pctile is 0.23; AF 25th-pctile is 0.19
    cv_range: float  = 0.20,    # span to reach full penalty (0.22->0.42)
    count_ceil: float = 18.0,   # beats: AF 75th-pctile is 18; Noise 25th-pctile is 14
    count_range: float = 8.0,
    alpha: float = 0.5,         # heart-rate weight (primary discriminator)
    beta:  float = 0.35,        # CVNN weight
    gamma: float = 0.15,        # beat-count weight
) -> pd.Series:
    """
    Assign a quality score in [0, 1] to every segment in `feat`.

    1 = almost certainly clean signal (pass everything)
    0 = almost certainly artifact (reject everything)

    WHY THESE THRESHOLDS?
    ---------------------
    From the empirical distributions (verified on the real data):

    Heart rate (60000/MeanNN_ms):
        NSR   P90 =  90 bpm   -- rarely above 90 at rest
        AF    P75 = 114 bpm   -- can be high (tachycardia AF), but not always
        Noise P25 =  93 bpm   -- starts elevated even at the lowest quartile
    → Ceiling at 90 bpm: penalises Noise from its 25th percentile up, penalises
      AF only above its median.

    CVNN:
        NSR   P90 = 0.23      -- rarely above 0.23
        AF    P25 = 0.19      -- starts high (irregular by definition)
        Noise P25 = 0.22      -- even the quietest Noise is irregular
    → Ceiling at 0.22: penalises both AF and Noise, but the PRODUCT with the
      heart-rate component is what discriminates them. AF can have high CVNN
      at a normal heart rate; Noise has both.

    Beat count:
        AF    P75 = 18 beats  -- can detect many beats (fast AF)
        Noise P25 = 14 beats  -- starts elevated
    → Mild supplementary signal.

    COMBINATION:
        noise_score = alpha * hr_cpt + beta * cv_cpt + gamma * count_cpt
    Simple weighted sum is used (not product) to preserve the continuous gradient
    that makes the operating curve smooth. The weights place 50% emphasis on
    the heart-rate signal because that is the cleanest separator.
    """
    # Implied heart rate
    meannn = feat["MeanNN"].fillna(feat["MeanNN"].median()).clip(lower=200.0)
    hr = 60_000.0 / meannn

    # --- Component 1: heart-rate plausibility ---
    hr_component = ((hr - hr_ceil) / hr_range).clip(0.0, 1.0)

    # --- Component 2: RR coefficient of variation ---
    cvnn = feat["CVNN"].fillna(0.0)
    cv_component = ((cvnn - cv_ceil) / cv_range).clip(0.0, 1.0)

    # --- Component 3: beat-count excess ---
    beats = feat["n_beats_detected"].fillna(0.0)
    count_component = ((beats - count_ceil) / count_range).clip(0.0, 1.0)

    # --- Weighted noise score (0 = clean, 1 = noisy) ---
    total_weight = alpha + beta + gamma
    noise_score = (
        alpha * hr_component +
        beta  * cv_component +
        gamma * count_component
    ) / total_weight

    # --- Quality score (invert: 1 = clean) ---
    quality_score = 1.0 - noise_score

    return quality_score.rename("quality_score")


# ============================================================================
# STEP 2 — VERIFY SCORE SEPARATION
# ============================================================================

def verify_score_separation(
    feat: pd.DataFrame,
    scores: pd.Series,
) -> dict:
    """
    Check that Noise segments score lower than AF and NSR segments.

    RETURNS
    -------
    dict with per-class medians and Mann-Whitney p-values.
    If Noise does not score significantly lower than AF, the gate is useless
    and the score design must be revised before continuing.
    """
    # feat already has quality_score joined from the caller
    merged = feat

    # Per-class quality score
    class_medians = merged.groupby("class_name")["quality_score"].median().round(4).to_dict()
    class_means   = merged.groupby("class_name")["quality_score"].mean().round(4).to_dict()

    noise_scores = merged.loc[merged["class_name"] == "Noise",  "quality_score"]
    af_scores    = merged.loc[merged["class_name"] == "AF",     "quality_score"]
    nsr_scores   = merged.loc[merged["class_name"] == "NSR",    "quality_score"]

    # Mann-Whitney: is Noise lower than AF? (one-sided, H1: Noise < AF)
    stat_noise_vs_af,  p_noise_vs_af  = stats.mannwhitneyu(noise_scores, af_scores,  alternative="less")
    stat_noise_vs_nsr, p_noise_vs_nsr = stats.mannwhitneyu(noise_scores, nsr_scores, alternative="less")

    result = {
        "class_medians":      class_medians,
        "class_means":        class_means,
        "noise_vs_af_p":      float(p_noise_vs_af),
        "noise_vs_nsr_p":     float(p_noise_vs_nsr),
        "noise_separates_af": bool(p_noise_vs_af  < 0.05),
        "noise_separates_nsr":bool(p_noise_vs_nsr < 0.05),
    }

    logger.info("Score separation check:")
    for cls, med in class_medians.items():
        logger.info("  %-8s median quality = %.3f", cls, med)
    logger.info("  Noise vs AF  Mann-Whitney p = %.4f  (significant: %s)",
                p_noise_vs_af, result["noise_separates_af"])
    logger.info("  Noise vs NSR Mann-Whitney p = %.4f  (significant: %s)",
                p_noise_vs_nsr, result["noise_separates_nsr"])

    if not result["noise_separates_af"]:
        logger.warning(
            "GATE USELESS: quality score does not separate Noise from AF. "
            "Revise the score formula before running the threshold sweep."
        )

    return result


def plot_score_histograms(
    feat: pd.DataFrame,
    scores: pd.Series,
    save_name: str = "quality_gate_scores.png",
) -> str:
    """
    Histogram of quality scores for each class, overlaid on the same axis.
    This is Step 2's visual verification — the Noise distribution must sit
    clearly to the LEFT of AF and NSR.
    """
    merged = feat  # quality_score already joined by the caller

    class_colours = {
        "AF":    COLOR_AFIB,
        "NSR":   COLOR_NORMAL,
        "Noise": "#e67e22",   # orange — distinct from both
        "Other": "#9b59b6",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 31)

    for class_name, colour in class_colours.items():
        vals = merged.loc[merged["class_name"] == class_name, "quality_score"]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, color=colour,
                label=f"{class_name}  (n={len(vals)}, median={vals.median():.2f})",
                edgecolor="none", density=True)

    ax.set_xlabel("Quality Score  (1 = clean signal,  0 = likely artifact)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Quality Score Distribution by Class\n"
                 "Noise must cluster near 0; AF and NSR near 1",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score histogram saved -> %s", out)
    return out


# ============================================================================
# STEP 3 — SWEEP THE THRESHOLD
# ============================================================================

def sweep_threshold(
    feat: pd.DataFrame,
    scores: pd.Series,
    oof_predictions: pd.DataFrame,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """
    For each threshold tau in [0, 1], compute:
        - how many Noise segments are rejected (quality_score < tau)
        - how many AF segments are rejected (lost sensitivity)
        - false-alarm rate on NSR that survive the gate
        - overall sensitivity on AF that survive

    PARAMETERS
    ----------
    feat : pd.DataFrame
        Full feature matrix (all 1602 rows including Noise).
    scores : pd.Series
        Quality scores aligned to feat.
    oof_predictions : pd.DataFrame
        Columns: segment_id, y_true, y_pred, y_prob.
        These are the XGBoost out-of-fold predictions from training on
        AF + NSR segments only (Noise was never in training, so OOF
        predictions for Noise come from running the trained model on them
        as held-out data). We need predictions for Noise segments too —
        they are loaded separately below.
    n_thresholds : int
        Resolution of the sweep.

    RETURNS
    -------
    pd.DataFrame with one row per threshold:
        tau, n_accepted, pct_accepted,
        af_sensitivity, noise_rejection_rate, nsr_fp_rate,
        n_af_accepted, n_noise_rejected, n_nsr_accepted
    """
    # feat already has quality_score from the caller
    merged = feat.copy()

    # We need classifier predictions for every segment type.
    # oof_predictions covers AF + NSR (training classes).
    # For Noise segments, use the saved all-segment predictions if available,
    # or re-use the OOF frame (Noise was never in training, so any prediction
    # on it is out-of-sample by construction).
    pred_map = oof_predictions.set_index("segment_id")["y_pred"].to_dict()
    merged["y_pred"] = merged["segment_id"].map(pred_map)

    # Identify the three populations
    is_af    = merged["class_name"] == "AF"
    is_nsr   = merged["class_name"] == "NSR"
    is_noise = merged["class_name"] == "Noise"

    total_af    = int(is_af.sum())
    total_nsr   = int(is_nsr.sum())
    total_noise = int(is_noise.sum())

    rows = []
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    for tau in thresholds:
        accepted = merged["quality_score"] >= tau

        # --- AF accepted (sensitivity = fraction of real AF that pass the gate) ---
        af_accepted  = int((is_af  & accepted).sum())
        af_sensitivity = af_accepted / max(1, total_af)

        # --- Noise rejected (rejection rate = fraction of Noise that the gate stops) ---
        noise_rejected = int((is_noise & ~accepted).sum())
        noise_rejection_rate = noise_rejected / max(1, total_noise)

        # --- NSR false-alarm rate ON ACCEPTED NSR SEGMENTS ---
        # Only meaningful on segments that passed the gate.
        # "False alarm" = NSR segment that the classifier calls AF.
        nsr_accepted = int((is_nsr & accepted).sum())
        nsr_fp = int(
            (is_nsr & accepted & (merged["y_pred"] == 1)).sum()
        )
        nsr_fp_rate = nsr_fp / max(1, nsr_accepted)

        # --- Noise false-alarm rate ON ACCEPTED NOISE SEGMENTS ---
        noise_accepted = int((is_noise & accepted).sum())
        noise_fp = int(
            (is_noise & accepted & (merged["y_pred"] == 1)).sum()
        )
        noise_fp_rate = noise_fp / max(1, noise_accepted)

        rows.append({
            "tau":                   float(tau),
            "n_accepted":            int(accepted.sum()),
            "pct_accepted":          float(accepted.mean() * 100),
            "af_sensitivity":        float(af_sensitivity),
            "af_accepted":           af_accepted,
            "noise_rejection_rate":  float(noise_rejection_rate),
            "noise_rejected":        noise_rejected,
            "noise_accepted":        noise_accepted,
            "noise_fp_rate":         float(noise_fp_rate),
            "nsr_fp_rate":           float(nsr_fp_rate),
            "nsr_accepted":          nsr_accepted,
        })

    return pd.DataFrame(rows)


# ============================================================================
# STEP 4 — PLOT THE OPERATING CURVE
# ============================================================================

def plot_operating_curve(
    sweep: pd.DataFrame,
    save_name: str = "quality_gate_curve.png",
) -> str:
    """
    Plot the quality gate operating curve.

    Left axis  (blue) : Noise false-alarm rate  vs AF sensitivity retained
    Right axis (red)  : NSR false-alarm rate     vs AF sensitivity retained
    X-axis            : AF sensitivity retained  (1 = gate accepts every AF)

    Each point on the curve corresponds to one threshold tau.
    Moving RIGHT → lower tau → gate accepts more → more false alarms.
    Moving LEFT  → higher tau → gate rejects more → fewer false alarms + fewer AF.

    The curve tells a device manufacturer: "if you can tolerate losing X% of AF
    detections, the gate reduces noise false-alarms by Y%."
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # --- Noise false-alarm rate (left y-axis) ---
    ax1.plot(
        sweep["af_sensitivity"] * 100,
        sweep["noise_fp_rate"]  * 100,
        lw=2.5, color="#e67e22",
        label="Noise false-alarm rate (left axis)"
    )
    ax1.fill_between(
        sweep["af_sensitivity"] * 100,
        sweep["noise_fp_rate"]  * 100,
        alpha=0.10, color="#e67e22"
    )
    ax1.set_xlabel("AF Sensitivity Retained  (%)", fontsize=12)
    ax1.set_ylabel("Noise False-Alarm Rate  (%)", fontsize=12, color="#e67e22")
    ax1.tick_params(axis="y", labelcolor="#e67e22")
    ax1.set_xlim(0, 101)
    ax1.set_ylim(0, 105)
    ax1.invert_xaxis()   # high sensitivity (lenient gate) on the right

    # --- NSR false-alarm rate (right y-axis) ---
    ax2 = ax1.twinx()
    ax2.plot(
        sweep["af_sensitivity"] * 100,
        sweep["nsr_fp_rate"]    * 100,
        lw=2.5, color=COLOR_AFIB, ls="--",
        label="NSR false-alarm rate (right axis)"
    )
    ax2.set_ylabel("NSR False-Alarm Rate  (%)", fontsize=12, color=COLOR_AFIB)
    ax2.tick_params(axis="y", labelcolor=COLOR_AFIB)
    ax2.set_ylim(0, 35)

    # --- Annotate the no-gate baseline ---
    baseline_noise = sweep.loc[sweep["tau"].abs().idxmin(), "noise_fp_rate"] * 100
    baseline_nsr   = sweep.loc[sweep["tau"].abs().idxmin(), "nsr_fp_rate"]   * 100
    ax1.axhline(baseline_noise, color="#e67e22", lw=0.8, ls=":", alpha=0.6)
    ax1.text(2, baseline_noise + 1.5, f"No-gate baseline {baseline_noise:.0f}%",
             color="#e67e22", fontsize=9)

    # --- Mark a recommended operating point (80% AF sensitivity) ---
    # Find the highest-tau point that still retains >= 80% AF sensitivity
    rec = sweep[sweep["af_sensitivity"] >= 0.80].iloc[-1]
    ax1.scatter(rec["af_sensitivity"] * 100, rec["noise_fp_rate"] * 100,
                s=100, zorder=5, color="#2ecc71",
                label=f"Recommended op. point (τ={rec['tau']:.2f})")
    ax2.scatter(rec["af_sensitivity"] * 100, rec["nsr_fp_rate"] * 100,
                s=100, zorder=5, color="#2ecc71", marker="^")
    ax1.annotate(
        f"τ = {rec['tau']:.2f}\nNoise FA: {rec['noise_fp_rate']*100:.0f}%\n"
        f"NSR FA: {rec['nsr_fp_rate']*100:.0f}%\nSen: {rec['af_sensitivity']*100:.0f}%",
        xy=(rec["af_sensitivity"] * 100, rec["noise_fp_rate"] * 100),
        xytext=(rec["af_sensitivity"] * 100 + 8, rec["noise_fp_rate"] * 100 + 8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8)
    )

    # --- Legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title(
        "Signal-Quality Gate Operating Curve\n"
        "Read right to left: stricter gate → lower false alarms, lower AF sensitivity",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.35)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Operating curve saved -> %s", out)
    return out


# ============================================================================
# STEP 5 — RECOMMENDED OPERATING POINT + RESULTS TABLE
# ============================================================================

def find_operating_point(
    sweep: pd.DataFrame,
    min_af_sensitivity: float = 0.80,
) -> pd.Series:
    """
    Return the highest-tau (strictest) operating point that still retains
    at least `min_af_sensitivity` fraction of AF segments.

    The 80% floor is a reasonable default for a screening device:
        - Missing 1 in 5 AF episodes is clinically uncomfortable but not
          catastrophic if the device is supplemented by periodic holter review.
        - 90% or higher floors are available on the curve; move the threshold
          if the application demands it.
    """
    viable = sweep[sweep["af_sensitivity"] >= min_af_sensitivity]
    if viable.empty:
        logger.warning("No operating point retains >= %.0f%% AF sensitivity.",
                       min_af_sensitivity * 100)
        return sweep.iloc[0]
    return viable.iloc[-1]   # highest tau = strictest viable point


def print_results_table(
    sweep: pd.DataFrame,
    separation: dict,
    rec_point: pd.Series,
) -> None:
    """Print the key numbers for the thesis results section."""
    print("\n" + "=" * 72)
    print("SIGNAL-QUALITY GATE — RESULTS")
    print("=" * 72)

    print("\n--- Step 2: Score Separation ---")
    print("  Class medians:")
    for cls, med in sorted(separation["class_medians"].items()):
        print(f"    {cls:<8}  {med:.3f}")
    print(f"\n  Mann-Whitney (Noise < AF):  p = {separation['noise_vs_af_p']:.4f}"
          f"  {'SIGNIFICANT' if separation['noise_separates_af'] else 'NOT SIGNIFICANT'}")
    print(f"  Mann-Whitney (Noise < NSR): p = {separation['noise_vs_nsr_p']:.4f}"
          f"  {'SIGNIFICANT' if separation['noise_separates_nsr'] else 'NOT SIGNIFICANT'}")

    print("\n--- No-gate baseline (tau = 0) ---")
    baseline = sweep.iloc[0]
    print(f"  Noise false-alarm rate : {baseline['noise_fp_rate']*100:.1f}%")
    print(f"  NSR   false-alarm rate : {baseline['nsr_fp_rate']*100:.1f}%")
    print(f"  AF sensitivity         : {baseline['af_sensitivity']*100:.1f}%")

    print(f"\n--- Recommended operating point (tau = {rec_point['tau']:.2f}) ---")
    print(f"  AF sensitivity retained    : {rec_point['af_sensitivity']*100:.1f}%")
    print(f"  AF segments accepted       : {int(rec_point['af_accepted'])} / {int(sweep.iloc[0]['af_accepted'])}")
    print(f"  Noise rejected             : {int(rec_point['noise_rejected'])} / {int(rec_point['noise_rejected'] + rec_point['noise_accepted'])}")
    print(f"  Noise rejection rate       : {rec_point['noise_rejection_rate']*100:.1f}%")
    print(f"  Noise false-alarm rate     : {rec_point['noise_fp_rate']*100:.1f}%  (was {sweep.iloc[0]['noise_fp_rate']*100:.1f}%)")
    print(f"  NSR   false-alarm rate     : {rec_point['nsr_fp_rate']*100:.1f}%  (was {sweep.iloc[0]['nsr_fp_rate']*100:.1f}%)")
    print(f"  Segments accepted total    : {int(rec_point['n_accepted'])} ({rec_point['pct_accepted']:.1f}%)")

    print("\n--- Sensitivity / false-alarm ladder ---")
    print(f"  {'tau':>5}  {'AF sen':>8}  {'Noise FA':>10}  {'NSR FA':>8}  {'Noise rej':>10}")
    print("  " + "-" * 46)
    for _, r in sweep[::sweep.shape[0]//10 or 1].iterrows():
        print(f"  {r['tau']:>5.2f}  {r['af_sensitivity']*100:>7.1f}%  "
              f"{r['noise_fp_rate']*100:>9.1f}%  {r['nsr_fp_rate']*100:>7.1f}%  "
              f"{r['noise_rejection_rate']*100:>9.1f}%")
    print("=" * 72)


# ============================================================================
# MAIN — runs the full experiment end-to-end
# ============================================================================

def run_quality_gate_experiment(
    feat_path:   str | None = None,
    oof_path:    str | None = None,
    sweep_path:  str | None = None,
) -> Tuple[pd.DataFrame, dict, pd.Series]:
    """
    Run the complete quality-gate experiment.

    PARAMETERS
    ----------
    feat_path  : path to hrv_features.csv (default: data/features/hrv_features.csv)
    oof_path   : path to oof_predictions_xgb.csv (default: data/features/)
    sweep_path : where to save quality_gate_results.csv

    RETURNS
    -------
    (sweep_df, separation_dict, recommended_point)
    """
    if feat_path is None:
        feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
    if oof_path is None:
        oof_path = os.path.join(FEATURES_DIR, "oof_predictions_xgb.csv")

    # Also try the noise FP predictions if available (scored by the trained model on Noise).
    # noise_fp_context_xgb.csv is produced by noise_fp_analysis.py and already has
    # segment_id, y_pred, y_prob for all 219 Noise segments.
    noise_pred_path = os.path.join(FEATURES_DIR, "noise_fp_context_xgb.csv")

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"Feature matrix not found: {feat_path}. Run Phase 3 first.")
    if not os.path.isfile(oof_path):
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}. Run Phase 6 first.")

    feat = pd.read_csv(feat_path).reset_index(drop=True)
    oof  = pd.read_csv(oof_path)

    # Load Noise predictions if they exist (from noise_fp_analysis.py)
    if os.path.isfile(noise_pred_path):
        noise_pred = pd.read_csv(noise_pred_path)[["segment_id", "y_pred", "y_prob"]]
        logger.info("Loaded %d Noise predictions from %s", len(noise_pred), noise_pred_path)
        oof_combined = pd.concat([oof, noise_pred], ignore_index=True)
    else:
        logger.warning(
            "No noise_predictions_xgb.csv found. Gate will show NaN for Noise FA rates. "
            "Run: python -m src.context_analysis.noise_fp_analysis"
        )
        oof_combined = oof

    logger.info("Loaded feature matrix: %d rows", len(feat))
    logger.info("Loaded OOF predictions: %d rows", len(oof_combined))

    # ---------------------------------------------------------------------------
    # STEP 1: Compute quality scores
    # ---------------------------------------------------------------------------
    print("\nStep 1: Computing quality scores...")
    scores = compute_quality_scores(feat)
    feat   = feat.join(scores)
    logger.info("Quality scores computed. Range: [%.3f, %.3f]",
                scores.min(), scores.max())

    # ---------------------------------------------------------------------------
    # STEP 2: Verify separation
    # ---------------------------------------------------------------------------
    print("\nStep 2: Verifying score separation across classes...")
    separation = verify_score_separation(feat, scores)

    if not separation["noise_separates_af"]:
        print("\n*** WARNING: Score does not separate Noise from AF. "
              "The gate will not be useful. Review the score formula. ***\n")

    # Plot histograms
    plot_score_histograms(feat, scores)

    # ---------------------------------------------------------------------------
    # STEP 3: Sweep the threshold
    # ---------------------------------------------------------------------------
    print("\nStep 3: Sweeping threshold (200 points)...")
    sweep_df = sweep_threshold(feat, scores, oof_combined)

    # Save the full sweep table
    if sweep_path is None:
        sweep_path = os.path.join(FEATURES_DIR, "quality_gate_results.csv")
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Sweep results saved -> %s", sweep_path)

    # ---------------------------------------------------------------------------
    # STEP 4: Plot the operating curve
    # ---------------------------------------------------------------------------
    print("\nStep 4: Plotting operating curve...")
    plot_operating_curve(sweep_df)

    # ---------------------------------------------------------------------------
    # STEP 5: Recommend an operating point and print results
    # ---------------------------------------------------------------------------
    rec = find_operating_point(sweep_df, min_af_sensitivity=0.80)
    print_results_table(sweep_df, separation, rec)

    return sweep_df, separation, rec


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("SIGNAL-QUALITY GATE EXPERIMENT")
    print("=" * 72)
    print(
        "\nThis experiment measures the trade-off from adding a signal-quality\n"
        "gate before the AF classifier. A strict gate cuts Noise false alarms\n"
        "but also rejects some real AF. The operating curve shows that trade-off.\n"
    )

    sweep, separation, rec = run_quality_gate_experiment()

    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Full sweep table: {os.path.join(FEATURES_DIR, 'quality_gate_results.csv')}")
