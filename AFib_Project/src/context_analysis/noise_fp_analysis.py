"""
=============================================================================
FILE: src/context_analysis/noise_fp_analysis.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 9: FALSE POSITIVES ON MOTION-CORRUPTED (NOISE) SEGMENTS
=============================================================================

WHY THIS FILE EXISTS - THE PROBLEM WITH THE ORIGINAL FP ANALYSIS
----------------------------------------------------------------
Phase 8 asks: among NSR segments the model wrongly called AF, was the model
misled by physical activity? It came back with nothing. Chi-square p = 0.56,
Mann-Whitney p = 0.45.

That null is not a bug, and it is not a finding either. It is a consequence of
how the dataset was labelled:

    MovementAcceleration, NSR segments:  median 0.008 g
    Segments above the 0.1 g threshold:  31 out of 634  (4.9%)

There is essentially no movement in the NSR set. A test for "does movement
explain the errors" cannot succeed when nothing moves. Asking it anyway and
reporting p = 0.45 would be reporting the absence of variance as the absence
of an effect - a different claim entirely.

WHY THE MOVEMENT IS MISSING - AND WHERE IT WENT
-----------------------------------------------
Cardiologists annotate segments they can READ. When a recording is corrupted
by motion artifact, the rhythm is not interpretable, so it does not get called
NSR - it gets called NOISE. There are 221 such segments, and they carry
roughly double the movement acceleration of the clean ones.

So the motion-artifact population was labelled into the Noise class, and the
pipeline then excluded Noise from training AND from the FP analysis. The
original analysis was looking in the one place the effect could not appear.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
It scores the trained model on all 221 Noise segments and treats every AF
prediction there as a false positive. That is not a modelling convenience - it
is the clinically correct reading:

    A Noise segment contains no interpretable rhythm. There is no AF in it to
    detect. If a screening device announces "atrial fibrillation" on an
    unreadable trace, it has produced a false alarm, and the patient is sent
    for an ECG they did not need. Deployment FP rate must include these.

This population has what the NSR population lacks: genuine variation in
movement. So the question "do motion artifacts drive false alarms?" finally
becomes answerable.

WHAT WE CAN AND CANNOT CONCLUDE
-------------------------------
CAN: quantify the false-alarm rate a wearable would experience on corrupted
     data, and test whether it scales with measured movement.
CANNOT: call these errors "misclassification" in the usual sense - there is no
     ground-truth rhythm to be wrong about. They are ALARM-RATE measurements,
     not accuracy measurements, and the write-up must keep the two separate.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (  # noqa: E402
    CLASS_AFIB, COLOR_AFIB, COLOR_NORMAL, CONTEXT_COLUMNS, FEATURES_DIR,
    FIGURES_DIR, FIGURE_DPI, MAI_MOTION_THRESHOLD, MODELS_DIR, PROCESSED_DIR,
    RAW_CLASS_NOISE,
)
from src.context_analysis.fp_analysis import attach_context  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MAI_COLUMN = "MovementAcceleration [g]"
ACTIVITY_COLUMN = "ActivityClass []"
STEP_COLUMN = "StepCount [steps]"


# ---------------------------------------------------------------------------
# SCORING THE MODEL ON NOISE SEGMENTS
# ---------------------------------------------------------------------------
def score_noise_segments(
    model_name: str = "xgb",
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run the trained model over every Noise segment and record its verdict.

    IMPORTANT - WHY THIS IS NOT DATA LEAKAGE:
        Noise segments were never in the training set (DROP_NOISE_FROM_TRAINING
        is True), so the final model has genuinely never seen them. Scoring
        them with the final model is legitimate out-of-sample inference.

    RETURNS
    -------
    DataFrame: segment_id, subject, y_prob, alarm (1 = model said AF)
    """
    import joblib

    model_path = os.path.join(MODELS_DIR, f"{model_name}_final.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run phases 5/6 first."
        )
    model = joblib.load(model_path)

    features = pd.read_csv(os.path.join(FEATURES_DIR, "hrv_features.csv"))
    index = pd.read_csv(os.path.join(PROCESSED_DIR, "segment_index.csv"))

    noise = index[index["raw_class"] == RAW_CLASS_NOISE][["segment_id", "subject"]]

    # hrv_features.csv already carries its own copy of several index columns
    # (subject, class_name, ...). Merging naively gives subject_x / subject_y
    # and then every later reference to "subject" raises a KeyError. Drop the
    # duplicates from the right-hand frame and keep the index as the single
    # source of truth for segment metadata.
    overlapping = [column for column in features.columns
                   if column in noise.columns and column != "segment_id"]
    data = noise.merge(features.drop(columns=overlapping),
                       on="segment_id", how="inner")

    # Only segments where feature extraction actually succeeded can be scored.
    usable = data[data["quality"] == "ok"].copy()
    logger.info(
        "Noise segments: %d labelled, %d with usable features (%d unreadable "
        "even for HRV)",
        len(noise), len(usable), len(noise) - len(usable),
    )

    if feature_names is None:
        import json
        selection_path = os.path.join(FEATURES_DIR, "selected_features.json")
        with open(selection_path) as handle:
            feature_names = json.load(handle)["selected_features"]

    matrix = usable[feature_names].to_numpy(dtype=float)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)[:, 1]
    else:
        probabilities = model.decision_function(matrix)

    predictions = model.predict(matrix)

    return pd.DataFrame({
        "segment_id": usable["segment_id"].to_numpy(),
        "subject": usable["subject"].to_numpy(),
        "y_prob": probabilities,
        "alarm": (predictions == CLASS_AFIB).astype(int),
    })


# ---------------------------------------------------------------------------
# STATISTICAL TESTS
# ---------------------------------------------------------------------------
def movement_vs_alarm_test(frame: pd.DataFrame) -> Dict[str, object]:
    """
    Mann-Whitney U: is movement higher when the model raises a false alarm?

    Mann-Whitney rather than a t-test because movement acceleration is heavily
    right-skewed (most of the time people are still, occasionally they walk),
    which violates the normality a t-test assumes.
    """
    alarmed = frame.loc[frame["alarm"] == 1, MAI_COLUMN].dropna()
    silent = frame.loc[frame["alarm"] == 0, MAI_COLUMN].dropna()

    if len(alarmed) < 5 or len(silent) < 5:
        return {"error": "too few segments in one group", "n_alarm": len(alarmed),
                "n_silent": len(silent)}

    statistic, p_value = stats.mannwhitneyu(alarmed, silent, alternative="two-sided")

    # Rank-biserial correlation: an effect size that goes with Mann-Whitney.
    # 0 = no difference, ±1 = complete separation. Reporting p alone says only
    # whether an effect exists, not whether it is large enough to matter.
    effect_size = 1 - (2 * statistic) / (len(alarmed) * len(silent))

    return {
        "test": "Mann-Whitney U",
        "u_statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "rank_biserial_r": float(effect_size),
        "alarm_mai_median": float(alarmed.median()),
        "silent_mai_median": float(silent.median()),
        "n_alarm": int(len(alarmed)),
        "n_silent": int(len(silent)),
    }


def alarm_rate_by_movement_quartile(frame: pd.DataFrame) -> pd.DataFrame:
    """
    False-alarm rate across movement quartiles.

    WHY QUARTILES INSTEAD OF THE FIXED 0.1 g THRESHOLD:
        The 0.1 g cut in config was a guess made before we saw the data. It
        splits the NSR set 603/31 - so lopsided that any comparison is
        underpowered. Quartiles are derived FROM the observed distribution, so
        every bin holds a quarter of the data by construction and the
        comparison has power regardless of what the sensor's units turn out
        to look like. A monotone rise across quartiles is also much harder to
        dismiss than a single significant threshold, which could be cherry-picked.
    """
    working = frame.dropna(subset=[MAI_COLUMN]).copy()
    if len(working) < 20:
        return pd.DataFrame()

    working["mai_quartile"] = pd.qcut(
        working[MAI_COLUMN], 4,
        labels=["Q1 (stillest)", "Q2", "Q3", "Q4 (most movement)"],
        duplicates="drop",
    )

    table = working.groupby("mai_quartile", observed=True).agg(
        n=("alarm", "size"),
        false_alarms=("alarm", "sum"),
        alarm_rate=("alarm", "mean"),
        mai_median=(MAI_COLUMN, "median"),
    )
    table["alarm_rate_pct"] = (table["alarm_rate"] * 100).round(1)
    return table


def trend_test(frame: pd.DataFrame) -> Dict[str, object]:
    """
    Spearman correlation between movement and the model's AF probability.

    This is the cleanest single statistic for the paper: it uses the model's
    continuous confidence rather than its thresholded decision, so it does not
    depend on where the 0.5 cut happens to fall, and it tests for a MONOTONE
    relationship without assuming the relationship is linear.
    """
    working = frame.dropna(subset=[MAI_COLUMN, "y_prob"])
    if len(working) < 10:
        return {"error": "too few segments"}

    rho, p_value = stats.spearmanr(working[MAI_COLUMN], working["y_prob"])
    return {
        "test": "Spearman rank correlation (movement vs AF probability)",
        "rho": float(rho),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n": int(len(working)),
    }


def compare_populations(
    noise_frame: pd.DataFrame,
    nsr_fp_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Contrast the false-alarm rate on Noise against the FP rate on clean NSR.

    This is the headline comparison. If a wearable produces alarms far more
    often on corrupted data than on clean data, then artifact rejection - not
    a better classifier - is the highest-value engineering fix, and that is a
    concrete recommendation rather than a general observation.
    """
    result = {
        "noise_segments_scored": int(len(noise_frame)),
        "noise_false_alarms": int(noise_frame["alarm"].sum()),
        "noise_alarm_rate": float(noise_frame["alarm"].mean()),
    }
    if nsr_fp_frame is not None and len(nsr_fp_frame):
        nsr_rate = float(nsr_fp_frame["is_fp"].mean())
        result["nsr_segments"] = int(len(nsr_fp_frame))
        result["nsr_fp_rate"] = nsr_rate
        if nsr_rate > 0:
            result["risk_ratio_noise_vs_nsr"] = result["noise_alarm_rate"] / nsr_rate
    return result


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
def plot_noise_fp(frame: pd.DataFrame, quartiles: pd.DataFrame,
                  model_name: str = "xgboost") -> str:
    """Three-panel summary of the Noise-segment false-alarm analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- movement distribution, alarm vs silent -------------------------
    alarmed = frame.loc[frame["alarm"] == 1, MAI_COLUMN].dropna()
    silent = frame.loc[frame["alarm"] == 0, MAI_COLUMN].dropna()
    box = axes[0].boxplot([silent, alarmed],
                          tick_labels=["No alarm", "False alarm"],
                          patch_artist=True, showfliers=False, widths=0.55)
    for patch, colour in zip(box["boxes"], [COLOR_NORMAL, COLOR_AFIB]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    axes[0].set_ylabel(MAI_COLUMN)
    axes[0].set_title("Movement when the model\nraises a false alarm",
                      fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # --- alarm rate by movement quartile ---------------------------------
    if not quartiles.empty:
        positions = np.arange(len(quartiles))
        bars = axes[1].bar(positions, quartiles["alarm_rate_pct"],
                           color=COLOR_AFIB, alpha=0.8, edgecolor="black",
                           linewidth=0.6)
        for bar, (_, row) in zip(bars, quartiles.iterrows()):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.6,
                         f"{row['alarm_rate_pct']:.1f}%\n(n={int(row['n'])})",
                         ha="center", fontsize=9)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(quartiles.index, rotation=20, ha="right")
        axes[1].set_ylabel("False alarm rate (%)")
        axes[1].set_title("Does the alarm rate rise\nwith movement?",
                          fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)

    # --- movement vs model confidence -------------------------------------
    working = frame.dropna(subset=[MAI_COLUMN, "y_prob"])
    axes[2].scatter(working[MAI_COLUMN], working["y_prob"], s=18, alpha=0.5,
                    color=COLOR_AFIB, edgecolors="none")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.9,
                    label="decision threshold")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_xlabel(MAI_COLUMN + "  (symlog scale)")
    axes[2].set_ylabel("Model P(AF)")
    axes[2].set_title("Movement vs model confidence\non unreadable segments",
                      fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.suptitle(
        f"False alarms on motion-corrupted (Noise) segments - {model_name}",
        fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, f"noise_fp_analysis_{model_name}.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Noise FP figure saved -> %s", out)
    return out


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
def run_noise_fp_analysis(model_name: str = "xgb") -> Dict[str, object]:
    """Score Noise segments, attach context, run the tests, draw the figure."""
    index = pd.read_csv(os.path.join(PROCESSED_DIR, "segment_index.csv"))

    scored = score_noise_segments(model_name=model_name)
    if scored.empty:
        logger.error("No Noise segments could be scored")
        return {}

    # attach_context expects a predictions-shaped frame; give it the columns
    # it needs without pretending these segments have a rhythm label.
    predictions = scored.rename(columns={"alarm": "y_pred"}).copy()
    predictions["y_true"] = np.nan
    # attach_context pulls "subject" from the index itself, so passing our own
    # copy would collide into subject_x / subject_y. Drop it and let the join
    # supply it.
    predictions = predictions.drop(columns=["subject"], errors="ignore")
    with_context = attach_context(index, predictions)
    with_context["alarm"] = with_context["y_pred"]

    quartiles = alarm_rate_by_movement_quartile(with_context)

    results: Dict[str, object] = {
        "model": model_name,
        "summary": compare_populations(with_context),
        "mann_whitney": movement_vs_alarm_test(with_context),
        "spearman": trend_test(with_context),
        "quartiles": quartiles,
    }

    # Compare against the clean-NSR FP rate from Phase 8, if it is on disk.
    oof_path = os.path.join(FEATURES_DIR, f"oof_predictions_{model_name}.csv")
    if os.path.isfile(oof_path):
        oof = pd.read_csv(oof_path)
        nsr = oof[oof["y_true"] == 0]
        if len(nsr):
            nsr_frame = pd.DataFrame({"is_fp": (nsr["y_pred"] == 1)})
            results["summary"] = compare_populations(with_context, nsr_frame)

    plot_noise_fp(with_context, quartiles, model_name=model_name)

    out_path = os.path.join(FEATURES_DIR, f"noise_fp_context_{model_name}.csv")
    with_context.to_csv(out_path, index=False)
    logger.info("Noise FP table saved -> %s", out_path)

    return results


def print_report(results: Dict[str, object]) -> None:
    """Human-readable summary for the console and the thesis."""
    print("=" * 74)
    print(f"FALSE ALARMS ON MOTION-CORRUPTED SEGMENTS - {results.get('model')}")
    print("=" * 74)

    summary = results.get("summary", {})
    print(f"\nNoise segments scored   : {summary.get('noise_segments_scored')}")
    print(f"False alarms raised     : {summary.get('noise_false_alarms')} "
          f"({summary.get('noise_alarm_rate', 0)*100:.1f}%)")
    if "nsr_fp_rate" in summary:
        print(f"FP rate on clean NSR    : {summary['nsr_fp_rate']*100:.1f}% "
              f"(n={summary.get('nsr_segments')})")
        if "risk_ratio_noise_vs_nsr" in summary:
            print(f"Risk ratio (noise/clean): "
                  f"{summary['risk_ratio_noise_vs_nsr']:.2f}x")

    print("\n--- Does movement predict the false alarm? ---")
    mw = results.get("mann_whitney", {})
    if "error" in mw:
        print(f"  Mann-Whitney not run: {mw['error']}")
    else:
        print(f"  Mann-Whitney U p = {mw['p_value']:.4f} "
              f"({'SIGNIFICANT' if mw['significant'] else 'not significant'})")
        print(f"  effect size (rank-biserial r) = {mw['rank_biserial_r']:+.3f}")
        print(f"  median movement: alarm {mw['alarm_mai_median']:.4f} g "
              f"vs silent {mw['silent_mai_median']:.4f} g")

    sp = results.get("spearman", {})
    if "error" not in sp:
        print(f"  Spearman rho = {sp['rho']:+.3f}, p = {sp['p_value']:.4f} "
              f"({'SIGNIFICANT' if sp['significant'] else 'not significant'})")

    quartiles = results.get("quartiles")
    if quartiles is not None and not quartiles.empty:
        print("\n--- False alarm rate by movement quartile ---")
        print(quartiles[["n", "false_alarms", "alarm_rate_pct",
                         "mai_median"]].to_string())

    print("\n" + "=" * 74)


if __name__ == "__main__":
    for model in ("xgb", "svm"):
        try:
            print_report(run_noise_fp_analysis(model_name=model))
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", model, exc)
