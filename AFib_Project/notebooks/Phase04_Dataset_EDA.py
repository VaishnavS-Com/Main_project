# -*- coding: utf-8 -*-
"""
=============================================================================
FILE: notebooks/Phase04_Dataset_EDA.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 4b: EXPLORATORY DATA ANALYSIS ON THE **REAL** DATASET
=============================================================================

WHAT MAKES THIS FILE DIFFERENT FROM PHASE 1 AND 2:
    Phases 1 and 2 ran on SYNTHETIC data - useful for explaining concepts
    before the dataset was in hand. This file touches the actual recordings.
    Every number it prints is real, and a few of them are inconvenient.

WHAT IT PRODUCES:
    Figure 1  Class distribution and per-subject composition
    Figure 2  Real ECG waveforms: AF vs NSR vs Noise, straight from ecg.bin
    Figure 3  RR-interval evidence - why we know class 1 is AF (Poincare plot)
    Figure 4  Context: movement, activity and heart rate around each class
    Plus a printed data-quality report that feeds the methodology section.

HOW TO RUN:
    cd AFib_Project
    .\\venv\\Scripts\\activate
    python notebooks\\Phase04_Dataset_EDA.py

PREREQUISITE:
    python src\\preprocessing\\dataset_index.py     (writes segment_index.csv)
=============================================================================
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")            # write files, never open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (  # noqa: E402
    COLOR_AFIB, COLOR_ECG, COLOR_NORMAL, CONTROL_SUBJECT_IDS, FIGURES_DIR,
    FIGURE_DPI, PROCESSED_DIR, SAMPLING_FREQUENCY, SEGMENT_DURATION_SEC,
)
from src.preprocessing.dataset_index import (  # noqa: E402
    get_modelling_subset, summarise_index,
)
from src.preprocessing.unisens_reader import (  # noqa: E402
    read_ecg_segment, read_unisens_metadata,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

os.makedirs(FIGURES_DIR, exist_ok=True)

CLASS_COLORS = {
    "AF": COLOR_AFIB,        # red
    "NSR": COLOR_NORMAL,     # green
    "Noise": "#7f8c8d",      # grey
    "Other": "#f39c12",      # orange
}
CLASS_ORDER = ["AF", "NSR", "Noise", "Other"]


# =============================================================================
# SMALL HELPERS
# =============================================================================
def load_index() -> pd.DataFrame:
    """Load the segment index built in Phase 4, with a helpful error if absent."""
    path = os.path.join(PROCESSED_DIR, "segment_index.csv")
    if not os.path.isfile(path):
        raise SystemExit(
            "segment_index.csv not found.\n"
            "Run this first:  python src/preprocessing/dataset_index.py"
        )
    return pd.read_csv(path)


def detect_r_peaks(signal: np.ndarray, fs: int = SAMPLING_FREQUENCY) -> np.ndarray:
    """
    A deliberately simple Pan-Tompkins-style R-peak detector, for EDA only.

    Steps, which are the classic ones:
      1. Bandpass 5-25 Hz  - the QRS complex lives here; P and T waves and
         baseline drift do not, so this alone removes most false peaks.
      2. Square the signal - makes big deflections dominate and removes sign.
      3. Moving-average over 150 ms - roughly one QRS width, turning each
         spike into a smooth bump that is easy to find.
      4. find_peaks with a 300 ms refractory distance - no human heart beats
         faster than 200 BPM, so peaks closer than that are artifacts.

    Phase 7 replaces this with the validated NeuroKit2 implementation. It is
    written out here because a detector you can read in ten lines is the best
    way to understand what the library is doing for you later.
    """
    nyquist = fs / 2
    b, a = butter(3, [5 / nyquist, 25 / nyquist], btype="band")
    filtered = filtfilt(b, a, signal.astype(float))

    energy = filtered ** 2
    window = int(0.15 * fs)
    energy = np.convolve(energy, np.ones(window) / window, mode="same")

    threshold = energy.mean() + 0.5 * energy.std()
    peaks, _ = find_peaks(energy, height=threshold, distance=int(0.3 * fs))
    return peaks


def rr_intervals(signal: np.ndarray, fs: int = SAMPLING_FREQUENCY) -> np.ndarray:
    """R-peak times converted to intervals in seconds, physiologically clipped."""
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 3:
        return np.array([])
    rr = np.diff(peaks) / fs
    return rr[(rr > 0.3) & (rr < 2.0)]


def load_segment(row: pd.Series) -> np.ndarray:
    """Fetch the raw ECG for one row of the index."""
    recording = read_unisens_metadata(row["signal_dir"])
    return read_ecg_segment(
        recording, int(row["start_sample"]), int(row["end_sample"])
    )


# =============================================================================
# FIGURE 1 - HOW THE LABELS ARE DISTRIBUTED
# =============================================================================
def figure_class_distribution(index: pd.DataFrame) -> None:
    summary = summarise_index(index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1, 2.2]})

    # --- left: overall counts -------------------------------------------
    counts = [int((index["class_name"] == c).sum()) for c in CLASS_ORDER]
    bars = axes[0].bar(CLASS_ORDER, counts,
                       color=[CLASS_COLORS[c] for c in CLASS_ORDER],
                       edgecolor="black", linewidth=0.6)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 12,
                     f"{count}\n{count / len(index) * 100:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Labelled 10 s segments by class\n(n = %d)" % len(index),
                      fontweight="bold")
    axes[0].set_ylabel("Number of segments")
    axes[0].set_ylim(0, max(counts) * 1.22)
    axes[0].grid(axis="y", alpha=0.3)

    # --- right: per-subject composition ----------------------------------
    subjects = summary.index.tolist()
    bottom = np.zeros(len(subjects))
    for class_name in CLASS_ORDER:
        values = summary[class_name].values.astype(float)
        axes[1].bar(subjects, values, bottom=bottom,
                    color=CLASS_COLORS[class_name], label=class_name,
                    edgecolor="white", linewidth=0.4)
        bottom += values

    axes[1].set_title(
        "Per-subject composition - note how few subjects contain BOTH classes",
        fontweight="bold")
    axes[1].set_ylabel("Number of segments")
    axes[1].legend(title="Class", ncol=4)
    axes[1].grid(axis="y", alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    # Mark the healthy controls so the two cohorts are visually distinct.
    for tick in axes[1].get_xticklabels():
        if tick.get_text() in CONTROL_SUBJECT_IDS:
            tick.set_color("#2980b9")
            tick.set_fontweight("bold")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "Phase04_Class_Distribution.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# =============================================================================
# FIGURE 2 - WHAT THE ACTUAL SIGNAL LOOKS LIKE
# =============================================================================
def figure_example_waveforms(index: pd.DataFrame, seed: int = 7) -> None:
    """
    Plot one real 10-second segment per class.

    Worth looking at closely: in the AF trace the R peaks are unevenly spaced
    and the flat P wave before each QRS is missing. In NSR the spacing is
    metronomic. The Noise panel shows why 221 segments had to be excluded -
    there is no recoverable rhythm in them.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(15, 11), sharex=True)

    for axis, class_name in zip(axes, CLASS_ORDER):
        candidates = index[index["class_name"] == class_name]
        if candidates.empty:
            axis.set_visible(False)
            continue

        row = candidates.iloc[int(rng.integers(len(candidates)))]
        signal = load_segment(row)
        time = np.arange(len(signal)) / SAMPLING_FREQUENCY

        axis.plot(time, signal, color=COLOR_ECG, linewidth=0.7)

        # Overlay detected R peaks for the interpretable classes.
        if class_name in ("AF", "NSR"):
            peaks = detect_r_peaks(signal)
            axis.plot(peaks / SAMPLING_FREQUENCY, signal[peaks], "v",
                      color=CLASS_COLORS[class_name], markersize=7,
                      label=f"{len(peaks)} R peaks")
            rr = rr_intervals(signal)
            if len(rr) > 2:
                axis.legend(
                    title=f"mean RR {rr.mean()*1000:.0f} ms | "
                          f"CV {rr.std()/rr.mean():.3f}",
                    loc="upper right", fontsize=9)

        axis.set_ylabel("mV")
        axis.set_title(
            f"{class_name}  -  subject {row['subject']}, "
            f"t = {row['t_start_sec']/3600:.2f} h into the recording",
            fontweight="bold", color=CLASS_COLORS[class_name], loc="left")
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel(f"Time (s)  -  one {SEGMENT_DURATION_SEC} s labelled segment")
    plt.suptitle("Real CACHET-CADB ECG segments, read directly from ecg.bin",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "Phase04_Example_Waveforms.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# =============================================================================
# FIGURE 3 - THE RR EVIDENCE THAT FIXES THE LABEL MAPPING
# =============================================================================
def figure_rr_evidence(index: pd.DataFrame, n_per_class: int = 120,
                       seed: int = 3) -> pd.DataFrame:
    """
    The figure that justifies "class 1 = AF" in config.py.

    A Poincare plot draws each RR interval against the NEXT one. Regular
    rhythm collapses to a tight cluster on the diagonal; AF scatters into a
    diffuse cloud because each beat is independent of the last. This is the
    single most convincing visual for the viva.
    """
    rng = np.random.default_rng(seed)
    records = []

    for class_name in ("AF", "NSR"):
        candidates = index[index["class_name"] == class_name]
        take = min(n_per_class, len(candidates))
        chosen = candidates.iloc[rng.choice(len(candidates), take, replace=False)]

        for _, row in chosen.iterrows():
            try:
                signal = load_segment(row)
            except Exception:      # noqa: BLE001 - one unreadable file is fine
                continue
            rr = rr_intervals(signal)
            if len(rr) < 4:
                continue
            records.append({
                "class_name": class_name,
                "subject": row["subject"],
                "mean_rr": rr.mean(),
                "hr": 60 / rr.mean(),
                "sdnn": rr.std() * 1000,
                "rmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000,
                "cv": rr.std() / rr.mean(),
                "rr": rr,
            })

    frame = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Poincare -------------------------------------------------------
    for class_name in ("NSR", "AF"):
        subset = frame[frame["class_name"] == class_name]
        x = np.concatenate([r[:-1] for r in subset["rr"]]) if len(subset) else []
        y = np.concatenate([r[1:] for r in subset["rr"]]) if len(subset) else []
        axes[0].scatter(x, y, s=9, alpha=0.35, label=class_name,
                        color=CLASS_COLORS[class_name], edgecolors="none")
    axes[0].plot([0.3, 1.8], [0.3, 1.8], "k--", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("RR$_n$ (s)")
    axes[0].set_ylabel("RR$_{n+1}$ (s)")
    axes[0].set_title("Poincare plot\ntight diagonal = regular, cloud = AF",
                      fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- RMSSD distribution ---------------------------------------------
    for class_name in ("NSR", "AF"):
        values = frame.loc[frame["class_name"] == class_name, "rmssd"]
        axes[1].hist(values, bins=35, alpha=0.6, label=class_name,
                     color=CLASS_COLORS[class_name], edgecolor="black",
                     linewidth=0.4)
    axes[1].set_xlabel("RMSSD (ms)")
    axes[1].set_ylabel("Segments")
    axes[1].set_title("Beat-to-beat variability\nseparates the classes almost alone",
                      fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # --- CV boxplot ------------------------------------------------------
    data = [frame.loc[frame["class_name"] == c, "cv"].values
            for c in ("NSR", "AF")]
    box = axes[2].boxplot(data, labels=["NSR", "AF"], patch_artist=True,
                          widths=0.55)
    for patch, class_name in zip(box["boxes"], ("NSR", "AF")):
        patch.set_facecolor(CLASS_COLORS[class_name])
        patch.set_alpha(0.75)
    axes[2].set_ylabel("Coefficient of variation of RR")
    axes[2].set_title("Irregularity index\n(the clinical definition of AF)",
                      fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Evidence that raw class 1 = AF and class 2 = NSR",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "Phase04_RR_Evidence.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)

    return frame.drop(columns="rr")


# =============================================================================
# FIGURE 4 - THE CONTEXT CHANNEL (the project's research contribution)
# =============================================================================
def figure_context(index: pd.DataFrame, max_files: int = 40) -> pd.DataFrame:
    """
    Join each segment to its row in context.xlsx and look at the distributions.

    The pairing is exact rather than interpolated: context.xlsx is sampled on
    a 10-second grid and the labelled segments are 10 seconds long, so segment
    k maps to context row floor(t_start / 10).

    What to look for: if Noise segments sit at systematically higher movement
    acceleration than clean ones, then motion is generating the artifacts -
    which is the hypothesis the whole false-positive analysis rests on.
    """
    context_columns = [
        "MovementAcceleration [g]", "ActivityClass []", "BodyPosition []",
        "StepCount [steps]", "Hr [1/min]", "MET []",
    ]
    records = []

    for path, group in list(index.groupby("context_path"))[:max_files]:
        if not isinstance(path, str) or not path or not os.path.isfile(path):
            continue
        try:
            context = pd.read_excel(path)
        except Exception:      # noqa: BLE001
            continue

        for _, row in group.iterrows():
            position = int(row["context_row"])
            if position >= len(context):
                continue
            entry = {"class_name": row["class_name"], "subject": row["subject"]}
            for column in context_columns:
                entry[column] = (context[column].iloc[position]
                                 if column in context.columns else np.nan)
            records.append(entry)

    frame = pd.DataFrame(records)
    if frame.empty:
        logger.warning("No context rows matched - skipping Figure 4")
        return frame

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- movement acceleration by class ----------------------------------
    present = [c for c in CLASS_ORDER
               if frame.loc[frame["class_name"] == c,
                            "MovementAcceleration [g]"].notna().any()]
    data = [frame.loc[frame["class_name"] == c,
                      "MovementAcceleration [g]"].dropna().values
            for c in present]
    box = axes[0].boxplot(data, labels=present, patch_artist=True, showfliers=False)
    for patch, class_name in zip(box["boxes"], present):
        patch.set_facecolor(CLASS_COLORS[class_name])
        patch.set_alpha(0.75)
    axes[0].set_ylabel("Movement acceleration (g)")
    axes[0].set_title("Motion at the time of each segment\n"
                      "is Noise driven by movement?", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # --- activity class composition --------------------------------------
    activity = (frame.dropna(subset=["ActivityClass []"])
                     .groupby(["class_name", "ActivityClass []"])
                     .size().unstack(fill_value=0))
    if not activity.empty:
        proportions = activity.div(activity.sum(axis=1), axis=0)
        proportions.plot(kind="bar", stacked=True, ax=axes[1],
                         colormap="viridis", edgecolor="white", linewidth=0.4)
        axes[1].set_ylabel("Proportion of segments")
        axes[1].set_xlabel("")
        axes[1].set_title("Activity class composition", fontweight="bold")
        axes[1].legend(title="Activity code", fontsize=8, ncol=2)
        plt.setp(axes[1].get_xticklabels(), rotation=0)

    # --- device heart rate -------------------------------------------------
    for class_name in ("NSR", "AF"):
        values = frame.loc[frame["class_name"] == class_name,
                           "Hr [1/min]"].dropna()
        if len(values):
            axes[2].hist(values, bins=30, alpha=0.6, label=class_name,
                         color=CLASS_COLORS[class_name], edgecolor="black",
                         linewidth=0.4)
    axes[2].set_xlabel("Device-reported heart rate (1/min)")
    axes[2].set_ylabel("Segments")
    axes[2].set_title("On-device HR\n(AF often shows rapid ventricular response)",
                      fontweight="bold")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Contextual conditions surrounding each labelled segment",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "Phase04_Context.png")
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)

    return frame


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=" * 74)
    print("PHASE 4b: EXPLORATORY DATA ANALYSIS ON THE REAL CACHET-CADB DATA")
    print("=" * 74)

    index = load_index()
    modelling = get_modelling_subset(index)

    # ---------------------------------------------------------------- report
    print(f"\nSegments indexed        : {len(index)}")
    print(f"Subjects with labels    : {index['subject'].nunique()} of 24")
    print(f"Recording parts         : "
          f"{index.groupby(['subject','session','part']).ngroups}")
    print(f"Total labelled ECG time : "
          f"{len(index) * SEGMENT_DURATION_SEC / 60:.1f} minutes "
          f"({len(index) * SEGMENT_DURATION_SEC / 3600:.2f} hours)")

    print("\n--- THINGS THE METHODOLOGY SECTION MUST DISCLOSE ---")

    missing = sorted(set(["P%d" % i for i in range(1, 20)] + ["P21", "P23"])
                     - set(index["subject"].unique()))
    print(f"  * Subjects with NO usable labels : {missing or 'none'}")

    per_subject = index.pivot_table(index="subject", columns="class_name",
                                    values="segment_id", aggfunc="count",
                                    fill_value=0)
    for name in ("AF", "NSR"):
        if name not in per_subject:
            per_subject[name] = 0
    both = per_subject[(per_subject["AF"] > 0) & (per_subject["NSR"] > 0)]
    print(f"  * Subjects containing BOTH AF and NSR : {len(both)} of "
          f"{len(per_subject)}  -> {sorted(both.index)}")
    print("    Most subjects are almost entirely one class. A plain GroupKFold")
    print("    can therefore produce a fold with no AF at all, making AUC")
    print("    undefined for that fold. Phase 12 must use a STRATIFIED group")
    print("    split and report per-fold class counts.")

    noise_rate = (index["class_name"] == "Noise").mean()
    print(f"  * Noise segments : {int((index['class_name']=='Noise').sum())} "
          f"({noise_rate*100:.1f}%) - excluded from training, retained for")
    print("    the false-positive analysis in Phase 14.")

    other_count = int((index["class_name"] == "Other").sum())
    print(f"  * 'Other' arrhythmia : {other_count} segments ({other_count/len(index)*100:.1f}%)")
    print("    Too few to model separately; grouped with non-AF. Say so.")

    print(f"  * Class balance for modelling : "
          f"{int((modelling['label']==0).sum())} non-AF vs "
          f"{int((modelling['label']==1).sum())} AF "
          f"({modelling['label'].mean()*100:.1f}% positive) - close to")
    print("    balanced, so accuracy is not a badly misleading metric here,")
    print("    though F1 and AUC remain the ones to report.")

    print("\n--- Generating figures ---")
    figure_class_distribution(index)
    print("  [1/4] class distribution")

    figure_example_waveforms(index)
    print("  [2/4] example waveforms")

    rr_frame = figure_rr_evidence(index)
    print("  [3/4] RR-interval evidence")

    context_frame = figure_context(index)
    print("  [4/4] context distributions")

    # ------------------------------------------------- printed RR summary
    if not rr_frame.empty:
        print("\n--- RR statistics measured on real segments ---")
        print(rr_frame.groupby("class_name")[["hr", "sdnn", "rmssd", "cv"]]
                      .agg(["mean", "median", "count"]).round(3).to_string())
        rr_path = os.path.join(PROCESSED_DIR, "phase04_rr_summary.csv")
        rr_frame.to_csv(rr_path, index=False)
        print(f"\nSaved -> {rr_path}")

    if not context_frame.empty:
        print("\n--- Movement acceleration (g) by class ---")
        print(context_frame.groupby("class_name")["MovementAcceleration [g]"]
                           .agg(["mean", "median", "std", "count"])
                           .round(4).to_string())
        context_path = os.path.join(PROCESSED_DIR, "phase04_context_sample.csv")
        context_frame.to_csv(context_path, index=False)
        print(f"\nSaved -> {context_path}")

    print(f"\nFigures written to: {FIGURES_DIR}")
    print("=" * 74)


if __name__ == "__main__":
    main()
