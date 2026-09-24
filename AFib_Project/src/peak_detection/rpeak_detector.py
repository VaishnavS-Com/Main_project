"""
=============================================================================
FILE: src/peak_detection/rpeak_detector.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 6 & 7: R-PEAK DETECTION + RR-INTERVAL EXTRACTION
=============================================================================

PURPOSE:
    Given a clean 10-second ECG segment (from ecg_filter.py), find every
    heartbeat's R-peak and convert peak locations into RR-interval series
    ready for HRV feature extraction.

THE PIPELINE:
    1. Detect R-peaks via NeuroKit2's Pan-Tompkins implementation
    2. Convert peak sample indices → RR intervals in milliseconds
    3. Remove ectopic/artefact beats (±20% local-median rule)
    4. Return (rr_ms, quality_flag) where quality_flag communicates
       how many beats were rejected

WHAT IS AN R-PEAK?
    The ECG has five named deflections: P, Q, R, S, T.
    The R peak is the tall, sharp spike — the most prominent feature.
    Its timing encodes the "beat timestamp" and therefore the
    inter-beat interval (RR interval) that HRV analysis is built on.

WHAT ARE RR INTERVALS?
    If R-peaks occur at sample indices [i₀, i₁, i₂, ...], then
        RR[k] = (i_{k+1} - i_k) / fs × 1000   (milliseconds)
    A healthy resting heart at 60 BPM has RR ≈ 1000 ms.
    AF is characterised by irregularly irregular RR - wild beat-to-beat
    variation (RMSSD ≫ 50 ms, SD1 ≫ 50 ms in our 10-s windows).

WHAT IS ECTOPIC BEAT REMOVAL?
    Occasionally a beat appears too early (premature contraction) or a
    detection is missed. These produce a very short or very long RR that
    would skew every HRV metric. We flag and remove them using a ±20%
    rule: any RR more than 20% different from the median of its 5
    nearest neighbours is removed. This is conservative enough to keep
    AF beats (which are genuinely irregular) but removes isolated spikes
    caused by noise.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    TARGET_SAMPLING_FREQ,
    SAMPLING_FREQUENCY,
    MIN_RR_MS,
    MAX_RR_MS,
    MIN_RR_BEATS,
    ECTOPIC_FILTER_ENABLED,
    ECTOPIC_THRESHOLD,
    ECTOPIC_WINDOW,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# QUALITY FLAGS  — returned alongside RR arrays so callers know WHY a segment
# was rejected without inspecting the log.
# ---------------------------------------------------------------------------
class SegmentQuality:
    OK                = "ok"           # ≥ MIN_RR_BEATS clean beats, usable
    TOO_FEW_BEATS     = "too_few"      # fewer than MIN_RR_BEATS after cleaning
    ALL_ECTOPIC       = "all_ectopic"  # nothing survived ectopic removal
    DETECTION_FAILED  = "detection_failed"  # NeuroKit2 threw an exception


# ---------------------------------------------------------------------------
# STEP 1: R-PEAK DETECTION
# ---------------------------------------------------------------------------

def detect_rpeaks(ecg_mv: np.ndarray, fs: int = TARGET_SAMPLING_FREQ) -> np.ndarray:
    """
    Return sample indices of R-peaks in `ecg_mv`.

    Uses NeuroKit2's 'pantompkins1985' method — the most widely used
    real-time QRS detection algorithm (Pan & Tompkins 1985, IEEE TBME).
    It was designed for ambulatory ECG, which is exactly CACHET-CADB.

    PARAMETERS
    ----------
    ecg_mv : np.ndarray, shape (n_samples,)
        Preprocessed ECG in millivolts.
    fs : int
        Sample rate of ecg_mv.

    RETURNS
    -------
    np.ndarray of int, shape (n_peaks,)
        Sample indices of detected R-peaks, sorted ascending.
        Empty array if detection fails or no peaks found.
    """
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(
            ecg_mv,
            sampling_rate=fs,
            method="pantompkins1985",
            correct_artifacts=True,   # NeuroKit2's built-in artifact correction
        )
        peaks = info["ECG_R_Peaks"]
        return np.asarray(peaks, dtype=int)

    except Exception as exc:
        logger.debug("R-peak detection error: %s", exc)
        return np.array([], dtype=int)


# ---------------------------------------------------------------------------
# STEP 2: RR INTERVAL EXTRACTION
# ---------------------------------------------------------------------------

def extract_rr_intervals(rpeaks: np.ndarray, fs: int = TARGET_SAMPLING_FREQ) -> np.ndarray:
    """
    Convert R-peak sample indices to RR intervals in milliseconds.

    FORMULA:
        rr_ms[k] = (rpeaks[k+1] - rpeaks[k]) / fs * 1000

    PARAMETERS
    ----------
    rpeaks : np.ndarray of int
        Sorted R-peak sample indices.
    fs : int
        Sample rate used to record those indices.

    RETURNS
    -------
    np.ndarray of float32, shape (len(rpeaks) - 1,)
        RR intervals in milliseconds.
    """
    if len(rpeaks) < 2:
        return np.array([], dtype=np.float32)

    rr_ms = np.diff(rpeaks.astype(np.float64)) / fs * 1000.0
    return rr_ms.astype(np.float32)


# ---------------------------------------------------------------------------
# STEP 3: PHYSIOLOGICAL RANGE FILTER
# ---------------------------------------------------------------------------

def _apply_physiological_limits(rr_ms: np.ndarray) -> np.ndarray:
    """
    Remove RR intervals outside [MIN_RR_MS, MAX_RR_MS].

    Intervals below MIN_RR_MS (300 ms = 200 BPM) are almost certainly
    double-detections (the same beat found twice). Intervals above
    MAX_RR_MS (2000 ms = 30 BPM) are missed beats or genuine pauses.
    Both corrupt HRV statistics.
    """
    return rr_ms[(rr_ms >= MIN_RR_MS) & (rr_ms <= MAX_RR_MS)]


# ---------------------------------------------------------------------------
# STEP 4: ECTOPIC BEAT REMOVAL
# ---------------------------------------------------------------------------

def clean_rr_intervals(
    rr_ms: np.ndarray,
    threshold: float = ECTOPIC_THRESHOLD,
    window: int = ECTOPIC_WINDOW,
    enabled: bool = ECTOPIC_FILTER_ENABLED,
) -> tuple[np.ndarray, float]:
    """
    Optionally remove ectopic RR intervals using a ±threshold local-median rule.

    !! THIS FILTER IS DISABLED BY DEFAULT, AND THAT IS DELIBERATE !!

        An earlier version of this function ran unconditionally at a 20%
        threshold. The reasoning in its docstring was that a LOCAL median
        (5 beats) would track AF variability as a "normal local pattern" and
        so preserve it. We measured whether that actually happened. It did not:

            class   beats removed   segments dropped entirely
            AF          48.5%              25.7%
            NSR         15.4%               0.5%

        The filter was removing half of all AF beats and destroying a quarter
        of AF segments outright, while leaving NSR essentially untouched. The
        local median does not rescue the method, because in AF there is no
        stable local rate for the median to track - successive intervals are
        independent, which is the definition of the arrhythmia.

        The deeper problem is WHICH AF segments get destroyed: the most
        irregular ones, i.e. the most certainly-AF ones. Removing them is
        selection on the outcome, and it inflates every metric downstream.

    WHAT WE DO INSTEAD:
        Physiological range filtering only (MIN_RR_MS to MAX_RR_MS, applied
        before this function). That discards intervals no human heart can
        produce - which are detector errors by definition - without assuming
        the underlying rhythm is regular.

    WHEN WOULD YOU ENABLE IT?
        If you extend this project to autonomic HRV analysis in sinus rhythm
        (stress, sleep staging, recovery), ectopic removal is correct and you
        should switch ECTOPIC_FILTER_ENABLED on. For AF detection, leave it off.

    ALGORITHM (when enabled):
        For each RR[k], compute the median of the nearest `window` RR values
        (window//2 on each side, clipped at boundaries). If
            |RR[k] - median| / median > threshold
        the interval is flagged as ectopic and removed.

    PARAMETERS
    ----------
    rr_ms : np.ndarray
        RR intervals already filtered through physiological limits.
    threshold : float
        Fraction (not percent) — default from config, 0.20 = 20%.
    window : int
        Number of neighbours to include in the local median.
    enabled : bool
        When False (the default), returns the input untouched.

    RETURNS
    -------
    (rr_clean, ectopic_fraction)
        rr_clean         — array of accepted RR intervals
        ectopic_fraction — fraction of beats removed (0.0 when disabled)
    """
    if len(rr_ms) == 0:
        return np.array([], dtype=np.float32), 0.0

    if not enabled:
        # Pass through unchanged. We still return the tuple shape so that
        # callers, the feature table and the CSV schema are unaffected.
        return rr_ms.astype(np.float32), 0.0

    n = len(rr_ms)
    half = window // 2
    keep = np.ones(n, dtype=bool)

    for k in range(n):
        lo = max(0, k - half)
        hi = min(n, k + half + 1)
        neighbours = np.concatenate([rr_ms[lo:k], rr_ms[k + 1:hi]])
        if len(neighbours) == 0:
            continue
        local_median = np.median(neighbours)
        if local_median > 0:
            deviation = abs(rr_ms[k] - local_median) / local_median
            if deviation > threshold:
                keep[k] = False

    ectopic_fraction = float((~keep).sum()) / n
    return rr_ms[keep].astype(np.float32), ectopic_fraction


# ---------------------------------------------------------------------------
# FULL SEGMENT PIPELINE
# ---------------------------------------------------------------------------

def process_segment(
    ecg_mv: np.ndarray,
    fs: int = TARGET_SAMPLING_FREQ,
) -> dict:
    """
    End-to-end: ECG segment → clean RR intervals + quality metadata.

    PARAMETERS
    ----------
    ecg_mv : np.ndarray, shape (n_samples,)
        Preprocessed ECG at `fs` Hz.
    fs : int
        Sample rate.

    RETURNS
    -------
    dict with keys:
        rr_ms            — clean RR intervals (ms), may be empty
        rpeaks           — raw detected peak indices
        n_beats_detected — total beats before cleaning
        n_beats_clean    — beats after ectopic removal
        ectopic_fraction — fraction removed
        quality          — SegmentQuality string constant
    """
    result = {
        "rr_ms": np.array([], dtype=np.float32),
        "rpeaks": np.array([], dtype=int),
        "n_beats_detected": 0,
        "n_beats_clean": 0,
        "ectopic_fraction": 0.0,
        "quality": SegmentQuality.DETECTION_FAILED,
    }

    # 1. Detect R-peaks
    rpeaks = detect_rpeaks(ecg_mv, fs=fs)
    result["rpeaks"] = rpeaks
    result["n_beats_detected"] = len(rpeaks)

    if len(rpeaks) < 2:
        result["quality"] = SegmentQuality.TOO_FEW_BEATS
        return result

    # 2. RR intervals
    rr_ms = extract_rr_intervals(rpeaks, fs=fs)

    # 3. Physiological limits
    rr_ms = _apply_physiological_limits(rr_ms)

    if len(rr_ms) == 0:
        result["quality"] = SegmentQuality.ALL_ECTOPIC
        return result

    # 4. Ectopic removal
    rr_clean, ectopic_frac = clean_rr_intervals(rr_ms)
    result["rr_ms"] = rr_clean
    result["ectopic_fraction"] = ectopic_frac
    result["n_beats_clean"] = len(rr_clean)

    if len(rr_clean) < MIN_RR_BEATS:
        result["quality"] = SegmentQuality.TOO_FEW_BEATS
        return result

    result["quality"] = SegmentQuality.OK
    return result


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config.config import FIGURES_DIR, PROCESSED_DIR
    from src.preprocessing.ecg_filter import preprocess_segment
    from src.preprocessing.unisens_reader import read_unisens_metadata, read_ecg_segment

    os.makedirs(FIGURES_DIR, exist_ok=True)

    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "processed", "segment_index.csv"
    )
    index = pd.read_csv(index_path)

    print("=" * 64)
    print("R-PEAK DETECTOR SELF-TEST")
    print("=" * 64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    labels_to_test = ["AF", "NSR"]

    for i, class_name in enumerate(labels_to_test):
        row = index[index["class_name"] == class_name].iloc[0]
        rec  = read_unisens_metadata(row["signal_dir"])
        raw  = read_ecg_segment(rec, int(row["start_sample"]), int(row["end_sample"]))
        ecg, out_fs = preprocess_segment(raw)
        result = process_segment(ecg, fs=out_fs)

        t = np.arange(len(ecg)) / out_fs
        axes[i].plot(t, ecg, lw=0.6, color="#2980b9", label="ECG")
        if len(result["rpeaks"]):
            peak_t = result["rpeaks"] / out_fs
            axes[i].scatter(peak_t, ecg[result["rpeaks"]],
                            color="#e74c3c", s=25, zorder=5, label="R-peaks")

        axes[i].set_title(
            f"{class_name} | beats={result['n_beats_detected']} → "
            f"clean={result['n_beats_clean']} | quality={result['quality']}"
        )
        axes[i].set_xlabel("Time (s)"); axes[i].set_ylabel("mV")
        axes[i].legend(fontsize=8)

        print(f"\n{class_name}:")
        print(f"  Beats detected   : {result['n_beats_detected']}")
        print(f"  Beats after clean: {result['n_beats_clean']}")
        print(f"  Ectopic fraction : {result['ectopic_fraction']:.2%}")
        print(f"  Quality          : {result['quality']}")
        if len(result["rr_ms"]):
            print(f"  RR mean          : {result['rr_ms'].mean():.1f} ms")
            print(f"  RR RMSSD         : "
                  f"{float(np.sqrt(np.mean(np.diff(result['rr_ms'])**2))):.1f} ms")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "rpeak_detector_demo.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    print("=" * 64)
