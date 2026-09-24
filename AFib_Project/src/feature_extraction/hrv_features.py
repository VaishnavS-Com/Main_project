"""
=============================================================================
FILE: src/feature_extraction/hrv_features.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 8 & 9: HRV FEATURE EXTRACTION
=============================================================================

PURPOSE:
    Given a clean RR-interval series (milliseconds), compute all Heart Rate
    Variability (HRV) features that are valid for 10-second windows.

WHICH FEATURES AND WHY ONLY THESE?
    HRV features fall into three families:

    1. TIME-DOMAIN (all valid for any window length)
       These are simple statistics of the RR series: mean, SD, RMSSD, etc.
       They work with as few as 5-10 beats (our minimum), so they are
       fully compatible with 10-second windows.

    2. NONLINEAR / GEOMETRIC (valid for short windows)
       Poincaré plot descriptors (SD1, SD2), entropy measures (Sample,
       Approximate, Shannon). These are well-established for short-term HRV
       and are the primary discriminators between AF (chaotic) and NSR
       (regular).

    3. FRAGMENTATION INDICES (valid for short windows)
       PAS, PIP, IALS, PSS — measures of how often the RR series changes
       direction. AF is "irregularly irregular", meaning the direction of
       change itself is random. These capture that property directly.

    4. FREQUENCY-DOMAIN — *** EXCLUDED ***
       LF power sits at 0.04–0.15 Hz. To resolve 0.04 Hz you need at
       least 1/0.04 = 25 seconds of signal (by the Rayleigh criterion),
       and ideally 5 minutes. Our windows are 10 seconds. Computing LF/HF
       on 10 s of data produces numerically plausible but mathematically
       meaningless numbers, and reporting them in a thesis would be a
       methodological error. This is documented in config.py and will be
       a stated limitation in the paper.

REFERENCE:
    Shaffer F, Ginsberg JP (2017). An Overview of Heart Rate Variability
    Metrics and Norms. Frontiers in Public Health 5:258.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Dict

# ---------------------------------------------------------------------------
# Guard: call compute_all_features() only through safe wrappers — if RR array
# is empty or has too few beats the individual computations will divide by
# zero or return NaN. The pipeline handles this, but this guard helps during
# development.
# ---------------------------------------------------------------------------
_MINIMUM_BEATS = 3  # Hard minimum for any feature to be non-NaN.


# ===========================================================================
# TIME-DOMAIN FEATURES
# ===========================================================================

def _mean_nn(rr: np.ndarray) -> float:
    """Mean of RR intervals (ms). The inverse is mean heart rate."""
    return float(np.mean(rr))


def _sdnn(rr: np.ndarray) -> float:
    """
    SDNN — Standard Deviation of NN intervals (ms).
    Measures OVERALL HRV. A low SDNN means very regular rhythm.
    In AF, SDNN is dramatically elevated.
    """
    return float(np.std(rr, ddof=1))


def _rmssd(rr: np.ndarray) -> float:
    """
    RMSSD — Root Mean Square of Successive Differences (ms).
    THE gold-standard short-term HRV metric. Reflects rapid beat-to-beat
    changes driven by the parasympathetic nervous system.
    In AF: RMSSD >> 50 ms (often > 200 ms in chaotic AF episodes).
    In NSR: RMSSD typically 20-50 ms at rest.
    """
    if len(rr) < 2:
        return float("nan")
    diffs = np.diff(rr)
    return float(np.sqrt(np.mean(diffs ** 2)))


def _pnn50(rr: np.ndarray) -> float:
    """
    pNN50 — Percentage of successive RR differences > 50 ms.
    Closely correlated with RMSSD. Simple to interpret:
    high pNN50 = many large jumps = irregular rhythm.
    """
    if len(rr) < 2:
        return float("nan")
    diffs = np.abs(np.diff(rr))
    return float(np.sum(diffs > 50) / len(diffs) * 100)


def _pnn20(rr: np.ndarray) -> float:
    """
    pNN20 — Percentage of successive RR differences > 20 ms.
    More sensitive than pNN50 for shorter windows where 50 ms
    jumps may be rare even in mild arrhythmia.
    """
    if len(rr) < 2:
        return float("nan")
    diffs = np.abs(np.diff(rr))
    return float(np.sum(diffs > 20) / len(diffs) * 100)


def _mednn(rr: np.ndarray) -> float:
    """Median of RR intervals (ms). Robust alternative to MeanNN."""
    return float(np.median(rr))


def _madnn(rr: np.ndarray) -> float:
    """
    MadNN — Median Absolute Deviation of NN intervals (ms).
    Robust analog of SDNN. Less sensitive to extreme outliers.
    """
    return float(np.median(np.abs(rr - np.median(rr))))


def _iqrnn(rr: np.ndarray) -> float:
    """
    IQRNN — Interquartile Range of NN intervals (ms).
    Q75 - Q25. Another robust dispersion measure.
    """
    return float(np.percentile(rr, 75) - np.percentile(rr, 25))


def _mcvnn(rr: np.ndarray) -> float:
    """
    MadNN Coefficient of Variation — MadNN / MeanNN.
    Normalises dispersion by the mean rate. Useful when comparing
    subjects with very different heart rates.
    """
    mean = _mean_nn(rr)
    if mean == 0:
        return float("nan")
    return float(_madnn(rr) / mean)


def _cvnn(rr: np.ndarray) -> float:
    """SDNN / MeanNN — coefficient of variation of RR intervals."""
    mean = _mean_nn(rr)
    if mean == 0:
        return float("nan")
    return float(_sdnn(rr) / mean)


# ===========================================================================
# NONLINEAR FEATURES
# ===========================================================================

def _sd1_sd2(rr: np.ndarray) -> tuple[float, float, float]:
    """
    Poincaré Plot descriptors.

    A Poincaré plot is a scatter-plot where:
        x-axis = RR[k]
        y-axis = RR[k+1]
    (each beat plotted against the next beat.)

    In NSR: the cloud is a tight ellipse (beats vary smoothly).
    In AF: the cloud is a wide, scattered blob (beats jump randomly).

    SD1 = width of the ellipse perpendicular to the identity line.
          = std of the first differences / sqrt(2)
          = short-term variability
    SD2 = length of the ellipse along the identity line.
          = long-term variability
    SD1/SD2 ratio: AF produces a high ratio (width ≈ length, no trend).

    FORMULA (Brennan et al. 2001):
        SD1 = RMSSD / sqrt(2)
        SD2 = sqrt(2 * SDNN^2 - SD1^2)
    """
    if len(rr) < 2:
        return float("nan"), float("nan"), float("nan")

    sd1 = float(np.std(np.diff(rr), ddof=1) / np.sqrt(2))
    sd2_sq = 2 * np.std(rr, ddof=1) ** 2 - sd1 ** 2
    sd2 = float(np.sqrt(max(0.0, sd2_sq)))
    ratio = (sd1 / sd2) if sd2 > 0 else float("nan")
    return sd1, sd2, ratio


def _sample_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) — a measure of signal complexity/unpredictability.

    SampEn is LOWER for regular signals (NSR) and HIGHER for irregular ones (AF).
    It is preferred over Approximate Entropy because it is less biased
    for short data series.

    PARAMETERS
    ----------
    m : int
        Template length (typically 2 for HRV).
    r_factor : float
        Tolerance as a fraction of SDNN (typically 0.1-0.25).
        r = r_factor * std(rr)

    ALGORITHM:
        Count pairs of m-length templates within r:    A_m
        Count pairs of (m+1)-length templates within r: B_m
        SampEn = -ln(B_m / A_m)
    """
    n = len(rr)
    if n < m + 2:
        return float("nan")

    r = r_factor * np.std(rr, ddof=1)
    if r <= 0:
        return float("nan")

    def _count_matches(length):
        count = 0
        for i in range(n - length):
            template = rr[i:i + length]
            # Count j != i where max absolute distance < r
            for j in range(i + 1, n - length):
                if np.max(np.abs(rr[j:j + length] - template)) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0 or A == 0:
        return float("nan")

    return float(-np.log(A / B))


def _approximate_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate Entropy (ApEn).

    Similar to SampEn but includes self-matches (i=j allowed). This makes
    ApEn slightly biased for short series, but it was historically popular
    and many papers report it, so we include it for comparability.
    """
    n = len(rr)
    if n < m + 1:
        return float("nan")

    r = r_factor * np.std(rr, ddof=1)
    if r <= 0:
        return float("nan")

    def _phi(length):
        counts = []
        for i in range(n - length + 1):
            template = rr[i:i + length]
            count = sum(
                1 for j in range(n - length + 1)
                if np.max(np.abs(rr[j:j + length] - template)) <= r
            )
            counts.append(np.log(count / (n - length + 1)))
        return np.mean(counts)

    try:
        return float(_phi(m) - _phi(m + 1))
    except Exception:
        return float("nan")


def _shannon_entropy(rr: np.ndarray, n_bins: int = 8) -> float:
    """
    Shannon Entropy of the RR histogram.

    Bins the RR distribution into `n_bins` equal-width bins and computes
    H = -Σ p_i log₂(p_i). A uniform distribution (high entropy = AF)
    vs. a peaked distribution (low entropy = regular NSR).
    """
    if len(rr) < 4:
        return float("nan")
    counts, _ = np.histogram(rr, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ===========================================================================
# FRAGMENTATION FEATURES
# ===========================================================================

def _fragmentation_features(rr: np.ndarray) -> Dict[str, float]:
    """
    Fragmentation indices (Costa et al. 2017).

    Fragmentation measures how often the RR series changes direction
    (beat k is longer than k-1, then shorter again). In NSR, the series
    has smooth trends — low fragmentation. In AF, the direction reverses
    almost every beat — high fragmentation.

    Returns:
        PAS  — Percentage of Alternation of Segments
               Fraction of successive difference sign changes.
        PIP  — Percentage of Inflection Points
               Fraction of beats that are local maxima or minima.
        IALS — Inverse Average of Long Alternating Segments
               Measures the length of sustained monotone runs.
        PSS  — Percentage of Short Segments
               Fraction of alternating runs of length exactly 2.
    """
    result = {"PAS": float("nan"), "PIP": float("nan"),
              "IALS": float("nan"), "PSS": float("nan")}

    if len(rr) < 4:
        return result

    diffs = np.diff(rr)
    signs = np.sign(diffs)

    # PAS — fraction of successive sign changes (alternations)
    sign_changes = np.diff(signs)
    n_alternations = int(np.sum(sign_changes != 0))
    result["PAS"] = float(n_alternations / max(1, len(signs) - 1) * 100)

    # PIP — fraction that are local maxima or minima
    is_inflection = np.zeros(len(rr), dtype=bool)
    for k in range(1, len(rr) - 1):
        if (rr[k] > rr[k - 1] and rr[k] > rr[k + 1]) or \
           (rr[k] < rr[k - 1] and rr[k] < rr[k + 1]):
            is_inflection[k] = True
    result["PIP"] = float(is_inflection.sum() / max(1, len(rr) - 2) * 100)

    # Identify alternating runs (consecutive sign changes)
    alternating = np.concatenate([[0], (sign_changes != 0).astype(int)])
    # Run-length encode to find segments
    runs = []
    current_len = 1
    for k in range(1, len(alternating)):
        if alternating[k] == 1:
            current_len += 1
        else:
            if current_len >= 2:
                runs.append(current_len)
            current_len = 1
    if current_len >= 2:
        runs.append(current_len)

    if runs:
        result["IALS"] = float(1.0 / np.mean(runs))
        result["PSS"] = float(sum(1 for r in runs if r == 2) / len(runs) * 100)
    else:
        result["IALS"] = 0.0
        result["PSS"] = 0.0

    return result


# ===========================================================================
# MASTER FUNCTION — computes all features from one RR series
# ===========================================================================

def _rate_features(rr: np.ndarray) -> Dict[str, float]:
    """
    Instantaneous heart-rate descriptors within the segment.

    WHERE THESE COME FROM:
        Hasan & Motin (SPICSCON 2025) include a "morphology" block containing
        ECG_Rate_Baseline / Max / Min / Mean. We adopt those four because,
        unlike the frequency-domain features in the same paper, they are
        perfectly well defined in a short window - they are simple arithmetic
        on the RR intervals we already have.

    WHAT WE DID *NOT* ADOPT, AND WHY:
        That paper also extracts ULF, VLF, LF, HF, VHF, TP and LF/HF from
        20-second windows, plus SDANN1/2/5 and SDNNI1/2/5 which are *defined*
        over 1, 2 and 5-minute windows. You cannot resolve a frequency whose
        period exceeds your window: ULF sits below 0.003 Hz (one cycle takes
        over five minutes) and LF needs at least ~25 seconds. NeuroKit2 will
        still return a number, but that number describes the algorithm's
        behaviour rather than the patient's physiology. Our windows are 10
        seconds, so those features stay excluded - see USE_FREQUENCY_DOMAIN
        in config.py.

    WHY THESE FOUR MIGHT ADD SOMETHING:
        Every other feature we compute measures the *spread* of RR intervals.
        These measure the *level* and its extremes. AF with rapid ventricular
        response runs fast as well as irregular, so rate carries information
        that dispersion alone does not.

    NOTE ON REDUNDANCY:
        ECG_Rate_Mean is close to a reciprocal of MeanNN, so MRMR may well
        discard it as redundant. That is the correct outcome and is worth
        reporting rather than hiding - it demonstrates the selector working.
    """
    # Instantaneous rate for each beat interval: 60000 ms per minute / RR ms.
    rate = 60000.0 / rr

    # "Baseline" = the typical rate, taken as the median so that one spurious
    # short interval cannot drag it around the way a mean would.
    return {
        "ECG_Rate_Mean":     float(np.mean(rate)),
        "ECG_Rate_Max":      float(np.max(rate)),
        "ECG_Rate_Min":      float(np.min(rate)),
        "ECG_Rate_Baseline": float(np.median(rate)),
    }


def compute_all_features(rr_ms: np.ndarray) -> Dict[str, float]:
    """
    Compute every HRV feature from a clean RR-interval array.

    PARAMETERS
    ----------
    rr_ms : np.ndarray
        Clean RR intervals in milliseconds (after ectopic removal).
        Should contain at least MIN_RR_BEATS values for sensible results.

    RETURNS
    -------
    Dict[str, float]
        Feature name → value. NaN for any feature that cannot be computed.
    """
    features: Dict[str, float] = {}

    if len(rr_ms) < _MINIMUM_BEATS:
        # Return all-NaN dict so downstream code gets consistent columns.
        for name in _FEATURE_NAMES:
            features[name] = float("nan")
        return features

    rr = rr_ms.astype(np.float64)

    # --- Time-domain -------------------------------------------------------
    features["MeanNN"]  = _mean_nn(rr)
    features["SDNN"]    = _sdnn(rr)
    features["RMSSD"]   = _rmssd(rr)
    features["pNN50"]   = _pnn50(rr)
    features["pNN20"]   = _pnn20(rr)
    features["MedNN"]   = _mednn(rr)
    features["MadNN"]   = _madnn(rr)
    features["IQRNN"]   = _iqrnn(rr)
    features["MCVNN"]   = _mcvnn(rr)
    features["CVNN"]    = _cvnn(rr)

    # --- Nonlinear / Poincaré ---------------------------------------------
    sd1, sd2, sd12 = _sd1_sd2(rr)
    features["SD1"]     = sd1
    features["SD2"]     = sd2
    features["SD1_SD2"] = sd12

    # --- Entropy ----------------------------------------------------------
    features["SampEn"]      = _sample_entropy(rr)
    features["ApEn"]        = _approximate_entropy(rr)
    features["ShannonEn"]   = _shannon_entropy(rr)

    # --- Fragmentation ----------------------------------------------------
    frag = _fragmentation_features(rr)
    features["PAS"]  = frag["PAS"]
    features["PIP"]  = frag["PIP"]
    features["IALS"] = frag["IALS"]
    features["PSS"]  = frag["PSS"]

    # --- Rate morphology (adopted from Hasan & Motin 2025) ----------------
    features.update(_rate_features(rr))

    return features


# Canonical list of feature names in a stable order (important for DataFrame columns)
_FEATURE_NAMES = [
    "MeanNN", "SDNN", "RMSSD", "pNN50", "pNN20",
    "MedNN", "MadNN", "IQRNN", "MCVNN", "CVNN",
    "SD1", "SD2", "SD1_SD2",
    "SampEn", "ApEn", "ShannonEn",
    "PAS", "PIP", "IALS", "PSS",
    "ECG_Rate_Mean", "ECG_Rate_Max", "ECG_Rate_Min", "ECG_Rate_Baseline",
]
FEATURE_NAMES = _FEATURE_NAMES  # public alias


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from config.config import PROCESSED_DIR, TARGET_SAMPLING_FREQ
    from src.preprocessing.ecg_filter import preprocess_segment
    from src.peak_detection.rpeak_detector import process_segment as detect
    from src.preprocessing.unisens_reader import read_unisens_metadata, read_ecg_segment

    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "processed", "segment_index.csv"
    )
    index = pd.read_csv(index_path)

    print("=" * 72)
    print("HRV FEATURE EXTRACTION SELF-TEST")
    print("=" * 72)
    print(f"Computing features for 3 AF segments and 3 NSR segments...\n")

    rows_out = []
    for class_name in ["AF", "NSR"]:
        sample_rows = index[index["class_name"] == class_name].head(3)
        for _, row in sample_rows.iterrows():
            rec   = read_unisens_metadata(row["signal_dir"])
            raw   = read_ecg_segment(rec, int(row["start_sample"]), int(row["end_sample"]))
            ecg, fs = preprocess_segment(raw)
            det   = detect(ecg, fs=fs)
            feats = compute_all_features(det["rr_ms"])
            feats["class"] = class_name
            feats["quality"] = det["quality"]
            rows_out.append(feats)

    df = pd.DataFrame(rows_out)
    print("Per-class mean feature values:")
    print("-" * 72)
    numeric_cols = [c for c in df.columns if c not in ["class", "quality"]]
    summary = df.groupby("class")[numeric_cols].mean().T
    print(summary.to_string())
    print("\nExpected: AF RMSSD >> NSR RMSSD, AF SD1 >> NSR SD1")
    print("=" * 72)
