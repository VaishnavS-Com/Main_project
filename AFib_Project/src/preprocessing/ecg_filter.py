"""
=============================================================================
FILE: src/preprocessing/ecg_filter.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 5: ECG SIGNAL PREPROCESSING
=============================================================================

PURPOSE:
    Turn a raw 10-second ECG segment (1024 Hz, mV, straight from the sensor)
    into a clean, downsampled signal (256 Hz) ready for R-peak detection.

    The pipeline has three stages, applied in this exact order:

    1. BANDPASS FILTER  (0.5 – 40 Hz)
       Removes two classes of noise simultaneously:
         - Baseline wander: slow drift caused by breathing and electrode
           movement. Sits below 0.5 Hz. The filter kills it.
         - High-frequency noise: muscle noise (EMG), thermal noise, radio
           interference. Sits above ~40 Hz for ECG-relevant content.
           The filter kills it.

    2. NOTCH FILTER  (50 Hz)
       India's power grid runs at 50 Hz. Any electrode lead is an antenna for
       it. The notch filter is a narrow stop-band at exactly 50 Hz, leaving
       everything else untouched. (USA labs use 60 Hz — change POWERLINE_FREQ_HZ
       in config.py, nothing else needs changing.)

    3. DOWNSAMPLE  (1024 Hz → 256 Hz)
       CACHET-CADB records at 1024 Hz. The highest frequency an ECG carries
       is ~40 Hz; 256 Hz is 6× the Nyquist limit for that, so no information
       is lost by downsampling. The payoff: every downstream operation
       (peak detection, feature extraction) is 4× faster and uses 4× less
       memory. scipy.signal.resample_poly does this anti-alias-safe.

WHY THIS ORDER?
    Always filter BEFORE downsampling. If we downsampled first, the 50 Hz
    powerline spike (which becomes a 50/1024 ≈ 0.049 normalised frequency)
    would alias into the lower frequency range and become unremovable. Filter
    first, then discard the samples you don't need.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt, resample_poly
from math import gcd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    SAMPLING_FREQUENCY,
    TARGET_SAMPLING_FREQ,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    POWERLINE_FREQ_HZ,
    FILTER_ORDER,
    DOWNSAMPLE_ECG,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# STEP 1: BANDPASS FILTER
# ---------------------------------------------------------------------------

def _bandpass_sos(fs: float, low: float, high: float, order: int) -> np.ndarray:
    """
    Build a Butterworth bandpass filter in second-order-sections (SOS) form.

    WHY SOS?
        The naive way to build a filter is as a single polynomial fraction
        (numerator/denominator, called BA form). For high-order filters
        this polynomial becomes numerically unstable - tiny floating-point
        rounding errors in the coefficients flip the filter's behaviour
        completely. SOS form breaks the same filter into a chain of small
        second-order stages and avoids this entirely. Always use SOS for
        biomedical signals.

    WHY BUTTERWORTH?
        Among IIR filter families (Butterworth, Chebyshev, Elliptic) the
        Butterworth is "maximally flat in the passband". That means it does
        not distort the ECG morphology inside 0.5-40 Hz - the P wave, QRS
        complex and T wave all come through with the same relative amplitudes,
        which is important if we ever want to look at morphology.
    """
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    return butter(order, [low_n, high_n], btype="band", output="sos")


def bandpass_filter(
    signal: np.ndarray,
    fs: float = SAMPLING_FREQUENCY,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to an ECG signal.

    ZERO-PHASE (sosfiltfilt):
        A regular filter (sosfilt) is causal — it processes samples left to
        right, introducing a time delay (phase shift). That delay is constant
        for all frequencies in an ideal filter, but in practice it is not
        perfectly constant, so the QRS spike can shift relative to the P and
        T waves. Zero-phase filtering runs the filter forwards AND backwards,
        cancelling the delay exactly. The cost is that the signal must be
        available in full (not real-time), which is fine for offline analysis.

    PARAMETERS
    ----------
    signal : np.ndarray, shape (n_samples,)
        Raw or partially-cleaned ECG in mV at native sample rate.
    fs : float
        Sample rate of `signal`.
    low, high : float
        Passband edges in Hz.
    order : int
        Filter order (higher = sharper roll-off, more computation).

    RETURNS
    -------
    np.ndarray
        Filtered ECG, same shape and dtype as input.
    """
    if len(signal) < 3 * order:
        # Too short for the filter to work without edge artefacts.
        logger.warning("Signal too short for bandpass (len=%d, order=%d). Returning as-is.",
                       len(signal), order)
        return signal.copy()

    sos = _bandpass_sos(fs, low, high, order)
    return sosfiltfilt(sos, signal).astype(np.float32)


# ---------------------------------------------------------------------------
# STEP 2: NOTCH FILTER (powerline interference)
# ---------------------------------------------------------------------------

def notch_filter(
    signal: np.ndarray,
    fs: float = SAMPLING_FREQUENCY,
    freq: float = POWERLINE_FREQ_HZ,
    quality: float = 35.0,
) -> np.ndarray:
    """
    Apply a notch (band-stop) filter to remove powerline noise.

    THE QUALITY FACTOR:
        Q = freq / bandwidth. Q=35 means the stop-band is 50/35 ≈ 1.4 Hz
        wide centred on 50 Hz. Wide enough to catch slight grid frequency
        drift (49-51 Hz is common); narrow enough to leave the ECG signal
        on either side intact.

    PARAMETERS
    ----------
    signal : np.ndarray, shape (n_samples,)
        ECG signal (already bandpass-filtered is fine).
    fs : float
        Sample rate.
    freq : float
        Target frequency to notch out (Hz). 50 for India, 60 for USA.
    quality : float
        Q factor (higher = narrower notch).

    RETURNS
    -------
    np.ndarray
        Signal with powerline interference suppressed.
    """
    b, a = iirnotch(freq, quality, fs)
    # Convert BA → SOS for numerical stability before filtering.
    # For a 2nd-order notch the BA→SOS conversion is exact.
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal).astype(np.float32)


# ---------------------------------------------------------------------------
# STEP 3: DOWNSAMPLE  1024 Hz → 256 Hz
# ---------------------------------------------------------------------------

def downsample_ecg(
    signal: np.ndarray,
    from_fs: int = SAMPLING_FREQUENCY,
    to_fs: int = TARGET_SAMPLING_FREQ,
) -> np.ndarray:
    """
    Downsample the ECG from `from_fs` to `to_fs` Hz using polyphase resampling.

    WHY POLYPHASE (resample_poly) AND NOT JUST SLICE EVERY 4th SAMPLE?
        Slicing every 4th sample is "naive downsampling" and it ALIASES.
        Any frequency above to_fs/2 = 128 Hz folds back into the spectrum and
        corrupts the signal. resample_poly applies an anti-alias low-pass
        filter automatically before decimating, so aliasing cannot happen.
        We have already bandpass-filtered to 40 Hz, so this is a belt-and-
        suspenders measure — but the cost is negligible.

    WHY POLYPHASE SPECIFICALLY (vs. scipy.signal.resample)?
        scipy.signal.resample works in the frequency domain (FFT). It handles
        non-integer ratios but assumes the signal is periodic, which introduces
        ringing at the edges. resample_poly works in the time domain and is
        correct for finite signals.

    PARAMETERS
    ----------
    signal : np.ndarray, shape (n_samples,)
        ECG signal at `from_fs` Hz.
    from_fs, to_fs : int
        Source and target sample rates.

    RETURNS
    -------
    np.ndarray, shape (n_samples * to_fs // from_fs,)
        Downsampled ECG.
    """
    if from_fs == to_fs:
        return signal.copy()

    # Simplify the up/down ratio to its lowest terms.
    # resample_poly(signal, up, down): up=1, down=4 for 1024→256.
    common = gcd(int(from_fs), int(to_fs))
    up = int(to_fs) // common
    down = int(from_fs) // common
    return resample_poly(signal, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# FULL SEGMENT PIPELINE
# ---------------------------------------------------------------------------

def preprocess_segment(
    raw_ecg: np.ndarray,
    native_fs: int = SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, int]:
    """
    Run the complete preprocessing pipeline on one 10-second ECG segment.

    Applies, in order:
        1. Bandpass filter   (0.5 – 40 Hz, Butterworth order-4, zero-phase)
        2. Notch filter      (50 Hz powerline, Q=35, zero-phase)
        3. Downsample        (native_fs → TARGET_SAMPLING_FREQ if DOWNSAMPLE_ECG)

    PARAMETERS
    ----------
    raw_ecg : np.ndarray, shape (n_raw_samples,)
        ECG in millivolts at `native_fs` Hz, as returned by read_ecg_segment().
    native_fs : int
        The recording's native sample rate (from unisens.xml; default 1024 Hz).

    RETURNS
    -------
    (ecg_clean, output_fs) : tuple
        ecg_clean  — preprocessed ECG, shape (n_output_samples,), float32
        output_fs  — sample rate of ecg_clean (TARGET_SAMPLING_FREQ or native_fs)
    """
    ecg = raw_ecg.astype(np.float32)

    # --- clip gross ADC saturation artefacts before filtering ---------------
    # Hard clips larger than ±5 mV are sensor saturation, not signal.
    # Clipping before filtering prevents the filter ringing on them.
    ecg = np.clip(ecg, -5.0, 5.0)

    # --- 1. bandpass --------------------------------------------------------
    ecg = bandpass_filter(ecg, fs=native_fs)

    # --- 2. notch -----------------------------------------------------------
    ecg = notch_filter(ecg, fs=native_fs)

    # --- 3. downsample ------------------------------------------------------
    if DOWNSAMPLE_ECG and native_fs != TARGET_SAMPLING_FREQ:
        ecg = downsample_ecg(ecg, from_fs=native_fs, to_fs=TARGET_SAMPLING_FREQ)
        output_fs = TARGET_SAMPLING_FREQ
    else:
        output_fs = native_fs

    return ecg, output_fs


# ---------------------------------------------------------------------------
# SELF-TEST
# Run: python src/preprocessing/ecg_filter.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config.config import FIGURES_DIR, CADB_SIGNAL_DIR, CADB_ANNOTATION_DIR
    from src.preprocessing.dataset_index import build_segment_index
    from src.preprocessing.unisens_reader import read_unisens_metadata, read_ecg_segment

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 64)
    print("ECG FILTER SELF-TEST")
    print("=" * 64)

    # Load the segment index (already built).
    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "processed", "segment_index.csv"
    )
    if os.path.isfile(index_path):
        index = __import__("pandas").read_csv(index_path)
        # Grab one AF and one NSR segment for visual comparison.
        af_row  = index[index["class_name"] == "AF"].iloc[0]
        nsr_row = index[index["class_name"] == "NSR"].iloc[0]
        test_rows = [("AF", af_row), ("NSR", nsr_row)]
    else:
        print("segment_index.csv not found — building now (slow first run)…")
        index = build_segment_index()
        af_row  = index[index["class_name"] == "AF"].iloc[0]
        nsr_row = index[index["class_name"] == "NSR"].iloc[0]
        test_rows = [("AF", af_row), ("NSR", nsr_row)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    fig.suptitle("ECG Preprocessing — Raw vs Filtered", fontsize=12, fontweight="bold")

    for col, (label, row) in enumerate(test_rows):
        rec  = read_unisens_metadata(row["signal_dir"])
        raw  = read_ecg_segment(rec, int(row["start_sample"]), int(row["end_sample"]))
        clean, out_fs = preprocess_segment(raw)

        t_raw   = np.arange(len(raw))   / SAMPLING_FREQUENCY
        t_clean = np.arange(len(clean)) / out_fs

        axes[0, col].plot(t_raw,   raw,   lw=0.6, color="#999999", label="Raw 1024 Hz")
        axes[0, col].set_title(f"{label} — Raw"); axes[0, col].set_ylabel("mV")
        axes[0, col].set_xlabel("Time (s)")

        axes[1, col].plot(t_clean, clean, lw=0.6, color="#2980b9", label="Filtered 256 Hz")
        axes[1, col].set_title(f"{label} — Filtered & Downsampled")
        axes[1, col].set_ylabel("mV"); axes[1, col].set_xlabel("Time (s)")

        print(f"\n{label}:")
        print(f"  Raw   : {len(raw)} samples @ {SAMPLING_FREQUENCY} Hz, "
              f"range [{raw.min():.3f}, {raw.max():.3f}] mV")
        print(f"  Clean : {len(clean)} samples @ {out_fs} Hz, "
              f"range [{clean.min():.3f}, {clean.max():.3f}] mV")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ecg_filter_demo.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    print("=" * 64)
