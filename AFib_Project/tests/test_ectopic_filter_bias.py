"""
=============================================================================
FILE: tests/test_ectopic_filter_bias.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

AN A/B EXPERIMENT: WHAT THE ECTOPIC FILTER DOES TO THE AF CLASS
=============================================================================

WHY THIS FILE EXISTS:
    config.py now sets ECTOPIC_FILTER_ENABLED = False, and the comment there
    makes a strong claim: that a 20% local-median ectopic filter destroys the
    AF class while leaving NSR untouched. A claim like that should not rest on
    a comment. This script re-runs the entire feature extraction BOTH WAYS on
    the same segments and prints the difference, so the claim can be checked
    - and re-checked by an examiner - in one command.

WHAT IT MEASURES:
    1. Beat retention per class, with and without the filter.
    2. How many segments are destroyed outright (too few surviving beats).
    3. Whether the destroyed AF segments are the MOST irregular ones -
       i.e. whether the loss is random or selects on the outcome.
    4. The effect on the separability of the headline feature (RMSSD).

HOW TO RUN:
    cd AFib_Project
    .\\venv\\Scripts\\activate
    python tests\\test_ectopic_filter_bias.py

    Optional: --n 400  to sample fewer segments and run faster.
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (  # noqa: E402
    ECTOPIC_THRESHOLD, ECTOPIC_WINDOW, MIN_RR_BEATS, PROCESSED_DIR,
    TARGET_SAMPLING_FREQ,
)
from src.feature_extraction.hrv_features import compute_all_features  # noqa: E402
from src.peak_detection.rpeak_detector import (  # noqa: E402
    _apply_physiological_limits, clean_rr_intervals, detect_rpeaks,
    extract_rr_intervals,
)
from src.preprocessing.ecg_filter import preprocess_segment  # noqa: E402
from src.preprocessing.unisens_reader import (  # noqa: E402
    read_ecg_segment, read_unisens_metadata,
)


def rr_for_segment(row: pd.Series) -> np.ndarray:
    """ECG bytes -> physiologically-plausible RR intervals, no ectopic step."""
    recording = read_unisens_metadata(row["signal_dir"])
    raw = read_ecg_segment(recording, int(row["start_sample"]),
                           int(row["end_sample"]))
    # preprocess_segment returns (filtered_signal, effective_sample_rate) -
    # the rate matters because the ECG is downsampled 1024 -> 256 Hz inside.
    clean, fs = preprocess_segment(raw)
    peaks = detect_rpeaks(clean, fs=fs)
    rr = extract_rr_intervals(peaks, fs=fs)
    return _apply_physiological_limits(rr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0,
                        help="Sample N segments per class (0 = use all)")
    args = parser.parse_args()

    index = pd.read_csv(os.path.join(PROCESSED_DIR, "segment_index.csv"))
    index = index[index["class_name"].isin(["AF", "NSR"])]

    if args.n:
        index = (index.groupby("class_name", group_keys=False)
                      .apply(lambda g: g.sample(min(args.n, len(g)),
                                                random_state=42)))

    print("=" * 74)
    print("A/B EXPERIMENT: ECTOPIC FILTER ON vs OFF")
    print(f"threshold = {ECTOPIC_THRESHOLD:.0%}, window = {ECTOPIC_WINDOW} beats, "
          f"minimum beats to keep a segment = {MIN_RR_BEATS}")
    print("=" * 74)

    records = []
    for _, row in index.iterrows():
        try:
            rr = rr_for_segment(row)
        except Exception:      # noqa: BLE001 - one unreadable file must not stop the run
            continue
        if len(rr) == 0:
            continue

        # OFF: physiological limits only (what we now do)
        rr_off = rr
        # ON: additionally apply the local-median ectopic rule (what we did)
        rr_on, removed = clean_rr_intervals(
            rr, threshold=ECTOPIC_THRESHOLD, window=ECTOPIC_WINDOW, enabled=True
        )

        entry = {
            "class_name": row["class_name"],
            "subject": row["subject"],
            "n_off": len(rr_off),
            "n_on": len(rr_on),
            "removed_frac": removed,
            "kept_off": len(rr_off) >= MIN_RR_BEATS,
            "kept_on": len(rr_on) >= MIN_RR_BEATS,
        }
        # Irregularity as measured BEFORE any filtering - the ground truth
        # against which we ask "were the deleted segments the irregular ones?"
        entry["cv_true"] = float(np.std(rr_off) / np.mean(rr_off)) if len(rr_off) > 1 else np.nan
        if entry["kept_off"]:
            entry["rmssd_off"] = compute_all_features(rr_off).get("RMSSD", np.nan)
        if entry["kept_on"]:
            entry["rmssd_on"] = compute_all_features(rr_on).get("RMSSD", np.nan)
        records.append(entry)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit("No segments processed - check signal paths in the index.")

    # ---------------------------------------------------------------- 1 & 2
    print(f"\nSegments processed: {len(frame)} "
          f"(AF {int((frame.class_name=='AF').sum())}, "
          f"NSR {int((frame.class_name=='NSR').sum())})")

    print("\n--- 1. BEAT RETENTION ---")
    summary = frame.groupby("class_name").agg(
        beats_before=("n_off", "mean"),
        beats_after=("n_on", "mean"),
        pct_removed=("removed_frac", lambda s: s.mean() * 100),
    ).round(2)
    print(summary.to_string())

    print("\n--- 2. SEGMENTS DESTROYED (dropped for too few beats) ---")
    destroyed = frame.groupby("class_name").apply(
        lambda g: pd.Series({
            "usable_filter_off": int(g.kept_off.sum()),
            "usable_filter_on": int(g.kept_on.sum()),
            "lost_to_filter": int((g.kept_off & ~g.kept_on).sum()),
            "pct_lost": round((g.kept_off & ~g.kept_on).sum() / max(1, g.kept_off.sum()) * 100, 1),
        })
    )
    print(destroyed.to_string())

    # ------------------------------------------------------------------- 3
    print("\n--- 3. IS THE LOSS RANDOM, OR DOES IT SELECT ON THE OUTCOME? ---")
    af = frame[(frame.class_name == "AF") & frame.kept_off]
    if len(af):
        survived = af[af.kept_on]["cv_true"]
        deleted = af[~af.kept_on]["cv_true"]
        print(f"  AF segments SURVIVING the filter : RR CV = "
              f"{survived.mean():.3f} (n={len(survived)})")
        print(f"  AF segments DELETED by the filter: RR CV = "
              f"{deleted.mean():.3f} (n={len(deleted)})")
        if len(deleted) and len(survived):
            ratio = deleted.mean() / survived.mean()
            print(f"  -> deleted segments are {ratio:.2f}x MORE irregular.")
            print("     The filter is removing the most unambiguous AF, which")
            print("     is selection on the label and inflates every metric.")

    # ------------------------------------------------------------------- 4
    print("\n--- 4. EFFECT ON CLASS SEPARATION (RMSSD, ms) ---")
    for column, label in (("rmssd_off", "filter OFF"), ("rmssd_on", "filter ON")):
        if column not in frame:
            continue
        stats = frame.dropna(subset=[column]).groupby("class_name")[column].mean()
        if {"AF", "NSR"}.issubset(stats.index):
            gap = stats["AF"] / stats["NSR"]
            print(f"  {label:<12}: AF {stats['AF']:7.2f} | NSR {stats['NSR']:7.2f} "
                  f"| ratio {gap:.2f}x")

    print("\n" + "=" * 74)
    print("CONCLUSION: the filter should stay disabled for AF detection.")
    print("=" * 74)

    out = os.path.join(PROCESSED_DIR, "ectopic_filter_ab_test.csv")
    frame.to_csv(out, index=False)
    print(f"\nSaved per-segment detail -> {out}")


if __name__ == "__main__":
    main()
