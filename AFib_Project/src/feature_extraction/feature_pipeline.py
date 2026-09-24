"""
=============================================================================
FILE: src/feature_extraction/feature_pipeline.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 10: FEATURE PIPELINE — BUILD THE FULL FEATURE MATRIX
=============================================================================

PURPOSE:
    Iterate every row of segment_index.csv, apply the full preprocessing +
    peak detection + HRV extraction chain, and write the result to
    data/features/hrv_features.csv.

    hrv_features.csv has one row per segment and columns:
        segment_id  — foreign key back to segment_index.csv
        subject     — for GroupKFold
        label       — binary target (0=non-AF, 1=AF)
        is_noise    — True for noise segments (excluded from training)
        class_name  — human-readable class
        quality     — SegmentQuality string from the peak detector
        ectopic_fraction
        MeanNN, SDNN, RMSSD, ... (20 HRV features)

DESIGN DECISIONS:
    - Failures are logged and stored as NaN rows, NOT as exceptions that
      crash the whole run. A 10 GB dataset with a handful of corrupted
      recordings is normal; losing 1% of data is fine, losing the run is not.
    - Progress is reported every 50 segments so you can monitor a long run.
    - The output file is written atomically (to a temp file first, then
      renamed) so a crash mid-run does not leave a half-written CSV.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    PROCESSED_DIR,
    FEATURES_DIR,
    SAMPLING_FREQUENCY,
    TARGET_SAMPLING_FREQ,
)
from src.preprocessing.ecg_filter import preprocess_segment
from src.preprocessing.unisens_reader import read_unisens_metadata, read_ecg_segment
from src.peak_detection.rpeak_detector import process_segment as detect_peaks, SegmentQuality
from src.feature_extraction.hrv_features import compute_all_features, FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PER-SEGMENT WORKER
# ---------------------------------------------------------------------------

def _process_one_segment(row: pd.Series) -> Dict:
    """
    Run the full pipeline for one row of the segment index.

    Returns a dict with all feature columns (NaN on failure).
    Never raises — failures are caught and returned as partial dicts.
    """
    out: Dict = {
        "segment_id":        row["segment_id"],
        "subject":           row["subject"],
        "label":             row.get("label", float("nan")),
        "raw_class":         row.get("raw_class", -1),
        "class_name":        row.get("class_name", "Unknown"),
        "is_noise":          row.get("is_noise", False),
        "is_control":        row.get("is_control", False),
        "quality":           SegmentQuality.DETECTION_FAILED,
        "n_beats_detected":  0,
        "n_beats_clean":     0,
        "ectopic_fraction":  float("nan"),
    }
    # Initialise all features to NaN (so every row has the same columns)
    for feat in FEATURE_NAMES:
        out[feat] = float("nan")

    try:
        # 1. Read ECG segment from binary file
        rec = read_unisens_metadata(row["signal_dir"])
        raw_ecg = read_ecg_segment(rec, int(row["start_sample"]), int(row["end_sample"]))

        # Determine native sample rate from unisens metadata (trust the file)
        ecg_entry = rec.ecg
        native_fs = int(ecg_entry.sample_rate) if ecg_entry else SAMPLING_FREQUENCY

        # 2. Preprocess (bandpass + notch + downsample)
        ecg_clean, out_fs = preprocess_segment(raw_ecg, native_fs=native_fs)

        # 3. R-peak detection + RR extraction + ectopic removal
        det = detect_peaks(ecg_clean, fs=out_fs)
        out["quality"]           = det["quality"]
        out["n_beats_detected"]  = det["n_beats_detected"]
        out["n_beats_clean"]     = det["n_beats_clean"]
        out["ectopic_fraction"]  = det["ectopic_fraction"]

        # 4. HRV feature extraction (only if quality is OK)
        if det["quality"] == SegmentQuality.OK and len(det["rr_ms"]) > 0:
            feats = compute_all_features(det["rr_ms"])
            for feat, val in feats.items():
                out[feat] = val

    except FileNotFoundError as exc:
        logger.debug("Missing file for segment %d: %s", row["segment_id"], exc)
        out["quality"] = "missing_file"
    except Exception as exc:
        logger.debug("Unexpected error for segment %d: %s", row["segment_id"], exc)
        out["quality"] = f"error:{type(exc).__name__}"

    return out


# ---------------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    index: Optional[pd.DataFrame] = None,
    save: bool = True,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Build hrv_features.csv from the segment index.

    PARAMETERS
    ----------
    index : pd.DataFrame, optional
        The segment index. If None, loads from data/processed/segment_index.csv.
    save : bool
        Write result to data/features/hrv_features.csv.
    resume : bool
        If hrv_features.csv already exists, skip already-processed segments.
        Allows recovering from an interrupted run without reprocessing.

    RETURNS
    -------
    pd.DataFrame
        Full feature matrix with one row per segment.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
    tmp_path = out_path + ".tmp"

    # --- Load segment index ------------------------------------------------
    if index is None:
        index_path = os.path.join(PROCESSED_DIR, "segment_index.csv")
        logger.info("Loading segment index from %s", index_path)
        index = pd.read_csv(index_path)

    logger.info("Segment index loaded: %d rows", len(index))

    # --- Resume logic -------------------------------------------------------
    done_ids: set = set()
    existing_rows: List[Dict] = []

    if resume and os.path.isfile(out_path):
        existing = pd.read_csv(out_path)
        done_ids = set(existing["segment_id"].tolist())
        existing_rows = existing.to_dict("records")
        logger.info("Resuming: %d segments already processed, %d remaining",
                    len(done_ids), len(index) - len(done_ids))

    pending = index[~index["segment_id"].isin(done_ids)]
    logger.info("Processing %d segments…", len(pending))

    # --- Process ------------------------------------------------------------
    t0 = time.time()
    new_rows: List[Dict] = []
    ok_count = 0
    fail_count = 0

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="HRV features",
                       unit="seg", dynamic_ncols=True):
        result = _process_one_segment(row)
        new_rows.append(result)

        if result["quality"] == SegmentQuality.OK:
            ok_count += 1
        else:
            fail_count += 1

        # Log progress every 100 segments
        if (ok_count + fail_count) % 100 == 0:
            elapsed = time.time() - t0
            per_seg = elapsed / (ok_count + fail_count)
            remaining = per_seg * (len(pending) - ok_count - fail_count)
            logger.info(
                "Progress: %d/%d | OK=%d fail=%d | ~%.0f min remaining",
                ok_count + fail_count, len(pending),
                ok_count, fail_count, remaining / 60,
            )

    # --- Combine + save -----------------------------------------------------
    all_rows = existing_rows + new_rows
    feature_df = pd.DataFrame(all_rows)

    # Reorder so metadata columns come first, features after
    meta_cols = [
        "segment_id", "subject", "label", "raw_class", "class_name",
        "is_noise", "is_control", "quality",
        "n_beats_detected", "n_beats_clean", "ectopic_fraction",
    ]
    feat_cols = [f for f in FEATURE_NAMES if f in feature_df.columns]
    ordered_cols = meta_cols + feat_cols
    remaining_cols = [c for c in feature_df.columns if c not in ordered_cols]
    feature_df = feature_df[ordered_cols + remaining_cols]

    if save:
        feature_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)   # atomic rename — no half-written file
        logger.info("Feature matrix saved → %s  (%d rows, %d features)",
                    out_path, len(feature_df), len(feat_cols))

    elapsed = time.time() - t0
    logger.info(
        "Feature pipeline complete in %.1f min | OK=%d fail=%d (%.1f%%)",
        elapsed / 60, ok_count, fail_count,
        fail_count / max(1, ok_count + fail_count) * 100,
    )

    return feature_df


# ---------------------------------------------------------------------------
# SUMMARY HELPER
# ---------------------------------------------------------------------------

def summarise_features(feature_df: pd.DataFrame) -> None:
    """Print a human-readable summary after the pipeline run."""
    ok = feature_df[feature_df["quality"] == "ok"]
    print("\n" + "=" * 72)
    print("FEATURE PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Total segments          : {len(feature_df)}")
    print(f"Quality OK              : {len(ok)}  "
          f"({len(ok) / max(1, len(feature_df)) * 100:.1f}%)")
    print(f"Failed / too few beats  : {len(feature_df) - len(ok)}")

    if len(ok):
        print("\nClass breakdown (OK segments only):")
        for name, grp in ok.groupby("class_name"):
            print(f"  {name:<8} n={len(grp)}")

        print("\nMean feature values by class (key features):")
        key = ["RMSSD", "SDNN", "SD1", "pNN50", "SampEn"]
        available = [f for f in key if f in ok.columns]
        if available:
            summary = ok.groupby("class_name")[available].mean()
            print(summary.round(2).to_string())

    print("\nExpected sanity check: AF RMSSD >> NSR RMSSD")
    print("=" * 72)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 10: BUILDING HRV FEATURE MATRIX")
    print("=" * 72)

    feature_df = run_feature_pipeline(resume=True)
    summarise_features(feature_df)
