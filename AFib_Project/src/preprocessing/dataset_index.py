"""
=============================================================================
FILE: src/preprocessing/dataset_index.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 4 (part 2 of 2): BUILDING THE SEGMENT INDEX
=============================================================================

PURPOSE:
    Turn a 10 GB tree of binary files into ONE table where each row is one
    labelled 10-second ECG segment, and every column needed to fetch that
    segment later is present.

    Everything downstream - preprocessing, features, training, the false
    positive analysis - reads this one table. Nothing else has to understand
    the folder layout.

THE FOLDER LAYOUT WE ARE FLATTENING:

    CACHET-CADB/
    ├── signal/
    │   └── P1/                        <- SUBJECT   (the grouping unit for CV)
    │       └── a2b3c4@cachet.dk1/     <- SESSION   (one wearing of the sensor)
    │           ├── 0/                 <- PART      (file rotation)
    │           │   ├── unisens.xml
    │           │   ├── ecg.bin
    │           │   └── acc.bin
    │           ├── 1/  2/  3/
    │           └── 4-last/
    └── annotations/
        └── P1/
            ├── a1b2c4@cachet.dk.json  <- patient-reported events (not used yet)
            └── a2b3c4@cachet.dk1/
                ├── 0/
                │   ├── annotation.csv <- Start,End,Class  (often EMPTY)
                │   └── context.xlsx   <- 32 context columns on a 10 s grid
                └── 4-last/ ...

TWO FACTS ABOUT THIS DATASET THAT ARE EASY TO GET WRONG
-------------------------------------------------------
1. SAMPLE INDICES ARE PART-RELATIVE, NOT SESSION-RELATIVE.
   In P1/a2b3c4@cachet.dk1 the annotated part is "4-last", holding 39,512,832
   samples, and its annotations run from 2,056,849 to 39,449,053 - they fit
   inside THAT part. If the indices were cumulative across the session's five
   parts they would have started above 295 million. So a segment is located
   by (subject, session, part, start_sample) and never by the session alone.
   `validate_index` re-checks this for every row and will tell you loudly if
   any recording breaks the pattern.

2. MOST annotation.csv FILES ARE EMPTY (0 bytes).
   The cardiologists labelled a sample of the recordings, not all 122 hours.
   An empty file is normal and is skipped silently; only a MALFORMED file is
   worth a warning.

OUTPUT COLUMNS
--------------
    subject, session, part      identity of the recording
    start_sample, end_sample    where in ecg.bin (native 1024 Hz)
    n_samples, duration_sec     derived, used as a sanity check
    raw_class, class_name       as written by the annotator (1..4)
    label                       binary model target: 0 = non-AF, 1 = AF
    is_noise                    True for raw_class 3 (excluded from training)
    is_control                  True for the PNSR-* healthy cohort
    t_start_sec                 offset into the part, for context lookup
    context_row                 matching row index in context.xlsx
    age, gender                 subject demographics from unisens.xml
    signal_dir, context_path    absolute paths for the loaders

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (  # noqa: E402
    CADB_SIGNAL_DIR,
    CADB_ANNOTATION_DIR,
    CONTEXT_GRID_SEC,
    CONTROL_SUBJECT_IDS,
    DROP_NOISE_FROM_TRAINING,
    LABEL_MAP,
    PROCESSED_DIR,
    RAW_CLASS_NAMES,
    RAW_CLASS_NOISE,
    SAMPLING_FREQUENCY,
    SEGMENT_SAMPLES_RAW,
)
from src.preprocessing.unisens_reader import read_unisens_metadata  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _natural_subject_key(subject: str):
    """
    Sort P1, P2, ... P10 in human order rather than P1, P10, P11, P2.

    Controls (PNSR-*) are pushed to the end so patient and control cohorts
    stay visually separated in every table and plot.
    """
    if subject.startswith("PNSR"):
        return (1, int(subject.split("-")[-1]))
    return (0, int(subject.lstrip("P")))


def list_subjects(signal_dir: str = CADB_SIGNAL_DIR) -> List[str]:
    """
    Return the subject folders that ACTUALLY EXIST, in natural order.

    We list the directory rather than generating IDs in a loop, because the
    numbering has holes: there is no P20 and no P22.
    """
    subjects = [
        name for name in os.listdir(signal_dir)
        if os.path.isdir(os.path.join(signal_dir, name))
    ]
    return sorted(subjects, key=_natural_subject_key)


def _read_annotation_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Read one annotation.csv, returning None if it holds no usable rows.

    Empty files are the common case and are not an error - see the module
    docstring. Anything else that fails to parse gets a warning, because a
    malformed file means we are silently losing labels.
    """
    if os.path.getsize(path) == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - we want to survive one bad file
        logger.warning("Could not parse %s: %s", path, exc)
        return None

    if not {"Start", "End", "Class"}.issubset(frame.columns):
        logger.warning("Unexpected columns in %s: %s", path, list(frame.columns))
        return None

    frame = frame.dropna(subset=["Start", "End", "Class"])
    return frame if len(frame) else None


def build_segment_index(
    signal_dir: str = CADB_SIGNAL_DIR,
    annotation_dir: str = CADB_ANNOTATION_DIR,
) -> pd.DataFrame:
    """
    Walk the whole dataset and return one row per labelled segment.

    The walk is driven by the ANNOTATIONS tree, not the signal tree: a
    recording without labels is of no use to a supervised model, so there is
    no reason to visit it. For each annotated part we then look up the
    matching signal folder and read its unisens.xml for metadata.
    """
    rows: List[Dict] = []
    metadata_cache: Dict[str, dict] = {}   # avoid re-parsing the same XML

    subjects = list_subjects(annotation_dir)
    logger.info("Scanning %d subjects for annotations", len(subjects))

    for subject in subjects:
        subject_ann_dir = os.path.join(annotation_dir, subject)
        is_control = subject in CONTROL_SUBJECT_IDS

        for session in sorted(os.listdir(subject_ann_dir)):
            session_ann_dir = os.path.join(subject_ann_dir, session)
            # Skip the patient-event JSON files that sit at subject level.
            if not os.path.isdir(session_ann_dir):
                continue

            for part in sorted(os.listdir(session_ann_dir)):
                part_ann_dir = os.path.join(session_ann_dir, part)
                if not os.path.isdir(part_ann_dir):
                    continue

                csv_path = os.path.join(part_ann_dir, "annotation.csv")
                if not os.path.isfile(csv_path):
                    continue

                annotations = _read_annotation_csv(csv_path)
                if annotations is None:
                    continue          # unlabelled part - expected, not an error

                # ---- locate the matching signal folder ---------------------
                part_signal_dir = os.path.join(signal_dir, subject, session, part)
                if not os.path.isdir(part_signal_dir):
                    logger.warning(
                        "Labels exist but no signal folder: %s/%s/%s",
                        subject, session, part,
                    )
                    continue

                # ---- read (and cache) the recording metadata ---------------
                if part_signal_dir not in metadata_cache:
                    try:
                        recording = read_unisens_metadata(part_signal_dir)
                    except FileNotFoundError as exc:
                        logger.warning("%s", exc)
                        continue

                    ecg_entry = recording.ecg
                    if ecg_entry is None:
                        logger.warning("No ECG channel in %s", part_signal_dir)
                        continue

                    # Trust the file, verify against config. A recording at a
                    # different rate would corrupt every timing calculation.
                    if int(ecg_entry.sample_rate) != SAMPLING_FREQUENCY:
                        logger.warning(
                            "%s records ECG at %s Hz, config says %s Hz",
                            part_signal_dir, ecg_entry.sample_rate,
                            SAMPLING_FREQUENCY,
                        )

                    ecg_path = os.path.join(part_signal_dir, ecg_entry.entry_id)
                    n_samples_in_part = (
                        os.path.getsize(ecg_path) // np.dtype(ecg_entry.dtype).itemsize
                        if os.path.isfile(ecg_path) else 0
                    )

                    metadata_cache[part_signal_dir] = {
                        "age": recording.subject_age,
                        "gender": recording.subject_gender,
                        "fs": ecg_entry.sample_rate,
                        "n_samples_in_part": n_samples_in_part,
                        "duration_sec": recording.duration_sec,
                    }

                meta = metadata_cache[part_signal_dir]
                context_path = os.path.join(part_ann_dir, "context.xlsx")
                has_context = os.path.isfile(context_path)

                # ---- one output row per annotated segment ------------------
                for _, annotation in annotations.iterrows():
                    start = int(annotation["Start"])
                    end = int(annotation["End"])
                    raw_class = int(annotation["Class"])
                    t_start_sec = start / SAMPLING_FREQUENCY

                    rows.append({
                        "subject": subject,
                        "session": session,
                        "part": part,
                        "start_sample": start,
                        "end_sample": end,
                        "n_samples": end - start,
                        "duration_sec": (end - start) / SAMPLING_FREQUENCY,
                        "raw_class": raw_class,
                        "class_name": RAW_CLASS_NAMES.get(raw_class, "Unknown"),
                        "label": LABEL_MAP.get(raw_class, np.nan),
                        "is_noise": raw_class == RAW_CLASS_NOISE,
                        "is_control": is_control,
                        "t_start_sec": t_start_sec,
                        "context_row": int(t_start_sec // CONTEXT_GRID_SEC),
                        "age": meta["age"],
                        "gender": meta["gender"],
                        "part_n_samples": meta["n_samples_in_part"],
                        "signal_dir": part_signal_dir,
                        "context_path": context_path if has_context else "",
                    })

    index = pd.DataFrame(rows)
    if index.empty:
        logger.error("No labelled segments found - check CADB_ROOT in config")
        return index

    index = index.sort_values(
        ["subject", "session", "part", "start_sample"],
        key=lambda col: col.map(_natural_subject_key) if col.name == "subject" else col,
    ).reset_index(drop=True)

    index.insert(0, "segment_id", range(len(index)))
    logger.info("Built index: %d segments from %d subjects",
                len(index), index["subject"].nunique())
    return index


def validate_index(index: pd.DataFrame) -> Dict[str, object]:
    """
    Check the index against everything we believe about the dataset.

    This is deliberately paranoid. A silent indexing bug here would not crash
    anything - it would just feed the model the wrong 10 seconds of ECG, and
    you would spend weeks blaming the classifier.

    Returns a dict of findings; also logs a readable report.
    """
    findings: Dict[str, object] = {}

    # 1. Every segment should be exactly SEGMENT_SAMPLES_RAW long.
    wrong_length = index[index["n_samples"] != SEGMENT_SAMPLES_RAW]
    findings["segments_wrong_length"] = len(wrong_length)
    if len(wrong_length):
        logger.warning(
            "%d segments are not %d samples long (values: %s)",
            len(wrong_length), SEGMENT_SAMPLES_RAW,
            sorted(wrong_length["n_samples"].unique())[:10],
        )

    # 2. Segments must fit inside their part's ecg.bin. This is the check that
    #    proves indices are part-relative rather than session-relative.
    overruns = index[index["end_sample"] > index["part_n_samples"]]
    findings["segments_out_of_bounds"] = len(overruns)
    if len(overruns):
        logger.error(
            "%d segments run past the end of their ecg.bin - the sample "
            "indices may not be part-relative after all. Affected: %s",
            len(overruns),
            overruns[["subject", "session", "part"]].drop_duplicates()
                    .to_dict("records")[:5],
        )

    # 3. Negative or reversed ranges.
    reversed_ranges = index[index["end_sample"] <= index["start_sample"]]
    findings["segments_reversed"] = len(reversed_ranges)

    # 4. Unknown class codes.
    unknown = index[~index["raw_class"].isin(RAW_CLASS_NAMES)]
    findings["unknown_class_codes"] = sorted(unknown["raw_class"].unique().tolist())

    # 5. Missing context files - these would silently drop out of Phase 14.
    findings["segments_without_context"] = int((index["context_path"] == "").sum())

    # 6. Overlapping segments within a part would leak identical beats across
    #    a train/test boundary.
    overlaps = 0
    for _, group in index.groupby(["subject", "session", "part"], sort=False):
        ordered = group.sort_values("start_sample")
        overlaps += int(
            (ordered["start_sample"].values[1:] <
             ordered["end_sample"].values[:-1]).sum()
        )
    findings["overlapping_segments"] = overlaps

    logger.info("Validation findings: %s", findings)
    return findings


def summarise_index(index: pd.DataFrame) -> pd.DataFrame:
    """Per-subject segment counts by class - the table for the report."""
    table = (
        index.pivot_table(
            index="subject", columns="class_name", values="segment_id",
            aggfunc="count", fill_value=0,
        )
        .reindex(columns=["NSR", "AF", "Noise", "Other"], fill_value=0)
    )
    table["Total"] = table.sum(axis=1)
    table["is_control"] = [
        subject in CONTROL_SUBJECT_IDS for subject in table.index
    ]
    return table.sort_index(key=lambda idx: idx.map(_natural_subject_key))


def get_modelling_subset(index: pd.DataFrame) -> pd.DataFrame:
    """
    The rows actually used for supervised training.

    Drops Noise segments (no rhythm to learn) when DROP_NOISE_FROM_TRAINING is
    set. The full index is kept on disk regardless, because the noise segments
    are evidence in the false-positive analysis.
    """
    subset = index.dropna(subset=["label"])
    if DROP_NOISE_FROM_TRAINING:
        subset = subset[~subset["is_noise"]]
    return subset.reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 72)
    print("PHASE 4: BUILDING THE CACHET-CADB SEGMENT INDEX")
    print("=" * 72)

    segment_index = build_segment_index()
    if segment_index.empty:
        raise SystemExit("No segments found - check CADB_ROOT.")

    print(f"\nTotal labelled segments : {len(segment_index)}")
    print(f"Subjects with labels    : {segment_index['subject'].nunique()}")
    print(f"Recording parts         : "
          f"{segment_index.groupby(['subject','session','part']).ngroups}")

    print("\n--- Class distribution (raw) ---")
    counts = segment_index["class_name"].value_counts()
    for name in ["NSR", "AF", "Noise", "Other"]:
        n = int(counts.get(name, 0))
        print(f"  {name:<6} {n:>5}  ({n / len(segment_index) * 100:5.1f}%)")

    print("\n--- Validation ---")
    for key, value in validate_index(segment_index).items():
        print(f"  {key:<28}: {value}")

    print("\n--- Per-subject breakdown ---")
    print(summarise_index(segment_index).to_string())

    model_subset = get_modelling_subset(segment_index)
    print(f"\n--- Modelling subset (noise dropped) ---")
    print(f"  Segments : {len(model_subset)}")
    print(f"  non-AF   : {int((model_subset['label'] == 0).sum())}")
    print(f"  AF       : {int((model_subset['label'] == 1).sum())}")
    print(f"  Subjects : {model_subset['subject'].nunique()}")
    af_subjects = model_subset[model_subset['label'] == 1]['subject'].nunique()
    print(f"  Subjects containing any AF : {af_subjects}")

    out_path = os.path.join(PROCESSED_DIR, "segment_index.csv")
    segment_index.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("=" * 72)
