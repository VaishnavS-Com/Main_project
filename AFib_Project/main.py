"""
=============================================================================
FILE: main.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis
         for Atrial Fibrillation Detection using CACHET-CADB

PYTHON VERSION: 3.13.7
=============================================================================

USAGE:
    # Run the complete pipeline end-to-end:
    python main.py

    # Run only specific phases:
    python main.py --phases 1 2 3        # Index + ECG filter test + Features
    python main.py --phases 4 5 6        # Feature selection + SVM + XGBoost
    python main.py --phases 7 8          # Evaluation + FP analysis

PHASES:
    1 — Validate dataset index (segment_index.csv)
    2 — ECG preprocessing self-test (saves demo figure)
    3 — HRV feature extraction (builds hrv_features.csv — SLOW, ~20 min)
    4 — MRMR feature selection  (builds selected_features.json)
    5 — SVM training + GroupKFold CV
    6 — XGBoost training + GroupKFold CV
    7 — Evaluation: metrics, ROC, confusion matrix, feature importance
    8 — Context-aware false positive analysis

DESIGN:
    Each phase is a self-contained function that:
    - Checks its required inputs exist (fails with a clear message if not)
    - Logs start/end times
    - Saves its outputs before returning
    This means a failed phase can be re-run without repeating earlier work.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of where this is run from
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR, LOGS_DIR,
    CADB_ROOT, SAMPLING_FREQUENCY, TOTAL_SUBJECTS,
    N_FOLDS, RANDOM_STATE,
    CLASS_AFIB, CLASS_NORMAL,
)
from src.utils.logger import get_logger

logger = get_logger("main")

# Create all output directories up front
for _d in [PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR, LOGS_DIR,
           os.path.join(PROJECT_ROOT, "models")]:
    os.makedirs(_d, exist_ok=True)


# ===========================================================================
# PHASE 1 — Dataset index validation
# ===========================================================================

def phase1_validate_index() -> pd.DataFrame:
    """
    Build (or reload) the segment index and print a validation report.

    Output: data/processed/segment_index.csv
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 1: Dataset Index Validation")
    logger.info("=" * 64)

    index_path = os.path.join(PROCESSED_DIR, "segment_index.csv")

    if os.path.isfile(index_path):
        logger.info("Found existing index — loading from disk")
        index = pd.read_csv(index_path)
    else:
        logger.info("Building index from %s …", CADB_ROOT)
        from src.preprocessing.dataset_index import build_segment_index
        index = build_segment_index()
        if index.empty:
            raise SystemExit("No segments found. Check CADB_ROOT in config.py.")
        index.to_csv(index_path, index=False)
        logger.info("Index saved → %s", index_path)

    from src.preprocessing.dataset_index import validate_index, summarise_index, get_modelling_subset

    print("\n--- Segment Index Summary ---")
    print(f"  Total segments      : {len(index)}")
    print(f"  Subjects            : {index['subject'].nunique()}")

    counts = index["class_name"].value_counts()
    for name in ["AF", "NSR", "Noise", "Other"]:
        n = int(counts.get(name, 0))
        print(f"  {name:<8}           : {n}  ({n/len(index)*100:.1f}%)")

    print("\n--- Validation ---")
    findings = validate_index(index)
    for key, val in findings.items():
        status = "OK" if val == 0 or val == [] else "!!"
        print(f"  {status} {key:<28}: {val}")

    print("\n--- Per-subject breakdown ---")
    print(summarise_index(index).to_string())

    subset = get_modelling_subset(index)
    print(f"\n--- Modelling subset (noise dropped) ---")
    print(f"  Segments : {len(subset)}")
    print(f"  AF       : {int((subset['label'] == 1).sum())}")
    print(f"  non-AF   : {int((subset['label'] == 0).sum())}")

    logger.info("Phase 1 done in %.1f s", time.time() - t0)
    return index


# ===========================================================================
# PHASE 2 — ECG preprocessing demo
# ===========================================================================

def phase2_preprocessing_demo(index: pd.DataFrame) -> None:
    """
    Run the ECG filter on one AF and one NSR segment, save demo figure.
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 2: ECG Preprocessing Demo")
    logger.info("=" * 64)

    from src.preprocessing.unisens_reader import read_unisens_metadata, read_ecg_segment
    from src.preprocessing.ecg_filter import preprocess_segment
    from src.peak_detection.rpeak_detector import process_segment as detect_peaks

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("ECG Preprocessing + R-Peak Detection", fontsize=13, fontweight="bold")

    for col_idx, class_name in enumerate(["AF", "NSR"]):
        row = index[index["class_name"] == class_name].dropna(subset=["signal_dir"]).iloc[0]
        rec  = read_unisens_metadata(row["signal_dir"])
        raw  = read_ecg_segment(rec, int(row["start_sample"]), int(row["end_sample"]))
        ecg_clean, out_fs = preprocess_segment(raw)
        det = detect_peaks(ecg_clean, fs=out_fs)

        t_raw   = np.arange(len(raw))       / SAMPLING_FREQUENCY
        t_clean = np.arange(len(ecg_clean)) / out_fs

        axes[0, col_idx].plot(t_raw, raw, lw=0.6, color="#aaaaaa")
        axes[0, col_idx].set_title(f"{class_name} — Raw ECG (1024 Hz)")
        axes[0, col_idx].set_ylabel("mV"); axes[0, col_idx].set_xlabel("s")

        axes[1, col_idx].plot(t_clean, ecg_clean, lw=0.6, color="#2980b9")
        if len(det["rpeaks"]):
            axes[1, col_idx].scatter(
                det["rpeaks"] / out_fs, ecg_clean[det["rpeaks"]],
                color="#e74c3c", s=20, zorder=5
            )
        axes[1, col_idx].set_title(
            f"{class_name} — Filtered 256 Hz | {det['n_beats_clean']} clean beats"
        )
        axes[1, col_idx].set_ylabel("mV"); axes[1, col_idx].set_xlabel("s")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "phase2_preprocessing_demo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPreprocessing demo figure → {out}")
    logger.info("Phase 2 done in %.1f s", time.time() - t0)


# ===========================================================================
# PHASE 3 — Feature extraction
# ===========================================================================

def phase3_feature_extraction() -> pd.DataFrame:
    """
    Run the full feature pipeline over all labelled segments.

    Output: data/features/hrv_features.csv
    Slow (~20 min for 1602 segments). Supports resume.
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 3: HRV Feature Extraction")
    logger.info("=" * 64)

    from src.feature_extraction.feature_pipeline import run_feature_pipeline, summarise_features

    feature_df = run_feature_pipeline(resume=True)
    summarise_features(feature_df)

    logger.info("Phase 3 done in %.1f min", (time.time() - t0) / 60)
    return feature_df


# ===========================================================================
# PHASE 4 — Feature selection
# ===========================================================================

def phase4_feature_selection(feature_df: pd.DataFrame) -> list:
    """
    Run MRMR feature selection.

    Output: data/features/selected_features.json
            data/features/mrmr_report.csv
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 4: MRMR Feature Selection")
    logger.info("=" * 64)

    # Try to reload if already done
    from src.feature_selection.mrmr_selector import load_selection
    json_path = os.path.join(FEATURES_DIR, "selected_features.json")
    if os.path.isfile(json_path):
        logger.info("Found existing selection — loading from disk")
        selected = load_selection()
        print(f"Selected features: {selected}")
        return selected

    from src.feature_selection.mrmr_selector import select_features, save_selection
    selected, report = select_features(feature_df)
    save_selection(selected, report)

    print("\nMRMR selected features (in order of selection):")
    print(report.to_string(index=False))

    logger.info("Phase 4 done in %.1f s", time.time() - t0)
    return selected


# ===========================================================================
# PHASE 5 — SVM training
# ===========================================================================

def phase5_svm(feature_cols: list) -> tuple:
    """
    Train SVM with GroupKFold CV.

    Outputs: data/features/svm_cv_results.csv
             models/svm_final.joblib
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 5: SVM Training + Cross-Validation")
    logger.info("=" * 64)

    from src.models.train_svm import load_training_data, cross_validate_svm, train_final_svm

    X, y, groups, df = load_training_data(feature_cols)
    print(f"Training SVM on {len(y)} segments, {len(np.unique(groups))} subjects, "
          f"{X.shape[1]} features …")

    cv_results, oof_preds, oof_probs = cross_validate_svm(X, y, groups)
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(FEATURES_DIR, "svm_cv_results.csv"), index=False)

    # Save OOF predictions for FP analysis
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_indices = []
    for _, test_idx in gkf.split(X, y, groups):
        all_indices.extend(test_idx.tolist())

    clean = df.iloc[all_indices]
    oof_df = pd.DataFrame({
        "segment_id": clean["segment_id"].values,
        "y_true": y[all_indices],
        "y_pred": np.concatenate(oof_preds),
        "y_prob": np.concatenate(oof_probs),
    })
    oof_df.to_csv(os.path.join(FEATURES_DIR, "oof_predictions_svm.csv"), index=False)

    train_final_svm(X, y, feature_cols)

    print("\n--- SVM CV Results ---")
    print(cv_df.to_string(index=False))

    logger.info("Phase 5 done in %.1f min", (time.time() - t0) / 60)
    return cv_df, oof_df


# ===========================================================================
# PHASE 6 — XGBoost training
# ===========================================================================

def phase6_xgboost(feature_cols: list) -> tuple:
    """
    Train XGBoost with GroupKFold CV.

    Outputs: data/features/xgb_cv_results.csv
             models/xgb_final.joblib
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 6: XGBoost Training + Cross-Validation")
    logger.info("=" * 64)

    from src.models.train_xgboost import load_training_data, cross_validate_xgb, train_final_xgb

    X, y, groups, df = load_training_data(feature_cols)
    print(f"Training XGBoost on {len(y)} segments, {len(np.unique(groups))} subjects, "
          f"{X.shape[1]} features …")

    cv_results, oof_preds, oof_probs = cross_validate_xgb(X, y, groups)
    cv_df = pd.DataFrame({k: v for k, v in cv_results.items() if k != "mean_importance"})
    cv_df.to_csv(os.path.join(FEATURES_DIR, "xgb_cv_results.csv"), index=False)

    # Save OOF predictions
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_indices = []
    for _, test_idx in gkf.split(X, y, groups):
        all_indices.extend(test_idx.tolist())

    clean = df.iloc[all_indices]
    oof_df = pd.DataFrame({
        "segment_id": clean["segment_id"].values,
        "y_true": y[all_indices],
        "y_pred": np.concatenate(oof_preds),
        "y_prob": np.concatenate(oof_probs),
    })
    oof_df.to_csv(os.path.join(FEATURES_DIR, "oof_predictions_xgb.csv"), index=False)

    xgb = train_final_xgb(X, y, feature_cols)

    print("\n--- XGBoost CV Results ---")
    print(cv_df.to_string(index=False))

    logger.info("Phase 6 done in %.1f min", (time.time() - t0) / 60)
    return cv_df, oof_df


# ===========================================================================
# PHASE 7 — Evaluation
# ===========================================================================

def phase7_evaluation(
    cv_svm: pd.DataFrame,
    cv_xgb: pd.DataFrame,
    oof_svm: pd.DataFrame,
    oof_xgb: pd.DataFrame,
) -> None:
    """
    Generate all evaluation figures and print the results table.
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 7: Evaluation")
    logger.info("=" * 64)

    from src.evaluation.metrics import (
        print_results_table, plot_roc_curves,
        plot_confusion_matrix, plot_cv_metrics,
        plot_feature_importance
    )

    print_results_table(cv_svm, cv_xgb)

    # ROC curves (overlay)
    plot_roc_curves({
        "SVM":     (oof_svm["y_true"].values, oof_svm["y_prob"].values),
        "XGBoost": (oof_xgb["y_true"].values, oof_xgb["y_prob"].values),
    })

    # Confusion matrices
    plot_confusion_matrix(oof_svm["y_true"].values, oof_svm["y_pred"].values,
                          model_name="SVM", save_name="confusion_matrix_svm.png")
    plot_confusion_matrix(oof_xgb["y_true"].values, oof_xgb["y_pred"].values,
                          model_name="XGBoost", save_name="confusion_matrix_xgb.png")

    # Feature importance
    imp_path = os.path.join(FEATURES_DIR, "xgb_feature_importance.csv")
    if os.path.isfile(imp_path):
        imp_df = pd.read_csv(imp_path)
        plot_feature_importance(imp_df["feature"].tolist(), imp_df["importance"].values)

    # CV comparison box plots
    plot_cv_metrics(cv_svm, cv_xgb)

    print(f"\nAll evaluation figures → {FIGURES_DIR}")
    logger.info("Phase 7 done in %.1f s", time.time() - t0)


# ===========================================================================
# PHASE 8 — Context-aware FP analysis
# ===========================================================================

def phase8_fp_analysis(
    oof_xgb: pd.DataFrame,
    index: pd.DataFrame,
) -> None:
    """
    Run the context-aware false positive analysis on XGBoost OOF predictions.
    """
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("PHASE 8: Context-Aware False Positive Analysis")
    logger.info("=" * 64)

    from src.context_analysis.fp_analysis import run_fp_analysis

    results = run_fp_analysis(
        y_true=oof_xgb["y_true"].values,
        y_pred=oof_xgb["y_pred"].values,
        segment_ids=oof_xgb["segment_id"].values,
        segment_index=index,
        model_name="XGBoost",
    )

    print(f"\nContext analysis figures → {FIGURES_DIR}")
    logger.info("Phase 8 done in %.1f s", time.time() - t0)


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AFib HRV Pipeline — run all phases or specific ones"
    )
    parser.add_argument(
        "--phases", "-p",
        nargs="*",
        type=int,
        default=None,
        help="Phases to run (1-8). Default: all. Example: --phases 1 3 5"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phases_to_run = set(args.phases) if args.phases else set(range(1, 9))

    print("\n" + "=" * 72)
    print("  AFib HRV Pipeline — Context-Aware False Positive Analysis")
    print("  CACHET-CADB Dataset | Python 3.13.7 | B.Tech Final Year Project")
    print("=" * 72)
    print(f"  CADB root       : {CADB_ROOT}")
    print(f"  Exists          : {os.path.isdir(CADB_ROOT)}")
    print(f"  Native ECG rate : {SAMPLING_FREQUENCY} Hz")
    print(f"  Subjects        : {TOTAL_SUBJECTS}")
    print(f"  Phases to run   : {sorted(phases_to_run)}")
    print("=" * 72 + "\n")

    # --- Phase 1 ---
    index = None
    if 1 in phases_to_run or any(p in phases_to_run for p in [2, 7, 8]):
        index = phase1_validate_index()

    # --- Phase 2 ---
    if 2 in phases_to_run and index is not None:
        phase2_preprocessing_demo(index)

    # --- Phase 3 ---
    feature_df = None
    if 3 in phases_to_run:
        feature_df = phase3_feature_extraction()
    elif any(p in phases_to_run for p in [4, 5, 6]):
        feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
        if os.path.isfile(feat_path):
            feature_df = pd.read_csv(feat_path)
            logger.info("Feature matrix loaded from disk (%d rows)", len(feature_df))
        else:
            logger.error("hrv_features.csv not found. Run Phase 3 first (--phases 3).")
            return

    # --- Phase 4 ---
    feature_cols = None
    if 4 in phases_to_run and feature_df is not None:
        feature_cols = phase4_feature_selection(feature_df)
    elif any(p in phases_to_run for p in [5, 6]):
        from src.feature_selection.mrmr_selector import load_selection
        try:
            feature_cols = load_selection()
        except FileNotFoundError:
            from src.feature_extraction.hrv_features import FEATURE_NAMES
            if feature_df is not None:
                feature_cols = [f for f in FEATURE_NAMES if f in feature_df.columns]

    # --- Phase 5 (SVM) ---
    cv_svm = None
    oof_svm = None
    if 5 in phases_to_run and feature_cols:
        cv_svm, oof_svm = phase5_svm(feature_cols)
    elif any(p in phases_to_run for p in [7]):
        svm_cv_path = os.path.join(FEATURES_DIR, "svm_cv_results.csv")
        svm_oof_path = os.path.join(FEATURES_DIR, "oof_predictions_svm.csv")
        if os.path.isfile(svm_cv_path):
            cv_svm = pd.read_csv(svm_cv_path)
            oof_svm = pd.read_csv(svm_oof_path) if os.path.isfile(svm_oof_path) else None

    # --- Phase 6 (XGBoost) ---
    cv_xgb = None
    oof_xgb = None
    if 6 in phases_to_run and feature_cols:
        cv_xgb, oof_xgb = phase6_xgboost(feature_cols)
    elif any(p in phases_to_run for p in [7, 8]):
        xgb_cv_path = os.path.join(FEATURES_DIR, "xgb_cv_results.csv")
        xgb_oof_path = os.path.join(FEATURES_DIR, "oof_predictions_xgb.csv")
        if os.path.isfile(xgb_cv_path):
            cv_xgb = pd.read_csv(xgb_cv_path)
            oof_xgb = pd.read_csv(xgb_oof_path) if os.path.isfile(xgb_oof_path) else None

    # --- Phase 7 (Evaluation) ---
    if 7 in phases_to_run:
        if cv_svm is not None and cv_xgb is not None:
            phase7_evaluation(cv_svm, cv_xgb, oof_svm, oof_xgb)
        else:
            logger.warning("Skipping Phase 7: run Phases 5 and 6 first.")

    # --- Phase 8 (FP Analysis) ---
    if 8 in phases_to_run:
        if oof_xgb is not None and index is not None:
            phase8_fp_analysis(oof_xgb, index)
        else:
            logger.warning("Skipping Phase 8: run Phase 6 and Phase 1 first.")

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print(f"  Figures  → {FIGURES_DIR}")
    print(f"  Features → {FEATURES_DIR}")
    print(f"  Models   → {os.path.join(PROJECT_ROOT, 'models')}")
    print(f"  Logs     → {LOGS_DIR}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
