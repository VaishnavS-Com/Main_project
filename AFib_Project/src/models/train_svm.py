"""
=============================================================================
FILE: src/models/train_svm.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 12 (Model 1): SVM CLASSIFIER WITH GroupKFold CROSS-VALIDATION
=============================================================================

PURPOSE:
    Train and evaluate a Support Vector Machine (SVM) for binary AF detection
    using subject-wise cross-validation to prevent data leakage.

WHY SVM FOR ECG/HRV?
    SVM with an RBF kernel is a strong baseline for small biomedical datasets.
    HRV feature spaces are typically low-dimensional (<30 features) and SVM
    handles the non-linear boundaries between AF and NSR well. Unlike neural
    networks it does not need thousands of samples to generalise — it relies
    on the support vectors (the critical boundary points), which can be a
    small subset of the data.

WHY GroupKFold, NOT KFold?
    Standard k-fold randomly shuffles samples into folds. If patient P3 has
    100 segments and they end up split across train and test, the model sees
    the same patient's ECG morphology in both — and learns patient identity,
    not rhythm type. This inflates accuracy by 10-20% and makes results
    unreproducible on new patients (the real use case).

    GroupKFold guarantees that ALL segments from the same patient are in the
    same fold. So in each iteration the model is tested on patients it has
    never seen. This is the only honest evaluation strategy for this dataset.

WHY SMOTE INSIDE EACH FOLD?
    The dataset has 747 AF vs 615 NSR segments — a mild imbalance. SMOTE
    (Synthetic Minority Oversampling TEchnique) generates synthetic minority-
    class samples by interpolating between real ones. IMPORTANT: SMOTE must
    be applied ONLY to the training fold, AFTER the train/test split. Applying
    it before splitting would place synthetic samples in the test set, which
    contaminates the evaluation.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    FEATURES_DIR,
    N_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    CLASS_AFIB,
    CLASS_NORMAL,
)
from src.feature_selection.mrmr_selector import load_selection
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def load_training_data(
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load hrv_features.csv and return X, y, groups, full_df.

    Only uses clean segments (quality == 'ok', label not NaN, not noise).

    Returns
    -------
    X      : (n, n_features) float array
    y      : (n,) int array — 0=NSR, 1=AF
    groups : (n,) array of subject ID strings — for GroupKFold
    df     : full DataFrame (for saving predictions later)
    """
    feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(
            f"hrv_features.csv not found at {feat_path}. "
            "Run feature_pipeline.py first."
        )

    df = pd.read_csv(feat_path)

    # Use MRMR-selected features if available, else all FEATURE_NAMES
    if feature_cols is None:
        try:
            feature_cols = load_selection()
            logger.info("Using MRMR-selected features: %s", feature_cols)
        except FileNotFoundError:
            from src.feature_extraction.hrv_features import FEATURE_NAMES
            feature_cols = [f for f in FEATURE_NAMES if f in df.columns]
            logger.info("MRMR selection not found; using all %d features", len(feature_cols))

    # Filter to clean modelling subset
    clean = df[
        (df["quality"] == "ok") &
        df["label"].notna() &
        (~df["is_noise"])
    ].copy()

    logger.info("Training data: %d clean segments", len(clean))

    X = clean[feature_cols].values.astype(np.float64)
    y = clean["label"].values.astype(int)
    groups = clean["subject"].values

    # Impute residual NaNs with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, col_idx] = float(np.nanmedian(col))

    return X, y, groups, clean


# ---------------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------------

def cross_validate_svm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[Dict, List[np.ndarray], List[np.ndarray]]:
    """
    Run GroupKFold cross-validation with SVM + SMOTE.

    Returns
    -------
    cv_results : dict of metric lists (one value per fold)
    oof_preds  : out-of-fold predictions (list of arrays, one per fold)
    oof_probs  : out-of-fold probabilities
    """
    gkf = GroupKFold(n_splits=n_folds)
    smote = SMOTE(random_state=RANDOM_STATE)

    # SVM with RBF kernel. probability=True enables predict_proba() for ROC.
    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              probability=True, random_state=RANDOM_STATE, class_weight="balanced")

    scaler = StandardScaler()

    cv_results: Dict[str, List] = {
        "fold": [], "accuracy": [], "f1": [],
        "auc": [], "sensitivity": [], "specificity": [],
        "n_train": [], "n_test": [],
        "n_af_train": [], "n_nsr_train": [],
    }
    oof_preds = []
    oof_probs = []

    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix
    )

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # --- Scale (fit on train only!) ---
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # --- SMOTE on training fold only ---
        n_minority_before = int((y_tr == CLASS_AFIB).sum())
        X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)
        n_minority_after = int((y_tr == CLASS_AFIB).sum())
        logger.info("Fold %d: SMOTE %d → %d AF samples",
                    fold + 1, n_minority_before, n_minority_after)

        # --- Train ---
        svm.fit(X_tr_s, y_tr)

        # --- Predict ---
        y_pred = svm.predict(X_te_s)
        y_prob = svm.predict_proba(X_te_s)[:, 1]  # probability of AF

        oof_preds.append(y_pred)
        oof_probs.append(y_prob)

        # --- Metrics ---
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / max(1, tp + fn)   # = Recall for AF class
        specificity = tn / max(1, tn + fp)

        cv_results["fold"].append(fold + 1)
        cv_results["accuracy"].append(accuracy_score(y_te, y_pred))
        cv_results["f1"].append(f1_score(y_te, y_pred, pos_label=CLASS_AFIB,
                                         zero_division=0))
        cv_results["auc"].append(roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1
                                 else float("nan"))
        cv_results["sensitivity"].append(sensitivity)
        cv_results["specificity"].append(specificity)
        cv_results["n_train"].append(len(y_tr))
        cv_results["n_test"].append(len(y_te))
        cv_results["n_af_train"].append(n_minority_after)
        cv_results["n_nsr_train"].append(int((y_tr == CLASS_NORMAL).sum()))

        logger.info(
            "Fold %d | Acc=%.3f F1=%.3f AUC=%.3f Sen=%.3f Spe=%.3f",
            fold + 1,
            cv_results["accuracy"][-1], cv_results["f1"][-1],
            cv_results["auc"][-1], cv_results["sensitivity"][-1],
            cv_results["specificity"][-1],
        )

    return cv_results, oof_preds, oof_probs


# ---------------------------------------------------------------------------
# FINAL MODEL (trained on all data)
# ---------------------------------------------------------------------------

def train_final_svm(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
) -> Pipeline:
    """
    Train the final SVM on the complete dataset.

    This model is saved and used for inference and false-positive analysis.
    It is NOT used for computing metrics (that is done via CV above).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    smote = SMOTE(random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_res, y_res = smote.fit_resample(X_s, y)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    svm.fit(X_res, y_res)

    # Wrap scaler + model together so inference only needs one call
    pipeline = Pipeline([("scaler", scaler), ("svm", svm)])
    pipeline.feature_names = feature_cols   # attach for documentation

    model_path = os.path.join(MODELS_DIR, "svm_final.joblib")
    joblib.dump(pipeline, model_path)
    logger.info("Final SVM saved → %s", model_path)

    return pipeline


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 12: SVM TRAINING + GroupKFold CROSS-VALIDATION")
    print("=" * 72)

    X, y, groups, df = load_training_data()
    print(f"Training data : {len(y)} segments | AF={int((y==1).sum())} NSR={int((y==0).sum())}")
    print(f"Subjects      : {len(np.unique(groups))}")
    print(f"Features      : {X.shape[1]}")

    try:
        feature_cols = load_selection()
    except FileNotFoundError:
        from src.feature_extraction.hrv_features import FEATURE_NAMES
        feature_cols = [f for f in FEATURE_NAMES if f in df.columns]

    cv_results, oof_preds, oof_probs = cross_validate_svm(X, y, groups)

    cv_df = pd.DataFrame(cv_results)
    print("\n--- Cross-Validation Results (per fold) ---")
    print(cv_df.to_string(index=False))

    print("\n--- Mean ± Std ---")
    for metric in ["accuracy", "f1", "auc", "sensitivity", "specificity"]:
        vals = cv_df[metric].dropna()
        print(f"  {metric:<15}: {vals.mean():.3f} ± {vals.std():.3f}")

    cv_out = os.path.join(FEATURES_DIR, "svm_cv_results.csv")
    cv_df.to_csv(cv_out, index=False)
    print(f"\nCV results saved → {cv_out}")

    print("\nTraining final model on all data…")
    pipeline = train_final_svm(X, y, feature_cols)
    print("Done.")
    print("=" * 72)
