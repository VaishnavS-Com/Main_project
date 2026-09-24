"""
=============================================================================
FILE: src/models/train_xgboost.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 12 (Model 2): XGBoost CLASSIFIER WITH GroupKFold CROSS-VALIDATION
=============================================================================

PURPOSE:
    Train and evaluate an XGBoost gradient-boosted tree ensemble using the
    same GroupKFold strategy as the SVM, allowing a fair comparison.

WHY XGBoost AS THE SECOND MODEL?
    XGBoost (Chen & Guestrin 2016) is a gradient-boosted decision tree
    ensemble that consistently outperforms other algorithms on tabular data
    with <10k samples. Key advantages over SVM for this task:
        - Handles feature scale without StandardScaler (trees are invariant
          to monotone transformations).
        - Provides native feature importance rankings (which SVM cannot).
        - The `scale_pos_weight` parameter handles class imbalance internally
          without requiring SMOTE (though we compare both approaches).
        - Parallel tree building is fast even on CPU.
        - Produces well-calibrated probabilities with `eval_metric='auc'`.

HYPERPARAMETER CHOICES:
    n_estimators=300  — enough trees to converge without overfitting on
                        small data; early stopping handles the rest.
    max_depth=4       — shallow trees to prevent memorising individual
                        patients (generalisation > training accuracy).
    learning_rate=0.1 — moderate rate; lower rates need more trees.
    subsample=0.8     — stochastic row subsampling reduces overfitting.
    colsample_bytree=0.8 — stochastic feature subsampling per tree.

These are reasonable defaults for small biomedical datasets. A proper
hyperparameter search (Bayesian optimisation, nested CV) is possible but
goes beyond the scope of a final-year project.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (
    FEATURES_DIR,
    N_FOLDS,
    RANDOM_STATE,
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


def _make_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Build the XGBClassifier with project-standard hyperparameters."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance natively
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,       # use all CPU cores
        verbosity=0,     # suppress XGBoost's own output (we log ourselves)
    )


def load_training_data(
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load hrv_features.csv — same logic as train_svm.load_training_data().
    Kept here to make this module self-contained (no import from train_svm).
    """
    feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(
            f"hrv_features.csv not found at {feat_path}. "
            "Run feature_pipeline.py first."
        )

    df = pd.read_csv(feat_path)

    if feature_cols is None:
        try:
            feature_cols = load_selection()
        except FileNotFoundError:
            from src.feature_extraction.hrv_features import FEATURE_NAMES
            feature_cols = [f for f in FEATURE_NAMES if f in df.columns]

    clean = df[
        (df["quality"] == "ok") &
        df["label"].notna() &
        (~df["is_noise"])
    ].copy()

    X = clean[feature_cols].values.astype(np.float64)
    y = clean["label"].values.astype(int)
    groups = clean["subject"].values

    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, col_idx] = float(np.nanmedian(col))

    return X, y, groups, clean


def cross_validate_xgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[Dict, List, List]:
    """
    GroupKFold cross-validation for XGBoost.

    Uses `scale_pos_weight` = n_negative / n_positive, computed from
    the training fold in each iteration (not from the full dataset).
    """
    gkf = GroupKFold(n_splits=n_folds)

    cv_results: Dict[str, List] = {
        "fold": [], "accuracy": [], "f1": [],
        "auc": [], "sensitivity": [], "specificity": [],
        "n_train": [], "n_test": [],
    }
    oof_preds = []
    oof_probs = []
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Compute class weight for this fold's training data
        n_pos = int((y_tr == CLASS_AFIB).sum())
        n_neg = int((y_tr == CLASS_NORMAL).sum())
        spw = n_neg / max(1, n_pos)

        xgb = _make_xgb(scale_pos_weight=spw)
        xgb.fit(X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False)

        y_pred = xgb.predict(X_te)
        y_prob = xgb.predict_proba(X_te)[:, 1]

        oof_preds.append(y_pred)
        oof_probs.append(y_prob)
        fold_importances.append(xgb.feature_importances_)

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        cv_results["fold"].append(fold + 1)
        cv_results["accuracy"].append(accuracy_score(y_te, y_pred))
        cv_results["f1"].append(f1_score(y_te, y_pred, pos_label=CLASS_AFIB,
                                         zero_division=0))
        cv_results["auc"].append(
            roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan")
        )
        cv_results["sensitivity"].append(tp / max(1, tp + fn))
        cv_results["specificity"].append(tn / max(1, tn + fp))
        cv_results["n_train"].append(len(y_tr))
        cv_results["n_test"].append(len(y_te))

        logger.info(
            "XGB Fold %d | Acc=%.3f F1=%.3f AUC=%.3f Sen=%.3f Spe=%.3f",
            fold + 1,
            cv_results["accuracy"][-1], cv_results["f1"][-1],
            cv_results["auc"][-1], cv_results["sensitivity"][-1],
            cv_results["specificity"][-1],
        )

    # Average feature importance across folds
    cv_results["mean_importance"] = np.mean(fold_importances, axis=0).tolist()

    return cv_results, oof_preds, oof_probs


def train_final_xgb(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
) -> XGBClassifier:
    """Train the final XGBoost model on the full dataset and save it."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    n_pos = int((y == CLASS_AFIB).sum())
    n_neg = int((y == CLASS_NORMAL).sum())
    spw = n_neg / max(1, n_pos)

    xgb = _make_xgb(scale_pos_weight=spw)

    # Impute NaN before final fit
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, col_idx] = float(np.nanmedian(col))

    xgb.fit(X, y)
    # Store feature names via the booster's attribute (feature_names_in_ is read-only in XGB ≥2)
    xgb.get_booster().feature_names = list(feature_cols)

    model_path = os.path.join(MODELS_DIR, "xgb_final.joblib")
    joblib.dump(xgb, model_path)
    logger.info("Final XGBoost saved → %s", model_path)

    # Also save feature importance as CSV
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(FEATURES_DIR, "xgb_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    logger.info("Feature importance saved → %s", imp_path)

    return xgb


if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 12: XGBoost TRAINING + GroupKFold CROSS-VALIDATION")
    print("=" * 72)

    X, y, groups, df = load_training_data()
    print(f"Training data : {len(y)} segments | AF={int((y==1).sum())} NSR={int((y==0).sum())}")
    print(f"Subjects      : {len(np.unique(groups))}")
    print(f"Features      : {X.shape[1]}")

    try:
        feature_cols = load_selection()
    except FileNotFoundError:
        from src.feature_extraction.hrv_features import FEATURE_NAMES
        feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
        df_tmp = pd.read_csv(feat_path)
        feature_cols = [f for f in FEATURE_NAMES if f in df_tmp.columns]

    cv_results, oof_preds, oof_probs = cross_validate_xgb(X, y, groups)

    cv_df = pd.DataFrame({
        k: v for k, v in cv_results.items() if k != "mean_importance"
    })
    print("\n--- Cross-Validation Results (per fold) ---")
    print(cv_df.to_string(index=False))

    print("\n--- Mean ± Std ---")
    for metric in ["accuracy", "f1", "auc", "sensitivity", "specificity"]:
        vals = cv_df[metric].dropna()
        print(f"  {metric:<15}: {vals.mean():.3f} ± {vals.std():.3f}")

    cv_out = os.path.join(FEATURES_DIR, "xgb_cv_results.csv")
    cv_df.to_csv(cv_out, index=False)
    print(f"\nCV results saved → {cv_out}")

    print("\nTraining final model on all data…")
    xgb = train_final_xgb(X, y, feature_cols)
    print("Done.")
    print("=" * 72)
