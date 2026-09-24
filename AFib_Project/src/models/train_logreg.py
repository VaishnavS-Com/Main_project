"""
=============================================================================
FILE: src/models/train_logreg.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 12 (Model 3): LOGISTIC REGRESSION BASELINE
=============================================================================

WHY ADD A THIRD, SIMPLER MODEL?
    Hasan & Motin (SPICSCON 2025) compare SVM, Logistic Regression and
    XGBoost on the same AF task, and we adopt that comparison for two
    reasons that matter independently of the paper:

    1. IT IS THE HONEST BASELINE.
       SVM and XGBoost are both non-linear. If a plain linear model gets
       within a point or two of them, then the extra machinery is not
       earning its place, and saying so is more useful than quietly
       reporting only the winner. Reviewers ask this question; better to
       answer it before they do.

    2. IT IS INTERPRETABLE.
       Logistic regression produces one coefficient per feature, with a
       sign. You can read straight off it that "higher CVNN pushes the
       prediction toward AF" - something neither an RBF kernel nor a forest
       of boosted trees will tell you. For a clinical audience that
       readability is worth real weight.

WHAT LOGISTIC REGRESSION ACTUALLY DOES:
    It fits a weighted sum of the features, then squashes that sum through
    a curve that maps any number onto the range 0 to 1, which we read as
    "probability this segment is AF":

        z = w1*feature1 + w2*feature2 + ... + b
        P(AF) = 1 / (1 + exp(-z))

    Learning means choosing the weights w so that this probability is high
    for real AF segments and low for the rest. The word "regression" in the
    name is a historical accident - it is a classifier.

PROTOCOL - IDENTICAL TO SVM AND XGBOOST ON PURPOSE:
    Same GroupKFold split, same SMOTE inside each training fold only, same
    scaler fitted on training data only, same random seed. If any of these
    differed, a difference in the results table could not be attributed to
    the model. Matching the protocol is what makes the comparison a
    comparison.

WHY SCALING IS NOT OPTIONAL HERE:
    Logistic regression with regularisation penalises large coefficients.
    Our features live on wildly different scales - MeanNN is around 800
    (milliseconds), CVNN is around 0.1, ECG_Rate_Max is around 100. Without
    standardisation the penalty would fall almost entirely on the
    small-valued features, effectively deleting them. StandardScaler puts
    every feature on comparable footing first.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import (  # noqa: E402
    CLASS_AFIB, CLASS_NORMAL, FEATURES_DIR, MODELS_DIR, N_FOLDS, RANDOM_STATE,
)
from src.feature_selection.mrmr_selector import load_selection  # noqa: E402
from src.models.train_svm import load_training_data  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _make_model() -> LogisticRegression:
    """
    Build the classifier.

    max_iter is raised to 2000 because the default of 100 is not always
    enough for the optimiser to converge on standardised biomedical
    features, and a silent non-convergence warning is easy to miss.

    class_weight="balanced" matches the SVM configuration. It is somewhat
    redundant alongside SMOTE, but keeping the two models configured
    identically is more important than shaving a redundancy.
    """
    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs",
    )


def cross_validate_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[Dict, List[np.ndarray], List[np.ndarray]]:
    """
    GroupKFold cross-validation, mirroring cross_validate_svm exactly.

    Returns
    -------
    cv_results : dict of metric lists, one entry per fold
    oof_preds  : out-of-fold predicted labels
    oof_probs  : out-of-fold AF probabilities
    """
    gkf = GroupKFold(n_splits=n_folds)
    smote = SMOTE(random_state=RANDOM_STATE)
    scaler = StandardScaler()
    model = _make_model()

    cv_results: Dict[str, List] = {
        "fold": [], "accuracy": [], "f1": [],
        "auc": [], "sensitivity": [], "specificity": [],
        "n_train": [], "n_test": [],
    }
    oof_preds: List[np.ndarray] = []
    oof_probs: List[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Scale using ONLY the training fold. Fitting the scaler on the test
        # data would leak information about the test distribution.
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # SMOTE on the training fold only, for the same reason.
        X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)

        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1]

        oof_preds.append(y_pred)
        oof_probs.append(y_prob)

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        cv_results["fold"].append(fold + 1)
        cv_results["accuracy"].append(accuracy_score(y_te, y_pred))
        cv_results["f1"].append(
            f1_score(y_te, y_pred, pos_label=CLASS_AFIB, zero_division=0))
        cv_results["auc"].append(
            roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1
            else float("nan"))
        cv_results["sensitivity"].append(tp / max(1, tp + fn))
        cv_results["specificity"].append(tn / max(1, tn + fp))
        cv_results["n_train"].append(len(y_tr))
        cv_results["n_test"].append(len(y_te))

        logger.info(
            "LogReg Fold %d | Acc=%.3f F1=%.3f AUC=%.3f Sen=%.3f Spe=%.3f",
            fold + 1, cv_results["accuracy"][-1], cv_results["f1"][-1],
            cv_results["auc"][-1], cv_results["sensitivity"][-1],
            cv_results["specificity"][-1],
        )

    return cv_results, oof_preds, oof_probs


def train_final_logreg(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
) -> Pipeline:
    """
    Fit the final model on all data, save it, and log the coefficients.

    The coefficient table is the reason this model earns its place in the
    thesis. A positive coefficient means "larger values of this feature push
    the prediction toward AF"; a negative one means the opposite. Because
    the inputs were standardised first, the magnitudes are directly
    comparable across features.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_s, y)

    model = _make_model()
    model.fit(X_res, y_res)

    pipeline = Pipeline([("scaler", scaler), ("logreg", model)])

    model_path = os.path.join(MODELS_DIR, "logreg_final.joblib")
    joblib.dump(pipeline, model_path)
    logger.info("Final Logistic Regression saved -> %s", model_path)

    coefficients = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
    })
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["pushes_toward"] = np.where(
        coefficients["coefficient"] > 0, "AF", "non-AF")
    coefficients = coefficients.sort_values(
        "abs_coefficient", ascending=False).reset_index(drop=True)

    coef_path = os.path.join(FEATURES_DIR, "logreg_coefficients.csv")
    coefficients.to_csv(coef_path, index=False)
    logger.info("Coefficients saved -> %s", coef_path)

    print("\n--- Logistic Regression coefficients (standardised features) ---")
    print(coefficients.to_string(index=False))

    return pipeline


if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 12 (Model 3): LOGISTIC REGRESSION BASELINE")
    print("=" * 72)

    X, y, groups, df = load_training_data()
    print(f"Training data : {len(y)} segments | "
          f"AF={int((y == CLASS_AFIB).sum())} "
          f"non-AF={int((y == CLASS_NORMAL).sum())}")
    print(f"Subjects      : {len(np.unique(groups))}")
    print(f"Features      : {X.shape[1]}")

    try:
        feature_cols = load_selection()
    except FileNotFoundError:
        from src.feature_extraction.hrv_features import FEATURE_NAMES
        feature_cols = [f for f in FEATURE_NAMES if f in df.columns]

    cv_results, oof_preds, oof_probs = cross_validate_logreg(X, y, groups)

    cv_df = pd.DataFrame(cv_results)
    print("\n--- Cross-Validation Results (per fold) ---")
    print(cv_df.to_string(index=False))

    print("\n--- Mean +/- Std ---")
    for metric in ["accuracy", "f1", "auc", "sensitivity", "specificity"]:
        values = cv_df[metric].dropna()
        print(f"  {metric:<15}: {values.mean():.3f} +/- {values.std():.3f}")

    cv_out = os.path.join(FEATURES_DIR, "logreg_cv_results.csv")
    cv_df.to_csv(cv_out, index=False)
    print(f"\nCV results saved -> {cv_out}")

    # Save out-of-fold predictions in the same format as the other models so
    # metrics.py and the false-positive analysis can consume them unchanged.
    gkf = GroupKFold(n_splits=N_FOLDS)
    test_indices = [test for _, test in gkf.split(X, y, groups)]
    oof = pd.DataFrame({
        "segment_id": df.iloc[np.concatenate(test_indices)]["segment_id"].to_numpy(),
        "y_true": y[np.concatenate(test_indices)],
        "y_pred": np.concatenate(oof_preds),
        "y_prob": np.concatenate(oof_probs),
    }).sort_values("segment_id")
    oof_path = os.path.join(FEATURES_DIR, "oof_predictions_logreg.csv")
    oof.to_csv(oof_path, index=False)
    print(f"OOF predictions saved -> {oof_path}")

    print("\nTraining final model on all data...")
    train_final_logreg(X, y, feature_cols)
    print("=" * 72)
