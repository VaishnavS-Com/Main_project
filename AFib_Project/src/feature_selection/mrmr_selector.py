"""
=============================================================================
FILE: src/feature_selection/mrmr_selector.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 11: FEATURE SELECTION — MRMR
=============================================================================

PURPOSE:
    Select the most informative subset of HRV features before training,
    reducing dimensionality and avoiding overfitting on a small dataset.

WHAT IS MRMR?
    Minimum Redundancy Maximum Relevance (Peng et al. 2005).

    Naive feature selection ranks features by their individual correlation
    with the target label and picks the top N. The problem: many HRV features
    are highly correlated with each other (e.g., RMSSD and SD1 carry almost
    identical information — SD1 = RMSSD / sqrt(2) by definition). Picking
    both wastes a "slot" and may even hurt a linear classifier.

    MRMR solves this by simultaneously maximising:
        RELEVANCE   — mutual information between feature and label
    and minimising:
        REDUNDANCY  — mutual information between the new feature and already-
                      selected features

    At each step it adds the feature with the best (relevance - redundancy)
    score. The result is a diverse, non-redundant set of features.

FALLBACK STRATEGY:
    The `mrmr-selection` PyPI package has historically had compatibility
    issues with new Python/numpy versions. We implement a pure-numpy fallback
    that scores features by (MI with label) / (mean MI with selected features),
    which is the MIQ (Mutual Information Quotient) variant of MRMR. It is
    slower but has zero extra dependencies.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config.config import FEATURES_DIR, N_FEATURES_TO_SELECT, RANDOM_STATE
from src.feature_extraction.hrv_features import FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MUTUAL INFORMATION HELPERS
# ---------------------------------------------------------------------------

def _mi_with_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute mutual information between each column of X and y.

    Uses sklearn's implementation which handles both continuous and
    discrete variables correctly via k-NN estimation.
    """
    return mutual_info_classif(X, y, discrete_features=False,
                               random_state=RANDOM_STATE)


def _mi_between_features(X: np.ndarray, selected_cols: List[int]) -> np.ndarray:
    """
    For each column in X, compute mean MI with all already-selected columns.

    Returns zero for the first feature (no selected yet → no redundancy).
    """
    if not selected_cols:
        return np.zeros(X.shape[1])

    redundancy = np.zeros(X.shape[1])
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx].reshape(-1, 1)
        mis = []
        for sel in selected_cols:
            mi = mutual_info_classif(col, X[:, sel], discrete_features=False,
                                     random_state=RANDOM_STATE)
            mis.append(float(mi[0]))
        redundancy[col_idx] = float(np.mean(mis)) if mis else 0.0
    return redundancy


# ---------------------------------------------------------------------------
# MRMR IMPLEMENTATION (pure numpy fallback — MIQ variant)
# ---------------------------------------------------------------------------

def mrmr_miq(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n: int = N_FEATURES_TO_SELECT,
) -> Tuple[List[str], List[float]]:
    """
    MRMR-MIQ (Mutual Information Quotient) greedy selection.

    At each step, adds the feature maximising:
        score(f) = MI(f, y) / mean(MI(f, already_selected))

    PARAMETERS
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (NaN-imputed, scaled).
    y : np.ndarray, shape (n_samples,)
        Binary labels.
    feature_names : list of str
        Column names corresponding to X.
    n : int
        Number of features to select.

    RETURNS
    -------
    (selected_names, scores)
        selected_names — ordered list of selected feature names
        scores         — MRMR score at each selection step
    """
    n = min(n, len(feature_names))
    relevance = _mi_with_target(X, y)
    logger.info("Relevance computed for %d features", len(feature_names))

    selected_idx: List[int] = []
    selected_names: List[str] = []
    scores: List[float] = []
    remaining = list(range(len(feature_names)))

    for step in range(n):
        if not remaining:
            break

        redundancy = _mi_between_features(X[:, remaining], selected_idx)

        # Score for remaining features: relevance / (1 + redundancy)
        # The +1 prevents division-by-zero on the first step.
        step_scores = {
            idx: relevance[idx] / (1.0 + redundancy[remaining.index(idx)])
            for idx in remaining
        }
        best_idx = max(step_scores, key=step_scores.__getitem__)
        best_score = step_scores[best_idx]

        selected_idx.append(best_idx)
        selected_names.append(feature_names[best_idx])
        scores.append(float(best_score))
        remaining.remove(best_idx)

        logger.info("Step %2d: selected %-12s  score=%.4f",
                    step + 1, feature_names[best_idx], best_score)

    return selected_names, scores


# ---------------------------------------------------------------------------
# MRMR WITH LIBRARY FALLBACK
# ---------------------------------------------------------------------------

def select_features(
    feature_df: pd.DataFrame,
    n: int = N_FEATURES_TO_SELECT,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select the top-N HRV features using MRMR.

    PARAMETERS
    ----------
    feature_df : pd.DataFrame
        Full feature matrix (hrv_features.csv), possibly containing NaNs.
    n : int
        Number of features to select.
    feature_cols : list of str, optional
        Which columns to consider. Defaults to FEATURE_NAMES.
    label_col : str
        Name of the label column (0/1).

    RETURNS
    -------
    (selected_features, selection_report)
        selected_features  — list of N feature names in selection order
        selection_report   — DataFrame with feature, relevance, rank
    """
    if feature_cols is None:
        feature_cols = [f for f in FEATURE_NAMES if f in feature_df.columns]

    # Use only clean rows (quality == 'ok', label not NaN)
    clean = feature_df[
        (feature_df["quality"] == "ok") &
        feature_df[label_col].notna()
    ].copy()
    logger.info("MRMR using %d clean segments", len(clean))

    if len(clean) < 20:
        raise ValueError(
            f"Only {len(clean)} clean segments available for MRMR. "
            "Run the feature pipeline first."
        )

    X = clean[feature_cols].values.astype(np.float64)
    y = clean[label_col].values.astype(int)

    # Impute NaN with column median (any feature that failed for some segments)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Scale to zero mean, unit variance (required for MI estimation stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Try the mrmr-selection library first --------------------------------
    selected_names = None
    scores = None

    try:
        from mrmr import mrmr_classif
        logger.info("Using mrmr-selection library")
        X_df = pd.DataFrame(X_scaled, columns=feature_cols)
        y_series = pd.Series(y)
        selected_names = mrmr_classif(X_df, y_series, K=n)
        scores = [float("nan")] * len(selected_names)  # library doesn't expose scores
    except ImportError:
        logger.info("mrmr-selection not available; using built-in MRMR-MIQ")
    except Exception as exc:
        logger.warning("mrmr library failed (%s); falling back to built-in", exc)

    if selected_names is None:
        selected_names, scores = mrmr_miq(X_scaled, y, feature_cols, n=n)

    # --- Build report --------------------------------------------------------
    relevance = _mi_with_target(X_scaled, y)
    relevance_map = dict(zip(feature_cols, relevance))

    report = pd.DataFrame({
        "rank":      range(1, len(selected_names) + 1),
        "feature":   selected_names,
        "relevance": [relevance_map.get(f, float("nan")) for f in selected_names],
        "mrmr_score": scores,
    })

    logger.info("MRMR selected %d features: %s", len(selected_names), selected_names)
    return selected_names, report


# ---------------------------------------------------------------------------
# SAVE / LOAD SELECTION
# ---------------------------------------------------------------------------

def save_selection(selected_features: List[str], report: pd.DataFrame) -> None:
    """Save the selected feature list and report to disk."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    json_path = os.path.join(FEATURES_DIR, "selected_features.json")
    with open(json_path, "w") as fh:
        json.dump({"selected_features": selected_features}, fh, indent=2)

    csv_path = os.path.join(FEATURES_DIR, "mrmr_report.csv")
    report.to_csv(csv_path, index=False)

    logger.info("MRMR selection saved → %s", json_path)
    logger.info("MRMR report saved    → %s", csv_path)


def load_selection() -> List[str]:
    """Load previously saved feature selection."""
    json_path = os.path.join(FEATURES_DIR, "selected_features.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"No saved selection at {json_path}. Run select_features() first."
        )
    with open(json_path) as fh:
        data = json.load(fh)
    return data["selected_features"]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    feat_path = os.path.join(FEATURES_DIR, "hrv_features.csv")
    if not os.path.isfile(feat_path):
        raise SystemExit(
            "hrv_features.csv not found. Run feature_pipeline.py first."
        )

    print("=" * 72)
    print("PHASE 11: MRMR FEATURE SELECTION")
    print("=" * 72)

    feature_df = pd.read_csv(feat_path)
    print(f"Loaded feature matrix: {len(feature_df)} rows")

    selected, report = select_features(feature_df, n=N_FEATURES_TO_SELECT)
    save_selection(selected, report)

    print("\nTop features selected by MRMR:")
    print(report.to_string(index=False))
    print("=" * 72)
