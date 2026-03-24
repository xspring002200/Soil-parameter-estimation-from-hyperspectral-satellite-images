#!/usr/bin/env python3
"""
Heavy-metal content prediction from hyperspectral CSV data.

Expected CSV format
-------------------
my_data.csv
  - Each row  : one sampling point (≈200 rows expected)
  - All columns except the last ``--n-targets`` ones : spectral band values
    (band1, band2, … bandN)
  - Last ``--n-targets`` columns : measured heavy-metal / soil-parameter
    concentrations to predict (e.g. Cd, Pb, As, …)

The script adapts the winning Random-Forest pipeline from the HYPERVIEW
challenge (experimental_3/random_forest_b/rf_train.py) so that it works with
plain CSV files instead of .npz hyperspectral patches.

Feature engineering (per sample, 1-D spectral curve)
-----------------------------------------------------
  arr       — raw reflectance / band values         [N]
  dXdl      — 1st derivative                        [N]
  d2Xdl2    — 2nd derivative                        [N]
  d3Xdl3    — 3rd derivative                        [N]
  real      — real  part of FFT(arr)                [N]
  imag      — imaginary part of FFT(arr)            [N]
  cA        — wavelet approximation coefs (sym3)    [≤N]  (optional)
  cD        — wavelet detail coefs (sym3)           [≤N]  (optional)

Total feature dimension: 6N (without wavelets) or ≈8N (with wavelets).

Usage example
-------------
python train_heavy_metals_csv.py \\
    --data   my_data.csv \\
    --n-targets 8 \\
    --folds  5 \\
    --n-estimators 500 1000 \\
    --save-model \\
    --output-dir results/
"""

import os
import sys
import random
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

# Optional dependencies -------------------------------------------------------
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Baseline regressor (mean prediction)
# ---------------------------------------------------------------------------

class BaselineRegressor:
    """Returns the per-column training mean for every test sample."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_ = np.mean(y, axis=0)
        self.n_outputs_ = y.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), self.n_outputs_), self.mean_)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(file_path: str, n_targets: int):
    """Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    n_targets : int
        Number of target columns at the right end of the file.

    Returns
    -------
    X : ndarray, shape (n_samples, n_bands)
    y : ndarray, shape (n_samples, n_targets)
    band_cols : list[str]
    target_cols : list[str]
    """
    df = pd.read_csv(file_path)
    if df.shape[1] <= n_targets:
        raise ValueError(
            f"CSV has {df.shape[1]} columns but n_targets={n_targets}. "
            "There must be at least one band column."
        )
    band_cols = list(df.columns[:-n_targets])
    target_cols = list(df.columns[-n_targets:])

    X = df[band_cols].values.astype(np.float64)
    y = df[target_cols].values.astype(np.float64)

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} bands, {y.shape[1]} targets.")
    print(f"Target columns: {target_cols}")
    return X, y, band_cols, target_cols


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    """Ensure a 1-D array has exactly *length* elements (zero-pad or trim)."""
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)))


def extract_features(X: np.ndarray, use_wavelets: bool = True) -> np.ndarray:
    """Extract spectral features from a 2-D array of band values.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_bands)
    use_wavelets : bool
        Whether to include wavelet coefficients (requires PyWavelets).

    Returns
    -------
    features : ndarray, shape (n_samples, n_features)
    """
    n_samples, n_bands = X.shape
    feature_list = []

    for i in tqdm(range(n_samples), desc="Extracting features", leave=True):
        arr = X[i]                          # raw spectral curve

        d1 = np.gradient(arr)               # 1st derivative
        d2 = np.gradient(d1)                # 2nd derivative
        d3 = np.gradient(d2)                # 3rd derivative

        fft = np.fft.fft(arr)
        real = np.real(fft)                 # FFT real part
        imag = np.imag(fft)                 # FFT imaginary part

        parts = [arr, d1, d2, d3, real, imag]

        if use_wavelets and _HAS_PYWT:
            cA, cD = pywt.dwt(arr, "sym3")
            parts.append(_pad_or_trim(cA, n_bands))
            parts.append(_pad_or_trim(cD, n_bands))

        feature_list.append(np.concatenate(parts))

    return np.array(feature_list)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_data(X: np.ndarray, y: np.ndarray,
                 noise_scale: float = 0.01,
                 n_copies: int = 1,
                 random_state: int = 42) -> tuple:
    """Add Gaussian noise copies to training data.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples, n_targets)
    noise_scale : float
        Noise magnitude relative to the per-feature standard deviation.
    n_copies : int
        How many noisy copies to append.
    random_state : int

    Returns
    -------
    X_aug : ndarray
    y_aug : ndarray
    """
    rng = np.random.RandomState(random_state)
    X_parts = [X]
    y_parts = [y]

    X_std = X.std(axis=0) + 1e-8
    y_std = y.std(axis=0) + 1e-8

    for _ in range(n_copies):
        X_noise = X + rng.randn(*X.shape) * noise_scale * X_std
        y_noise = y + rng.randn(*y.shape) * noise_scale * y_std
        X_parts.append(X_noise)
        y_parts.append(y_noise)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluation_score(y_true: np.ndarray, y_pred: np.ndarray,
                     y_baseline: np.ndarray,
                     target_names: list) -> float:
    """Compute the mean normalised MSE score (lower = better).

    Score = (1/T) * Σ_t  MSE(model_t) / MSE(baseline_t)
    """
    n_targets = y_true.shape[1]
    total = 0.0
    for i in range(n_targets):
        mse_model = mean_squared_error(y_true[:, i], y_pred[:, i])
        mse_bl = mean_squared_error(y_true[:, i], y_baseline[:, i])
        ratio = mse_model / (mse_bl + 1e-12)
        total += ratio
        print(
            f"  {target_names[i]:>10s}  baseline MSE={mse_bl:.4f}  "
            f"model MSE={mse_model:.4f}  ratio={ratio:.4f}"
        )
    score = total / n_targets
    print(f"  Mean score (lower is better): {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------

def _build_rf(n_estimators: int, max_depth: int | None,
              min_samples_leaf: int, n_outputs: int):
    """Return a fitted-ready RF regressor (multi-output).

    Parameters
    ----------
    n_estimators:     Number of trees.
    max_depth:        Maximum tree depth (None = unlimited).
    min_samples_leaf: Minimum samples per leaf.
    n_outputs:        Number of target columns; wraps in MultiOutputRegressor
                      when > 1.
    """
    base = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        criterion="squared_error",
        random_state=42,
    )
    if n_outputs == 1:
        return base
    return MultiOutputRegressor(base, n_jobs=1)


def _build_xgb(n_estimators: int, max_depth: int | None, eta: float,
               n_outputs: int):
    """Return an XGBoost regressor (multi-output).

    Parameters
    ----------
    n_estimators: Number of boosting rounds.
    max_depth:    Maximum tree depth (None defaults to 6).
    eta:          Learning rate.
    n_outputs:    Number of target columns; wraps in MultiOutputRegressor
                  when > 1.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed.")
    params = dict(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        eta=eta,
        max_depth=max_depth if max_depth is not None else 6,
        verbosity=0,
    )
    base = xgb.XGBRegressor(**params)
    if n_outputs == 1:
        return base
    return MultiOutputRegressor(base, n_jobs=1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------ load
    X_raw, y, band_cols, target_cols = load_csv(args.data, args.n_targets)
    n_samples, n_bands = X_raw.shape
    n_targets = y.shape[1]

    # --------------------------------------------------- feature engineering
    print("\nExtracting features ...")
    use_wavelets = args.use_wavelets and _HAS_PYWT
    if args.use_wavelets and not _HAS_PYWT:
        print("WARNING: PyWavelets (pywt) not found – wavelet features disabled.")
    X = extract_features(X_raw, use_wavelets=use_wavelets)
    print(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------- cross-validation loop
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_scores = []
    all_y_true, all_y_pred, all_y_bl = [], [], []
    best_models = []

    print(f"\nRunning {args.folds}-fold cross-validation ...")
    for fold_idx, (ix_train, ix_val) in enumerate(kfold.split(np.arange(n_samples))):
        print(f"\n--- Fold {fold_idx + 1}/{args.folds} ---")

        X_fold_train = X[ix_train]
        y_fold_train = y[ix_train]

        # Apply augmentation only to the training fold
        if args.augment_copies > 0:
            X_t, y_t = augment_data(
                X_fold_train, y_fold_train,
                noise_scale=args.noise_scale,
                n_copies=args.augment_copies,
                random_state=fold_idx,
            )
        else:
            X_t, y_t = X_fold_train, y_fold_train

        X_v = X[ix_val]
        y_v = y[ix_val]

        # Baseline
        baseline = BaselineRegressor().fit(X_t, y_t)
        y_b = baseline.predict(X_v)

        # Build model
        if args.regressor == "RandomForest":
            model = _build_rf(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                n_outputs=n_targets,
            )
        else:
            model = _build_xgb(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                eta=args.eta,
                n_outputs=n_targets,
            )

        model.fit(X_t, y_t if n_targets > 1 else y_t.ravel())
        y_hat = model.predict(X_v)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)

        score = evaluation_score(y_v, y_hat, y_b, target_cols)
        fold_scores.append(score)
        best_models.append(model)

        all_y_true.append(y_v)
        all_y_pred.append(y_hat)
        all_y_bl.append(y_b)

    # ----------------------------------------------- overall CV evaluation
    Y_true = np.concatenate(all_y_true, axis=0)
    Y_pred = np.concatenate(all_y_pred, axis=0)
    Y_bl = np.concatenate(all_y_bl, axis=0)

    print("\n=== Overall CV performance ===")
    overall_score = evaluation_score(Y_true, Y_pred, Y_bl, target_cols)
    print(f"\nCV fold scores : {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean CV score  : {np.mean(fold_scores):.4f}")

    # ----------------------------------------------- train final model on all data
    print("\nTraining final model on full dataset ...")
    if args.augment_copies > 0:
        print(f"Augmenting training data ({args.augment_copies} noisy copies) ...")
        X_full, y_full = augment_data(
            X, y,
            noise_scale=args.noise_scale,
            n_copies=args.augment_copies,
            random_state=42,
        )
    else:
        X_full, y_full = X, y

    if args.regressor == "RandomForest":
        final_model = _build_rf(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_outputs=n_targets,
        )
    else:
        final_model = _build_xgb(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            eta=args.eta,
            n_outputs=n_targets,
        )

    final_model.fit(X_full, y_full if n_targets > 1 else y_full.ravel())

    # Training-set performance (for information only)
    y_train_pred = final_model.predict(X)
    if y_train_pred.ndim == 1:
        y_train_pred = y_train_pred.reshape(-1, 1)
    baseline_full = BaselineRegressor().fit(X, y)
    y_train_bl = baseline_full.predict(X)
    print("\nFull training set performance (optimistic – no hold-out):")
    evaluation_score(y, y_train_pred, y_train_bl, target_cols)

    # ------------------------------------------------------ save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        model_path = os.path.join(
            args.output_dir,
            f"model_{args.regressor}_{timestamp}_ne{args.n_estimators}.pkl",
        )
        joblib.dump(final_model, model_path)
        print(f"\nModel saved to: {model_path}")

    # Save CV predictions
    pred_df = pd.DataFrame(
        Y_pred,
        columns=[f"pred_{c}" for c in target_cols],
    )
    true_df = pd.DataFrame(
        Y_true,
        columns=[f"true_{c}" for c in target_cols],
    )
    cv_results = pd.concat([true_df, pred_df], axis=1)
    cv_path = os.path.join(args.output_dir, "cv_predictions.csv")
    cv_results.to_csv(cv_path, index=False)
    print(f"CV predictions saved to: {cv_path}")

    # Full-data predictions (for ensemble / submission use)
    full_pred_df = pd.DataFrame(
        y_train_pred,
        columns=[f"pred_{c}" for c in target_cols],
    )
    full_path = os.path.join(args.output_dir, "full_predictions.csv")
    full_pred_df.to_csv(full_path, index=False)
    print(f"Full-data predictions saved to: {full_path}")

    return final_model, overall_score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a Random-Forest (or XGBoost) regressor on "
                    "hyperspectral CSV data to predict heavy-metal contents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the input CSV file (bands + targets).",
    )
    parser.add_argument(
        "--n-targets", type=int, default=8,
        help="Number of target columns at the right end of the CSV "
             "(default: 8 heavy metals).",
    )

    # --- features
    parser.add_argument(
        "--use-wavelets", action="store_true", default=False,
        help="Include sym3 wavelet features (requires PyWavelets).",
    )

    # --- augmentation
    parser.add_argument(
        "--augment-copies", type=int, default=2,
        help="Number of noisy augmentation copies to append during training.",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.01,
        help="Noise scale relative to per-feature std for augmentation.",
    )

    # --- cross-validation
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of K-fold cross-validation splits.",
    )

    # --- model selection
    parser.add_argument(
        "--regressor", type=str, default="RandomForest",
        choices=["RandomForest", "XGB"],
        help="Which regressor to use.",
    )

    # --- RandomForest hyperparameters
    parser.add_argument(
        "--n-estimators", type=int, default=500,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Maximum tree depth (None = unlimited).",
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=1,
        help="Minimum samples per leaf for RandomForest.",
    )

    # --- XGBoost hyperparameters
    parser.add_argument(
        "--eta", type=float, default=0.1,
        help="XGBoost learning rate (eta).",
    )

    # --- output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for saved models and prediction CSVs.",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False,
        help="Persist the final model to disk as a .pkl file.",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=== Heavy-metal prediction from hyperspectral CSV ===")
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    start = time.time()
    train(args)
    print(f"\nTotal runtime: {time.time() - start:.1f}s")
