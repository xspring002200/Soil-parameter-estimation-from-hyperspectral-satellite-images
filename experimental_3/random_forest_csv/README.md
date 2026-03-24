# Heavy-Metal Prediction from Hyperspectral CSV

This directory contains a self-contained script that adapts the winning
Random-Forest pipeline from the HYPERVIEW challenge
(`experimental_3/random_forest_b/rf_train.py`) so that it works with a plain
CSV file instead of `.npz` hyperspectral image patches.

## Expected CSV format (`my_data.csv`)

| band1 | band2 | … | bandN | Cd | Pb | As | Hg | Cu | Zn | Ni | Cr |
|-------|-------|---|-------|----|----|----|----|----|----|----|-----|
| …     | …     | … | …     | …  | …  | …  | …  | …  | …  | …  | …  |

- **Rows** – one sampling point per row (≈200 rows expected).
- **All columns except the last `--n-targets` ones** – spectral band values
  (`band1`, `band2`, …, `bandN`).  The number of bands `N` is detected
  automatically.
- **Last `--n-targets` columns (default 8)** – measured heavy-metal /
  soil-parameter concentrations to predict (e.g. Cd, Pb, As, …).

## Feature engineering

For each sample the script computes a 1-D spectral feature vector:

| Feature | Size | Description |
|---------|------|-------------|
| `arr`   | N    | Raw reflectance / band values |
| `dXdl`  | N    | 1st derivative (np.gradient) |
| `d2Xdl2`| N    | 2nd derivative |
| `d3Xdl3`| N    | 3rd derivative |
| `real`  | N    | Real part of FFT(arr) |
| `imag`  | N    | Imaginary part of FFT(arr) |
| `cA`    | ≤N   | Wavelet approximation coefs (`sym3`) — optional |
| `cD`    | ≤N   | Wavelet detail coefs (`sym3`) — optional |

Total: **6 × N** (or **8 × N** with `--use-wavelets`).

SVD-based features are intentionally omitted because the CSV contains 1-D
spectral curves only (no 2-D spatial patch data).

## Quick start

```bash
# Install dependencies (if not already present)
pip install scikit-learn tqdm pandas numpy joblib
# Optional but recommended:
pip install PyWavelets xgboost

# Run with default settings (5-fold CV, RandomForest, 8 targets)
python train_heavy_metals_csv.py \
    --data   my_data.csv \
    --n-targets 8 \
    --folds  5 \
    --n-estimators 500 \
    --save-model \
    --output-dir results/

# Use wavelets and more trees
python train_heavy_metals_csv.py \
    --data   my_data.csv \
    --use-wavelets \
    --n-estimators 1000 \
    --augment-copies 3 \
    --output-dir results/

# Use XGBoost instead of Random Forest
python train_heavy_metals_csv.py \
    --data   my_data.csv \
    --regressor XGB \
    --n-estimators 500 \
    --eta 0.05 \
    --output-dir results/
```

## Output files

| File | Description |
|------|-------------|
| `results/cv_predictions.csv` | Cross-validation true vs. predicted values |
| `results/full_predictions.csv` | Predictions on the full training set |
| `results/model_RandomForest_<timestamp>_ne<N>.pkl` | Serialised final model |

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | — | Path to input CSV (**required**) |
| `--n-targets` | 8 | Number of target columns at right end of CSV |
| `--folds` | 5 | K-fold cross-validation splits |
| `--regressor` | `RandomForest` | `RandomForest` or `XGB` |
| `--n-estimators` | 500 | Number of trees |
| `--max-depth` | `None` | Maximum tree depth |
| `--min-samples-leaf` | 1 | Minimum samples per leaf (RF) |
| `--augment-copies` | 2 | Noisy copies appended during training |
| `--noise-scale` | 0.01 | Noise magnitude (relative to feature std) |
| `--use-wavelets` | off | Add sym3 wavelet features (needs PyWavelets) |
| `--save-model` | off | Save final model to disk |
| `--output-dir` | `results` | Output directory |

## Evaluation metric

```
Score = (1/T) × Σ_t  MSE(model_t) / MSE(baseline_t)
```

where `baseline_t` predicts the training mean for target `t`.  Lower is
better; 1.0 means the model is no better than the mean baseline.
