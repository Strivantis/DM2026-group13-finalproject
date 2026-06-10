# Drought Severity Forecasting — DM2026 Group 13

Multi-output time-series regression on the Kaggle "Natural Disaster Severity Prediction" competition.
Best public leaderboard MAE: **0.8217** (v52, Dual-Tree Hurdle with L1 objective).

---

## Repository Layout

```
project_root/
├── data/                        ← Create manually; place raw CSVs here
│   ├── train.csv                ← Kaggle training set  (daily, ~12M rows)
│   ├── test.csv                 ← Kaggle test set      (daily, ~205K rows)
│   └── v51_processed/           ← Auto-created by v51_preprocess.py
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── region_stats.csv
├── models/                      ← Auto-created by v52_train.py
│   └── v52_models/
│       ├── lgbm_a_fold{0-3}_week{0-4}.pkl   (20 files, Model A checkpoints)
│       ├── lgbm_b_fold{0-3}_week{0-4}.pkl   (20 files, Model B checkpoints)
│       └── v52_raw_test_preds.pkl            (serialised ensemble outputs)
├── plots/
│   └── final_eda/               ← EDA figures (pre-generated; committed)
├── src/                         ← All runnable source code
│   ├── v51_preprocess.py
│   ├── v52_dataset.py
│   ├── v52_train.py
│   ├── inference.py
│   ├── anasubmission.py
│   └── final_eda.py
└── submission_v52_<mode>.csv    ← Auto-created by inference.py
```

---

## Prerequisites

```
Python  >=3.10
lightgbm>=4.0   (GPU build recommended; set device="cpu" if unavailable)
numpy, pandas, scikit-learn, joblib, matplotlib, seaborn, scipy
```

Install dependencies:
```bash
pip install lightgbm numpy pandas scikit-learn joblib matplotlib seaborn scipy
```

All scripts must be run from the **project root** so that `src.*` imports resolve correctly.

---

## Data Setup

Download `train.csv` and `test.csv` from the Kaggle competition page and place them under `data/`:

```
project_root/
└── data/
    ├── train.csv
    └── test.csv
```

The preprocessing script reads these paths directly; no renaming or subdirectory nesting is required.

---

## Execution Pipeline

Run the three steps in order:

```bash
# Step 1 – Preprocess raw daily data into weekly feature matrices
python src/v51_preprocess.py

# Step 2 – Train the Dual-Tree Hurdle model (GPU recommended; ~2–3 h)
python src/v52_train.py

# Step 3 – Assemble and export the Kaggle submission file
python src/inference.py
```

The submission CSV is written to the project root as `submission_v52_<mode>.csv`.

---

## Module Descriptions

### `src/v51_preprocess.py`

**Purpose:** Full daily-to-weekly preprocessing pipeline for both train and test sets.

**Inputs:**
- `data/train.csv`
- `data/test.csv`

**Steps (in order):**
1. Per-region `ffill`/`bfill` imputation on all meteorological columns.
2. Global Z = 3.5 outlier clipping on temperature-related columns.
3. Parallelised day-of-year climatology pre-padding of the test set (3 weeks = 21 days per region).
4. Absolute-index weekly aggregation (sum for precipitation, mean/max/min/std for others).
5. 4-week rolling sums and means; 1- and 2-week lag features.
6. Physical drought feature engineering: `pet`, `deficit`, `deficit_roll_cum_4w`, `aridity_index`, `heat_shock`, `tmp_anomaly`.
7. K-Means (K = 10) climate clustering using per-region score and climate statistics.
8. `log1p` transform on `prec`, `prec_week_max`, `prec_roll_sum_4w`.

**Outputs:**
```
data/v51_processed/
├── train_processed.csv   (1,757,936 rows × 49 cols)
├── test_processed.csv    (    29,224 rows × 48 cols)
└── region_stats.csv      (     2,248 rows; region-level cluster assignments)
```

---

### `src/v52_dataset.py`

**Purpose:** Library module — not run directly. Imported by `v52_train.py`.

Provides:
- `FEATURE_COLS`: list of 32 feature names used for model input.
- `WINDOW_SIZE = 13`, `HORIZON = 5`: sliding-window and forecast-horizon constants.
- `refine_features()`: applies `DROP_COLS` pruning and `log1p` (for non-preprocessed input only).
- `build_time_seasonal_cv_folds()`: constructs 4-fold Cross-Seasonal time-based CV splits,
  with one strict 5-week validation window per region per fold, spaced 13 weeks apart.
- `build_tabular_dataset()`: flattens `W × D` sliding windows into 448-dimensional row vectors
  with appended trend-delta features.
- `build_tabular_test()`: same flattening for test regions (last 13 weeks per region).
- `extract_training_targets_for_te()`: extracts scores from the training portion of a fold
  for leakage-free target encoding computation.

**Outputs:** None (pure library).

---

### `src/v52_train.py`

**Purpose:** Trains the Dual-Tree Hurdle model under 4-fold Cross-Seasonal CV.

**Inputs:**
- `data/v51_processed/train_processed.csv`
- `data/v51_processed/test_processed.csv`

**Architecture:**
- **Model A** — `LGBMRegressor` with `objective="regression_l1"` (MAE loss); trained on all samples per fold-week.
- **Model B** — `LGBMClassifier` with `objective="binary"`; trained to predict P(score > 0).
- Per-fold, per-horizon-week: one Model A + one Model B = 4 folds × 5 weeks × 2 models = **40 checkpoints total**.

**Outputs:**
```
models/v52_models/
├── lgbm_a_fold{0-3}_week{0-4}.pkl    (20 files; Model A severity regressors)
├── lgbm_b_fold{0-3}_week{0-4}.pkl    (20 files; Model B occurrence classifiers)
└── v52_raw_test_preds.pkl             (dict with keys:
                                          preds_a_stack  shape (4, 2248, 5)
                                          probs_b_stack  shape (4, 2248, 5)
                                          region_ids     shape (2248,))
_training_log_52nd.txt                 (per-fold/per-week training diagnostics)
```

---

### `src/inference.py`

**Purpose:** Assembles the final Kaggle submission from saved model outputs. Supports
configurable gating strategies without retraining.

**Inputs:**
- `models/v52_models/v52_raw_test_preds.pkl`

**Key configuration (top of file):**

| Variable | Default | Description |
|---|---|---|
| `VERSION` | `"v52"` | Model version; determines which `{VERSION}_models/` directory to load. |
| `GATING_STRATEGY` | `"HARD"` | `"HARD"` thresholds `prob_mean`; `"SOFT"` computes expected value `score × prob`. |
| `USE_AUTO_THRESHOLD` | `False` | If `True`, uses OOF-tuned threshold stored in the pkl. |
| `MANUAL_THRESHOLD` | `0.5` | Hard-gate cutoff when `USE_AUTO_THRESHOLD=False`. |
| `APPLY_ROUNDING` | `True` | Round predictions to nearest integer. |

**Ensemble logic:**
```
l1_median  = median(preds_a_stack, axis=folds)   # robust to outlier folds
prob_mean  = mean(probs_b_stack,   axis=folds)
final      = where(prob_mean < τ, 0.0, l1_median)   # HARD mode
```

**Output:**
```
submission_v52_{mode}.csv    (2,248 rows × 6 cols: region_id, pred_week1 … pred_week5)
```
where `{mode}` is `SOFT`, `AUTO`, or `MANUAL_{τ:.2f}` depending on configuration.

---

### `src/anasubmission.py`

**Purpose:** Post-hoc diagnostic tool for comparing prediction distributions across
multiple submission CSVs. Prints percentile breakdowns, physical drought interval
bucket fractions, and exports a KDE density comparison plot.

**Inputs:**
- One or more `submission_*.csv` files at the project root (paths hard-coded in the
  `FILES` dict near the top of the file; edit to select which submissions to compare).

**Outputs:**
- Console: percentile table, interval fraction table.
- `submission_kde_comparison.png` (saved at project root).

---

### `src/final_eda.py`

### `src/final_eda.py`

**Purpose:** Comprehensive EDA pipeline covering score distribution, covariate shift, feature correlations, time-series visualisations, climate clustering, and adversarial validation. The pipeline analyzes the dataset generated by `v54_preprocess.py`, which stemmed from a subsequent, ultimately unsuccessful feature engineering iteration. 
Pre-generated outputs are committed under `plots/final_eda/`.

**Note:** Certain feature columns utilized in this script are missing from the `v52_processed` data. To verify or re-run the pipeline, `v54_preprocess.py` must be executed prior to running this script.

**Outputs:** Multiple PNG figures saved to `plots/final_eda/` (see source comments for full list).
