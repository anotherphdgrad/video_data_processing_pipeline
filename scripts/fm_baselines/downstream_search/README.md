# Framewise FM Downstream Optuna Search

This folder tunes Torch downstream classifiers on cached framewise foundation-model embeddings.

## Input Contract

- Embedding root: `outputs_rgb_depth_fm/embeddings_zarr2/`
- Store layout: `{encoder}/{feature}.zarr`
- Required arrays: `X`, `y`, `window_id`, `base_subject_id`
- Expected `X` shape: `num_windows x 150 x embedding_dim`
- Metadata CSV: `{encoder}/{feature}_metadata.csv`

## Evaluation Contract

- Participant-disjoint outer split: `GroupKFold(n_splits=5)` by `base_subject_id`
- Validation split: `GroupShuffleSplit(test_size=0.2, random_state=42 + fold_id)`
- Optuna tunes on fold 1 only by default.
- Final best config is evaluated on all folds.
- Threshold is selected on validation balanced accuracy and reused for train/val/test metrics.

## Model Families

- `attn_pool_mlp`: learned attention pooling over frame embeddings plus MLP.
- `rnn_attn`: GRU/LSTM encoder plus attention pooling.
- `tcn`: temporal convolution stack plus attention pooling.
- `transformer_encoder`: Transformer encoder with CLS or attention pooling.

## Smoke Test

```bash
python scripts/fm_baselines/downstream_search/run_optuna_downstream.py \
  --embedding-root outputs_rgb_depth_fm/embeddings_zarr2 \
  --output-root outputs_rgb_depth_fm_downstream_search_smoke \
  --encoders dinov2 \
  --features masked_rgb \
  --model-families attn_pool_mlp \
  --optuna-trials 2 \
  --max-windows-per-feature 120 \
  --max-folds 2 \
  --device cuda \
  --random-seed 42 \
  --overwrite
```

## Full Single-Feature Search

```bash
python scripts/fm_baselines/downstream_search/run_optuna_downstream.py \
  --embedding-root outputs_rgb_depth_fm/embeddings_zarr2 \
  --output-root outputs_rgb_depth_fm_downstream_search \
  --encoders imagebind omnivore dinov2 \
  --features masked_rgb masked_depth motion_prev_rgb motion_prev_depth flow_edge_rgb flow_edge_depth \
  --model-families attn_pool_mlp rnn_attn tcn transformer_encoder \
  --optuna-trials 30 \
  --tune-fold-id 1 \
  --n-splits 5 \
  --val-ratio 0.2 \
  --random-seed 42
```

## Output Files

Each run writes:

- `config.json`
- `available_embedding_stores.csv`
- `per_fold_metrics.csv`
- `fold_predictions.csv`
- `fold_assignments.csv`
- `summary_metrics.csv`
- `all_window_mode_results_concise.csv`
- per-combo `optuna_study.csv`, `best_params.json`, histories, predictions, metrics, and checkpoints

The aggregate concise summary is written to:

```text
outputs_rgb_depth_fm_downstream_search/all_window_mode_results_concise.csv
```

## Saved-Prediction Calibration Audit

Use this when the downstream neural models are already trained and you want to
check whether validation-only score calibration helps without retraining or
overwriting the main results.

```bash
python scripts/fm_baselines/downstream_search/calibrate_saved_predictions.py \
  --input-root outputs_downstream_pub \
  --output-root outputs_downstream_pub_calibrated \
  --overwrite
```

The audit compares three per-fold strategies:

- `current_saved_threshold`: original saved score, prediction, and validation-selected threshold.
- `val_platt_lr_0p5`: logistic-regression/Platt calibrator fit only on that fold's validation split, then classified at calibrated score `0.5`.
- `val_platt_lr_val_threshold`: same validation-only calibrator, then a balanced-accuracy threshold selected on calibrated validation scores and applied to test.

Useful filtered smoke test:

```bash
python scripts/fm_baselines/downstream_search/calibrate_saved_predictions.py \
  --input-root outputs_downstream_pub \
  --output-root outputs_downstream_pub_calibrated_smoke \
  --encoders dinov2 \
  --features motion_prev_rgb \
  --model-families tcn \
  --max-combos 1 \
  --overwrite
```

Calibration audit outputs:

- `calibrated_fold_predictions.csv`
- `calibration_per_fold_metrics.csv`
- `calibrator_params.csv`
- `calibration_test_metrics.csv`
- `calibration_all_splits_metrics.csv`
- `calibration_all_window_mode_results_concise.csv`

## Subject-Normalized Inference

**Script:** `infer_subject_normalized.py`

### Why

During training, a single population mean/std is fit across all training
participants and reused for val and test subjects.  Test subjects are
held-out people whose embedding distributions may differ (body size, movement
style, camera distance).  Per-subject normalization re-centres each test
subject's embeddings into the space the model was trained in, reducing
inter-subject distributional shift without retraining.

### What it does

For each saved fold checkpoint:

1. Loads the raw zarr embeddings for test subjects (bypasses population normalization).
2. Computes each test subject's own mean/std across their test windows.
3. Normalizes their embeddings with those subject-specific statistics.
4. Runs the saved model and collects scores.
5. Applies the original validation-selected threshold.

Subjects with fewer than `--min-windows-for-subject-norm` (default 3) windows
fall back to population normalization — not enough data to estimate reliable
per-subject statistics.

Both strategies (`population_norm` and `subject_norm`) are written side by
side so the comparison is direct.

### Requirements

Must be run on the server where **both** the entropy-selected zarr stores and
the downstream checkpoints are present.  The script uses the same zarr + model
infrastructure as the training pipeline — no extra dependencies.

### Usage

```bash
# Full run — motion features, all encoders, all model families:
python infer_subject_normalized.py \
    --embedding-root /scratch/hsharm62/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected \
    --input-root     /scratch/hsharm62/video_data_processing_pipeline/outputs_downstream_pub \
    --output-root    /scratch/hsharm62/video_data_processing_pipeline/outputs_downstream_pub_subjectnorm \
    --device cuda

# Smoke test — one combo, two folds:
python infer_subject_normalized.py \
    --embedding-root /scratch/hsharm62/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected \
    --input-root     /scratch/hsharm62/video_data_processing_pipeline/outputs_downstream_pub \
    --output-root    /scratch/hsharm62/video_data_processing_pipeline/outputs_downstream_pub_subjectnorm_smoke \
    --encoders dinov2 \
    --features motion_prev_rgb \
    --model-families transformer_encoder \
    --max-folds 2 \
    --device cuda
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--embedding-root` | required | Entropy-selected zarr store root |
| `--input-root` | required | Downstream run root (contains checkpoints + fold_predictions.csv) |
| `--output-root` | required | New directory for outputs — source never overwritten |
| `--features` | `motion_prev_rgb motion_prev_depth` | Feature filter |
| `--min-windows-for-subject-norm` | `3` | Fallback threshold to population norm |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--max-folds` | None | Limit folds per combo for smoke tests |

### Outputs

| File | Description |
|---|---|
| `subject_norm_per_fold_metrics.csv` | Metrics per fold × strategy (`population_norm` / `subject_norm`) |
| `subject_norm_fold_predictions.csv` | Per-window scores for both strategies |
| `subject_norm_test_metrics.csv` | Mean ± std across folds per combo × strategy |
| `subject_norm_comparison.csv` | Side-by-side delta table: subject_norm − population_norm |
| `config.json` | Run configuration |
| `failures.csv` | Any combos that errored (if present) |

### Interpreting the delta table

A positive `delta_auc` means subject normalization improved AUC for that
combo.  If most deltas are near zero, the main bottleneck is genuine
inter-subject variability in stress responses rather than distributional shift
in embedding statistics — in that case, AUC is already your honest ceiling.
