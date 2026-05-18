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
