# Cross-Modal Training — Copy-Paste Commands

All paths are stored in `config.json`. Scripts read paths from there automatically.
Run all commands from the `cross_modal/` directory inside a tmux session.

```bash
tmux new-session -s cross_modal
conda activate imagebind
cd /home/harshit/2024/video_data_processing_pipeline/scripts/fm_baselines/cross_modal
```

---

## Step 1 — Verify alignment

Confirms both zarr stores open and windows pair correctly.

```bash
python verify_paired_alignment.py --config config.json --no-normalizer
```

Expected: `All checks passed. Pairs aligned: 3040`

---

## Step 2 — Extract teacher embeddings (all 5 folds, ~5 minutes total)

Reads all fold checkpoints from config, saves `fold_1.npz` ... `fold_5.npz`
to `outputs_cross_modal/teacher_embeddings/`.

```bash
python extract_teacher_embeddings.py --config config.json
```

Expected output per fold:
```
  Fold 1: fold_1_model.pt
  Saved 3040 embeddings (shape (3040, 72)) → .../fold_1.npz
```

Already-extracted folds are skipped automatically on re-run.

---

## Step 3 — Smoke test (2 folds, 3 Optuna trials, ~5 minutes)

Confirms training loop runs end-to-end before committing to full run.

```bash
python train_cross_modal.py \
    --config config.json \
    --run-name smoke_test \
    --max-folds 2 \
    --optuna-trials 3
```

Expected: 3 Optuna trials, 2 folds complete, test metrics printed.
Check: `outputs_cross_modal/smoke_test/test_summary.csv`

---

## Step 4 — Full run (all 5 folds, 30 Optuna trials, ~2-3 hours)

```bash
python train_cross_modal.py \
    --config config.json \
    --run-name cross_modal_dinov2_depth_limu
```

Results: `outputs_cross_modal/cross_modal_dinov2_depth_limu/test_summary.csv`

Run inside tmux so it survives SSH disconnection:
```bash
tmux new-session -s full_run
conda activate imagebind
cd /home/harshit/2024/video_data_processing_pipeline/scripts/fm_baselines/cross_modal
python train_cross_modal.py --config config.json --run-name cross_modal_dinov2_depth_limu
```

---

## Step 5 — Compare against baseline

```bash
python -c "
import pandas as pd

baseline = pd.read_csv('/home/harshit/2024/video_data_processing_pipeline/outputs_downstream_pub/per_fold_metrics_full.csv')
baseline = baseline[
    (baseline.encoder=='dinov2') & (baseline.feature=='motion_prev_depth') &
    (baseline.model_family=='tcn') & (baseline.split=='test')
]

cross = pd.read_csv('/home/harshit/2024/video_data_processing_pipeline/outputs_cross_modal/cross_modal_dinov2_depth_limu/per_fold_metrics.csv')
cross = cross[cross.split=='test']

print('=== Baseline (downstream TCN, no IMU guidance) ===')
print(baseline[['fold_id','auc','balanced_accuracy','mcc']].to_string(index=False))
print(f'  Mean AUC: {baseline.auc.mean():.3f}  BA: {baseline.balanced_accuracy.mean():.3f}')

print()
print('=== Cross-modal (depth + LIMU-BERT guidance) ===')
print(cross[['fold_id','auc','balanced_accuracy','mcc']].to_string(index=False))
print(f'  Mean AUC: {cross.auc.mean():.3f}  BA: {cross.balanced_accuracy.mean():.3f}')

delta_auc = cross.auc.mean() - baseline.auc.mean()
print(f'  Delta AUC: {delta_auc:+.3f}')
"
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'imu_dataset'` | Must run from `cross_modal/` directory |
| `LIMU-BERT-Public not found` | `cd /home/harshit/2024/IMU_stress_sensing_src && git submodule update --init modules/LIMU-BERT-Public` |
| `checkpoint has model_family=rnn_attn` | Verify `depth_ckpt_dir` in config.json points to `tcn/checkpoints` |
| CUDA OOM | Add `--batch-size 32` to training command |
| Teacher embeddings dir empty | Re-run Step 2 |
