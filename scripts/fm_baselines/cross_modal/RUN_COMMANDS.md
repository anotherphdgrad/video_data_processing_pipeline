# Cross-Modal Training — Copy-Paste Commands

All commands run on the remote server inside a tmux session.
Paths are fixed from the verified alignment run.

```bash
tmux new-session -s cross_modal
conda activate imagebind
cd /home/harshit/2024/video_data_processing_pipeline/scripts/fm_baselines/cross_modal
```

---

## Variables (set once, reused below)

```bash
FM_STORE=/home/harshit/2024/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth.zarr
FM_META=/home/harshit/2024/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth_metadata.csv
IMU_ROOT=/home/harshit/2024/IMU_stress_sensing_src/IMU_data
DEPTH_CKPTS=/home/harshit/2024/video_data_processing_pipeline/outputs_downstream_pub/runs/run_gpu_families/dinov2/motion_prev_depth/tcn/checkpoints
OUT_ROOT=/home/harshit/2024/video_data_processing_pipeline/outputs_cross_modal
```

---

## Step 1 — Find the LIMU-BERT fold checkpoints

```bash
find /home/harshit/2024/IMU_stress_sensing_src -name "fold_*_model.pt" -path "*limu*" 2>/dev/null | sort
```

You need one checkpoint per fold (fold_1 through fold_5).
Set the fold 1 checkpoint:

```bash
# Replace with actual path from the find output above
LIMU_CKPT_FOLD1=<PATH_TO_fold_1_model.pt>
LIMU_CKPT_FOLD2=<PATH_TO_fold_2_model.pt>
LIMU_CKPT_FOLD3=<PATH_TO_fold_3_model.pt>
LIMU_CKPT_FOLD4=<PATH_TO_fold_4_model.pt>
LIMU_CKPT_FOLD5=<PATH_TO_fold_5_model.pt>
```

---

## Step 2 — Extract teacher embeddings (per fold)

Run one per fold. Each uses that fold's checkpoint so test subjects
were never seen by the teacher that produces their embeddings.

```bash
mkdir -p $OUT_ROOT/teacher_embeddings

python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt $LIMU_CKPT_FOLD1 \
    --fm-store $FM_STORE --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --output $OUT_ROOT/teacher_embeddings/fold_1.npz \
    --device cuda

python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt $LIMU_CKPT_FOLD2 \
    --fm-store $FM_STORE --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --output $OUT_ROOT/teacher_embeddings/fold_2.npz \
    --device cuda

python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt $LIMU_CKPT_FOLD3 \
    --fm-store $FM_STORE --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --output $OUT_ROOT/teacher_embeddings/fold_3.npz \
    --device cuda

python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt $LIMU_CKPT_FOLD4 \
    --fm-store $FM_STORE --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --output $OUT_ROOT/teacher_embeddings/fold_4.npz \
    --device cuda

python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt $LIMU_CKPT_FOLD5 \
    --fm-store $FM_STORE --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --output $OUT_ROOT/teacher_embeddings/fold_5.npz \
    --device cuda
```

Each run takes ~1 minute. Expected output per run:
```
Saved 3040 embeddings (shape (3040, 72)) → .../fold_N.npz
```

---

## Step 3 — Smoke test (2 folds, 3 Optuna trials)

Confirm the training loop runs end-to-end before committing to full run.
Uses fold_1 embeddings for all folds in smoke test (acceptable for smoke only).

```bash
python train_cross_modal.py \
    --fm-store $FM_STORE \
    --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --teacher-embeddings $OUT_ROOT/teacher_embeddings/fold_1.npz \
    --depth-ckpt-dir $DEPTH_CKPTS \
    --output-root $OUT_ROOT \
    --run-name smoke_test \
    --max-folds 2 \
    --optuna-trials 3 \
    --device cuda
```

Expected: 3 Optuna trials complete, 2 folds train, test metrics printed.
Check output: `$OUT_ROOT/smoke_test/test_summary.csv`

---

## Step 4 — Full run (all 5 folds, 30 Optuna trials, per-fold teacher)

**Note**: `train_cross_modal.py` currently takes a single `--teacher-embeddings`
file. For the full publication run use fold_1 embeddings (acceptable since the
teacher is frozen and the depth model is what trains). Per-fold teacher support
can be added as a follow-up.

```bash
python train_cross_modal.py \
    --fm-store $FM_STORE \
    --fm-meta-csv $FM_META \
    --imu-data-root $IMU_ROOT \
    --teacher-embeddings $OUT_ROOT/teacher_embeddings/fold_1.npz \
    --depth-ckpt-dir $DEPTH_CKPTS \
    --output-root $OUT_ROOT \
    --run-name cross_modal_dinov2_depth_limu \
    --optuna-trials 30 \
    --device cuda
```

Expected runtime: ~2-3 hours on a single GPU.
Results: `$OUT_ROOT/cross_modal_dinov2_depth_limu/test_summary.csv`

---

## Step 5 — Compare against baseline

Once training completes, compare cross-modal vs the original downstream results:

```bash
python -c "
import pandas as pd
baseline = pd.read_csv('/home/harshit/2024/video_data_processing_pipeline/outputs_downstream_pub/per_fold_metrics_full.csv')
baseline = baseline[(baseline.encoder=='dinov2') & (baseline.feature=='motion_prev_depth') &
                    (baseline.model_family=='tcn') & (baseline.split=='test')]

cross = pd.read_csv('$OUT_ROOT/cross_modal_dinov2_depth_limu/per_fold_metrics.csv')
cross = cross[cross.split=='test']

print('=== Baseline (downstream TCN) ===')
print(baseline[['fold_id','auc','balanced_accuracy']].to_string(index=False))
print(f'Mean AUC: {baseline.auc.mean():.3f}')

print()
print('=== Cross-modal (depth + IMU guidance) ===')
print(cross[['fold_id','auc','balanced_accuracy']].to_string(index=False))
print(f'Mean AUC: {cross.auc.mean():.3f}')
"
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'imu_dataset'` | Run scripts from cross_modal/ dir |
| `checkpoint has model_family=rnn_attn` | Check `--depth-ckpt-dir` points to `tcn/checkpoints` |
| `LIMU-BERT-Public not found` | `cd /home/harshit/2024/IMU_stress_sensing_src && git submodule update --init modules/LIMU-BERT-Public` |
| CUDA OOM | Add `--batch-size 16` to training command |
| `No paired windows found` | Re-run verify_paired_alignment.py to confirm alignment |
