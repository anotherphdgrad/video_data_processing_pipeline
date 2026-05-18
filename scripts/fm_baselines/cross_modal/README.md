# Cross-Modal Depth + IMU Training

Improves the depth motion TCN representation by aligning it with a frozen IMU
teacher signal during training.  At inference only depth is needed.

## Contents

| File | Purpose |
|---|---|
| `paired_dataset.py` | Aligns FM depth zarr + IMU shared zarr into paired windows |
| `verify_paired_alignment.py` | Standalone check — run before training |
| `imu_models.py` | Standalone IMU teacher model classes (FLIRT-LSTM, LIMU-BERT) |
| `depth_models.py` | Standalone depth TCN encoder (split from downstream checkpoint) |
| `cross_modal_model.py` | Shared projection + alignment/reconstruction losses |
| `extract_teacher_embeddings.py` | Pre-extract frozen IMU embeddings (run once) |
| `train_cross_modal.py` | Full training loop with Optuna + 5-fold CV |

---

## External Dependencies

### Required

| Package | Install | Used by |
|---|---|---|
| `torch` ≥ 2.0 | conda env `imagebind` already has it | all |
| `zarr` < 3 | conda env `imagebind` already has it | all |
| `numpy`, `pandas`, `scikit-learn` | conda env `imagebind` already has it | all |
| `optuna` | `pip install optuna` | `train_cross_modal.py` |

### Required for LIMU-BERT teacher (recommended path)

The `LimuBertTeacher` class in `imu_models.py` loads the `Transformer` class
from the LIMU-BERT-Public repository.

**Setup:**
```bash
# From the IMU-Stress-sensing directory:
git submodule update --init modules/LIMU-BERT-Public
```
The script looks for the module at:
- `modules/LIMU-BERT-Public/models.py`  (relative to CWD)
- `IMU-Stress-sensing/modules/LIMU-BERT-Public/models.py`  (relative to this file)

### Required for FLIRT-LSTM teacher only

| Package | Install | Notes |
|---|---|---|
| `flirt` | `pip install flirt` | FLIRT ACC feature extraction |
| `imu_stress` package | lives in `IMU-Stress-sensing/` | used for `IMUStressWindowDataset` and `FlirtACCFeatureExtractor` |

The `flirt_lstm` path is the slower and more complex option.  Use
`limu_bert` unless you specifically need FLIRT-LSTM as the teacher.

---

## Step-by-Step Usage

### 1. Verify alignment

```bash
conda activate imagebind
cd /home/harshit/2024/video_data_processing_pipeline/scripts/fm_baselines/cross_modal

python verify_paired_alignment.py \
    --fm-store    /home/harshit/2024/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth.zarr \
    --fm-meta-csv /home/harshit/2024/video_data_processing_pipeline/outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth_metadata.csv \
    --imu-store   /home/harshit/2024/video_data_processing_pipeline/IMU_shared_stores/window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr
```

Check: "All checks passed" and paired window count > 0.

### 2. Extract teacher embeddings (run once per IMU checkpoint)

**LIMU-BERT teacher (recommended):**
```bash
python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt /home/harshit/2024/video_data_processing_pipeline/outputs_flirt_torch/flirt_limu_bert_scratch/fold_1_model.pt \
    --fm-store    /path/to/dinov2/motion_prev_depth.zarr \
    --fm-meta-csv /path/to/dinov2/motion_prev_depth_metadata.csv \
    --imu-store   /path/to/imu_shared_store.zarr \
    --output      /home/harshit/2024/video_data_processing_pipeline/teacher_embeddings/limu_bert_fold1.npz \
    --device cuda
```

**FLIRT-LSTM teacher (requires flirt + IMU data root):**
```bash
python extract_teacher_embeddings.py \
    --teacher-type flirt_lstm \
    --teacher-ckpt /path/to/flirt_lstm_attention/fold_1_model.pt \
    --imu-data-root /home/harshit/2024/video_data_processing_pipeline/assets/IMU_data \
    --fm-store    /path/to/dinov2/motion_prev_depth.zarr \
    --fm-meta-csv /path/to/dinov2/motion_prev_depth_metadata.csv \
    --imu-store   /path/to/imu_shared_store.zarr \
    --output      ./teacher_embeddings/flirt_lstm_fold1.npz \
    --device cuda
```

### 3. Run cross-modal training

```bash
python train_cross_modal.py \
    --fm-store    /path/to/dinov2/motion_prev_depth.zarr \
    --fm-meta-csv /path/to/dinov2/motion_prev_depth_metadata.csv \
    --imu-store   /path/to/imu_shared_store.zarr \
    --teacher-embeddings ./teacher_embeddings/limu_bert_fold1.npz \
    --depth-ckpt-dir /path/to/outputs_downstream_pub/runs/run_gpu_families/dinov2/motion_prev_depth/tcn/checkpoints \
    --output-root ./outputs_cross_modal \
    --run-name dinov2_depth_limu_bert \
    --optuna-trials 30 \
    --device cuda
```

**Smoke test (2 folds, 5 Optuna trials):**
```bash
python train_cross_modal.py \
    --fm-store ... --fm-meta-csv ... --imu-store ... \
    --teacher-embeddings ... --depth-ckpt-dir ... \
    --output-root ./outputs_cross_modal_smoke \
    --max-folds 2 --optuna-trials 5 --device cuda
```

---

## Architecture Details

```
depth_x (B, T=75, D=384)          imu_embed (B, 72)  [frozen, pre-extracted]
       |                                   |
  DepthTCNEncoder                  SharedProjectionMLP
  (warm from checkpoint)           (same weights)
       |                                   |
  depth_repr (B, hidden_dim)       z_imu (B, shared_dim)
       |                                   |
  SharedProjectionMLP ──────────────────── cosine → L_align
       |
  z_depth (B, shared_dim)
       |─────────────── StressClassifier → logits → L_cls
       |─────────────── IMUDecoder → imu_recon → MSE(imu_embed) → L_recon
```

**Total loss:**
```
L = L_cls + λ_align · L_align + λ_recon · L_recon
```

Both `λ_align` and `λ_recon` are tuned by Optuna.  Setting `λ_recon=0`
disables the decoder (same as `--use-decoder False`).

---

## IMU Shared Store Path

The IMU shared zarr store is built by:
```bash
python IMU-Stress-sensing/scripts/write_shared_cache.py
```
Expected store name format:
`window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr`

---

## Missing Module Notes

If you see this error:
```
ImportError: LIMU-BERT-Public not found
```
Run: `git submodule update --init modules/LIMU-BERT-Public` from the
`IMU-Stress-sensing/` directory.

If you see:
```
SystemExit: flirt package not installed
```
Run: `pip install flirt` (only needed for FLIRT-LSTM teacher path).
