# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All scripts require the `imagebind` conda environment:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
```

There are no tests or a build system. Scripts are run directly as CLI tools.

## Project Overview

Research pipeline for OUD/stress detection using RGB and depth video streams from a clinical study. Participants performed stress and non-stress tasks while recorded by RGB-D cameras and IMU sensors. The pipeline aligns these modalities, extracts temporal clips, computes derived features, and trains/evaluates classifiers.

**Binary stress label contract** (shared with the companion IMU pipeline):
- Non-stress tasks: `jelly`, `count`, `baseline`
- Stress tasks: `bad`, `stress`, `arithmetic`, `stroop`
- Ignored tasks: `good`, `nature_video`, `song`, `speech`

**Evaluation contract** (must remain consistent with `/home/harshit/2024/IMU_stress_sensing_src/imu_stress/`):
- Participant-disjoint folds via `GroupKFold(n_splits=5)` grouped by `base_subject_id`
- Inner validation via `GroupShuffleSplit(test_size=0.2, random_state=42 + fold_id)`

## Pipeline Stages

### Stage 1 — Session manifest

Maps IMU participant stems to RGB/depth H5 sessions. Rebuilds modality paths from filenames + runtime roots so the manifest is portable across local mounts and HPC.

```bash
python scripts/generate_imu_video_mapping_manifest.py \
  --imu-root assets/IMU_data \
  --manifest-csv assets/manifest_mapping_clean_updated_sol.csv \
  --candidate-csv assets/imu_participant_mapping_candidates.csv \
  --depth-root /home/harshit/mnt/sol_scratch/OUD_Stress_depth/depth_hdf5 \
  --rgb-root /home/harshit/mnt/sol_scratch/OUD_Stress_RGB/rgb_hdf5 \
  --output-root assets/imu_video_mapping
```

Outputs: `assets/imu_video_mapping/imu_to_video_session_manifest.csv`

### Stage 2 — Task/frame manifest

Maps IMU task intervals onto RGB/depth H5 frame indices. IMU timestamps are reconstructed as `first_imu_ts + row_index / 32Hz` (stored timestamps are coarse 1-second anchors with ~32 samples/second).

```bash
python scripts/generate_rgb_depth_task_frame_manifest.py \
  --session-manifest assets/imu_video_mapping/imu_to_video_session_manifest.csv \
  --output-csv assets/imu_video_mapping/rgb_depth_task_frame_manifest.csv
```

Smoke test (one participant, skip H5 reads):
```bash
python scripts/generate_rgb_depth_task_frame_manifest.py \
  --session-manifest assets/imu_video_mapping/imu_to_video_session_manifest.csv \
  --output-csv /tmp/smoke.csv \
  --participants 0001 --max-session-rows 2 --skip-h5
```

### Stage 3 — Raw 5 Hz Zarr extraction

Reads RGB/depth H5 frames for each task interval, downsamples to 5 Hz, resizes to 224×224, and writes compressed Zarr v2 stores (Blosc/Zstandard, chunk size 150 frames = 30s). H5 timestamps are normalized from epoch-milliseconds to epoch-seconds before IMU alignment.

```bash
python scripts/preprocess_rgb_depth_task_zarr.py \
  --task-frame-manifest assets/imu_video_mapping/rgb_depth_task_frame_manifest.csv \
  --output-root /tmp/rgb_depth_zarr_smoke \
  --participants 0001 --tasks jelly --views frontal --max-rows 1 --overwrite
```

Zarr structure per participant store: `group/base_subject_id/{view}_{task}/rgb`, `depth`, `timestamps`, `metadata`.

### Stage 4 — Derived feature Zarr stores

Input: raw Zarr from Stage 3. Three representations:

**Human-masked** (YOLO → SAM2, mask applied to both RGB and depth):
```bash
python scripts/preprocess_rgb_depth_derived_zarr.py \
  --input-root <raw_root> --output-root <masked_root> \
  --representation human_masked \
  --sam2-config /path/to/config.yaml --sam2-checkpoint /path/to/ckpt.pt \
  --participants 0001 --tasks stress --views frontal --max-task-groups 1 --overwrite
```

Mask carry-forward: up to 15 frames (3s). Masks outside 1%–70% area are rejected. SAM2 is required; there is no SAM v1 fallback.

**Motion difference** (requires masked output as `--mask-source-root`):
```bash
python scripts/preprocess_rgb_depth_derived_zarr.py \
  --representation motion_previous \
  --mask-source-root <masked_root> ...
```

Variants: `motion_previous` (consecutive frames), `motion_jelly_mean3` (vs. mean of first 3 jelly windows).

**RAFT flow + Sobel edges** (RAFT from `torchvision.models.optical_flow`; depth uses `depth_diff_sobel`):
```bash
python scripts/preprocess_rgb_depth_derived_zarr.py \
  --representation flow_edge_raft \
  --mask-source-root <masked_root> \
  --flow-lag 5 --flow-edge-clamp 2 --flow-edge-gamma 0.5 ...
```

### Stage 5 — Training pipeline

Orchestrates all derive/window/train stages end-to-end:

```bash
python scripts/run_rgb_depth_feature_training_pipeline.py \
  --stage all \
  --raw-root OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw_zarr2 \
  --feature-root processed_rgb_depth_features \
  --run-root outputs_rgb_depth \
  --sam2-config "$SAM2_CONFIG" --sam2-checkpoint "$SAM2_CKPT"
```

If derived features already exist, resume from `--stage train`. Window manifest applies tiered mask-quality filtering (`clean`, `usable`, `inspect`, `drop`); production runs use `clean usable` only.

### Foundation model baselines

Frozen encoder embeddings cached as framewise Zarr v2 under `outputs_rgb_depth_fm/embeddings_zarr2/` (`X`: `num_windows × frames_per_window × embed_dim`):

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage embed \
  --window-manifest processed_rgb_depth_features/manifests/window_manifest.csv \
  --output-root outputs_rgb_depth_fm \
  --encoders imagebind dinov2 \
  --features masked_rgb masked_depth \
  --embedding-cache-mode framewise --embedding-frame-selection all \
  --model-repo-root checkpoints/repos \
  --torch-hub-dir checkpoints/torch_hub \
  --imagebind-checkpoint checkpoints/imagebind/imagebind_huge.pth
```

## HPC (SLURM / Sol)

The launcher `scripts/hpc_rgb_depth_preprocessing.sh` runs the full pipeline with HPC paths:
- Depth root: `/scratch/hsharm62/OUD_Stress_depth/depth_hdf5`
- RGB root: `/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5`
- Output: `/scratch/hsharm62/OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw`

Key env var overrides: `STAGE` (`all`/`session`/`task_frames`/`manifests`/`zarr`), `PARTICIPANTS`, `TASKS`, `VIEWS`, `LOCAL_ZARR_WORKERS`, `NUM_SHARDS`/`SHARD_INDEX`, `COMPRESSION_LEVEL`, `OVERWRITE_ZARR=1`.

Multi-worker Zarr (CPU/IO bound, no GPU needed):
```bash
STAGE=zarr LOCAL_ZARR_WORKERS=4 CPU_THREADS_PER_WORKER=1 COMPRESSION_LEVEL=1 \
  bash scripts/hpc_rgb_depth_preprocessing.sh
```

## Key Data Assets

| File | Purpose |
|---|---|
| `assets/manifest_mapping_clean_updated_sol.csv` | Session-level manifest with HPC depth paths, timing offsets, `view_type` |
| `assets/imu_participant_mapping_candidates.csv` | IMU-stem → video participant crosswalk |
| `assets/imu_video_mapping/imu_to_video_session_manifest.csv` | Generated session manifest (78 IMU files, 0 missing) |
| `assets/imu_video_mapping/rgb_depth_task_frame_manifest.csv` | Generated frame-index manifest |

Hardcoded special cases in the session mapper: skip IMU stem `8876`; map `9933v2` to `9933` session-2 rows; preserve `xianfei` with both `group` and `manifest_group` columns.

## Validate Raw Zarr Stores

```bash
python scripts/validate_rgb_depth_raw_zarr.py \
  --input-root processed_rgb_depth_zarr_5hz_raw \
  --output-csv assets/derived_feature_visual_checks/raw_zarr_validation_report.csv
```

Incomplete task groups are skipped by derived-feature extraction.

## Smoke/Debug Artifacts

Use repo-local ignored folder `derived_smoke/` for smoke outputs — not `/tmp` — so artifacts persist between sessions and are easy to inspect with `--write-smoke-png`.
