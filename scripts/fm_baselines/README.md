# Foundation-Model Baseline Runner

This directory contains the first RGB/depth foundation-model baseline branch. It is intentionally separate from `scripts/run_rgb_depth_feature_training_pipeline.py` so the current CNN/Transformer experiments can keep running without interference.

The first implementation is OOM-safe by design:

- frozen encoders only
- deterministic sampled frames per 30s window, default `16`
- embedding caches written before training
- fold-local Torch `mlp_probe` heads on cached vectors by default
- same participant-disjoint folds and metrics as the current RGB/depth and IMU baselines

## Inputs

Before running this branch, build the full window manifest:

```bash
python scripts/run_rgb_depth_feature_training_pipeline.py \
  --stage build_windows \
  --raw-root OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw_zarr2 \
  --feature-root processed_rgb_depth_features \
  --allowed-mask-quality-tiers clean usable \
  --overwrite
```

Expected input:

```text
processed_rgb_depth_features/manifests/window_manifest.csv
```

## Checkpoint Layout

Keep all checkpoints and hub caches under the repo-local ignored folder:

```text
checkpoints/
checkpoints/imagebind/imagebind_huge.pth
checkpoints/torch_hub/
checkpoints/repos/omnivore/
```

`.gitignore` excludes `checkpoints/`.

## Getting Checkpoints Locally

### ImageBind

ImageBind is already importable in the current `imagebind` env on this machine. Put the official checkpoint here:

```bash
mkdir -p checkpoints/imagebind
wget -O checkpoints/imagebind/imagebind_huge.pth \
  https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
```

If this file is absent, the official ImageBind package tries to use its hardcoded `.checkpoints/imagebind_huge.pth` download behavior.

### DINOv2

DINOv2 can be used as a local Python-importable repo plus Torch Hub cache. Clone it locally:

```bash
mkdir -p checkpoints/repos checkpoints/torch_hub
git clone https://github.com/facebookresearch/dinov2.git checkpoints/repos/dinov2
TORCH_HOME="$PWD/checkpoints/torch_hub" python - <<'PY'
import sys
import torch
sys.path.insert(0, "checkpoints/repos/dinov2")
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
PY
```

Do **not** run `pip install -r checkpoints/repos/dinov2/requirements.txt` inside the existing `imagebind` env. That file pins older `torch==2.0.0`, `torchvision==0.15.0`, and `xformers==0.0.18`, which can break the current SAM2/ImageBind/RAFT environment. For our frozen-backbone inference use case, the local repo plus the current PyTorch stack is enough in most cases.

If a missing lightweight dependency appears, install only that dependency, not the whole requirements file. Safe examples:

```bash
pip install omegaconf fvcore iopath
```

Other useful model names:

```text
dinov2_vits14
dinov2_vitb14
dinov2_vitl14
dinov2_vitg14
```

Start with `dinov2_vitb14`; `vitl` and `vitg` are more expensive.

### Omnivore

Omnivore is not currently importable in `imagebind`, so install it as a local Python-importable repo:

```bash
mkdir -p checkpoints/torch_hub checkpoints/repos
git clone https://github.com/facebookresearch/omnivore.git checkpoints/repos/omnivore
pip install -e checkpoints/repos/omnivore
pip install einops pytorchvideo timm
TORCH_HOME="$PWD/checkpoints/torch_hub" python - <<'PY'
import sys
import torch
sys.path.insert(0, "checkpoints/repos/omnivore")
torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB")
PY
```

Start with `omnivore_swinB`. The runner uses Torch Hub and will fail clearly if dependencies/checkpoints are missing.

## Run All FM Baselines

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage all \
  --window-manifest processed_rgb_depth_features/manifests/window_manifest.csv \
  --output-root outputs_rgb_depth_fm \
  --encoders imagebind dinov2 omnivore \
  --features masked_rgb masked_depth motion_prev_rgb motion_prev_depth flow_edge_rgb flow_edge_depth \
  --heads mlp_probe \
  --probe-epochs 30 \
  --probe-batch-size 64 \
  --num-sampled-frames 16 \
  --embedding-batch-size 2 \
  --model-repo-root checkpoints/repos \
  --skip-unavailable-encoders
```

Use `--skip-unavailable-encoders` while setting up checkpoints so DINOv2/ImageBind can run even if Omnivore is not ready yet.

## Run Specific Models And Modalities

ImageBind on RGB and depth only:

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage all \
  --encoders imagebind \
  --features masked_rgb masked_depth \
  --heads mlp_probe \
  --num-sampled-frames 16 \
  --embedding-batch-size 2
```

DINOv2 on motion and flow maps:

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage all \
  --encoders dinov2 \
  --features motion_prev_rgb motion_prev_depth flow_edge_rgb flow_edge_depth \
  --dinov2-model dinov2_vitb14 \
  --heads mlp_probe
```

Omnivore on RGB only:

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage all \
  --encoders omnivore \
  --features masked_rgb \
  --omnivore-model omnivore_swinB \
  --heads mlp_probe
```

## Outputs

Embedding caches:

```text
outputs_rgb_depth_fm/embeddings_zarr2/<encoder>/<feature>.zarr
outputs_rgb_depth_fm/embeddings_zarr2/<encoder>/<feature>_metadata.csv
outputs_rgb_depth_fm/embeddings_zarr2/<encoder>/<feature>_embedding_config.json
```

Embeddings are stored as compressed Zarr v2 arrays using Blosc/Zstd chunks. This is safer for full-dataset runs than a single `.npz` because it avoids rewriting one large archive and stays consistent with the rest of the RGB/depth pipeline.

Run outputs:

```text
outputs_rgb_depth_fm/runs/<run_name>/per_fold_metrics.csv
outputs_rgb_depth_fm/runs/<run_name>/fold_predictions.csv
outputs_rgb_depth_fm/runs/<run_name>/summary_metrics.csv
outputs_rgb_depth_fm/runs/<run_name>/all_window_mode_results_concise.csv
outputs_rgb_depth_fm/all_window_mode_results_concise.csv
```

## Compatibility Notes

- `masked_rgb`: native for all three encoders.
- `masked_depth`: native-ish for ImageBind depth branch and Omnivore; DINOv2 uses depth rendered as 3-channel image.
- `motion_prev_*` and `flow_edge_*`: technically supported as rendered visual maps, but out-of-distribution for these encoders. Treat these as compatibility baselines.
- ImageBind uses its depth branch only for `masked_depth`; motion and flow use the vision branch to avoid muddy interpretation.
- The first branch freezes encoders. Finetuning should come later, after frozen embedding baselines show promise.
