# RGB/Depth SOTA Foundation-Model Baselines

This note tracks candidate pretrained encoders to evaluate after the current lightweight RGB/depth baselines. The goal is to improve stress detection performance while keeping the existing preprocessing, Zarr stores, 30s/15s windowing, participant-disjoint folds, and metric reporting unchanged.

## Current Baseline Contract

- Input data remains the derived 5 Hz Zarr feature stores under `processed_rgb_depth_features/`.
- Windowing remains `30s` at `5 Hz` = `150` frames with `15s` overlap = `75` frames.
- Evaluation remains participant-disjoint: `GroupKFold(n_splits=5)` by `base_subject_id`, with inner `GroupShuffleSplit(test_size=0.2, random_state=42 + fold_id)`.
- Metrics remain aligned with IMU baselines: validation-threshold-selected balanced accuracy, macro F1, and ROC AUC.
- First foundation-model scope should still be single-representation first, not fusion.

## Recommended Integration Pattern

Prefer frozen or lightly fine-tuned encoders with small trainable decision heads:

1. Read the same `window_manifest.csv`.
2. Load the same Zarr window slice.
3. Apply encoder-specific resize/normalization only inside the dataset/model path.
4. Extract frame-level or clip-level embeddings with the pretrained encoder.
5. Train a small temporal pooling/Transformer/classifier head fold-locally.
6. Save outputs using the same `per_fold_metrics.csv`, `summary_metrics.csv`, and `all_window_mode_results_concise.csv` schema.

This avoids rewriting preprocessing and keeps split/evaluation behavior comparable to the current CNN/Transformer baselines.

## Priority Candidates

| Candidate | Modalities To Try | Why It Is Useful | Compatibility Notes |
|---|---|---|---|
| ImageBind | `masked_rgb`, `masked_depth` | Strong first choice because it explicitly supports image/video and depth in a shared embedding space. | Use RGB/image branch for RGB windows and depth branch for depth windows. Start frozen. Good fit for minimal preprocessing changes. |
| Omnivore | `masked_rgb`, `masked_depth`, possibly motion/flow rendered as visual inputs | Designed as one model for images, videos, and single-view 3D/RGB-D style data. | Check available checkpoints and exact input adapters. Start as frozen feature extractor before finetuning. |
| DINOv2 | `masked_rgb`; simple depth-as-3-channel baseline | Strong general-purpose visual features with easy frame-level embedding extraction. | Best low-friction RGB baseline. For depth, normalize depth and replicate to 3 channels, but note this is not truly depth-native. |
| VideoMAE V2 | `masked_rgb`, possibly `motion_prev_rgb` | Video-pretrained encoder can model short temporal clips better than per-frame image encoders. | Requires sampling/packing 150-frame windows into expected video tensor shape. More integration work than DINOv2/ImageBind. |
| Video Swin / TimeSformer | `masked_rgb` | Mature video recognition baselines for temporal posture/motion patterns. | RGB-only. Useful if pretrained video action features transfer to stress-related body movement. |
| CLIP / OpenCLIP / SigLIP | `masked_rgb` | Easy robust RGB image embeddings; good sanity check against DINOv2. | RGB-specific. Depth/motion/flow can be rendered as 3-channel images, but that should be treated as a weaker compatibility baseline. |

## Depth-Relevant But Not Native Measured-Depth Encoders

These models are useful for geometry-aware RGB features, but they primarily take RGB input and predict depth rather than consuming measured depth as the native modality.

| Candidate | Possible Use | Compatibility Notes |
|---|---|---|
| Depth Anything V2 | RGB-derived geometry representation or encoder features | Strong modern monocular depth model. Useful on masked RGB, but not a direct measured-depth encoder without adaptation. |
| MiDaS / DPT | RGB-derived geometry representation | Older robust depth-estimation family. Lower priority than Depth Anything V2. |
| ZoeDepth / UniDepth | RGB-derived metric depth cues | Potentially useful if metric geometry helps, but adds complexity and is not directly aligned with measured depth Zarr inputs. |

## Suggested Evaluation Order

1. ImageBind frozen embeddings for `masked_rgb` and `masked_depth`.
2. Omnivore frozen embeddings for RGB and depth if implementation/checkpoints are clean.
3. DINOv2 frozen per-frame embeddings for `masked_rgb`, plus depth-as-3-channel baseline.
4. VideoMAE V2 or Video Swin for RGB temporal windows.
5. CLIP/SigLIP/OpenCLIP as RGB image-embedding baselines.

## Implementation Notes

- Keep foundation-model baselines in a separate model family rather than modifying `small_frame_cnn_transformer` or `temporal_pooling_cnn`.
- Cache embeddings only after confirming the split contract. If cached, include `feature_name`, `encoder_name`, `encoder_checkpoint`, `window_id`, `base_subject_id`, `task_id`, `label`, and `view_type`.
- Current cache format is framewise Zarr v2 under `outputs_rgb_depth_fm/embeddings_zarr2/`: `X` is `num_windows x frames_per_window x embedding_dim`, with `y`, `window_id`, and `base_subject_id` stored beside it. For full 30s windows at 5 Hz, `frames_per_window=150`.
- The inspection notebook lives with the cache at `outputs_rgb_depth_fm/embeddings_zarr2/inspect_framewise_fm_embeddings.ipynb`.
- Avoid feature extraction that mixes participants before fold creation unless embeddings are purely frozen and label-free.
- For finetuning, use fold-local training only and save one checkpoint per fold.
- For depth input to RGB-only encoders, record the rendering rule explicitly: percentile normalization, clipping range, 3-channel replication, and whether zeros/background were preserved.
- For video encoders, keep frame selection deterministic. The default framewise cache uses all 150 frames; optional sampled caches should record the sampled frame indices/config.

## Framewise Cache Commands

The old pooled cache was removed to avoid mixing cache semantics. Rebuild framewise caches with:

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage embed \
  --window-manifest processed_rgb_depth_features/manifests/window_manifest.csv \
  --output-root outputs_rgb_depth_fm \
  --encoders imagebind omnivore dinov2 \
  --features masked_rgb masked_depth motion_prev_rgb motion_prev_depth flow_edge_rgb flow_edge_depth \
  --embedding-cache-mode framewise \
  --embedding-frame-selection all \
  --embedding-batch-size 1 \
  --frame-embed-batch-size 16 \
  --model-repo-root checkpoints/repos \
  --torch-hub-dir checkpoints/torch_hub \
  --imagebind-checkpoint checkpoints/imagebind/imagebind_huge.pth
```

For a quick shape smoke test before a full run:

```bash
python scripts/fm_baselines/run_fm_baseline_eval.py \
  --stage embed \
  --window-manifest processed_rgb_depth_features/manifests/window_manifest.csv \
  --output-root outputs_rgb_depth_fm_framewise_smoke \
  --encoders dinov2 \
  --features masked_rgb \
  --max-windows-per-feature 4 \
  --embedding-cache-mode framewise \
  --embedding-frame-selection all \
  --embedding-batch-size 1 \
  --frame-embed-batch-size 16 \
  --model-repo-root checkpoints/repos \
  --torch-hub-dir checkpoints/torch_hub \
  --overwrite
```

## Source Pointers

- ImageBind: https://ai.meta.com/research/publications/imagebind-one-embedding-space-to-bind-them-all/
- ImageBind blog: https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/
- Omnivore: https://ai.meta.com/research/publications/omnivore-a-single-model-for-many-visual-modalities/
- DINOv2: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/
- CLIP: https://openai.com/index/clip/
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- VideoMAE V2: https://github.com/OpenGVLab/VideoMAEv2
- Video Swin Transformer: https://github.com/SwinTransformer/Video-Swin-Transformer
- TimeSformer: https://huggingface.co/docs/transformers/main/model_doc/timesformer
