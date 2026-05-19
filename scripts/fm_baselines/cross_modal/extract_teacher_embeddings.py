#!/usr/bin/env python3
"""
Pre-extract frozen IMU teacher embeddings for all paired windows.

Run this ONCE before training.  The output is a numpy archive that the
training loop loads instead of running the IMU model every epoch.

Supported teacher types
-----------------------
limu_bert   Uses segment-mean-pooled raw ACC sequences (no flirt needed).
            Input: (B, seq_len, C_imu) pooled ACC sequences.
            Output: (B, 72)  first-token LIMU-BERT representation.

flirt_lstm  Uses FLIRT feature sequences (requires the flirt package).
            Input: (B, seq_len, flirt_dim) FLIRT feature sequences.
            Output: (B, hidden_dim) attention-pooled LSTM representation.
            NOTE: flirt package must be installed — see README.

Usage
-----
# LIMU-BERT teacher (recommended, no flirt needed):
python extract_teacher_embeddings.py \
    --teacher-type limu_bert \
    --teacher-ckpt /path/to/imu_runs/fold_1_model.pt \
    --fm-store     .../dinov2/motion_prev_depth.zarr \
    --fm-meta-csv  .../dinov2/motion_prev_depth_metadata.csv \
    --imu-store    .../window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr \
    --output       ./teacher_embeddings_limu_bert_fold1.npz \
    --device cuda

# FLIRT-LSTM teacher:
python extract_teacher_embeddings.py \
    --teacher-type flirt_lstm \
    --teacher-ckpt /path/to/imu_runs/fold_1_model.pt \
    --imu-data-root /path/to/IMU_data \
    --fm-store     .../dinov2/motion_prev_depth.zarr \
    --fm-meta-csv  .../dinov2/motion_prev_depth_metadata.csv \
    --imu-store    .../window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr \
    --output       ./teacher_embeddings_flirt_lstm_fold1.npz \
    --device cuda

Output format (.npz)
--------------------
  pair_indices   : (N,)  int64 — pair_index values from PairedDepthIMUDataset
  embeddings     : (N, D_imu)  float32 — frozen teacher representations
  teacher_type   : str attribute
  teacher_ckpt   : str attribute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from paired_dataset import PairedDepthIMUDataset
from imu_models import (
    load_limu_bert_teacher,
    load_flirt_lstm_teacher,
    segment_mean_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-extract IMU teacher embeddings.")
    parser.add_argument("--config", default=None,
                        help="Path to config.json. When provided, extracts all folds automatically.")
    parser.add_argument("--teacher-type", choices=["flirt_limu_bert", "flirt_lstm"], default=None)
    parser.add_argument("--teacher-ckpt", default=None,
                        help="Single fold checkpoint (ignored when --config is used).")
    parser.add_argument("--fm-store", default=None)
    parser.add_argument("--fm-meta-csv", default=None)
    parser.add_argument("--imu-data-root", default=None)
    parser.add_argument("--imu-channel-mode", default=None)
    parser.add_argument("--output", default=None,
                        help="Output .npz path (single fold). Ignored when --config is used.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for per-fold .npz files (overrides config output_root).")
    parser.add_argument("--imu-seq-len", type=int, default=None)
    parser.add_argument("--imu-feature-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()
    if args.config:
        from config_utils import load_config, apply_config
        apply_config(args, load_config(args.config))
    # Fill remaining defaults
    if args.imu_channel_mode is None:
        args.imu_channel_mode = "raw_absdelta"
    if args.imu_seq_len is None:
        args.imu_seq_len = 12
    if args.imu_feature_dim is None:
        args.imu_feature_dim = 6
    if args.device is None:
        args.device = "cuda"
    if args.teacher_type is None:
        args.teacher_type = "flirt_limu_bert"
    return args


def _extract_limu_bert(
    teacher,
    dataset: PairedDepthIMUDataset,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract LIMU-BERT embeddings from raw ACC sequences via segment mean pooling.
    Reads directly from IMUStressWindowDataset — no zarr cache needed.
    """
    all_pairs = dataset._all_pairs
    N = len(all_pairs)
    pair_indices = np.array([p.pair_index for p in all_pairs], dtype=np.int64)
    embeddings_list = []

    teacher.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_pairs = all_pairs[start:end]
            # Load raw (non-transformed) ACC sequences from IMUStressWindowDataset
            raw_seqs = np.stack([
                dataset.imu_dataset.get_raw_sequence_array(p.imu_dataset_index)
                for p in batch_pairs
            ], axis=0).astype(np.float32)
            # Segment mean pool to seq_len tokens
            pooled = segment_mean_pool(raw_seqs, target_len=seq_len)
            x = torch.from_numpy(pooled).to(device=device, dtype=torch.float32)
            reprs = teacher.get_representation(x)
            embeddings_list.append(reprs.cpu().numpy())
            if (start // batch_size) % 10 == 0:
                print(f"  [{end}/{N}] extracted", flush=True)

    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    return pair_indices, embeddings


def _extract_flirt_lstm(
    teacher,
    dataset: PairedDepthIMUDataset,
    imu_data_root: str,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract FLIRT-LSTM embeddings.
    Requires the flirt package and raw IMU CSV data.
    The sequence construction follows the same logic as build_sequence_split
    in baselines_flirt_torch/common.py.
    """
    try:
        import flirt.acc  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "flirt package not installed. Run: pip install flirt\n"
            "See README for details."
        ) from exc

    # FlirtACCFeatureExtractor is in the IMU pipeline — only required for this path.
    # It wraps the flirt package and the already-loaded IMUStressWindowDataset.
    imu_pipeline_root = Path(__file__).resolve().parents[3] / "IMU-Stress-sensing"
    if str(imu_pipeline_root) not in sys.path:
        sys.path.insert(0, str(imu_pipeline_root))
    try:
        from imu_stress.features import FlirtACCFeatureExtractor
    except ImportError as exc:
        raise SystemExit(
            "Cannot import FlirtACCFeatureExtractor.\n"
            "Ensure IMU-Stress-sensing is present and flirt is installed.\n"
            f"Looked in: {imu_pipeline_root}"
        ) from exc

    # Reuse the already-loaded IMUStressWindowDataset from paired_dataset
    # (it was built with raw_absdelta which includes reserved jelly baselines)
    imu_dataset = dataset.imu_dataset
    extractor = FlirtACCFeatureExtractor(
        imu_dataset,
        num_cores=1,
        use_jelly_baseline_delta=True,
        feature_mode="raw_delta",
        use_abs_delta=True,
    )

    all_pairs = dataset._all_pairs
    pair_indices = np.array([p.pair_index for p in all_pairs], dtype=np.int64)

    # Build FLIRT sequences in task/subject order (same logic as build_sequence_split)
    sorted_pairs = sorted(all_pairs, key=lambda p: (p.base_subject_id, p.task_id, p.imu_dataset_index))

    feature_sequences = {}  # pair_index → (seq_len, flirt_dim) array
    from itertools import groupby
    for (subj, task), group_iter in groupby(sorted_pairs, key=lambda p: (p.base_subject_id, p.task_id)):
        group = list(group_iter)
        # Match to imu_dataset windows by (subject, task)
        # Use imu_dataset_index from the pairs directly — already aligned
        imu_window_indices = [p.imu_dataset_index for p in group]
        n_windows = len(imu_window_indices)
        if n_windows == 0:
            continue
        group_features = extractor.feature_matrix(imu_window_indices[:n_windows])
        flirt_dim = group_features.shape[1]
        for pos, pair in enumerate(group[:n_windows]):
            seq = np.zeros((seq_len, flirt_dim), dtype=np.float32)
            mask = np.zeros(seq_len, dtype=np.float32)
            start_pos = max(0, pos - seq_len + 1)
            slice_f = group_features[start_pos: pos + 1]
            seq[-len(slice_f):] = slice_f
            mask[-len(slice_f):] = 1.0
            feature_sequences[pair.pair_index] = (seq, mask)

    # Run through teacher
    ordered_pairs = [p for p in all_pairs if p.pair_index in feature_sequences]
    N = len(ordered_pairs)
    embeddings_list = []
    pair_indices_out = []

    teacher.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_pairs = ordered_pairs[start:end]
            seqs = np.stack([feature_sequences[p.pair_index][0] for p in batch_pairs])
            masks = np.stack([feature_sequences[p.pair_index][1] for p in batch_pairs])
            x = torch.from_numpy(seqs).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(masks).to(device=device, dtype=torch.float32)
            reprs = teacher.get_representation(x, m)
            embeddings_list.append(reprs.cpu().numpy())
            pair_indices_out.extend(p.pair_index for p in batch_pairs)
            if (start // batch_size) % 10 == 0:
                print(f"  [{end}/{N}] extracted", flush=True)

    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    return np.array(pair_indices_out, dtype=np.int64), embeddings


def _extract_one(teacher_type, ckpt_path, dataset, args, device):
    if teacher_type == "flirt_limu_bert":
        # LIMU-BERT trained on FLIRT features — infer dims from checkpoint weights
        teacher = load_limu_bert_teacher(
            Path(ckpt_path),
            feature_dim=None,   # inferred from transformer.embed.lin.weight
            seq_len=None,       # inferred from checkpoint params
            device=device,
            limu_repo=getattr(args, "limu_bert_public_repo", None),
        )
        print(f"    Teacher feature_dim={teacher.transformer.embed.lin.weight.shape[1]}, "
              f"repr_dim={teacher.repr_dim}")
        # Uses the same FLIRT sequence building as flirt_lstm
        return _extract_flirt_lstm(teacher, dataset, args.imu_seq_len, args.batch_size, device)
    else:
        teacher = load_flirt_lstm_teacher(Path(ckpt_path), device=device)
        print(f"    Teacher repr_dim={teacher.repr_dim}")
        return _extract_flirt_lstm(teacher, dataset, args.imu_seq_len, args.batch_size, device)


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print("Loading paired dataset...")
    dataset = PairedDepthIMUDataset(
        fm_store_path=Path(args.fm_store),
        fm_metadata_csv_path=Path(args.fm_meta_csv),
        imu_data_root=Path(args.imu_data_root),
        imu_channel_mode=args.imu_channel_mode,
        verbose=True,
    )
    print(f"Total paired windows: {dataset.n_pairs_total}")

    if args.config:
        # Config mode: extract all folds automatically
        from config_utils import load_config, teacher_checkpoints, teacher_embeddings_dir
        cfg = load_config(args.config)
        fold_ckpts = teacher_checkpoints(cfg)
        out_dir = Path(args.output_dir) if args.output_dir else teacher_embeddings_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nConfig mode: extracting {len(fold_ckpts)} fold(s) → {out_dir}")
        for fold_id in sorted(fold_ckpts):
            ckpt = fold_ckpts[fold_id]
            out_path = out_dir / f"fold_{fold_id}.npz"
            if out_path.exists():
                print(f"  Fold {fold_id}: already exists, skipping ({out_path.name})")
                continue
            print(f"\n  Fold {fold_id}: {Path(ckpt).name}")
            pair_indices, embeddings = _extract_one(args.teacher_type, ckpt, dataset, args, device)
            np.savez(str(out_path), pair_indices=pair_indices, embeddings=embeddings)
            print(f"  Saved {len(pair_indices)} embeddings (shape {embeddings.shape}) → {out_path}")
        print(f"\nAll folds extracted to {out_dir}")
    else:
        # Single-fold mode
        if not args.teacher_ckpt:
            raise SystemExit("--teacher-ckpt is required when not using --config")
        if not args.output:
            raise SystemExit("--output is required when not using --config")
        print(f"\nExtracting with {args.teacher_type} from {args.teacher_ckpt}")
        pair_indices, embeddings = _extract_one(
            args.teacher_type, args.teacher_ckpt, dataset, args, device
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), pair_indices=pair_indices, embeddings=embeddings)
    print(f"\nSaved {len(pair_indices)} embeddings (shape {embeddings.shape}) → {output_path}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Mean: {embeddings.mean():.4f}  Std: {embeddings.std():.4f}")


if __name__ == "__main__":
    main()
