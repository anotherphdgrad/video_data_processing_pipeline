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
    parser.add_argument("--teacher-type", choices=["limu_bert", "flirt_lstm"], required=True)
    parser.add_argument("--teacher-ckpt", required=True, help="Path to IMU fold checkpoint .pt")
    parser.add_argument("--fm-store", required=True)
    parser.add_argument("--fm-meta-csv", required=True)
    parser.add_argument("--imu-store", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--imu-data-root", default=None,
                        help="[flirt_lstm only] Path to IMU CSV data root")
    parser.add_argument("--imu-seq-len", type=int, default=12,
                        help="Sequence length for LIMU-BERT pooling (default: 12)")
    parser.add_argument("--imu-feature-dim", type=int, default=6,
                        help="[limu_bert] ACC feature channels (default: 6 for raw_absdelta)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def _extract_limu_bert(
    teacher,
    dataset: PairedDepthIMUDataset,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract LIMU-BERT embeddings from raw ACC sequences via segment mean pooling."""
    import zarr

    root = zarr.open_group(str(dataset.imu_store_path), mode="r")
    seq_array = root["samples"]["sequences"]

    all_pairs = dataset._all_pairs
    N = len(all_pairs)
    pair_indices = np.array([p.pair_index for p in all_pairs], dtype=np.int64)
    imu_zarr_indices = np.array([p.imu_zarr_index for p in all_pairs], dtype=np.int64)

    embeddings_list = []

    teacher.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_zarr_idx = imu_zarr_indices[start:end]
            # Load raw ACC sequences
            raw_seqs = np.asarray(seq_array.oindex[batch_zarr_idx], dtype=np.float32)
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

    # Import IMU dataset from sister repo (optional dep — noted in README)
    imu_pipeline_root = Path(__file__).resolve().parents[3] / "IMU-Stress-sensing"
    if str(imu_pipeline_root) not in sys.path:
        sys.path.insert(0, str(imu_pipeline_root))

    try:
        from imu_stress.dataset import IMUStressWindowDataset
        from imu_stress.features import FlirtACCFeatureExtractor
    except ImportError as exc:
        raise SystemExit(
            "Cannot import IMU pipeline modules. Ensure IMU-Stress-sensing is in the repo.\n"
            f"Looked in: {imu_pipeline_root}"
        ) from exc

    print("Building IMU dataset for FLIRT feature extraction (this may take a while)...")
    imu_dataset = IMUStressWindowDataset(
        data_root=imu_data_root,
        window_seconds=30.0,
        overlap_seconds=15.0,
        sample_rate_hz=64.0,
        raw_sample_rate_hz=32.0,
        reserve_jelly_baseline_windows=True,
        use_jelly_baseline_delta=False,
        jelly_sequence_mode="raw_absdelta",
    )
    extractor = FlirtACCFeatureExtractor(
        imu_dataset,
        num_cores=1,
        use_jelly_baseline_delta=True,
        feature_mode="raw_delta",
        use_abs_delta=True,
    )

    meta_df = dataset.pairs_metadata_frame()
    all_pairs = dataset._all_pairs
    pair_indices = np.array([p.pair_index for p in all_pairs], dtype=np.int64)

    # Build FLIRT sequences in task/subject order (same logic as build_sequence_split)
    imu_meta = _load_imu_zarr_meta(dataset.imu_store_path)
    sorted_pairs = sorted(all_pairs, key=lambda p: (p.base_subject_id, p.task_id, p.imu_zarr_index))

    feature_sequences = {}  # pair_index → (seq_len, flirt_dim) array
    from itertools import groupby
    for (subj, task), group_iter in groupby(sorted_pairs, key=lambda p: (p.base_subject_id, p.task_id)):
        group = list(group_iter)
        # Match to imu_dataset windows by (subject, task)
        imu_window_indices = [
            i for i, w in enumerate(imu_dataset.windows)
            if w.base_subject_id == subj and w.task_id == task
        ]
        n_windows = min(len(group), len(imu_window_indices))
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


def _load_imu_zarr_meta(imu_store_path: Path) -> dict:
    import zarr
    root = zarr.open_group(str(imu_store_path), mode="r")
    return {
        "base_subject_ids": np.asarray(root["samples"]["base_subject_ids"][:]).astype(str),
        "task_ids": np.asarray(root["samples"]["task_ids"][:]).astype(str),
        "window_start": np.asarray(root["samples"]["window_start"][:], dtype=np.float64),
    }


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print(f"Loading paired dataset...")
    dataset = PairedDepthIMUDataset(
        fm_store_path=Path(args.fm_store),
        fm_metadata_csv_path=Path(args.fm_meta_csv),
        imu_store_path=Path(args.imu_store),
        verbose=True,
    )
    print(f"Total paired windows: {dataset.n_pairs_total}")

    print(f"\nLoading {args.teacher_type} teacher from {args.teacher_ckpt}")
    if args.teacher_type == "limu_bert":
        teacher = load_limu_bert_teacher(
            Path(args.teacher_ckpt),
            feature_dim=args.imu_feature_dim,
            seq_len=args.imu_seq_len,
            device=device,
        )
        print(f"Teacher repr_dim: {teacher.repr_dim}")
        print("Extracting LIMU-BERT embeddings...")
        pair_indices, embeddings = _extract_limu_bert(
            teacher, dataset, args.imu_seq_len, args.batch_size, device
        )
    else:
        teacher = load_flirt_lstm_teacher(Path(args.teacher_ckpt), device=device)
        print(f"Teacher repr_dim: {teacher.repr_dim}")
        if not args.imu_data_root:
            raise SystemExit("--imu-data-root is required for flirt_lstm teacher.")
        print("Extracting FLIRT-LSTM embeddings (slow — FLIRT runs per window)...")
        pair_indices, embeddings = _extract_flirt_lstm(
            teacher, dataset, args.imu_data_root, args.imu_seq_len, args.batch_size, device
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        pair_indices=pair_indices,
        embeddings=embeddings,
    )
    print(f"\nSaved {len(pair_indices)} embeddings (shape {embeddings.shape}) → {output_path}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Mean: {embeddings.mean():.4f}  Std: {embeddings.std():.4f}")


if __name__ == "__main__":
    main()
