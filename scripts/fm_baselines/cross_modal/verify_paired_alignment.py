#!/usr/bin/env python3
"""
Verify that FM depth and IMU zarr stores can be loaded and aligned.

Run this on the server before any training to confirm:
  1. Both zarr stores open correctly
  2. The FM metadata CSV has the required task_id column
  3. Windows align by (base_subject_id, task_id, window_position)
  4. Labels are consistent across modalities
  5. Returned tensor shapes are correct

This script does NOT train anything.  It is a data-readiness check.

Usage
-----
python verify_paired_alignment.py \
    --fm-store      /path/to/embeddings_zarr2_entropy75/dinov2/motion_prev_depth.zarr \
    --fm-meta-csv   /path/to/embeddings_zarr2_entropy75/dinov2/motion_prev_depth_metadata.csv \
    --imu-store     /path/to/IMU_shared_stores/window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr \
    --n-samples     5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FM-IMU paired alignment.")
    parser.add_argument("--config", default=None, help="Path to config.json (provides path defaults).")
    parser.add_argument("--fm-store", default=None)
    parser.add_argument("--fm-meta-csv", default=None)
    parser.add_argument("--imu-data-root", default=None)
    parser.add_argument("--imu-channel-mode", default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--no-normalizer", action="store_true")
    args = parser.parse_args()
    if args.config:
        from config_utils import load_config, apply_config
        apply_config(args, load_config(args.config))
    if args.imu_channel_mode is None:
        args.imu_channel_mode = "raw_absdelta"
    for required in ["fm_store", "fm_meta_csv", "imu_data_root"]:
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_','-')} is required (or set in --config)")
    return args


def _sep(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'='*pad} {title} {'='*pad}")
    else:
        print("=" * width)


def main() -> None:
    args = parse_args()

    # Add script directory to path so paired_dataset can be imported
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from paired_dataset import PairedDepthIMUDataset

    fm_store = Path(args.fm_store).resolve()
    fm_meta = Path(args.fm_meta_csv).resolve()
    imu_data_root = Path(args.imu_data_root).resolve()

    _sep("PATHS")
    print(f"FM zarr store   : {fm_store}")
    print(f"FM metadata CSV : {fm_meta}")
    print(f"IMU data root   : {imu_data_root}")
    print(f"IMU channel mode: {args.imu_channel_mode}")

    for p in [fm_store, fm_meta, imu_data_root]:
        if not p.exists():
            print(f"\nERROR: not found: {p}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load and align
    # -----------------------------------------------------------------------
    _sep("ALIGNMENT")
    try:
        dataset = PairedDepthIMUDataset(
            fm_store_path=fm_store,
            fm_metadata_csv_path=fm_meta,
            imu_data_root=imu_data_root,
            imu_channel_mode=args.imu_channel_mode,
            verbose=True,
        )
    except Exception as exc:
        print(f"\nFATAL: dataset init failed: {exc}")
        sys.exit(1)

    _sep("SHAPES")
    print(f"FM zarr X shape    : {dataset.fm_shape}  (N x T_depth x D_depth)")
    print(f"IMU sequences shape: {dataset.imu_shape}  (N x T_imu x C_imu)")
    print(f"Paired examples    : {len(dataset)}")

    # -----------------------------------------------------------------------
    # 2. Label balance
    # -----------------------------------------------------------------------
    _sep("LABEL BALANCE")
    labels = dataset.labels_array
    n_stress = int(labels.sum())
    n_total = len(labels)
    print(f"Stress    : {n_stress} ({100*n_stress/n_total:.1f}%)")
    print(f"Non-stress: {n_total - n_stress} ({100*(n_total-n_stress)/n_total:.1f}%)")

    subjects = dataset.subjects
    print(f"\nSubjects present: {len(subjects)}")
    print(f"  {subjects}")

    # -----------------------------------------------------------------------
    # 3. Per-subject coverage
    # -----------------------------------------------------------------------
    _sep("PER-SUBJECT PAIRS")
    meta_df = dataset.pairs_metadata_frame()
    subject_summary = (
        meta_df.groupby("base_subject_id")
        .agg(n_pairs=("pair_index", "count"), n_stress=("label", "sum"))
        .reset_index()
    )
    subject_summary["pct_stress"] = (
        100 * subject_summary["n_stress"] / subject_summary["n_pairs"]
    ).round(1)
    print(subject_summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # 4. Sample pairs
    # -----------------------------------------------------------------------
    _sep(f"SAMPLE PAIRS (n={args.n_samples})")
    sample_indices = np.linspace(0, len(dataset) - 1, args.n_samples, dtype=int)
    all_ok = True
    for i in sample_indices:
        try:
            item = dataset[int(i)]
        except Exception as exc:
            print(f"  [pair {i}] ERROR loading: {exc}")
            all_ok = False
            continue

        depth = item["depth_embedding"]
        imu = item["imu_sequence"]
        label = item["label"].item()
        subj = item["base_subject_id"]
        task = item["task_id"]

        depth_ok = depth.isfinite().all().item()
        imu_ok = imu.isfinite().all().item()
        flag = "" if (depth_ok and imu_ok) else " *** NaN/Inf detected ***"

        print(
            f"  [{i:4d}] subj={subj} task={task:12s} label={label} "
            f"depth={tuple(depth.shape)} imu={tuple(imu.shape)}"
            f"  depth_finite={depth_ok} imu_finite={imu_ok}{flag}"
        )
        if not (depth_ok and imu_ok):
            all_ok = False

    # -----------------------------------------------------------------------
    # 5. Optional: normalizer test
    # -----------------------------------------------------------------------
    if not args.no_normalizer and len(dataset) >= 10:
        _sep("NORMALIZER CHECK (first 80% of pairs as mock train set)")
        n_train = int(0.8 * len(dataset))
        train_pair_indices = list(range(n_train))

        try:
            fm_mean, fm_std = dataset.compute_fm_normalizer(train_pair_indices)
            imu_mean, imu_std = dataset.compute_imu_normalizer(train_pair_indices)
            print(f"FM  mean range : [{fm_mean.min():.4f}, {fm_mean.max():.4f}]")
            print(f"FM  std  range : [{fm_std.min():.4f}, {fm_std.max():.4f}]")
            print(f"IMU mean range : [{imu_mean.min():.4f}, {imu_mean.max():.4f}]")
            print(f"IMU std  range : [{imu_std.min():.4f}, {imu_std.max():.4f}]")
            print("Normalizer computation: OK")
        except Exception as exc:
            print(f"Normalizer computation failed: {exc}")
            all_ok = False

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    _sep("RESULT")
    if all_ok:
        print("All checks passed. Dataset is ready for cross-modal training.")
        print()
        print("Next step: run your training script pointing to these paths.")
        print(f"  --fm-store        {fm_store}")
        print(f"  --fm-meta-csv     {fm_meta}")
        print(f"  --imu-data-root   {imu_data_root}")
        print(f"  --imu-channel-mode {args.imu_channel_mode}")
    else:
        print("Some checks FAILED. Review warnings above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
