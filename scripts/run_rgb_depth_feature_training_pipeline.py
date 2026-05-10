#!/usr/bin/env python3
"""End-to-end RGB/depth feature processing and stress training pipeline.

This launcher intentionally keeps the heavy artifacts repo-local and ignored:
derived Zarr features go under ``processed_rgb_depth_features/`` by default and
model outputs go under ``outputs_rgb_depth/``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


TARGET_TASKS = {"jelly", "count", "baseline", "bad", "stress", "arithmetic", "stroop"}
NON_STRESS_TASKS = {"jelly", "count", "baseline"}
STRESS_TASKS = {"bad", "stress", "arithmetic", "stroop"}
FEATURE_SPECS = {
    "masked_rgb": ("human_masked_sam2_yolo", "rgb_masked", "rgb"),
    "masked_depth": ("human_masked_sam2_yolo", "depth_masked", "depth"),
    "motion_prev_rgb": ("motion_previous", "motion_rgb", "rgb"),
    "motion_prev_depth": ("motion_previous", "motion_depth", "depth"),
    "flow_edge_rgb": ("flow_edge_raft", "flow_edge_rgb", "depth"),
    "flow_edge_depth": ("flow_edge_raft", "flow_edge_depth", "depth"),
}
MODEL_NAMES = ("temporal_pooling_cnn", "small_frame_cnn_transformer")
CONCISE_COLUMNS = [
    "model_name",
    "module_name",
    "window_strategy",
    "input_mode",
    "representation_family",
    "representation_equation",
    "sequence_pooling",
    "sequence_length",
    "optuna_trials",
    "n_folds",
    "person_disjoint_setting",
    "train_balanced_accuracy_mean",
    "train_auc_mean",
    "val_balanced_accuracy_mean",
    "val_auc_mean",
    "test_balanced_accuracy_mean",
    "test_auc_mean",
]


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


@contextmanager
def timer(label: str):
    start = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START {label}", flush=True)
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE  {label} in {format_seconds(elapsed)}", flush=True)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RGB/depth features, windows, folds, and stress models.")
    parser.add_argument(
        "--raw-root",
        default="OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw_zarr2",
        help="Full raw 5 Hz Zarr root.",
    )
    parser.add_argument("--feature-root", default="processed_rgb_depth_features")
    parser.add_argument("--run-root", default="outputs_rgb_depth")
    parser.add_argument("--run-name", default=None, help="Training run folder name. Defaults to timestamp.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "derive",
            "train",
            "validate_raw",
            "derive_masked",
            "derive_motion",
            "derive_flow_edge",
            "build_windows",
            "summarize",
        ],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--view", default="frontal", choices=["frontal", "side"])
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=float, default=5.0)
    parser.add_argument("--features", nargs="*", default=list(FEATURE_SPECS), choices=sorted(FEATURE_SPECS))
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mask-carry-forward-frames", type=int, default=15)
    parser.add_argument("--mask-min-area-fraction", type=float, default=0.01)
    parser.add_argument("--mask-max-area-fraction", type=float, default=0.70)
    parser.add_argument("--flow-lag", type=int, default=5)
    parser.add_argument("--flow-edge-clamp", type=float, default=2.0)
    parser.add_argument("--flow-edge-gamma", type=float, default=0.5)
    parser.add_argument("--depth-flow-mode", choices=["depth_diff_sobel", "raft_depth_pseudo"], default="depth_diff_sobel")
    parser.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "blosclz", "zlib"])
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--frames-per-chunk", type=int, default=150)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--normalizer-max-windows", type=int, default=256)
    parser.add_argument("--max-task-groups", type=int, default=None, help="Debug limit for derive stages.")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None, help="Debug cap after window filtering.")
    parser.add_argument("--smoke-participant", default=None, help="Convenience alias for --participants one participant.")
    parser.add_argument("--skip-training-existing", action="store_true", help="Skip completed model/feature/fold outputs.")
    parser.add_argument(
        "--max-missing-mask-frames-per-window",
        type=int,
        default=None,
        help=(
            "Optional absolute cap for mask_source==0 frames. If omitted, the fraction cap is used. "
            "Set 0 for strict no-missing-mask windows."
        ),
    )
    parser.add_argument(
        "--max-missing-mask-fraction-per-window",
        type=float,
        default=0.50,
        help="Hard drop cap for missing-mask fraction before tier filtering. Default 0.50 matches inspect-tier maximum.",
    )
    parser.add_argument(
        "--min-window-mask-area-mean",
        type=float,
        default=0.01,
        help="Drop windows whose mean mask area fraction is below this value.",
    )
    parser.add_argument(
        "--max-window-mask-area-mean",
        type=float,
        default=0.70,
        help="Drop windows whose mean mask area fraction is above this value.",
    )
    parser.add_argument(
        "--max-window-mask-age-frames",
        type=int,
        default=15,
        help="Drop windows whose carried-forward mask age exceeds this value.",
    )
    parser.add_argument(
        "--allowed-mask-quality-tiers",
        nargs="*",
        default=["clean", "usable"],
        choices=["clean", "usable", "inspect"],
        help="Mask-quality tiers retained for training windows. Production default keeps clean+usable.",
    )
    parser.add_argument(
        "--debug-allow-window-split",
        action="store_true",
        help="Debug only: allow one non-participant-disjoint fold when too few participants are available.",
    )
    return parser.parse_args()


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr<3 in the active environment before running this pipeline.") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"Unsupported zarr version {zarr.__version__}; this pipeline expects zarr<3.")
    return zarr


def task_to_label(task_id: str) -> int:
    if task_id in NON_STRESS_TASKS:
        return 0
    if task_id in STRESS_TASKS:
        return 1
    raise ValueError(f"Unsupported task label: {task_id}")


def canonical_subject_id(subject_id: object) -> str:
    text = str(subject_id).strip()
    lower = text.lower()
    if lower.endswith("v2"):
        return text[:-2]
    return text


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(np.int64)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, 19):
        score = float(balanced_accuracy_score(y_true, apply_threshold(y_score, float(threshold))))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc": safe_auc(y_true, y_score),
    }


def create_grouped_folds(metadata_df: pd.DataFrame, n_splits: int, val_ratio: float, random_seed: int) -> list[FoldSplit]:
    indices = metadata_df.index.to_numpy()
    groups = metadata_df["base_subject_id"].to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} participants after filtering, found {len(np.unique(groups))}.")
    outer = GroupKFold(n_splits=n_splits)
    folds: list[FoldSplit] = []
    for fold_id, (train_val_pos, test_pos) in enumerate(outer.split(indices, metadata_df["label"], groups), start=1):
        train_val_indices = indices[train_val_pos]
        train_val_groups = groups[train_val_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed + fold_id)
        train_pos, val_pos = next(inner.split(train_val_indices, groups=train_val_groups))
        folds.append(
            FoldSplit(
                fold_id=fold_id,
                train_indices=np.sort(train_val_indices[train_pos]),
                val_indices=np.sort(train_val_indices[val_pos]),
                test_indices=np.sort(indices[test_pos]),
            )
        )
    return folds


def create_debug_window_fold(metadata_df: pd.DataFrame, val_ratio: float, random_seed: int) -> list[FoldSplit]:
    indices = metadata_df.index.to_numpy()
    labels = metadata_df["label"].to_numpy()
    train_val, test = train_test_split(
        indices,
        test_size=max(0.2, val_ratio),
        random_state=random_seed + 1,
        stratify=labels if len(np.unique(labels)) == 2 else None,
    )
    train_labels = metadata_df.loc[train_val, "label"].to_numpy()
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_seed + 2,
        stratify=train_labels if len(np.unique(train_labels)) == 2 else None,
    )
    return [FoldSplit(fold_id=1, train_indices=np.sort(train), val_indices=np.sort(val), test_indices=np.sort(test))]


def repo_script(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage_validate_raw(args: argparse.Namespace, feature_root: Path) -> Path:
    report_path = feature_root / "manifests" / "raw_validation_report.csv"
    if report_path.exists() and not args.overwrite:
        print(f"Validation report exists, skipping: {report_path}")
        return report_path
    command = [
        sys.executable,
        repo_script("validate_rgb_depth_raw_zarr.py"),
        "--input-root",
        str(Path(args.raw_root).resolve()),
        "--output-csv",
        str(report_path),
        "--expected-sample-rate-hz",
        str(args.sample_rate_hz),
    ]
    run_command(command)
    return report_path


def derive_common_args(args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        repo_script("preprocess_rgb_depth_derived_zarr.py"),
        "--input-root",
        str(Path(args.raw_root).resolve()),
        "--output-root",
        str(output_root),
        "--views",
        args.view,
        "--sample-rate-hz",
        str(args.sample_rate_hz),
        "--frames-per-chunk",
        str(args.frames_per_chunk),
        "--compressor",
        args.compressor,
        "--compression-level",
        str(args.compression_level),
        "--device",
        args.device,
    ]
    participants = args.participants or ([args.smoke_participant] if args.smoke_participant else None)
    if participants:
        command.extend(["--participants", *participants])
    if args.max_task_groups is not None:
        command.extend(["--max-task-groups", str(args.max_task_groups)])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_progress:
        command.append("--no-progress")
    return command


def stage_derive_masked(args: argparse.Namespace, feature_root: Path) -> Path:
    output_root = feature_root / "human_masked_sam2_yolo"
    if not args.sam2_config or not args.sam2_checkpoint:
        raise SystemExit("--sam2-config and --sam2-checkpoint are required for derive_masked/all/derive.")
    command = derive_common_args(args, output_root)
    command.extend(
        [
            "--representation",
            "human_masked",
            "--allow-missing-masks",
            "--yolo-model",
            args.yolo_model,
            "--sam2-config",
            args.sam2_config,
            "--sam2-checkpoint",
            args.sam2_checkpoint,
            "--mask-carry-forward-frames",
            str(args.mask_carry_forward_frames),
            "--mask-min-area-fraction",
            str(args.mask_min_area_fraction),
            "--mask-max-area-fraction",
            str(args.mask_max_area_fraction),
        ]
    )
    run_command(command)
    return output_root


def stage_derive_motion(args: argparse.Namespace, feature_root: Path) -> Path:
    output_root = feature_root / "motion_previous"
    command = derive_common_args(args, output_root)
    command.extend(
        [
            "--representation",
            "motion_previous",
            "--mask-source-root",
            str((feature_root / "human_masked_sam2_yolo").resolve()),
        ]
    )
    run_command(command)
    return output_root


def stage_derive_flow_edge(args: argparse.Namespace, feature_root: Path) -> Path:
    output_root = feature_root / "flow_edge_raft"
    command = derive_common_args(args, output_root)
    command.extend(
        [
            "--representation",
            "flow_edge_raft",
            "--mask-source-root",
            str((feature_root / "human_masked_sam2_yolo").resolve()),
            "--flow-lag",
            str(args.flow_lag),
            "--flow-edge-clamp",
            str(args.flow_edge_clamp),
            "--flow-edge-gamma",
            str(args.flow_edge_gamma),
            "--depth-flow-mode",
            args.depth_flow_mode,
        ]
    )
    run_command(command)
    return output_root


def zarr_task_group(root, row: pd.Series):
    return root[row["zarr_task_group"]]


def iter_feature_task_rows(zarr, feature_root: Path, features: Iterable[str], view: str):
    for feature_name in features:
        family, array_name, _kind = FEATURE_SPECS[feature_name]
        for store_path in sorted((feature_root / family).glob("*/*.zarr")):
            try:
                root = zarr.open_group(str(store_path), mode="r")
                if "tasks" not in root:
                    continue
                for task_name in sorted(root["tasks"].keys()):
                    task_group = root["tasks"][task_name]
                    attrs = dict(task_group.attrs)
                    task_id = str(attrs.get("task_id", ""))
                    if str(attrs.get("view_type")) != view or task_id not in TARGET_TASKS:
                        continue
                    if array_name not in task_group:
                        continue
                    yield {
                        "feature": feature_name,
                        "family": family,
                        "array_name": array_name,
                        "zarr_path": str(store_path),
                        "zarr_task_group": f"tasks/{task_name}",
                        "group": str(attrs.get("group", "")),
                        "base_subject_id": canonical_subject_id(attrs.get("base_subject_id", "")),
                        "task_id": task_id,
                        "label": int(attrs.get("stress_label", task_to_label(task_id))),
                        "view_type": str(attrs.get("view_type", "")),
                        "num_samples": int(task_group[array_name].shape[0]),
                    }
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping unreadable feature store {store_path}: {type(exc).__name__}: {exc}")


def mask_task_lookup(zarr, masked_root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for store_path in sorted(masked_root.glob("*/*.zarr")):
        try:
            root = zarr.open_group(str(store_path), mode="r")
            for task_name in sorted(root["tasks"].keys()):
                tg = root["tasks"][task_name]
                attrs = dict(tg.attrs)
                key = (canonical_subject_id(attrs.get("base_subject_id", "")), f"tasks/{task_name}")
                lookup[key] = (str(store_path), f"tasks/{task_name}")
        except Exception:
            continue
    return lookup


def mask_window_stats(zarr, masked_cache: dict[str, object], mask_path: str, mask_group: str, start: int, end: int) -> dict:
    if not mask_path:
        return {
            "missing_mask_frames": end - start,
            "missing_mask_fraction": 1.0,
            "mask_direct_fraction": 0.0,
            "mask_carry_fraction": 0.0,
            "mask_area_mean": np.nan,
            "mask_area_median": np.nan,
            "mask_area_std": np.nan,
            "mask_area_jump_p95": np.nan,
            "mask_age_max": np.nan,
            "yolo_box_mean": np.nan,
            "masked_rgb_nonzero_fraction": np.nan,
            "masked_depth_nonzero_fraction": np.nan,
        }
    root = masked_cache.get(mask_path)
    if root is None:
        root = zarr.open_group(mask_path, mode="r")
        masked_cache[mask_path] = root
    tg = root[mask_group]
    mask_source = np.asarray(tg["mask_source"][start:end], dtype=np.int16) if "mask_source" in tg else np.zeros(end - start)
    mask_area = np.asarray(tg["mask_area_fraction"][start:end], dtype=np.float32) if "mask_area_fraction" in tg else np.full(end - start, np.nan)
    mask_age = np.asarray(tg["mask_age_frames"][start:end], dtype=np.float32) if "mask_age_frames" in tg else np.full(end - start, np.nan)
    yolo = np.asarray(tg["yolo_box_count"][start:end], dtype=np.float32) if "yolo_box_count" in tg else np.full(end - start, np.nan)
    area_diffs = np.abs(np.diff(mask_area[np.isfinite(mask_area)]))
    return {
        "missing_mask_frames": int(np.count_nonzero(mask_source == 0)),
        "missing_mask_fraction": float(np.count_nonzero(mask_source == 0)) / float(max(end - start, 1)),
        "mask_direct_fraction": float(np.count_nonzero(mask_source == 1)) / float(max(end - start, 1)),
        "mask_carry_fraction": float(np.count_nonzero(mask_source == 2)) / float(max(end - start, 1)),
        "mask_area_mean": float(np.nanmean(mask_area)) if len(mask_area) else np.nan,
        "mask_area_median": float(np.nanmedian(mask_area)) if len(mask_area) else np.nan,
        "mask_area_std": float(np.nanstd(mask_area)) if len(mask_area) else np.nan,
        "mask_area_jump_p95": float(np.nanpercentile(area_diffs, 95)) if len(area_diffs) else 0.0,
        "mask_age_max": float(np.nanmax(mask_age)) if len(mask_age) else np.nan,
        "yolo_box_mean": float(np.nanmean(yolo)) if len(yolo) else np.nan,
        "masked_rgb_nonzero_fraction": np.nan,
        "masked_depth_nonzero_fraction": np.nan,
    }


def mask_quality_tier_and_reason(stats: dict, args: argparse.Namespace, window: int) -> tuple[str, str]:
    missing_frames = int(stats["missing_mask_frames"])
    missing_fraction = float(stats["missing_mask_fraction"])
    frame_cap = args.max_missing_mask_frames_per_window
    if frame_cap is None:
        frame_cap = int(math.floor(float(args.max_missing_mask_fraction_per_window) * float(window)))
    frame_cap = int(frame_cap)
    fraction_cap = float(args.max_missing_mask_fraction_per_window)
    if missing_frames > frame_cap:
        return "drop", f"missing_mask_frames>{frame_cap}"
    if missing_fraction > fraction_cap:
        return "drop", f"missing_mask_fraction>{fraction_cap:g}"
    area_mean = float(stats["mask_area_mean"]) if pd.notna(stats["mask_area_mean"]) else np.nan
    if not np.isfinite(area_mean):
        return "drop", "mask_area_mean_not_finite"
    if area_mean < float(args.min_window_mask_area_mean):
        return "drop", f"mask_area_mean<{float(args.min_window_mask_area_mean):g}"
    if area_mean > float(args.max_window_mask_area_mean):
        return "drop", f"mask_area_mean>{float(args.max_window_mask_area_mean):g}"
    age_max = float(stats["mask_age_max"]) if pd.notna(stats["mask_age_max"]) else np.nan
    if np.isfinite(age_max) and age_max > float(args.max_window_mask_age_frames):
        return "drop", f"mask_age_max>{int(args.max_window_mask_age_frames)}"

    direct_fraction = float(stats["mask_direct_fraction"])
    area_jump_p95 = float(stats["mask_area_jump_p95"]) if pd.notna(stats["mask_area_jump_p95"]) else np.inf
    rgb_nonzero = float(stats["masked_rgb_nonzero_fraction"]) if pd.notna(stats["masked_rgb_nonzero_fraction"]) else np.nan
    depth_nonzero = float(stats["masked_depth_nonzero_fraction"]) if pd.notna(stats["masked_depth_nonzero_fraction"]) else np.nan
    if np.isfinite(rgb_nonzero) and rgb_nonzero <= 0.001:
        return "drop", "masked_rgb_nonzero_fraction<=0.001"
    if np.isfinite(depth_nonzero) and depth_nonzero <= 0.001:
        return "drop", "masked_depth_nonzero_fraction<=0.001"

    if missing_fraction <= 0.10 and direct_fraction >= 0.70 and age_max <= 10 and area_jump_p95 <= 0.20:
        return "clean", ""
    if missing_fraction <= 0.25 and direct_fraction >= 0.50 and age_max <= 15 and area_jump_p95 <= 0.25:
        return "usable", ""
    if missing_fraction <= 0.50 and direct_fraction >= 0.25 and age_max <= 15 and area_jump_p95 <= 0.35:
        return "inspect", ""
    return (
        "drop",
        "tier_threshold_failed:"
        f"missing={missing_fraction:.3f},direct={direct_fraction:.3f},age={age_max:.1f},jump_p95={area_jump_p95:.3f}",
    )


def stage_build_windows(args: argparse.Namespace, feature_root: Path) -> Path:
    zarr = require_zarr()
    manifest_dir = feature_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "window_manifest.csv"
    included_path = manifest_dir / "included_participants.csv"
    excluded_path = manifest_dir / "excluded_participants.csv"
    counts_path = manifest_dir / "window_filter_summary.csv"
    if manifest_path.exists() and not args.overwrite:
        if window_manifest_is_current(manifest_path, feature_root, args.features):
            print(f"Window manifest exists and is current, skipping: {manifest_path}")
            return manifest_path
        print(f"Window manifest exists but is empty/stale, rebuilding: {manifest_path}")

    window = int(round(args.window_seconds * args.sample_rate_hz))
    stride = int(round(args.stride_seconds * args.sample_rate_hz))
    if window <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive.")
    mask_lookup = mask_task_lookup(zarr, feature_root / "human_masked_sam2_yolo")
    masked_cache: dict[str, object] = {}
    feature_cache: dict[str, object] = {}
    rows: list[dict] = []
    count_rows: list[dict] = []
    task_rows = list(iter_feature_task_rows(zarr, feature_root, args.features, args.view))
    iterator = tqdm(task_rows, desc="Building feature windows", unit="task") if tqdm and not args.no_progress else task_rows
    for task_row in iterator:
        n = int(task_row["num_samples"])
        total = 0
        dropped_short = 0
        dropped_missing_mask = 0
        dropped_mask_area = 0
        dropped_mask_age = 0
        dropped_tier = 0
        dropped_other_mask_quality = 0
        tier_counts = {"clean": 0, "usable": 0, "inspect": 0, "drop": 0}
        kept = 0
        mask_key = (task_row["base_subject_id"], task_row["zarr_task_group"])
        mask_path, mask_group = mask_lookup.get(mask_key, ("", ""))
        if n < window:
            dropped_short = 1
        else:
            for start in range(0, n - window + 1, stride):
                end = start + window
                total += 1
                stats = mask_window_stats(zarr, masked_cache, mask_path, mask_group, start, end)
                quality_tier, reject_reason = mask_quality_tier_and_reason(stats, args, window)
                tier_counts[quality_tier] = tier_counts.get(quality_tier, 0) + 1
                if quality_tier not in set(args.allowed_mask_quality_tiers):
                    reject_reason = reject_reason or f"mask_quality_tier_not_allowed:{quality_tier}"
                if reject_reason:
                    if reject_reason.startswith("missing_mask"):
                        dropped_missing_mask += 1
                    elif reject_reason.startswith("mask_area"):
                        dropped_mask_area += 1
                    elif reject_reason.startswith("mask_age"):
                        dropped_mask_age += 1
                    elif reject_reason.startswith("mask_quality_tier"):
                        dropped_tier += 1
                    else:
                        dropped_other_mask_quality += 1
                    continue
                timestamps_root = feature_cache.get(task_row["zarr_path"])
                if timestamps_root is None:
                    timestamps_root = zarr.open_group(task_row["zarr_path"], mode="r")
                    feature_cache[task_row["zarr_path"]] = timestamps_root
                tg = timestamps_root[task_row["zarr_task_group"]]
                target_ts = np.asarray(tg["target_timestamps"][start:end], dtype=np.float64)
                rows.append(
                    {
                        **task_row,
                        "window_start": int(start),
                        "window_end": int(end),
                        "window_num_frames": int(window),
                        "window_stride_frames": int(stride),
                        "window_start_timestamp": float(target_ts[0]),
                        "window_end_timestamp": float(target_ts[-1]),
                        "mask_zarr_path": mask_path,
                        "mask_zarr_task_group": mask_group,
                        "mask_quality_tier": quality_tier,
                        "mask_reject_reason": "",
                        **stats,
                    }
                )
                kept += 1
        count_rows.append(
            {
                **task_row,
                "candidate_windows": total,
                "kept_windows": kept,
                "dropped_short_task": dropped_short,
                "dropped_missing_mask_windows": dropped_missing_mask,
                "dropped_mask_area_windows": dropped_mask_area,
                "dropped_mask_age_windows": dropped_mask_age,
                "dropped_mask_quality_tier_windows": dropped_tier,
                "dropped_other_mask_quality_windows": dropped_other_mask_quality,
                "clean_windows": tier_counts.get("clean", 0),
                "usable_windows": tier_counts.get("usable", 0),
                "inspect_windows": tier_counts.get("inspect", 0),
                "drop_tier_windows": tier_counts.get("drop", 0),
                "allowed_mask_quality_tiers": " ".join(args.allowed_mask_quality_tiers),
                "max_missing_mask_frames_per_window": (
                    int(args.max_missing_mask_frames_per_window)
                    if args.max_missing_mask_frames_per_window is not None
                    else int(math.floor(float(args.max_missing_mask_fraction_per_window) * float(window)))
                ),
                "max_missing_mask_fraction_per_window": float(args.max_missing_mask_fraction_per_window),
                "min_window_mask_area_mean": float(args.min_window_mask_area_mean),
                "max_window_mask_area_mean": float(args.max_window_mask_area_mean),
                "max_window_mask_age_frames": int(args.max_window_mask_age_frames),
            }
        )

    full_df = pd.DataFrame(rows)
    pd.DataFrame(count_rows).to_csv(counts_path, index=False)
    if full_df.empty:
        pd.DataFrame(
            columns=[
                "base_subject_id",
                "keep",
                "reason",
                "num_windows",
                "label_0_windows",
                "label_1_windows",
                "tasks",
            ]
        ).to_csv(included_path, index=False)
        pd.DataFrame(
            [
                {
                    "base_subject_id": "",
                    "keep": False,
                    "reason": (
                        "no_windows_after_filtering; check "
                        f"{counts_path} for dropped_short_task and dropped_missing_mask_windows"
                    ),
                    "num_windows": 0,
                    "label_0_windows": 0,
                    "label_1_windows": 0,
                    "tasks": "",
                }
            ]
        ).to_csv(excluded_path, index=False)
        raise SystemExit(
            "No windows were generated after filtering. "
            f"Wrote diagnostics to {counts_path}. "
            "To relax production filtering, adjust --max-missing-mask-fraction-per-window, "
            "--min-window-mask-area-mean, --max-window-mask-area-mean, and --max-window-mask-age-frames."
        )
    full_df = full_df.sort_values(["feature", "base_subject_id", "view_type", "task_id", "window_start_timestamp"]).reset_index(drop=True)

    participant_rows = []
    for subject, group_df in full_df.groupby("base_subject_id"):
        labels = sorted(group_df["label"].unique().tolist())
        keep = labels == [0, 1]
        participant_rows.append(
            {
                "base_subject_id": subject,
                "keep": bool(keep),
                "reason": "included" if keep else f"missing_label_{sorted(set([0, 1]) - set(labels))}",
                "num_windows": int(len(group_df)),
                "label_0_windows": int((group_df["label"] == 0).sum()),
                "label_1_windows": int((group_df["label"] == 1).sum()),
                "tasks": ",".join(sorted(group_df["task_id"].unique())),
            }
        )
    participant_df = pd.DataFrame(participant_rows).sort_values("base_subject_id")
    included = participant_df[participant_df["keep"]].copy()
    excluded = participant_df[~participant_df["keep"]].copy()
    keep_subjects = set(included["base_subject_id"].astype(str))
    final_df = full_df[full_df["base_subject_id"].astype(str).isin(keep_subjects)].copy()
    if args.max_windows is not None:
        final_df = final_df.groupby("feature", group_keys=False).head(int(args.max_windows)).copy()
    final_df = final_df.reset_index(drop=True)
    final_df.insert(0, "window_id", np.arange(len(final_df), dtype=np.int64))
    final_df.to_csv(manifest_path, index=False)
    included.to_csv(included_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    print(f"Wrote window manifest: {manifest_path}")
    print(f"Windows retained: {len(final_df)}")
    print(f"Included participants: {len(included)}; excluded participants: {len(excluded)}")
    return manifest_path


def read_manifest_head(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def newest_feature_metadata_mtime(feature_root: Path, features: Iterable[str]) -> float:
    mtimes = []
    for feature in features:
        family = FEATURE_SPECS[feature][0]
        for path in (feature_root / family).glob("*_metadata.csv"):
            mtimes.append(path.stat().st_mtime)
        for path in (feature_root / family).glob("*/*.zarr"):
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def window_manifest_is_current(manifest_path: Path, feature_root: Path, features: Iterable[str]) -> bool:
    manifest_df = read_manifest_head(manifest_path)
    if manifest_df.empty:
        return False
    if "feature" not in manifest_df:
        return False
    missing_features = set(features) - set(manifest_df["feature"].astype(str).unique())
    if missing_features:
        return False
    return manifest_path.stat().st_mtime >= newest_feature_metadata_mtime(feature_root, features)


def import_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
    except ImportError as exc:
        raise SystemExit("Training requires PyTorch in the active environment.") from exc
    return torch, F, nn, DataLoader, Dataset, Subset, WeightedRandomSampler


def build_models(nn):
    class FrameEncoder(nn.Module):
        def __init__(self, in_channels: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.GELU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.project = nn.Linear(96, hidden_dim)

        def forward(self, x):
            b, t, c, h, w = x.shape
            y = self.net(x.reshape(b * t, c, h, w)).flatten(1)
            y = self.project(y)
            return y.reshape(b, t, -1)

    class TemporalPoolingCNN(nn.Module):
        def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.2):
            super().__init__()
            self.encoder = FrameEncoder(in_channels, hidden_dim)
            self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 2))

        def forward(self, x):
            tokens = self.encoder(x)
            weights = self.attention(tokens).squeeze(-1).softmax(dim=1).unsqueeze(-1)
            attended = (weights * tokens).sum(dim=1)
            mean = tokens.mean(dim=1)
            return self.classifier(torch_cat([mean, attended], dim=1))

    class SmallFrameCNNTransformer(nn.Module):
        def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.2):
            super().__init__()
            self.encoder = FrameEncoder(in_channels, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

        def forward(self, x):
            tokens = self.transformer(self.encoder(x))
            return self.classifier(tokens.mean(dim=1))

    return TemporalPoolingCNN, SmallFrameCNNTransformer


def torch_cat(items, dim: int):
    import torch

    return torch.cat(items, dim=dim)


class WindowDatasetBase:
    pass


def make_window_dataset_class(torch, Dataset, zarr):
    class RGBDepthWindowDataset(Dataset):
        def __init__(self, rows: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
            self.rows = rows.reset_index(drop=True)
            self.mean = mean
            self.std = std
            self._roots: dict[str, object] = {}

        def __len__(self) -> int:
            return len(self.rows)

        def _root(self, path: str):
            root = self._roots.get(path)
            if root is None:
                root = zarr.open_group(path, mode="r")
                self._roots[path] = root
            return root

        def load_array(self, row: pd.Series) -> np.ndarray:
            root = self._root(str(row["zarr_path"]))
            tg = root[str(row["zarr_task_group"])]
            data = np.asarray(tg[str(row["array_name"])][int(row["window_start"]) : int(row["window_end"])])
            if data.ndim == 3:
                data = data[..., None]
            data = np.moveaxis(data, -1, 1).astype(np.float32)
            feature = str(row["feature"])
            if feature in {"masked_rgb"}:
                data = data / 255.0
            elif feature == "motion_prev_rgb":
                data = data / 255.0
            if self.mean is not None and self.std is not None:
                data = (data - self.mean) / self.std
            return data.astype(np.float32)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows.iloc[int(idx)]
            return {
                "x": torch.from_numpy(self.load_array(row)),
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "window_id": torch.tensor(int(row["window_id"]), dtype=torch.long),
            }

    return RGBDepthWindowDataset


def compute_normalizer(dataset, indices: np.ndarray, max_windows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = np.asarray(indices, dtype=np.int64)
    if max_windows > 0 and len(chosen) > max_windows:
        chosen = np.sort(rng.choice(chosen, size=max_windows, replace=False))
    sums = None
    sq_sums = None
    count = 0
    iterator = tqdm(chosen, desc="Fold-local normalization", unit="window", leave=False) if tqdm else chosen
    for idx in iterator:
        x = dataset.load_array(dataset.rows.iloc[int(idx)])
        if sums is None:
            sums = np.zeros((1, x.shape[1], 1, 1), dtype=np.float64)
            sq_sums = np.zeros_like(sums)
        sums += x.sum(axis=(0, 2, 3), keepdims=True)
        sq_sums += np.square(x, dtype=np.float64).sum(axis=(0, 2, 3), keepdims=True)
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = sums / max(count, 1)
    var = (sq_sums / max(count, 1)) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def collate_batch(batch: list[dict]) -> dict:
    import torch

    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "window_id": torch.stack([item["window_id"] for item in batch], dim=0),
    }


def weighted_sampler(labels: np.ndarray, WeightedRandomSampler):
    import torch

    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0.0] = 1.0
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)


def collect_scores(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    labels = []
    scores = []
    window_ids = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            labels.append(batch["label"].cpu().numpy())
            scores.append(prob.detach().cpu().numpy())
            window_ids.append(batch["window_id"].cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(window_ids)


def train_one_fold(args: argparse.Namespace, feature_df: pd.DataFrame, fold: FoldSplit, feature: str, model_name: str, run_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    torch, F, nn, DataLoader, Dataset, Subset, WeightedRandomSampler = import_torch()
    zarr = require_zarr()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device
    torch.manual_seed(args.random_seed + fold.fold_id)
    np.random.seed(args.random_seed + fold.fold_id)
    RGBDepthWindowDataset = make_window_dataset_class(torch, Dataset, zarr)
    base_dataset = RGBDepthWindowDataset(feature_df)
    mean, std = compute_normalizer(base_dataset, fold.train_indices, args.normalizer_max_windows, args.random_seed + fold.fold_id)
    dataset = RGBDepthWindowDataset(feature_df, mean=mean, std=std)
    train_y = feature_df.iloc[fold.train_indices]["label"].to_numpy(dtype=np.int64)
    sampler = weighted_sampler(train_y, WeightedRandomSampler)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "collate_fn": collate_batch}
    train_loader = DataLoader(Subset(dataset, fold.train_indices.tolist()), sampler=sampler, **loader_kwargs)
    train_eval_loader = DataLoader(Subset(dataset, fold.train_indices.tolist()), shuffle=False, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, fold.val_indices.tolist()), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(dataset, fold.test_indices.tolist()), shuffle=False, **loader_kwargs)
    channels = 3 if FEATURE_SPECS[feature][2] == "rgb" else 1
    TemporalPoolingCNN, SmallFrameCNNTransformer = build_models(nn)
    model = (TemporalPoolingCNN if model_name == "temporal_pooling_cnn" else SmallFrameCNNTransformer)(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_val = -np.inf
    best_threshold = 0.5
    history_rows = []
    epoch_iter = range(1, args.epochs + 1)
    if tqdm and not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc=f"{model_name}/{feature}/fold{fold.fold_id}", unit="epoch", leave=False)
    for epoch in epoch_iter:
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y_val, val_scores, _ = collect_scores(model, val_loader, device)
        threshold, _ = find_best_threshold(y_val, val_scores)
        val_metrics = compute_metrics(y_val, apply_threshold(val_scores, threshold), val_scores)
        epoch_loss = float(np.mean(losses)) if losses else np.nan
        history_rows.append({"epoch": epoch, "loss": epoch_loss, "threshold": threshold, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if tqdm and not args.no_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                {
                    "loss": f"{epoch_loss:.4f}" if np.isfinite(epoch_loss) else "nan",
                    "val_ba": f"{val_metrics['balanced_accuracy']:.3f}",
                    "val_auc": f"{val_metrics['auc']:.3f}" if np.isfinite(val_metrics["auc"]) else "nan",
                    "thr": f"{threshold:.2f}",
                }
            )
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    fold_dir = run_dir / model_name / feature / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature": feature,
            "model_name": model_name,
            "mean": mean,
            "std": std,
            "config": vars(args),
        },
        fold_dir / "model.pt",
    )
    pd.DataFrame(history_rows).to_csv(fold_dir / "history.csv", index=False)
    metric_rows = []
    pred_rows = []
    for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_score, window_ids = collect_scores(model, loader, device)
        y_pred = apply_threshold(y_score, best_threshold)
        metrics = compute_metrics(y_true, y_pred, y_score)
        metric_rows.append(
            {
                "model_name": model_name,
                "feature": feature,
                "fold_id": fold.fold_id,
                "split": split,
                "threshold": best_threshold,
                **metrics,
            }
        )
        for window_id, truth, score, pred in zip(window_ids, y_true, y_score, y_pred):
            pred_rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "fold_id": fold.fold_id,
                    "split": split,
                    "window_id": int(window_id),
                    "label": int(truth),
                    "score": float(score),
                    "prediction": int(pred),
                    "threshold": float(best_threshold),
                }
            )
    pd.DataFrame(pred_rows).to_csv(fold_dir / "predictions.csv", index=False)
    return metric_rows, pd.DataFrame(pred_rows)


def save_fold_assignments(df: pd.DataFrame, folds: list[FoldSplit], output_path: Path) -> None:
    rows = []
    for fold in folds:
        for split, indices in [("train", fold.train_indices), ("val", fold.val_indices), ("test", fold.test_indices)]:
            subjects = sorted(df.iloc[indices]["base_subject_id"].astype(str).unique())
            rows.append({"fold_id": fold.fold_id, "split": split, "base_subject_ids": ",".join(subjects), "num_subjects": len(subjects), "num_windows": len(indices)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def summarize_metrics(
    metric_df: pd.DataFrame,
    run_dir: Path,
    n_folds: int,
    sequence_length: int,
    person_disjoint_setting: str,
) -> pd.DataFrame:
    rows = []
    for (model_name, feature), group in metric_df.groupby(["model_name", "feature"]):
        row = {
            "model_name": model_name,
            "module_name": "RGBDepth-Torch",
            "window_strategy": "30s window / 15s overlap",
            "input_mode": feature,
            "representation_family": "RGBDepth single representation",
            "representation_equation": FEATURE_SPECS[feature][1],
            "sequence_pooling": model_name,
            "sequence_length": sequence_length,
            "optuna_trials": 0,
            "n_folds": n_folds,
            "person_disjoint_setting": person_disjoint_setting,
        }
        for split in ["train", "val", "test"]:
            split_df = group[group["split"] == split]
            row[f"{split}_balanced_accuracy_mean"] = float(split_df["balanced_accuracy"].mean())
            row[f"{split}_balanced_accuracy_std"] = float(split_df["balanced_accuracy"].std(ddof=0))
            row[f"{split}_f1_macro_mean"] = float(split_df["f1_macro"].mean())
            row[f"{split}_auc_mean"] = float(split_df["auc"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    summary[CONCISE_COLUMNS].to_csv(run_dir / "all_window_mode_results_concise.csv", index=False)
    return summary


def stage_train(args: argparse.Namespace, feature_root: Path, run_root: Path) -> Path:
    manifest_path = feature_root / "manifests" / "window_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing window manifest: {manifest_path}. Run --stage build_windows first.")
    run_name = args.run_name or datetime.now().strftime("rgb_depth_%Y%m%d_%H%M%S")
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    df = pd.read_csv(manifest_path)
    metric_rows = []
    all_pred_frames = []
    sequence_length = int(round(args.window_seconds * args.sample_rate_hz))
    used_debug_window_split = False
    combo_iter = [(feature, model) for feature in args.features for model in args.models]
    if tqdm and not args.no_progress:
        combo_iter = tqdm(combo_iter, desc="Training feature/model combos", unit="combo")
    for feature, model_name in combo_iter:
        feature_df = df[df["feature"] == feature].reset_index(drop=True)
        if feature_df.empty:
            print(f"Skipping {feature}: no windows")
            continue
        unique_groups = feature_df["base_subject_id"].nunique()
        if unique_groups < args.n_splits and args.debug_allow_window_split:
            print(
                f"DEBUG window split for {feature}/{model_name}: "
                f"{unique_groups} participant(s) is too few for GroupKFold(n_splits={args.n_splits})."
            )
            folds = create_debug_window_fold(feature_df, args.val_ratio, args.random_seed)
            used_debug_window_split = True
        else:
            folds = create_grouped_folds(feature_df, args.n_splits, args.val_ratio, args.random_seed)
        if args.max_folds is not None:
            folds = folds[: int(args.max_folds)]
        save_fold_assignments(feature_df, folds, run_dir / model_name / feature / "fold_assignments.csv")
        for fold in folds:
            fold_metric_path = run_dir / model_name / feature / f"fold_{fold.fold_id}" / "predictions.csv"
            if fold_metric_path.exists() and args.skip_training_existing and not args.overwrite:
                continue
            rows, preds = train_one_fold(args, feature_df, fold, feature, model_name, run_dir)
            metric_rows.extend(rows)
            all_pred_frames.append(preds)
    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        raise SystemExit("No training metrics were produced.")
    metric_df.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    if all_pred_frames:
        pd.concat(all_pred_frames, ignore_index=True).to_csv(run_dir / "fold_predictions.csv", index=False)
    split_setting = (
        "DEBUG window-level split; not participant-disjoint"
        if used_debug_window_split
        else "GroupKFold(base_subject_id) with GroupShuffleSplit validation"
    )
    summarize_metrics(
        metric_df,
        run_dir,
        int(metric_df["fold_id"].nunique()),
        sequence_length,
        split_setting,
    )
    print(f"Wrote training run: {run_dir}")
    return run_dir


def stage_summarize(args: argparse.Namespace, run_root: Path) -> None:
    run_dirs = [run_root / args.run_name] if args.run_name else sorted([p for p in run_root.iterdir() if p.is_dir()])
    rows = []
    for run_dir in run_dirs:
        concise = run_dir / "all_window_mode_results_concise.csv"
        if concise.exists():
            df = pd.read_csv(concise)
            df.insert(0, "run_dir", str(run_dir))
            rows.append(df)
    if not rows:
        raise SystemExit(f"No concise result CSVs found under {run_root}")
    output = run_root / "all_window_mode_results_concise.csv"
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)
    print(f"Wrote aggregate concise results: {output}")


def main() -> None:
    args = parse_args()
    if args.smoke_participant and not args.participants:
        args.participants = [args.smoke_participant]
    feature_root = Path(args.feature_root).resolve()
    run_root = Path(args.run_root).resolve()
    (feature_root / "manifests").mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    stage_sequence = {
        "all": ["validate_raw", "derive_masked", "derive_motion", "derive_flow_edge", "build_windows", "train", "summarize"],
        "derive": ["derive_masked", "derive_motion", "derive_flow_edge"],
        "train": ["build_windows", "train", "summarize"],
    }.get(args.stage, [args.stage])

    for stage in stage_sequence:
        with timer(stage):
            if stage == "validate_raw":
                stage_validate_raw(args, feature_root)
            elif stage == "derive_masked":
                stage_derive_masked(args, feature_root)
            elif stage == "derive_motion":
                stage_derive_motion(args, feature_root)
            elif stage == "derive_flow_edge":
                stage_derive_flow_edge(args, feature_root)
            elif stage == "build_windows":
                stage_build_windows(args, feature_root)
            elif stage == "train":
                stage_train(args, feature_root, run_root)
            elif stage == "summarize":
                stage_summarize(args, run_root)
            else:
                raise ValueError(stage)


if __name__ == "__main__":
    main()
