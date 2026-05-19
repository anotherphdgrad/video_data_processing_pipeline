#!/usr/bin/env python3
"""
Paired depth-IMU dataset for cross-modal representation learning.

Loads FM depth embeddings (from a framewise zarr store) and IMU ACC sequences
(directly from CSV files via IMUStressWindowDataset — no shared zarr cache
needed) and aligns them into paired training examples.

Alignment strategy
------------------
Both pipelines window the same recordings with identical parameters
(30 s window / 15 s overlap by default).  Windows are paired by:

    (base_subject_id, task_id, window_position_within_task)

For each (subject, task) group, FM windows are ordered by source_index and
IMU windows are ordered by window_start timestamp.  The N-th FM window is
paired with the N-th IMU window.  Groups with mismatched counts use the
smaller count (excess windows dropped with a warning).

Returned items
--------------
Each __getitem__ returns a dict:
    depth_embedding : FloatTensor (T_depth, D_depth)
    imu_sequence    : FloatTensor (T_imu, C_imu)
    label           : LongTensor scalar
    base_subject_id : str
    task_id         : str
    fm_source_index : int
    imu_dataset_index : int   — index into IMUStressWindowDataset.windows
    pair_index      : int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from imu_dataset import IMUStressWindowDataset, build_imu_dataset


# ---------------------------------------------------------------------------
# Helpers for FM zarr loading (self-contained)
# ---------------------------------------------------------------------------

def _require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr<3: pip install 'zarr<3'") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"zarr {zarr.__version__} not supported; need zarr<3")
    return zarr


# ---------------------------------------------------------------------------
# Alignment index entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedWindow:
    pair_index: int
    base_subject_id: str
    task_id: str
    label: int
    fm_source_index: int       # row in FM zarr X array
    fm_window_id: int          # window_id stored in FM zarr
    imu_dataset_index: int     # index into IMUStressWindowDataset.windows


# ---------------------------------------------------------------------------
# FM zarr index loader
# ---------------------------------------------------------------------------

def _load_fm_index(store_path: Path, metadata_csv_path: Path) -> pd.DataFrame:
    """
    Returns DataFrame: source_index, window_id, base_subject_id, task_id, label
    """
    zarr = _require_zarr()
    group = zarr.open_group(str(store_path), mode="r")
    n = int(group["X"].shape[0])
    fm_df = pd.DataFrame({
        "source_index": np.arange(n, dtype=np.int64),
        "window_id": np.asarray(group["window_id"][:], dtype=np.int64),
        "base_subject_id": np.asarray(group["base_subject_id"][:]).astype(str),
        "label": np.asarray(group["y"][:], dtype=np.int64),
    })
    if not metadata_csv_path.exists():
        raise FileNotFoundError(
            f"FM metadata CSV not found: {metadata_csv_path}\n"
            "Required to obtain task_id for alignment."
        )
    meta = pd.read_csv(metadata_csv_path)
    if "task_id" not in meta.columns:
        raise ValueError(
            f"FM metadata CSV missing 'task_id' column.\n"
            f"Available: {meta.columns.tolist()}"
        )
    if "window_id" not in meta.columns:
        raise ValueError(
            f"FM metadata CSV missing 'window_id' column.\n"
            f"Available: {meta.columns.tolist()}"
        )
    meta_slim = meta[["window_id", "task_id"]].drop_duplicates("window_id")
    fm_df = fm_df.merge(meta_slim, on="window_id", how="left")
    n_missing = fm_df["task_id"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing}/{len(fm_df)} FM windows have no task_id — excluded from pairing.")
        fm_df = fm_df.dropna(subset=["task_id"])
    return fm_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IMU index loader — direct from IMUStressWindowDataset, no zarr
# ---------------------------------------------------------------------------

def _load_imu_index(imu_dataset: IMUStressWindowDataset) -> pd.DataFrame:
    """
    Returns DataFrame: imu_dataset_index, base_subject_id, task_id, label, window_start
    Built directly from IMUStressWindowDataset.windows — no zarr store needed.
    """
    rows = [
        {
            "imu_dataset_index": idx,
            "base_subject_id": str(w.base_subject_id),
            "task_id": str(w.task_id),
            "label": int(w.label),
            "window_start": float(w.window_start),
        }
        for idx, w in enumerate(imu_dataset.windows)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _build_alignment(
    fm_df: pd.DataFrame,
    imu_df: pd.DataFrame,
) -> tuple[list[PairedWindow], dict]:
    pairs: list[PairedWindow] = []
    stats = {
        "n_subject_task_groups": 0,
        "n_groups_with_mismatch": 0,
        "n_groups_imu_only": 0,
        "n_groups_fm_only": 0,
        "n_pairs_total": 0,
        "n_fm_windows_total": len(fm_df),
        "n_imu_windows_total": len(imu_df),
        "subjects_covered": [],
    }
    fm_groups = fm_df.groupby(["base_subject_id", "task_id"])
    imu_groups = imu_df.groupby(["base_subject_id", "task_id"])
    fm_keys = set(fm_groups.groups.keys())
    imu_keys = set(imu_groups.groups.keys())
    all_keys = fm_keys | imu_keys
    stats["n_subject_task_groups"] = len(all_keys)
    stats["n_groups_imu_only"] = len(imu_keys - fm_keys)
    stats["n_groups_fm_only"] = len(fm_keys - imu_keys)
    covered: set[str] = set()
    for key in sorted(all_keys):
        if key not in fm_keys or key not in imu_keys:
            continue
        fm_group = fm_groups.get_group(key).sort_values("source_index").reset_index(drop=True)
        imu_group = imu_groups.get_group(key).sort_values("window_start").reset_index(drop=True)
        n_fm, n_imu = len(fm_group), len(imu_group)
        if n_fm != n_imu:
            stats["n_groups_with_mismatch"] += 1
        subj, task = key
        covered.add(subj)
        for pos in range(min(n_fm, n_imu)):
            fm_row = fm_group.iloc[pos]
            imu_row = imu_group.iloc[pos]
            if int(fm_row["label"]) != int(imu_row["label"]):
                print(f"  WARNING: label mismatch ({subj}, {task}, pos={pos}): "
                      f"FM={fm_row['label']}, IMU={imu_row['label']} — using FM label")
            pairs.append(PairedWindow(
                pair_index=len(pairs),
                base_subject_id=subj,
                task_id=task,
                label=int(fm_row["label"]),
                fm_source_index=int(fm_row["source_index"]),
                fm_window_id=int(fm_row["window_id"]),
                imu_dataset_index=int(imu_row["imu_dataset_index"]),
            ))
    stats["n_pairs_total"] = len(pairs)
    stats["subjects_covered"] = sorted(covered)
    return pairs, stats


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class PairedDepthIMUDataset(Dataset):
    """
    Paired FM depth embeddings + IMU ACC sequences for cross-modal training.

    IMU sequences are read directly from CSV files via IMUStressWindowDataset.
    No shared zarr cache is required.

    Parameters
    ----------
    fm_store_path : Path
        FM zarr store, e.g. .../dinov2/motion_prev_depth.zarr
    fm_metadata_csv_path : Path
        Sibling metadata CSV containing task_id column.
    imu_data_root : Path | str
        Root directory of IMU CSV files (contains population subdirs with
        left_*_acc.csv files).  Same path used by the FLIRT-Torch baseline.
    imu_channel_mode : str
        Sequence transformation mode.  Must match the mode used to train
        the IMU teacher.  Default: 'raw_absdelta' (matches FLIRT-Torch best).
    fm_mean / fm_std : np.ndarray or None
        Population normalization for FM embeddings (fit on train fold only).
    imu_mean / imu_std : np.ndarray or None
        Population normalization for IMU sequences (fit on train fold only).
    pair_indices : sequence of int or None
        Restrict to this subset of pair indices (e.g. one fold's split).
    verbose : bool
    """

    def __init__(
        self,
        fm_store_path: Path | str,
        fm_metadata_csv_path: Path | str,
        imu_data_root: Path | str,
        imu_channel_mode: str = "raw_absdelta",
        imu_window_seconds: float = 30.0,
        imu_overlap_seconds: float = 15.0,
        imu_sample_rate_hz: float = 64.0,
        imu_raw_sample_rate_hz: float = 32.0,
        fm_mean: np.ndarray | None = None,
        fm_std: np.ndarray | None = None,
        imu_mean: np.ndarray | None = None,
        imu_std: np.ndarray | None = None,
        pair_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> None:
        self.fm_store_path = Path(fm_store_path)
        self.fm_metadata_csv_path = Path(fm_metadata_csv_path)
        self.imu_data_root = Path(imu_data_root)
        self.imu_channel_mode = imu_channel_mode
        self.fm_mean = np.asarray(fm_mean, dtype=np.float32) if fm_mean is not None else None
        self.fm_std = np.asarray(fm_std, dtype=np.float32) if fm_std is not None else None
        self.imu_mean = np.asarray(imu_mean, dtype=np.float32) if imu_mean is not None else None
        self.imu_std = np.asarray(imu_std, dtype=np.float32) if imu_std is not None else None
        self._fm_group = None  # lazy zarr handle

        if verbose:
            print(f"[PairedDepthIMUDataset] Loading FM index from {self.fm_store_path.name} ...")
        fm_df = _load_fm_index(self.fm_store_path, self.fm_metadata_csv_path)

        if verbose:
            print(f"[PairedDepthIMUDataset] Loading IMU dataset from {self.imu_data_root} ...")
        self.imu_dataset = build_imu_dataset(
            data_root=imu_data_root,
            channel_mode=imu_channel_mode,
            window_seconds=imu_window_seconds,
            overlap_seconds=imu_overlap_seconds,
            sample_rate_hz=imu_sample_rate_hz,
            raw_sample_rate_hz=imu_raw_sample_rate_hz,
        )

        if verbose:
            print(f"[PairedDepthIMUDataset] Building alignment index ...")
        imu_df = _load_imu_index(self.imu_dataset)
        all_pairs, stats = _build_alignment(fm_df, imu_df)

        if verbose:
            self._print_stats(stats, fm_df, imu_df)

        if not all_pairs:
            raise ValueError(
                "No paired windows found. Check that base_subject_id and task_id "
                "match between the FM zarr store and the IMU CSV data."
            )

        self._all_pairs = all_pairs
        self._pairs = (
            [p for p in all_pairs if p.pair_index in set(pair_indices)]
            if pair_indices is not None else all_pairs
        )
        self.stats = stats

        zarr = _require_zarr()
        fm_group = zarr.open_group(str(self.fm_store_path), mode="r")
        self.fm_shape: tuple[int, int, int] = tuple(int(v) for v in fm_group["X"].shape)
        self.imu_feature_dim: int = self.imu_dataset.sequence_feature_dim
        self.imu_seq_len: int = self.imu_dataset.window_size
        self.imu_shape: tuple[int, int, int] = (
            len(self.imu_dataset), self.imu_seq_len, self.imu_feature_dim
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_pairs_total(self) -> int:
        return len(self._all_pairs)

    @property
    def subjects(self) -> list[str]:
        return self.stats["subjects_covered"]

    @property
    def labels_array(self) -> np.ndarray:
        return np.asarray([p.label for p in self._pairs], dtype=np.int64)

    @property
    def base_subject_ids_array(self) -> np.ndarray:
        return np.asarray([p.base_subject_id for p in self._pairs])

    def pairs_metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "pair_index": p.pair_index,
                "base_subject_id": p.base_subject_id,
                "task_id": p.task_id,
                "label": p.label,
                "fm_source_index": p.fm_source_index,
                "fm_window_id": p.fm_window_id,
                "imu_dataset_index": p.imu_dataset_index,
            }
            for p in self._pairs
        ])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self._pairs[index]

        # Lazy-open FM zarr handle
        zarr = _require_zarr()
        if self._fm_group is None:
            self._fm_group = zarr.open_group(str(self.fm_store_path), mode="r")

        # FM depth embedding (T_depth, D_depth)
        fm_x = np.asarray(self._fm_group["X"][pair.fm_source_index], dtype=np.float32)
        if self.fm_mean is not None and self.fm_std is not None:
            fm_x = (fm_x - self.fm_mean[None, :]) / self.fm_std[None, :]

        # IMU sequence — read directly from IMUStressWindowDataset
        imu_x = self.imu_dataset.get_sequence_array(pair.imu_dataset_index).astype(np.float32)
        if self.imu_mean is not None and self.imu_std is not None:
            imu_x = (imu_x - self.imu_mean[None, :]) / self.imu_std[None, :]

        return {
            "depth_embedding": torch.from_numpy(fm_x),
            "imu_sequence": torch.from_numpy(imu_x),
            "label": torch.tensor(pair.label, dtype=torch.long),
            "base_subject_id": pair.base_subject_id,
            "task_id": pair.task_id,
            "fm_source_index": pair.fm_source_index,
            "imu_dataset_index": pair.imu_dataset_index,
            "pair_index": pair.pair_index,
        }

    # ------------------------------------------------------------------
    # Normalization helpers (fit on train fold only)
    # ------------------------------------------------------------------

    def compute_fm_normalizer(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        zarr = _require_zarr()
        group = zarr.open_group(str(self.fm_store_path), mode="r")
        x_array = group["X"]
        pair_map = {p.pair_index: p for p in self._all_pairs}
        source_indices = sorted(pair_map[i].fm_source_index for i in pair_indices if i in pair_map)
        D = self.fm_shape[2]
        sum_x = np.zeros(D, dtype=np.float64)
        sum_x2 = np.zeros(D, dtype=np.float64)
        total = 0
        for start in range(0, len(source_indices), 128):
            batch = np.array(source_indices[start:start + 128], dtype=np.int64)
            x = np.asarray(
                x_array.get_orthogonal_selection((batch, slice(None), slice(None))),
                dtype=np.float32,
            )
            flat = x.reshape(-1, D).astype(np.float64)
            sum_x += flat.sum(0)
            sum_x2 += np.square(flat).sum(0)
            total += flat.shape[0]
        mean = (sum_x / total).astype(np.float32)
        std = np.sqrt(np.maximum(sum_x2 / total - np.square(mean.astype(np.float64)), 1e-8)).astype(np.float32)
        std[std < 1e-4] = 1.0
        return mean, std

    def compute_imu_normalizer(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        pair_map = {p.pair_index: p for p in self._all_pairs}
        C = self.imu_feature_dim
        sum_x = np.zeros(C, dtype=np.float64)
        sum_x2 = np.zeros(C, dtype=np.float64)
        total = 0
        for i in pair_indices:
            if i not in pair_map:
                continue
            seq = self.imu_dataset.get_sequence_array(pair_map[i].imu_dataset_index).astype(np.float64)
            sum_x += seq.sum(0)
            sum_x2 += np.square(seq).sum(0)
            total += seq.shape[0]
        mean = (sum_x / total).astype(np.float32)
        std = np.sqrt(np.maximum(sum_x2 / total - np.square(mean.astype(np.float64)), 1e-8)).astype(np.float32)
        std[std < 1e-4] = 1.0
        return mean, std

    # ------------------------------------------------------------------

    @staticmethod
    def _print_stats(stats: dict, fm_df: pd.DataFrame, imu_df: pd.DataFrame) -> None:
        print(f"  FM windows loaded         : {stats['n_fm_windows_total']}")
        print(f"  IMU windows loaded        : {stats['n_imu_windows_total']}")
        print(f"  (subject, task) groups    : {stats['n_subject_task_groups']}")
        print(f"  FM-only groups (no IMU)   : {stats['n_groups_fm_only']}")
        print(f"  IMU-only groups (no FM)   : {stats['n_groups_imu_only']}")
        print(f"  Groups with count mismatch: {stats['n_groups_with_mismatch']}")
        print(f"  Pairs aligned             : {stats['n_pairs_total']}")
        print(f"  Subjects covered          : {len(stats['subjects_covered'])}")
        coverage = 100 * stats["n_pairs_total"] / max(stats["n_fm_windows_total"], 1)
        print(f"  FM coverage               : {coverage:.1f}%")
        print(f"  FM stress/total           : {int(fm_df['label'].sum())}/{len(fm_df)}")
        print(f"  IMU stress/total          : {int(imu_df['label'].sum())}/{len(imu_df)}")
