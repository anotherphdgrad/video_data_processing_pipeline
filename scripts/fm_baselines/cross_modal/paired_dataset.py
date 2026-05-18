#!/usr/bin/env python3
"""
Paired depth-IMU dataset for cross-modal representation learning.

Loads FM depth embeddings (from a framewise zarr store) and IMU ACC sequences
(from a shared IMU zarr store) and aligns them into paired training examples.

Alignment strategy
------------------
Both pipelines window the same recording with identical parameters
(30 s window / 15 s overlap by default).  Windows are paired by:

    (base_subject_id, task_id, window_position_within_task)

For each (subject, task) group, FM windows are ordered by their source index
and IMU windows are ordered by window_start timestamp.  The N-th FM window is
paired with the N-th IMU window.  Groups with mismatched counts are reported
and the smaller count is used (excess windows are dropped with a warning).

If the FM metadata CSV contains a ``task_id`` column the pairing is exact.
If it is absent, a ``task_id`` argument must be inferred from another source
or the dataset raises a clear error at init time.

Returned items
--------------
Each __getitem__ returns a dict:
    depth_embedding : FloatTensor (T_depth, D_depth)   — FM frame embeddings
    imu_sequence    : FloatTensor (T_imu, C_imu)       — raw ACC sequence
    label           : LongTensor scalar
    base_subject_id : str
    task_id         : str
    fm_source_index : int   — row in FM zarr
    imu_window_id   : int   — IMU window_id (from IMU zarr)
    pair_index      : int   — position in this dataset
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers for zarr loading (self-contained — no pipeline imports)
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
    """One aligned (depth, IMU) window pair."""
    pair_index: int
    base_subject_id: str
    task_id: str
    label: int
    fm_source_index: int     # row in FM zarr X array
    fm_window_id: int        # window_id stored in FM zarr
    imu_zarr_index: int      # row in IMU shared zarr sequences array
    imu_window_id: int       # window_id stored in IMU zarr


# ---------------------------------------------------------------------------
# FM zarr helpers
# ---------------------------------------------------------------------------

def _load_fm_index(
    store_path: Path,
    metadata_csv_path: Path,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        source_index, window_id, base_subject_id, task_id, label
    source_index is the row index in the FM zarr X array.
    """
    zarr = _require_zarr()
    group = zarr.open_group(str(store_path), mode="r")

    n = int(group["X"].shape[0])
    window_ids = np.asarray(group["window_id"][:], dtype=np.int64)
    base_subject_ids = np.asarray(group["base_subject_id"][:]).astype(str)
    labels = np.asarray(group["y"][:], dtype=np.int64)

    fm_df = pd.DataFrame({
        "source_index": np.arange(n, dtype=np.int64),
        "window_id": window_ids,
        "base_subject_id": base_subject_ids,
        "label": labels,
    })

    if not metadata_csv_path.exists():
        raise FileNotFoundError(
            f"FM metadata CSV not found: {metadata_csv_path}\n"
            "This file is required to obtain task_id for alignment."
        )
    meta = pd.read_csv(metadata_csv_path)
    if "task_id" not in meta.columns:
        raise ValueError(
            f"FM metadata CSV {metadata_csv_path} does not contain a 'task_id' column.\n"
            f"Available columns: {meta.columns.tolist()}"
        )
    # Join on window_id (global manifest ID stored in both zarr and CSV)
    if "window_id" not in meta.columns:
        raise ValueError(
            f"FM metadata CSV {metadata_csv_path} does not contain 'window_id'.\n"
            f"Available columns: {meta.columns.tolist()}"
        )
    meta_slim = meta[["window_id", "task_id"]].drop_duplicates("window_id")
    fm_df = fm_df.merge(meta_slim, on="window_id", how="left")
    n_missing = fm_df["task_id"].isna().sum()
    if n_missing > 0:
        print(
            f"  WARNING: {n_missing}/{len(fm_df)} FM windows have no task_id after join — "
            "they will be excluded from pairing."
        )
        fm_df = fm_df.dropna(subset=["task_id"])
    return fm_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IMU shared zarr helpers
# ---------------------------------------------------------------------------

def _load_imu_index(imu_store_path: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        imu_zarr_index, imu_window_id, base_subject_id, task_id,
        label, window_start, window_end
    """
    zarr = _require_zarr()
    root = zarr.open_group(str(imu_store_path), mode="r")
    samples = root["samples"]

    n = int(samples["labels"].shape[0])
    imu_df = pd.DataFrame({
        "imu_zarr_index": np.arange(n, dtype=np.int64),
        "imu_window_id": np.asarray(samples["window_ids"][:], dtype=np.int64),
        "base_subject_id": np.asarray(samples["base_subject_ids"][:]).astype(str),
        "task_id": np.asarray(samples["task_ids"][:]).astype(str),
        "label": np.asarray(samples["labels"][:], dtype=np.int64),
        "window_start": np.asarray(samples["window_start"][:], dtype=np.float64),
        "window_end": np.asarray(samples["window_end"][:], dtype=np.float64),
    })
    return imu_df


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _build_alignment(
    fm_df: pd.DataFrame,
    imu_df: pd.DataFrame,
) -> tuple[list[PairedWindow], dict]:
    """
    Pair FM and IMU windows by (base_subject_id, task_id, window_position).

    Returns (pairs, stats_dict).
    """
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
    imu_keys = set(imu_groups.groups.keys())
    fm_keys = set(fm_groups.groups.keys())

    all_keys = fm_keys | imu_keys
    stats["n_subject_task_groups"] = len(all_keys)
    stats["n_groups_imu_only"] = len(imu_keys - fm_keys)
    stats["n_groups_fm_only"] = len(fm_keys - imu_keys)

    covered_subjects: set[str] = set()
    for key in sorted(all_keys):
        if key not in fm_keys or key not in imu_keys:
            continue

        fm_group = fm_groups.get_group(key).sort_values("source_index").reset_index(drop=True)
        imu_group = imu_groups.get_group(key).sort_values("window_start").reset_index(drop=True)

        n_fm, n_imu = len(fm_group), len(imu_group)
        n_pairs = min(n_fm, n_imu)
        if n_fm != n_imu:
            stats["n_groups_with_mismatch"] += 1

        subj, task = key
        covered_subjects.add(subj)
        for pos in range(n_pairs):
            fm_row = fm_group.iloc[pos]
            imu_row = imu_group.iloc[pos]
            # Verify label consistency
            if int(fm_row["label"]) != int(imu_row["label"]):
                print(
                    f"  WARNING: label mismatch at ({subj}, {task}, pos={pos}): "
                    f"FM={fm_row['label']}, IMU={imu_row['label']} — using FM label"
                )
            pairs.append(PairedWindow(
                pair_index=len(pairs),
                base_subject_id=subj,
                task_id=task,
                label=int(fm_row["label"]),
                fm_source_index=int(fm_row["source_index"]),
                fm_window_id=int(fm_row["window_id"]),
                imu_zarr_index=int(imu_row["imu_zarr_index"]),
                imu_window_id=int(imu_row["imu_window_id"]),
            ))

    stats["n_pairs_total"] = len(pairs)
    stats["subjects_covered"] = sorted(covered_subjects)
    return pairs, stats


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class PairedDepthIMUDataset(Dataset):
    """
    Paired FM depth embeddings + IMU ACC sequences for cross-modal training.

    Parameters
    ----------
    fm_store_path : Path
        Path to the FM zarr store, e.g.
        outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth.zarr
    fm_metadata_csv_path : Path
        Sibling metadata CSV, e.g.
        outputs_rgb_depth_fm/embeddings_zarr2_entropy75/dinov2/motion_prev_depth_metadata.csv
    imu_store_path : Path
        Path to the IMU shared zarr store, e.g.
        IMU_shared_stores/window_30s_overlap_15s_sr_64hz_channels_raw_absdelta.zarr
    fm_mean : np.ndarray or None
        Population mean (D_depth,) for FM embedding normalization.
        If None, no normalization is applied. Should be computed on training
        fold indices only.
    fm_std : np.ndarray or None
        Population std (D_depth,) for FM embedding normalization.
    imu_mean : np.ndarray or None
        Per-channel mean (C_imu,) for IMU sequence normalization.
    imu_std : np.ndarray or None
        Per-channel std (C_imu,) for IMU sequence normalization.
    pair_indices : sequence of int or None
        If provided, restricts to this subset of pair indices (e.g. a fold's
        train/val/test split).  If None, uses all pairs.
    verbose : bool
        Print alignment statistics on init.
    """

    def __init__(
        self,
        fm_store_path: Path | str,
        fm_metadata_csv_path: Path | str,
        imu_store_path: Path | str,
        fm_mean: np.ndarray | None = None,
        fm_std: np.ndarray | None = None,
        imu_mean: np.ndarray | None = None,
        imu_std: np.ndarray | None = None,
        pair_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> None:
        self.fm_store_path = Path(fm_store_path)
        self.fm_metadata_csv_path = Path(fm_metadata_csv_path)
        self.imu_store_path = Path(imu_store_path)
        self.fm_mean = np.asarray(fm_mean, dtype=np.float32) if fm_mean is not None else None
        self.fm_std = np.asarray(fm_std, dtype=np.float32) if fm_std is not None else None
        self.imu_mean = np.asarray(imu_mean, dtype=np.float32) if imu_mean is not None else None
        self.imu_std = np.asarray(imu_std, dtype=np.float32) if imu_std is not None else None

        self._fm_group = None   # lazy zarr handle
        self._imu_root = None   # lazy zarr handle

        if verbose:
            print(f"[PairedDepthIMUDataset] Loading FM index from {self.fm_store_path.name} ...")
        fm_df = _load_fm_index(self.fm_store_path, self.fm_metadata_csv_path)

        if verbose:
            print(f"[PairedDepthIMUDataset] Loading IMU index from {self.imu_store_path.name} ...")
        imu_df = _load_imu_index(self.imu_store_path)

        if verbose:
            print("[PairedDepthIMUDataset] Building alignment index ...")
        all_pairs, stats = _build_alignment(fm_df, imu_df)

        if verbose:
            self._print_stats(stats, fm_df, imu_df)

        if not all_pairs:
            raise ValueError(
                "No paired windows found. Check that base_subject_id and task_id "
                "match between the FM and IMU zarr stores."
            )

        # Optionally restrict to a fold subset
        if pair_indices is not None:
            idx_set = set(pair_indices)
            self._pairs: list[PairedWindow] = [p for p in all_pairs if p.pair_index in idx_set]
        else:
            self._pairs = all_pairs

        self._all_pairs = all_pairs   # keep full index for fold creation
        self.stats = stats

        # Cache shape info without loading data
        zarr = _require_zarr()
        fm_group = zarr.open_group(str(self.fm_store_path), mode="r")
        imu_root = zarr.open_group(str(self.imu_store_path), mode="r")
        self.fm_shape: tuple[int, int, int] = tuple(int(v) for v in fm_group["X"].shape)
        self.imu_shape: tuple[int, int, int] = tuple(int(v) for v in imu_root["samples"]["sequences"].shape)

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
        """All pairs as a DataFrame — useful for fold creation."""
        return pd.DataFrame([
            {
                "pair_index": p.pair_index,
                "base_subject_id": p.base_subject_id,
                "task_id": p.task_id,
                "label": p.label,
                "fm_source_index": p.fm_source_index,
                "fm_window_id": p.fm_window_id,
                "imu_zarr_index": p.imu_zarr_index,
                "imu_window_id": p.imu_window_id,
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

        # Lazy-open zarr handles (per-process, survives DataLoader fork)
        zarr = _require_zarr()
        if self._fm_group is None:
            self._fm_group = zarr.open_group(str(self.fm_store_path), mode="r")
        if self._imu_root is None:
            self._imu_root = zarr.open_group(str(self.imu_store_path), mode="r")

        # Load FM depth embedding (T_depth, D_depth)
        fm_x = np.asarray(
            self._fm_group["X"][pair.fm_source_index],
            dtype=np.float32,
        )
        if self.fm_mean is not None and self.fm_std is not None:
            fm_x = (fm_x - self.fm_mean[None, :]) / self.fm_std[None, :]

        # Load IMU sequence (T_imu, C_imu)
        imu_x = np.asarray(
            self._imu_root["samples"]["sequences"][pair.imu_zarr_index],
            dtype=np.float32,
        )
        if self.imu_mean is not None and self.imu_std is not None:
            imu_x = (imu_x - self.imu_mean[None, :]) / self.imu_std[None, :]

        return {
            "depth_embedding": torch.from_numpy(fm_x),
            "imu_sequence": torch.from_numpy(imu_x),
            "label": torch.tensor(pair.label, dtype=torch.long),
            "base_subject_id": pair.base_subject_id,
            "task_id": pair.task_id,
            "fm_source_index": pair.fm_source_index,
            "imu_window_id": pair.imu_window_id,
            "pair_index": pair.pair_index,
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def compute_fm_normalizer(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Compute FM embedding mean/std from a subset of pair indices (train fold only)."""
        zarr = _require_zarr()
        group = zarr.open_group(str(self.fm_store_path), mode="r")
        x_array = group["X"]
        pair_map = {p.pair_index: p for p in self._all_pairs}
        source_indices = sorted(pair_map[i].fm_source_index for i in pair_indices if i in pair_map)
        batch_size = 128
        sum_x = np.zeros(self.fm_shape[2], dtype=np.float64)
        sum_x2 = np.zeros(self.fm_shape[2], dtype=np.float64)
        total = 0
        for start in range(0, len(source_indices), batch_size):
            batch_idx = np.array(source_indices[start: start + batch_size], dtype=np.int64)
            x = np.asarray(
                x_array.get_orthogonal_selection((batch_idx, slice(None), slice(None))),
                dtype=np.float32,
            )
            flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
            sum_x += flat.sum(0)
            sum_x2 += np.square(flat).sum(0)
            total += flat.shape[0]
        mean = (sum_x / total).astype(np.float32)
        std = np.sqrt(np.maximum(sum_x2 / total - np.square(mean.astype(np.float64)), 1e-8)).astype(np.float32)
        std[std < 1e-4] = 1.0
        return mean, std

    def compute_imu_normalizer(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Compute IMU sequence mean/std from a subset of pair indices (train fold only)."""
        zarr = _require_zarr()
        root = zarr.open_group(str(self.imu_store_path), mode="r")
        seq_array = root["samples"]["sequences"]
        pair_map = {p.pair_index: p for p in self._all_pairs}
        zarr_indices = sorted(pair_map[i].imu_zarr_index for i in pair_indices if i in pair_map)
        batch_size = 256
        n_channels = self.imu_shape[2]
        sum_x = np.zeros(n_channels, dtype=np.float64)
        sum_x2 = np.zeros(n_channels, dtype=np.float64)
        total = 0
        for start in range(0, len(zarr_indices), batch_size):
            batch_idx = np.array(zarr_indices[start: start + batch_size], dtype=np.int64)
            x = np.asarray(seq_array.oindex[batch_idx], dtype=np.float32)
            flat = x.reshape(-1, n_channels).astype(np.float64)
            sum_x += flat.sum(0)
            sum_x2 += np.square(flat).sum(0)
            total += flat.shape[0]
        mean = (sum_x / total).astype(np.float32)
        std = np.sqrt(np.maximum(sum_x2 / total - np.square(mean.astype(np.float64)), 1e-8)).astype(np.float32)
        std[std < 1e-4] = 1.0
        return mean, std

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_stats(stats: dict, fm_df: pd.DataFrame, imu_df: pd.DataFrame) -> None:
        print(f"  FM windows loaded       : {stats['n_fm_windows_total']}")
        print(f"  IMU windows loaded      : {stats['n_imu_windows_total']}")
        print(f"  (subject, task) groups  : {stats['n_subject_task_groups']}")
        print(f"  FM-only groups (no IMU) : {stats['n_groups_fm_only']}")
        print(f"  IMU-only groups (no FM) : {stats['n_groups_imu_only']}")
        print(f"  Groups with count mismatch: {stats['n_groups_with_mismatch']}")
        print(f"  Pairs aligned           : {stats['n_pairs_total']}")
        print(f"  Subjects covered        : {len(stats['subjects_covered'])}")
        coverage = 100 * stats["n_pairs_total"] / max(stats["n_fm_windows_total"], 1)
        print(f"  FM coverage             : {coverage:.1f}%")
        # Label balance
        fm_stress = int(fm_df["label"].sum()) if "label" in fm_df.columns else "?"
        imu_stress = int(imu_df["label"].sum())
        print(f"  FM stress/total         : {fm_stress}/{len(fm_df)}")
        print(f"  IMU stress/total        : {imu_stress}/{len(imu_df)}")
