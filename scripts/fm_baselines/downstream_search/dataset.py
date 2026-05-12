#!/usr/bin/env python3
"""Lazy datasets for framewise RGB/depth FM embedding stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def require_zarr():
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install zarr<3 in the active environment.") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"Unsupported zarr version {zarr.__version__}; this pipeline expects zarr<3.")
    return zarr


@dataclass(frozen=True)
class EmbeddingStore:
    encoder: str
    feature: str
    store_path: Path
    metadata_path: Path
    metadata: pd.DataFrame
    y: np.ndarray
    window_id: np.ndarray
    base_subject_id: np.ndarray
    x_shape: tuple[int, int, int]
    attrs: dict


def discover_embedding_stores(root: Path) -> pd.DataFrame:
    rows = []
    zarr = require_zarr()
    for store_path in sorted(root.glob("*/*.zarr")):
        encoder = store_path.parent.name
        feature = store_path.stem
        metadata_path = store_path.parent / f"{feature}_metadata.csv"
        if not metadata_path.exists():
            continue
        group = zarr.open_group(str(store_path), mode="r")
        rows.append(
            {
                "encoder": encoder,
                "feature": feature,
                "store_path": str(store_path),
                "metadata_path": str(metadata_path),
                "X_shape": tuple(group["X"].shape),
                "X_chunks": tuple(group["X"].chunks),
                "y_shape": tuple(group["y"].shape),
                "embedding_dim": int(group.attrs.get("embedding_dim", group["X"].shape[-1])),
                "frames_per_window": int(group.attrs.get("frames_per_window", group["X"].shape[1])),
                "embedding_cache_mode": group.attrs.get("embedding_cache_mode", ""),
            }
        )
    return pd.DataFrame(rows)


def load_embedding_store(root: Path, encoder: str, feature: str) -> EmbeddingStore:
    zarr = require_zarr()
    store_path = root / encoder / f"{feature}.zarr"
    metadata_path = root / encoder / f"{feature}_metadata.csv"
    if not store_path.exists():
        raise FileNotFoundError(f"Missing embedding Zarr store: {store_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")
    group = zarr.open_group(str(store_path), mode="r")
    x_shape = tuple(int(v) for v in group["X"].shape)
    if len(x_shape) != 3:
        raise ValueError(f"Expected framewise X with shape N x T x D, got {x_shape} in {store_path}")
    metadata = pd.read_csv(metadata_path)
    y = np.asarray(group["y"][:], dtype=np.int64)
    window_id = np.asarray(group["window_id"][:], dtype=np.int64)
    base_subject_id = np.asarray(group["base_subject_id"][:]).astype(str)
    if len(metadata) != x_shape[0] or len(y) != x_shape[0]:
        raise ValueError(f"Metadata/label length mismatch for {encoder}/{feature}: X={x_shape}, meta={len(metadata)}, y={len(y)}")
    return EmbeddingStore(
        encoder=encoder,
        feature=feature,
        store_path=store_path,
        metadata_path=metadata_path,
        metadata=metadata,
        y=y,
        window_id=window_id,
        base_subject_id=base_subject_id,
        x_shape=x_shape,
        attrs=dict(group.attrs),
    )


def balanced_feature_sample_indices(labels: np.ndarray, subjects: np.ndarray, max_windows: int) -> np.ndarray:
    df = pd.DataFrame({"index": np.arange(len(labels)), "label": labels, "base_subject_id": subjects.astype(str)})
    ordered = df.sort_values(["label", "base_subject_id", "index"]).copy()
    ordered["_rank"] = ordered.groupby(["label", "base_subject_id"]).cumcount()
    sampled = ordered.sort_values(["_rank", "label", "base_subject_id", "index"]).head(int(max_windows))
    return np.sort(sampled["index"].to_numpy(dtype=np.int64))


def compute_standardizer(store: EmbeddingStore, indices: Iterable[int], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    zarr = require_zarr()
    group = zarr.open_group(str(store.store_path), mode="r")
    x_array = group["X"]
    indices = np.asarray(list(indices), dtype=np.int64)
    total = 0
    sum_x = np.zeros((store.x_shape[2],), dtype=np.float64)
    sum_x2 = np.zeros((store.x_shape[2],), dtype=np.float64)
    for start in range(0, len(indices), int(batch_size)):
        batch_idx = np.sort(indices[start : start + int(batch_size)])
        x = np.asarray(x_array.get_orthogonal_selection((batch_idx, slice(None), slice(None))), dtype=np.float32)
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        sum_x += flat.sum(axis=0)
        sum_x2 += np.square(flat).sum(axis=0)
        total += flat.shape[0]
    mean = sum_x / max(total, 1)
    var = np.maximum(sum_x2 / max(total, 1) - np.square(mean), 1e-8)
    std = np.sqrt(var)
    std[std < 1e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class FramewiseEmbeddingDataset(Dataset):
    def __init__(
        self,
        store: EmbeddingStore,
        indices: Iterable[int],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.store = store
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.mean = mean
        self.std = std
        self._group = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _x_array(self):
        if self._group is None:
            zarr = require_zarr()
            self._group = zarr.open_group(str(self.store.store_path), mode="r")
        return self._group["X"]

    def __getitem__(self, item: int):
        source_idx = int(self.indices[item])
        x = np.asarray(self._x_array()[source_idx], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[None, :]) / self.std[None, :]
        y = int(self.store.y[source_idx])
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long), source_idx
