#!/usr/bin/env python3
"""Frozen foundation-model embedding baselines for RGB/depth stress windows.

This runner is intentionally conservative for memory:
- encoders are frozen
- embeddings are cached before fold-local classifier training
- framewise caches preserve the per-frame sequence inside each 30s window
- classifiers operate on cached vectors after an explicit fold-stage pooling step
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


FEATURE_SPECS = {
    "masked_rgb": ("rgb_masked", "rgb"),
    "masked_depth": ("depth_masked", "depth"),
    "motion_prev_rgb": ("motion_rgb", "motion_rgb"),
    "motion_prev_depth": ("motion_depth", "motion_depth"),
    "flow_edge_rgb": ("flow_edge_rgb", "flow_edge"),
    "flow_edge_depth": ("flow_edge_depth", "flow_edge"),
}
DEFAULT_FEATURES = list(FEATURE_SPECS)
DEFAULT_ENCODERS = ["imagebind", "omnivore", "dinov2"]
DEFAULT_HEADS = ["mlp_probe"]
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
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE  {label} in {format_seconds(time.time() - start)}", flush=True)


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
    parser = argparse.ArgumentParser(description="Evaluate frozen FM embeddings on RGB/depth stress windows.")
    parser.add_argument("--window-manifest", default="processed_rgb_depth_features/manifests/window_manifest.csv")
    parser.add_argument("--output-root", default="outputs_rgb_depth_fm")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", choices=["all", "embed", "train", "summarize"], default="all")
    parser.add_argument("--encoders", nargs="*", default=DEFAULT_ENCODERS, choices=DEFAULT_ENCODERS)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, choices=DEFAULT_FEATURES)
    parser.add_argument("--heads", nargs="*", default=DEFAULT_HEADS, choices=["linear_probe", "mlp_probe"])
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--torch-hub-dir", default="checkpoints/torch_hub")
    parser.add_argument(
        "--model-repo-root",
        default="checkpoints/repos",
        help="Local repo root containing optional dinov2/omnivore checkouts.",
    )
    parser.add_argument("--imagebind-checkpoint", default="checkpoints/imagebind/imagebind_huge.pth")
    parser.add_argument("--dinov2-model", default="dinov2_vitb14")
    parser.add_argument("--omnivore-model", default="omnivore_swinB")
    parser.add_argument("--num-sampled-frames", type=int, default=16)
    parser.add_argument(
        "--embedding-cache-mode",
        choices=["framewise", "pooled"],
        default="framewise",
        help="framewise writes X as windows x frames x dim; pooled writes one mean vector per window.",
    )
    parser.add_argument(
        "--embedding-frame-selection",
        choices=["all", "sampled"],
        default="all",
        help="Use all frames in each 30s window for the cache, or deterministic sampled frames.",
    )
    parser.add_argument(
        "--frame-embed-batch-size",
        type=int,
        default=32,
        help="Number of individual frames to send through image encoders at once.",
    )
    parser.add_argument(
        "--cached-sequence-pooling",
        choices=["mean"],
        default="mean",
        help="Temporary probe pooling for framewise caches. Future sequence models can consume X directly.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument("--max-windows-per-feature", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-unavailable-encoders", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def add_local_repo_to_path(args: argparse.Namespace, repo_name: str) -> Path | None:
    import sys

    repo_root = Path(args.model_repo_root) / repo_name
    if repo_root.exists():
        text = str(repo_root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
        return repo_root
    return None


def load_local_hubconf(args: argparse.Namespace, repo_name: str):
    repo_root = add_local_repo_to_path(args, repo_name)
    if repo_root is None:
        return None
    hubconf_path = repo_root / "hubconf.py"
    if not hubconf_path.exists():
        return None
    module_name = f"_fm_baselines_{repo_name}_hubconf"
    spec = importlib.util.spec_from_file_location(module_name, hubconf_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr<3 in the active environment.") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"Unsupported zarr version {zarr.__version__}; this pipeline expects zarr<3.")
    return zarr


def require_zarr_write_modules():
    zarr = require_zarr()
    try:
        from numcodecs import Blosc
    except ImportError as exc:
        raise SystemExit("Install numcodecs for compressed Zarr embedding caches.") from exc
    return zarr, Blosc


def import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("Foundation-model embedding extraction requires PyTorch.") from exc
    return torch, F


def resolve_device(args: argparse.Namespace):
    torch, _ = import_torch()
    if args.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_grouped_folds(metadata_df: pd.DataFrame, n_splits: int, val_ratio: float, random_seed: int) -> list[FoldSplit]:
    indices = metadata_df.index.to_numpy()
    groups = metadata_df["base_subject_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} participants, found {len(np.unique(groups))}.")
    folds = []
    outer = GroupKFold(n_splits=n_splits)
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


def load_manifest(path: Path, features: Iterable[str], max_windows_per_feature: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["feature"].isin(features)].copy()
    if max_windows_per_feature is not None:
        df = df.groupby("feature", group_keys=False).apply(
            lambda g: balanced_feature_sample(g, int(max_windows_per_feature))
        )
    if df.empty:
        raise SystemExit(f"No rows found for requested features in {path}.")
    return df.reset_index(drop=True)


def balanced_feature_sample(group_df: pd.DataFrame, max_windows: int) -> pd.DataFrame:
    ordered = group_df.sort_values(["label", "base_subject_id", "window_start_timestamp", "task_id"]).copy()
    ordered["_label_subject_rank"] = ordered.groupby(["label", "base_subject_id"]).cumcount()
    sampled = (
        ordered.sort_values(["_label_subject_rank", "label", "base_subject_id", "window_start_timestamp"])
        .head(int(max_windows))
        .drop(columns=["_label_subject_rank"])
    )
    return sampled.sort_values(["label", "base_subject_id", "window_start_timestamp"]).copy()


def sampled_indices(start: int, end: int, num_frames: int) -> np.ndarray:
    n = int(end) - int(start)
    if n <= 0:
        raise ValueError("Window has no frames.")
    count = min(int(num_frames), n)
    offsets = np.linspace(0, n - 1, count).round().astype(np.int64)
    return int(start) + offsets


def robust_unit_scale(values: np.ndarray, signed: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    if signed:
        scale = float(np.percentile(np.abs(finite), 98))
        scale = max(scale, 1e-6)
        return np.clip(values / (2.0 * scale) + 0.5, 0.0, 1.0)
    nonzero = finite[np.abs(finite) > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [2, 98])
    else:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def render_frames(frames: np.ndarray, feature: str) -> np.ndarray:
    kind = FEATURE_SPECS[feature][1]
    frames = np.asarray(frames)
    if kind == "rgb":
        return np.clip(frames.astype(np.float32) / 255.0, 0.0, 1.0)
    if kind == "motion_rgb":
        return robust_unit_scale(frames, signed=True)
    if frames.ndim == 3:
        rendered = robust_unit_scale(frames, signed=kind == "motion_depth")
        return np.repeat(rendered[..., None], 3, axis=-1)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        rendered = robust_unit_scale(frames[..., 0], signed=kind == "motion_depth")
        return np.repeat(rendered[..., None], 3, axis=-1)
    return np.clip(frames.astype(np.float32), 0.0, 1.0)


def normalize_for_encoder(torch, frames: np.ndarray, encoder: str):
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    if encoder in {"imagebind", "clip"}:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def normalize_for_imagebind_depth(torch, frames: np.ndarray):
    tensor = torch.from_numpy(frames).float()
    if tensor.ndim == 4:
        if tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        elif tensor.shape[-1] == 3:
            tensor = tensor[..., 0]
        else:
            raise ValueError(f"Unsupported ImageBind depth frame shape: {tuple(tensor.shape)}")
    if tensor.ndim != 3:
        raise ValueError(f"ImageBind depth expects N x H x W or N x H x W x C frames, got {tuple(tensor.shape)}")
    return tensor.unsqueeze(1)


class EncoderAdapter:
    name = "base"
    embedding_dim: int | None = None

    def encode(self, frames_by_window, feature: str, framewise: bool) -> np.ndarray:
        raise NotImplementedError


class DINOv2Adapter(EncoderAdapter):
    name = "dinov2"

    def __init__(self, args: argparse.Namespace, device: str):
        torch, _ = import_torch()
        self.torch = torch
        self.device = device
        os.environ["TORCH_HOME"] = str(Path(args.torch_hub_dir).resolve())
        self.frame_embed_batch_size = int(args.frame_embed_batch_size)
        self.model = self._load_model(args)
        self.model.eval()
        self.model.to(device)

    def _load_model(self, args: argparse.Namespace):
        torch = self.torch
        try:
            hubconf = load_local_hubconf(args, "dinov2")
            if hasattr(hubconf, args.dinov2_model):
                return getattr(hubconf, args.dinov2_model)(pretrained=True)
        except Exception:
            pass
        return torch.hub.load("facebookresearch/dinov2", args.dinov2_model)

    def encode(self, frames_by_window, feature: str, framewise: bool) -> np.ndarray:
        torch = self.torch
        counts = [len(frames) for frames in frames_by_window]
        flat = np.concatenate(frames_by_window, axis=0)
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(flat), self.frame_embed_batch_size):
                x = normalize_for_encoder(torch, flat[start : start + self.frame_embed_batch_size], "dinov2").to(self.device)
                out = self.model(x)
                if isinstance(out, dict):
                    out = out.get("x_norm_clstoken", next(iter(out.values())))
                encoded.append(out.detach().cpu().numpy().astype(np.float32))
        out = np.concatenate(encoded, axis=0)
        pieces = np.split(out, np.cumsum(counts)[:-1], axis=0)
        if framewise:
            return np.stack(pieces, axis=0)
        return np.stack([piece.mean(axis=0) for piece in pieces], axis=0)


class ImageBindAdapter(EncoderAdapter):
    name = "imagebind"

    def __init__(self, args: argparse.Namespace, device: str):
        torch, _ = import_torch()
        self.torch = torch
        self.device = device
        try:
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
        except ImportError as exc:
            raise RuntimeError("ImageBind is not importable. Install the official ImageBind repo in this env.") from exc
        self.ModalityType = ModalityType
        self.frame_embed_batch_size = int(args.frame_embed_batch_size)
        ckpt = Path(args.imagebind_checkpoint)
        self.model = imagebind_model.imagebind_huge(pretrained=False)
        if ckpt.exists():
            self.model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        else:
            print(
                f"ImageBind checkpoint not found at {ckpt}; falling back to imagebind_huge(pretrained=True), "
                "which uses the package's .checkpoints/imagebind_huge.pth behavior."
            )
            self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

    def modality_for_feature(self, feature: str):
        if feature.endswith("_depth") or feature == "masked_depth":
            return self.ModalityType.DEPTH
        return self.ModalityType.VISION

    def encode(self, frames_by_window, feature: str, framewise: bool) -> np.ndarray:
        torch = self.torch
        counts = [len(frames) for frames in frames_by_window]
        flat = np.concatenate(frames_by_window, axis=0)
        modality = self.modality_for_feature(feature)
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(flat), self.frame_embed_batch_size):
                chunk = flat[start : start + self.frame_embed_batch_size]
                if modality == self.ModalityType.DEPTH:
                    x = normalize_for_imagebind_depth(torch, chunk).to(self.device)
                else:
                    x = normalize_for_encoder(torch, chunk, "imagebind").to(self.device)
                encoded.append(self.model({modality: x})[modality].detach().cpu().numpy().astype(np.float32))
        out = np.concatenate(encoded, axis=0)
        pieces = np.split(out, np.cumsum(counts)[:-1], axis=0)
        if framewise:
            return np.stack(pieces, axis=0)
        return np.stack([piece.mean(axis=0) for piece in pieces], axis=0)


class OmnivoreAdapter(EncoderAdapter):
    name = "omnivore"

    def __init__(self, args: argparse.Namespace, device: str):
        torch, _ = import_torch()
        self.torch = torch
        self.device = device
        os.environ["TORCH_HOME"] = str(Path(args.torch_hub_dir).resolve())
        self.frame_embed_batch_size = int(args.frame_embed_batch_size)
        self.model = self._load_model(args)
        self.model.eval()
        self.model.to(device)

    def _load_model(self, args: argparse.Namespace):
        torch = self.torch
        local_error = None
        try:
            hubconf = load_local_hubconf(args, "omnivore")
            if hasattr(hubconf, args.omnivore_model):
                model = getattr(hubconf, args.omnivore_model)(pretrained=True, load_heads=False)
                if model is not None:
                    return model
                local_error = RuntimeError(f"{args.omnivore_model}(load_heads=False) returned None")
        except Exception as exc:
            local_error = exc
        try:
            model = torch.hub.load("facebookresearch/omnivore:main", args.omnivore_model, load_heads=False)
            if model is not None:
                return model
        except Exception as exc:
            if local_error is not None:
                raise RuntimeError(
                    "Failed to load Omnivore from both the local repo and torch.hub. "
                    f"Local error: {local_error}. Torch hub error: {exc}"
                ) from exc
            raise
        raise RuntimeError("Omnivore loader returned None from both local repo and torch.hub.")

    def encode(self, frames_by_window, feature: str, framewise: bool) -> np.ndarray:
        torch = self.torch
        if framewise:
            counts = [len(frames) for frames in frames_by_window]
            flat = np.concatenate(frames_by_window, axis=0)
            encoded = []
            with torch.inference_mode():
                for start in range(0, len(flat), self.frame_embed_batch_size):
                    x = normalize_for_encoder(torch, flat[start : start + self.frame_embed_batch_size], "omnivore").to(self.device)
                    out = self.model(x)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, dict):
                        out = next(iter(out.values()))
                    encoded.append(out.reshape(out.shape[0], -1).detach().cpu().numpy().astype(np.float32))
            out = np.concatenate(encoded, axis=0)
            pieces = np.split(out, np.cumsum(counts)[:-1], axis=0)
            return np.stack(pieces, axis=0)
        embeddings = []
        with torch.inference_mode():
            for frames in frames_by_window:
                x = normalize_for_encoder(torch, frames, "omnivore").permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                out = self.model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                embeddings.append(out.reshape(out.shape[0], -1).detach().cpu().numpy()[0].astype(np.float32))
        return np.stack(embeddings, axis=0)


def build_adapter(encoder: str, args: argparse.Namespace, device: str) -> EncoderAdapter:
    if encoder == "dinov2":
        return DINOv2Adapter(args, device)
    if encoder == "imagebind":
        return ImageBindAdapter(args, device)
    if encoder == "omnivore":
        return OmnivoreAdapter(args, device)
    raise ValueError(encoder)


def embedding_cache_paths(output_root: Path, encoder: str, feature: str) -> tuple[Path, Path]:
    root = output_root / "embeddings_zarr2" / encoder
    return root / f"{feature}.zarr", root / f"{feature}_metadata.csv"


def write_embedding_zarr(
    store_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    window_id: np.ndarray,
    base_subject_id: np.ndarray,
    attrs: dict,
) -> None:
    zarr, Blosc = require_zarr_write_modules()
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(store_path), mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    chunk_n = max(1, min(64 if X.ndim == 3 else 256, X.shape[0]))
    if X.ndim == 3:
        x_chunks = (chunk_n, max(1, min(32, X.shape[1])), X.shape[2])
        embedding_dim = int(X.shape[2])
        frames_per_window = int(X.shape[1])
    elif X.ndim == 2:
        x_chunks = (chunk_n, X.shape[1])
        embedding_dim = int(X.shape[1])
        frames_per_window = 1
    else:
        raise ValueError(f"Expected X to be 2D or 3D, got shape {X.shape}")
    root.create_dataset("X", data=X.astype(np.float32), chunks=x_chunks, compressor=compressor, overwrite=True)
    root.create_dataset("y", data=y.astype(np.int64), chunks=(chunk_n,), compressor=compressor, overwrite=True)
    root.create_dataset("window_id", data=window_id.astype(np.int64), chunks=(chunk_n,), compressor=compressor, overwrite=True)
    root.create_dataset(
        "base_subject_id",
        data=np.asarray(base_subject_id, dtype=object),
        object_codec=zarr.codecs.VLenUTF8(),
        chunks=(chunk_n,),
        compressor=compressor,
        overwrite=True,
    )
    root.attrs.update(
        {
            "format": "rgb_depth_fm_embeddings_zarr2",
            "num_windows": int(X.shape[0]),
            "frames_per_window": frames_per_window,
            "embedding_dim": embedding_dim,
            **attrs,
        }
    )


def read_embedding_zarr(store_path: Path) -> dict[str, np.ndarray]:
    zarr = require_zarr()
    root = zarr.open_group(str(store_path), mode="r")
    return {
        "X": np.asarray(root["X"][:], dtype=np.float32),
        "y": np.asarray(root["y"][:], dtype=np.int64),
        "window_id": np.asarray(root["window_id"][:], dtype=np.int64),
        "base_subject_id": np.asarray(root["base_subject_id"][:]).astype(str),
        "attrs": dict(root.attrs),
    }


def load_feature_batch(zarr, rows: pd.DataFrame, num_frames: int | None, feature: str) -> list[np.ndarray]:
    roots: dict[str, object] = {}
    rendered = []
    array_name = FEATURE_SPECS[feature][0]
    for row in rows.itertuples(index=False):
        root = roots.get(row.zarr_path)
        if root is None:
            root = zarr.open_group(row.zarr_path, mode="r")
            roots[row.zarr_path] = root
        tg = root[row.zarr_task_group]
        start = int(row.window_start)
        end = int(row.window_end)
        if num_frames is None:
            window = np.asarray(tg[array_name][start:end])
        else:
            indices = sampled_indices(start, end, num_frames)
            local = indices - start
            window = np.asarray(tg[array_name][start:end])[local]
        rendered.append(render_frames(window, feature))
    return rendered


def selected_frame_count(args: argparse.Namespace) -> int | None:
    if args.embedding_frame_selection == "all":
        return None
    return int(args.num_sampled_frames)


def stage_embed(args: argparse.Namespace, manifest: pd.DataFrame, output_root: Path) -> None:
    zarr = require_zarr()
    device = resolve_device(args)
    framewise = args.embedding_cache_mode == "framewise"
    num_frames = selected_frame_count(args)
    for encoder in args.encoders:
        try:
            adapter = build_adapter(encoder, args, device)
        except Exception as exc:  # noqa: BLE001
            if args.skip_unavailable_encoders:
                print(f"Skipping unavailable encoder {encoder}: {type(exc).__name__}: {exc}")
                continue
            raise
        for feature in args.features:
            feature_df = manifest[manifest["feature"] == feature].reset_index(drop=True)
            if feature_df.empty:
                print(f"Skipping {encoder}/{feature}: no manifest windows")
                continue
            store_path, meta_path = embedding_cache_paths(output_root, encoder, feature)
            if store_path.exists() and meta_path.exists() and not args.overwrite:
                print(f"Embedding cache exists, skipping {encoder}/{feature}: {store_path}")
                continue
            store_path.parent.mkdir(parents=True, exist_ok=True)
            vectors = []
            iterator = range(0, len(feature_df), int(args.embedding_batch_size))
            if tqdm and not args.no_progress:
                iterator = tqdm(iterator, desc=f"Embedding {encoder}/{feature}", unit="batch")
            for start in iterator:
                batch_df = feature_df.iloc[start : start + int(args.embedding_batch_size)]
                frames = load_feature_batch(zarr, batch_df, num_frames, feature)
                vectors.append(adapter.encode(frames, feature, framewise=framewise))
            X = np.concatenate(vectors, axis=0).astype(np.float32)
            write_embedding_zarr(
                store_path,
                X,
                feature_df["label"].to_numpy(dtype=np.int64),
                feature_df["window_id"].to_numpy(dtype=np.int64),
                feature_df["base_subject_id"].astype(str).to_numpy(),
                {
                    "encoder": encoder,
                    "feature": feature,
                    "embedding_cache_mode": args.embedding_cache_mode,
                    "embedding_frame_selection": args.embedding_frame_selection,
                    "num_sampled_frames": int(args.num_sampled_frames),
                    "frame_embed_batch_size": int(args.frame_embed_batch_size),
                },
            )
            feature_df.to_csv(meta_path, index=False)
            with (store_path.parent / f"{feature}_embedding_config.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "encoder": encoder,
                        "feature": feature,
                        "embedding_cache_mode": args.embedding_cache_mode,
                        "embedding_frame_selection": args.embedding_frame_selection,
                        "num_sampled_frames": args.num_sampled_frames,
                        "frame_embed_batch_size": args.frame_embed_batch_size,
                        "X_shape": list(X.shape),
                    },
                    handle,
                    indent=2,
                )
            print(f"Wrote embeddings: {store_path} shape={X.shape}")
        del adapter


def make_head(name: str, seed: int):
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed)


def evaluate_head(model, X: np.ndarray, y: np.ndarray, threshold: float | None = None) -> tuple[dict, np.ndarray, float]:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        raw = model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-raw))
    if threshold is None:
        threshold, _ = find_best_threshold(y, scores)
    pred = apply_threshold(scores, threshold)
    return compute_metrics(y, pred, scores), scores, float(threshold)


def torch_mlp_scores(model, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    torch, _ = import_torch()
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            batch = torch.from_numpy(X[start : start + int(batch_size)]).float().to(device)
            logits = model(batch)
            scores.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def train_torch_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[object, float]:
    torch, _ = import_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    device = resolve_device(args)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class TorchMLPProbe(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            hidden = min(512, max(128, input_dim // 2))
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden // 2, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = TorchMLPProbe(train_x.shape[1]).to(device)
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.probe_learning_rate),
        weight_decay=float(args.probe_weight_decay),
    )
    sample_weights = 1.0 / counts[train_y]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_y),
        replacement=True,
        generator=generator,
    )
    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=int(args.probe_batch_size), sampler=sampler)
    best_state = None
    best_threshold = 0.5
    best_val = -np.inf
    epoch_iter = range(1, int(args.probe_epochs) + 1)
    if tqdm and not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc="Torch MLP probe", unit="epoch", leave=False)
    for _epoch in epoch_iter:
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = torch_mlp_scores(model, val_x, device, int(args.probe_batch_size))
        threshold, _ = find_best_threshold(val_y, val_scores)
        val_metrics = compute_metrics(val_y, apply_threshold(val_scores, threshold), val_scores)
        if tqdm and not args.no_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                {
                    "loss": f"{np.mean(losses):.4f}" if losses else "nan",
                    "val_ba": f"{val_metrics['balanced_accuracy']:.3f}",
                    "val_auc": f"{val_metrics['auc']:.3f}" if np.isfinite(val_metrics["auc"]) else "nan",
                    "thr": f"{threshold:.2f}",
                }
            )
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold


def evaluate_torch_mlp(model, X: np.ndarray, y: np.ndarray, threshold: float, args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    device = resolve_device(args)
    scores = torch_mlp_scores(model, X, device, int(args.probe_batch_size))
    pred = apply_threshold(scores, threshold)
    return compute_metrics(y, pred, scores), scores


def pool_cached_embeddings_for_probe(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if X.ndim == 2:
        return X.astype(np.float32)
    if X.ndim == 3 and args.cached_sequence_pooling == "mean":
        return X.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported cached embedding shape for probe training: {X.shape}")


def stage_train(args: argparse.Namespace, output_root: Path) -> Path:
    run_name = args.run_name or datetime.now().strftime("fm_rgb_depth_%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    metric_rows = []
    prediction_rows = []
    for encoder in args.encoders:
        for feature in args.features:
            store_path, meta_path = embedding_cache_paths(output_root, encoder, feature)
            if not store_path.exists():
                print(f"Missing embeddings, skipping {encoder}/{feature}: {store_path}")
                continue
            data = read_embedding_zarr(store_path)
            meta = pd.read_csv(meta_path)
            X = pool_cached_embeddings_for_probe(data["X"], args)
            y = data["y"].astype(np.int64)
            groups = data["base_subject_id"].astype(str)
            fold_df = pd.DataFrame({"label": y, "base_subject_id": groups})
            folds = create_grouped_folds(fold_df, args.n_splits, args.val_ratio, args.random_seed)
            for head_name in args.heads:
                iterator = folds
                if tqdm and not args.no_progress:
                    iterator = tqdm(folds, desc=f"Training {encoder}/{feature}/{head_name}", unit="fold")
                for fold in iterator:
                    train_x, train_y = X[fold.train_indices], y[fold.train_indices]
                    val_x, val_y = X[fold.val_indices], y[fold.val_indices]
                    test_x, test_y = X[fold.test_indices], y[fold.test_indices]
                    mean = train_x.mean(axis=0, keepdims=True)
                    std = train_x.std(axis=0, keepdims=True)
                    std = np.where(std < 1e-6, 1.0, std)
                    train_x = (train_x - mean) / std
                    val_x = (val_x - mean) / std
                    test_x = (test_x - mean) / std
                    if head_name == "mlp_probe":
                        head, threshold = train_torch_mlp_probe(
                            train_x,
                            train_y,
                            val_x,
                            val_y,
                            args,
                            args.random_seed + fold.fold_id,
                        )
                    else:
                        head = make_head(head_name, args.random_seed + fold.fold_id)
                        head.fit(train_x, train_y)
                        _val_metrics, _val_scores, threshold = evaluate_head(head, val_x, val_y, threshold=None)
                    for split, split_x, split_y, indices in [
                        ("train", train_x, train_y, fold.train_indices),
                        ("val", val_x, val_y, fold.val_indices),
                        ("test", test_x, test_y, fold.test_indices),
                    ]:
                        if head_name == "mlp_probe":
                            metrics, scores = evaluate_torch_mlp(head, split_x, split_y, threshold, args)
                        else:
                            metrics, scores, _ = evaluate_head(head, split_x, split_y, threshold=threshold)
                        metric_rows.append(
                            {
                                "encoder": encoder,
                                "feature": feature,
                                "head": head_name,
                                "model_name": f"{encoder}_{head_name}",
                                "fold_id": fold.fold_id,
                                "split": split,
                                "threshold": threshold,
                                "sequence_length": int(data["attrs"].get("frames_per_window", args.num_sampled_frames)),
                                "embedding_cache_mode": data["attrs"].get("embedding_cache_mode", "pooled"),
                                "cached_sequence_pooling": args.cached_sequence_pooling,
                                **metrics,
                            }
                        )
                        for idx, label, score in zip(indices, split_y, scores):
                            prediction_rows.append(
                                {
                                    "encoder": encoder,
                                    "feature": feature,
                                    "head": head_name,
                                    "fold_id": fold.fold_id,
                                    "split": split,
                                    "window_id": int(meta.iloc[int(idx)]["window_id"]),
                                    "label": int(label),
                                    "score": float(score),
                                    "prediction": int(score >= threshold),
                                    "threshold": threshold,
                                }
                            )
    if not metric_rows:
        raise SystemExit("No FM metrics produced. Check embedding caches and selected encoders/features.")
    metric_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(prediction_rows)
    metric_df.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    pred_df.to_csv(run_dir / "fold_predictions.csv", index=False)
    summarize(metric_df, run_dir, args.n_splits, int(args.num_sampled_frames))
    print(f"Wrote FM run: {run_dir}")
    return run_dir


def summarize(metric_df: pd.DataFrame, run_dir: Path, n_folds: int, sequence_length: int) -> pd.DataFrame:
    rows = []
    for (encoder, feature, head), group in metric_df.groupby(["encoder", "feature", "head"]):
        group_sequence_length = int(group["sequence_length"].iloc[0]) if "sequence_length" in group else sequence_length
        cache_mode = str(group["embedding_cache_mode"].iloc[0]) if "embedding_cache_mode" in group else "pooled"
        pooling = str(group["cached_sequence_pooling"].iloc[0]) if "cached_sequence_pooling" in group else "mean"
        row = {
            "model_name": f"{encoder}_{head}",
            "module_name": "RGBDepth-FM-Frozen",
            "window_strategy": "30s window / 15s overlap; framewise cached embeddings",
            "input_mode": feature,
            "representation_family": "RGBDepth foundation model frozen embedding",
            "representation_equation": f"{encoder}:{feature}:{cache_mode}_frame_embedding",
            "sequence_pooling": f"{pooling} cached frame embeddings + fold-local probe",
            "sequence_length": group_sequence_length,
            "optuna_trials": 0,
            "n_folds": n_folds,
            "person_disjoint_setting": "GroupKFold(base_subject_id) with GroupShuffleSplit validation",
            "encoder": encoder,
            "head": head,
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


def stage_summarize(output_root: Path) -> None:
    frames = []
    for path in sorted((output_root / "runs").glob("*/all_window_mode_results_concise.csv")):
        df = pd.read_csv(path)
        df.insert(0, "run_dir", str(path.parent))
        frames.append(df)
    if not frames:
        raise SystemExit(f"No FM concise result CSVs found under {output_root / 'runs'}")
    out = output_root / "all_window_mode_results_concise.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote aggregate FM concise results: {out}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.window_manifest), args.features, args.max_windows_per_feature)
    stages = {"all": ["embed", "train", "summarize"], "embed": ["embed"], "train": ["train", "summarize"], "summarize": ["summarize"]}[args.stage]
    for stage in stages:
        with timer(stage):
            if stage == "embed":
                stage_embed(args, manifest, output_root)
            elif stage == "train":
                stage_train(args, output_root)
            elif stage == "summarize":
                stage_summarize(output_root)


if __name__ == "__main__":
    main()
