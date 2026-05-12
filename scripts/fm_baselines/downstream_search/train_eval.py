#!/usr/bin/env python3
"""Training/evaluation utilities for framewise FM downstream search."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from dataset import EmbeddingStore, FramewiseEmbeddingDataset, compute_standardizer
from models import build_model


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_grouped_folds(labels: np.ndarray, subjects: np.ndarray, n_splits: int, val_ratio: float, random_seed: int) -> list[FoldSplit]:
    indices = np.arange(len(labels), dtype=np.int64)
    groups = subjects.astype(str)
    if len(np.unique(groups)) < int(n_splits):
        raise ValueError(f"Need at least {n_splits} participants, found {len(np.unique(groups))}.")
    folds = []
    outer = GroupKFold(n_splits=int(n_splits))
    for fold_id, (train_val_pos, test_pos) in enumerate(outer.split(indices, labels, groups), start=1):
        train_val_indices = indices[train_val_pos]
        train_val_groups = groups[train_val_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(random_seed) + fold_id)
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
    return (y_score >= float(threshold)).astype(np.int64)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = apply_threshold(y_score, float(threshold))
        score = float(balanced_accuracy_score(y_true, pred))
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


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float, label_smoothing: float):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_loaders(
    store: EmbeddingStore,
    fold: FoldSplit,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    weighted_sampler: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_ds = FramewiseEmbeddingDataset(store, fold.train_indices, mean=mean, std=std)
    train_eval_ds = FramewiseEmbeddingDataset(store, fold.train_indices, mean=mean, std=std)
    val_ds = FramewiseEmbeddingDataset(store, fold.val_indices, mean=mean, std=std)
    test_ds = FramewiseEmbeddingDataset(store, fold.test_indices, mean=mean, std=std)
    sampler = None
    if weighted_sampler:
        labels = store.y[fold.train_indices]
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        sample_weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=sampler is None, sampler=sampler, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    return train_loader, train_eval_loader, val_loader, test_loader


def collect_scores(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, scores, source_indices = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            logits = model(x.to(device=device, dtype=torch.float32))
            prob = torch.softmax(logits, dim=1)[:, 1]
            labels.append(y.numpy())
            scores.append(prob.detach().cpu().numpy())
            source_indices.append(idx.numpy())
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(source_indices)


def make_criterion(labels: np.ndarray, params: dict, device: str) -> nn.Module:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    if params["loss"] == "focal":
        return FocalLoss(class_weights, gamma=float(params["focal_gamma"]), label_smoothing=float(params["label_smoothing"]))
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(params["label_smoothing"]))


def train_one_fold(
    *,
    store: EmbeddingStore,
    fold: FoldSplit,
    model_family: str,
    params: dict,
    device: str,
    random_seed: int,
    output_dir: Path | None = None,
    trial=None,
    no_progress: bool = False,
) -> tuple[list[dict], list[dict], pd.DataFrame, float]:
    set_seed(int(random_seed) + int(fold.fold_id))
    device = resolve_device(device)
    mean, std = compute_standardizer(store, fold.train_indices, batch_size=64)
    train_loader, train_eval_loader, val_loader, test_loader = make_loaders(
        store,
        fold,
        mean,
        std,
        batch_size=int(params["batch_size"]),
        weighted_sampler=True,
    )
    model = build_model(model_family, input_dim=store.x_shape[2], seq_len=store.x_shape[1], params=params).to(device)
    criterion = make_criterion(store.y[fold.train_indices], params, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))

    best_state = None
    best_threshold = 0.5
    best_val = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    epoch_iter = range(1, int(params["max_epochs"]) + 1)
    if tqdm and not no_progress:
        epoch_iter = tqdm(epoch_iter, desc=f"fold {fold.fold_id} train", unit="epoch", leave=False)
    for epoch in epoch_iter:
        model.train()
        losses = []
        for x, y, _idx in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if float(params["grad_clip_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(params["grad_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_y, val_scores, _ = collect_scores(model, val_loader, device)
        threshold, val_ba = find_best_threshold(val_y, val_scores)
        train_y, train_scores, _ = collect_scores(model, train_eval_loader, device)
        train_metrics = compute_metrics(train_y, apply_threshold(train_scores, threshold), train_scores)
        val_metrics = compute_metrics(val_y, apply_threshold(val_scores, threshold), val_scores)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "threshold": threshold,
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)
        if tqdm and not no_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                {
                    "loss": f"{row['loss']:.4f}",
                    "val_ba": f"{val_metrics['balanced_accuracy']:.3f}",
                    "val_auc": f"{val_metrics['auc']:.3f}" if np.isfinite(val_metrics["auc"]) else "nan",
                    "thr": f"{threshold:.2f}",
                }
            )
        if trial is not None:
            trial.report(val_metrics["balanced_accuracy"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()  # type: ignore[union-attr]
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_threshold = float(threshold)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(params["patience"]):
            break

    if best_state is None:
        raise ValueError("No valid training epoch completed.")
    model.load_state_dict(best_state)
    metric_rows = []
    pred_rows = []
    for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_score, source_idx = collect_scores(model, loader, device)
        y_pred = apply_threshold(y_score, best_threshold)
        metrics = compute_metrics(y_true, y_pred, y_score)
        metric_rows.append(
            {
                "fold_id": fold.fold_id,
                "split": split,
                "threshold": best_threshold,
                "best_epoch": best_epoch,
                **metrics,
            }
        )
        for idx, label, score, pred in zip(source_idx, y_true, y_score, y_pred):
            pred_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "split": split,
                    "source_index": int(idx),
                    "window_id": int(store.window_id[int(idx)]),
                    "base_subject_id": str(store.base_subject_id[int(idx)]),
                    "label": int(label),
                    "score": float(score),
                    "prediction": int(pred),
                    "threshold": best_threshold,
                }
            )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "params": params,
                "model_family": model_family,
                "input_dim": store.x_shape[2],
                "seq_len": store.x_shape[1],
                "mean": mean,
                "std": std,
                "threshold": best_threshold,
                "best_epoch": best_epoch,
            },
            output_dir / f"fold_{fold.fold_id}.pt",
        )
    return metric_rows, history, pd.DataFrame(pred_rows), float(best_val)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def fold_assignments_frame(store: EmbeddingStore, folds: list[FoldSplit]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        for split, indices in [("train", fold.train_indices), ("val", fold.val_indices), ("test", fold.test_indices)]:
            for idx in indices:
                rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "split": split,
                        "source_index": int(idx),
                        "window_id": int(store.window_id[int(idx)]),
                        "base_subject_id": str(store.base_subject_id[int(idx)]),
                        "label": int(store.y[int(idx)]),
                    }
                )
    return pd.DataFrame(rows)
