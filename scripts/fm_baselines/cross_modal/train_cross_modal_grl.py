#!/usr/bin/env python3
"""
Cross-modal depth training with Gradient Reversal Layer (GRL) alignment.

Variant of train_cross_modal.py that replaces the cosine alignment loss with
a domain-adversarial discriminator attached via a GRL.  Everything else is
identical: same folds, Optuna protocol, subject-norm eval, checkpointing.

Architecture difference
-----------------------
  train_cross_modal.py  : L = L_cls + λ_align * (1 - cos(z_d, z_i))
  this script           : L = L_cls + λ_grl   * L_domain

  L_domain = CE(D(GRL(z_d) ∥ z_i.detach()), [depth_labels ∥ imu_labels])

The GRL (alpha=1) reverses gradients flowing back from L_domain into the
shared MLP and depth encoder, forcing them to produce depth representations
that the discriminator cannot distinguish from IMU representations.

The discriminator weights receive un-reversed gradients so they keep improving
at telling the modalities apart — this is the adversarial tug-of-war.

At inference the discriminator and GRL are discarded; only the depth path is
used (identical to train_cross_modal.py).

Optuna search space changes vs train_cross_modal.py
----------------------------------------------------
  Removed : lambda_align, lambda_recon, use_decoder
  Added   : lambda_grl, use_grl_schedule

Usage
-----
python train_cross_modal_grl.py --config config.json \
    --run-name cross_modal_grl_v1

Smoke test:
python train_cross_modal_grl.py --config config.json \
    --run-name grl_smoke --max-folds 1 --optuna-trials 3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise SystemExit("Install optuna: pip install optuna") from exc

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from paired_dataset import PairedDepthIMUDataset
from depth_models import load_depth_tcn_encoder
from cross_modal_model import CrossModalDepthModel


# ---------------------------------------------------------------------------
# GRL components  (new in this variant)
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Identity in the forward pass; reverses (and scales) gradients in backward."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


class ModalityDiscriminator(nn.Module):
    """
    2-class MLP: predicts whether a shared-space vector came from depth (0) or IMU (1).
    Receives gradients normally; the adversarial effect comes from the GRL upstream.
    """

    def __init__(self, shared_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(32, shared_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Metrics (identical to train_cross_modal.py)
# ---------------------------------------------------------------------------

THRESHOLDS = np.linspace(0.05, 0.95, 19)


def _safe(fn, y_true, y_score_or_pred):
    try:
        return float(fn(y_true, y_score_or_pred))
    except Exception:
        return float("nan")


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_thr, best_ba = 0.5, -np.inf
    for thr in THRESHOLDS:
        ba = float(balanced_accuracy_score(y_true, (y_score >= thr).astype(int)))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr, best_ba


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": _safe(roc_auc_score, y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_precision": _safe(average_precision_score, y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Dataset (identical to train_cross_modal.py)
# ---------------------------------------------------------------------------

class PairedWithTeacher(torch.utils.data.Dataset):
    def __init__(self, base_dataset, teacher_embeddings, pair_index_to_teacher_pos,
                 indices, fm_mean, fm_std):
        self.base = base_dataset
        self.teacher_embeddings = teacher_embeddings
        self.pair_index_to_pos = pair_index_to_teacher_pos
        self.indices = indices
        self.fm_mean = fm_mean
        self.fm_std = fm_std
        self.pairs = [base_dataset._all_pairs[i] for i in indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.base[int(self.indices[idx])]
        depth_x = item["depth_embedding"].float()
        if self.fm_mean is not None:
            mean = torch.as_tensor(self.fm_mean, dtype=torch.float32)
            std = torch.as_tensor(self.fm_std, dtype=torch.float32)
            depth_x = (depth_x - mean) / std
        label = item["label"]
        pair_idx = item["pair_index"]
        teacher_pos = self.pair_index_to_pos.get(pair_idx, -1)
        if teacher_pos >= 0:
            imu_embed = torch.from_numpy(self.teacher_embeddings[teacher_pos]).float()
        else:
            imu_embed = torch.zeros(self.teacher_embeddings.shape[1], dtype=torch.float32)
        return depth_x, imu_embed, label


# ---------------------------------------------------------------------------
# Fold / seed utilities (identical to train_cross_modal.py)
# ---------------------------------------------------------------------------

def create_folds(meta_df, n_splits, val_ratio, random_seed):
    indices = meta_df.index.to_numpy()
    groups = meta_df["base_subject_id"].to_numpy()
    labels = meta_df["label"].to_numpy()
    folds = []
    outer = GroupKFold(n_splits=n_splits)
    for fold_id, (tv_pos, test_pos) in enumerate(outer.split(indices, labels, groups), start=1):
        tv_idx = indices[tv_pos]
        test_idx = indices[test_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed + fold_id)
        train_pos, val_pos = next(inner.split(tv_idx, groups=groups[tv_pos]))
        folds.append({
            "fold_id": fold_id,
            "train": np.sort(tv_idx[train_pos]),
            "val": np.sort(tv_idx[val_pos]),
            "test": np.sort(test_idx),
        })
    return folds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_sampler(labels):
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Inference helpers (identical to train_cross_modal.py)
# ---------------------------------------------------------------------------

def collect_scores(model, loader, device):
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for depth_x, _imu_embed, label in loader:
            depth_x = depth_x.to(device=device, dtype=torch.float32)
            probs = model.predict_scores(depth_x)
            ys.append(label.numpy())
            scores.append(probs.cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def collect_scores_subject_normalized(model, dataset, pair_indices, pop_mean, pop_std,
                                       device, batch_size=64, min_windows=3):
    from collections import defaultdict
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("zarr not installed") from exc

    group = zarr.open_group(str(dataset.fm_store_path), mode="r")
    x_array = group["X"]
    pair_map = {p.pair_index: p for p in dataset._all_pairs}
    pairs = [pair_map[i] for i in pair_indices if i in pair_map]
    N = len(pairs)
    T, D = dataset.fm_shape[1], dataset.fm_shape[2]
    X_norm = np.zeros((N, T, D), dtype=np.float32)

    subject_positions = defaultdict(list)
    for pos, pair in enumerate(pairs):
        subject_positions[pair.base_subject_id].append(pos)

    for subj, positions in subject_positions.items():
        source_indices = np.array([pairs[p].fm_source_index for p in positions], dtype=np.int64)
        sort_order = np.argsort(source_indices)
        X_raw = np.asarray(
            x_array.get_orthogonal_selection((source_indices[sort_order], slice(None), slice(None))),
            dtype=np.float32,
        )
        if len(positions) >= min_windows:
            flat = X_raw.reshape(-1, D).astype(np.float64)
            s_mean = flat.mean(0).astype(np.float32)
            s_std = flat.std(0).astype(np.float32)
            s_std[s_std < 1e-4] = 1.0
        else:
            s_mean, s_std = pop_mean, pop_std
        X_subj_norm = (X_raw - s_mean[None, None, :]) / s_std[None, None, :]
        for local_i, pos in enumerate(np.array(positions)[sort_order]):
            X_norm[pos] = X_subj_norm[local_i]

    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(X_norm[start:end]).to(device=device, dtype=torch.float32)
            probs = model.predict_scores(batch)
            scores.append(probs.cpu().numpy())
            ys.append(np.array([pairs[i].label for i in range(start, end)]))
    return np.concatenate(ys), np.concatenate(scores)


# ---------------------------------------------------------------------------
# Training loop  — GRL variant
# ---------------------------------------------------------------------------

def train_one_fold(
    *,
    model: CrossModalDepthModel,
    discriminator: ModalityDiscriminator,
    grl: GradientReversalLayer,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    params: dict,
    device: str,
    fold_id: int,
    output_dir: Path | None,
    random_seed: int,
    paired_dataset=None,
    val_pair_indices=None,
    test_pair_indices=None,
    pop_mean=None,
    pop_std=None,
) -> tuple[list[dict], list[dict]]:
    set_seed(random_seed + fold_id)
    model = model.to(device)
    discriminator = discriminator.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    lambda_grl = float(params["lambda_grl"])
    use_schedule = bool(params.get("use_grl_schedule", False))
    max_epochs = int(params["max_epochs"])

    best_state = None
    best_disc_state = None
    best_threshold = 0.5
    best_val_ba = -np.inf
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        # DANN-style lambda schedule: ramp from ~0 to lambda_grl
        if use_schedule:
            progress = (epoch - 1) / max(max_epochs - 1, 1)
            current_lambda = lambda_grl * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        else:
            current_lambda = lambda_grl
        grl.set_alpha(current_lambda)

        model.train()
        discriminator.train()
        total_loss = 0.0
        n_batches = 0

        for depth_x, imu_embed, labels in train_loader:
            depth_x  = depth_x.to(device=device, dtype=torch.float32)
            imu_embed = imu_embed.to(device=device, dtype=torch.float32)
            labels   = labels.to(device=device, dtype=torch.long)

            # Depth classification path
            z_depth  = model.encode_depth(depth_x)
            logits_d = model.classifier(z_depth)
            loss_cls = F.cross_entropy(logits_d, labels)

            # IMU path — imu_embed frozen (detached), but imu_in_proj trainable.
            # loss_cls_imu trains imu_in_proj so z_imu is stress-discriminative
            # before it serves as the stable reference distribution for the GRL.
            z_imu        = model.encode_imu(imu_embed.detach())
            logits_imu   = model.classifier(z_imu)
            loss_cls_imu = F.cross_entropy(logits_imu, labels)

            # GRL domain alignment.
            # z_depth gets reversed gradients (confuses discriminator).
            # z_imu is detached here: the IMU side is a stable reference —
            # the discriminator learns to recognise it, the depth encoder
            # learns to match it via GRL, independently of loss_cls_imu.
            z_all = torch.cat([grl(z_depth), z_imu.detach()], dim=0)
            domain_labels = torch.cat([
                torch.zeros(z_depth.size(0), dtype=torch.long, device=device),
                torch.ones(z_imu.size(0),   dtype=torch.long, device=device),
            ], dim=0)
            loss_domain = F.cross_entropy(discriminator(z_all), domain_labels)

            # grl.alpha scales the reversed gradient — no lambda multiplier here.
            loss = loss_cls + loss_cls_imu + loss_domain

            optimizer.zero_grad()
            loss.backward()
            if float(params.get("grad_clip_norm", 0.0)) > 0:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(discriminator.parameters()),
                    float(params["grad_clip_norm"]),
                )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Val eval — subject-normalized (threshold selection under deployment conditions)
        model.eval()
        if paired_dataset is not None and val_pair_indices is not None:
            val_y, val_scores = collect_scores_subject_normalized(
                model, paired_dataset, val_pair_indices, pop_mean, pop_std,
                device, batch_size=int(params.get("batch_size", 64)),
            )
        else:
            val_y, val_scores = collect_scores(model, val_loader, device)

        threshold, val_ba = find_best_threshold(val_y, val_scores)

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_balanced_accuracy": val_ba,
            "threshold": threshold,
            "lambda_grl": current_lambda,
        })

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_disc_state = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(params["patience"]):
            break

    if best_state is None:
        raise ValueError("No valid training epoch completed.")
    model.load_state_dict(best_state)

    metric_rows = []
    for split, loader, sn_indices in [
        ("train", train_eval_loader, None),
        ("val",   val_loader,        val_pair_indices),
        ("test",  test_loader,       test_pair_indices),
    ]:
        if paired_dataset is not None and sn_indices is not None:
            y, scores = collect_scores_subject_normalized(
                model, paired_dataset, sn_indices, pop_mean, pop_std,
                device, batch_size=int(params.get("batch_size", 64)),
            )
        else:
            y, scores = collect_scores(model, loader, device)
        metrics = compute_metrics(y, scores, best_threshold)
        metric_rows.append({"fold_id": fold_id, "split": split, "threshold": best_threshold, **metrics})

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save only the depth model (discriminator discarded at inference)
        torch.save({
            "state_dict": best_state,
            "params": params,
            "threshold": best_threshold,
            "fold_id": fold_id,
        }, output_dir / f"fold_{fold_id}.pt")
        with open(output_dir / f"fold_{fold_id}_history.json", "w") as f:
            json.dump(history, f, indent=2)

    return metric_rows, history


# ---------------------------------------------------------------------------
# Optuna search space  — GRL variant
# ---------------------------------------------------------------------------

def sample_params(trial: optuna.Trial) -> dict:
    use_grl_schedule = trial.suggest_categorical("use_grl_schedule", [True, False])
    return {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "lambda_grl":       trial.suggest_float("lambda_grl", 0.05, 1.0),
        "use_grl_schedule": use_grl_schedule,
        "shared_dim":       trial.suggest_categorical("shared_dim", [64, 128, 256]),
        "proj_dropout":     trial.suggest_float("proj_dropout", 0.0, 0.3),
        "batch_size":       trial.suggest_categorical("batch_size", [32, 64, 128]),
        "max_epochs":       trial.suggest_categorical("max_epochs", [30, 50, 75]),
        "patience":         trial.suggest_categorical("patience", [8, 12, 16]),
        "grad_clip_norm":   trial.suggest_categorical("grad_clip_norm", [0.0, 1.0, 5.0]),
    }


# ---------------------------------------------------------------------------
# CLI (identical to train_cross_modal.py)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-modal depth TCN training with GRL alignment.")
    p.add_argument("--config", default=None)
    p.add_argument("--fm-store", default=None)
    p.add_argument("--fm-meta-csv", default=None)
    p.add_argument("--imu-data-root", default=None)
    p.add_argument("--imu-channel-mode", default=None)
    p.add_argument("--teacher-embeddings", default=None)
    p.add_argument("--depth-ckpt-dir", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--optuna-trials", type=int, default=None)
    p.add_argument("--tune-fold-id", type=int, default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--best-params-json", default=None,
                   help="Path to best_params.json from a previous run. Skips Optuna.")
    args = p.parse_args()
    if args.config:
        from config_utils import load_config, apply_config, teacher_embeddings_dir
        cfg = load_config(args.config)
        apply_config(args, cfg)
        if args.teacher_embeddings is None:
            args.teacher_embeddings = str(teacher_embeddings_dir(cfg))
    for attr, default in [("n_splits", 5), ("val_ratio", 0.2), ("random_seed", 42),
                           ("optuna_trials", 50), ("tune_fold_id", 1),
                           ("device", "cuda"), ("imu_channel_mode", "raw_absdelta"),
                           ("output_root", "outputs_cross_modal")]:
        if getattr(args, attr) is None:
            setattr(args, attr, default)
    for required in ["fm_store", "fm_meta_csv", "imu_data_root", "depth_ckpt_dir", "teacher_embeddings"]:
        if getattr(args, required) is None:
            p.error(f"--{required.replace('_','-')} is required (or set in --config)")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    run_name = args.run_name or datetime.now().strftime("cross_modal_grl_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading paired dataset...")
    dataset = PairedDepthIMUDataset(
        fm_store_path=Path(args.fm_store),
        fm_metadata_csv_path=Path(args.fm_meta_csv),
        imu_data_root=Path(args.imu_data_root),
        imu_channel_mode=args.imu_channel_mode,
        verbose=True,
    )

    # Load teacher embeddings — single file or per-fold directory
    teacher_path = Path(args.teacher_embeddings)
    if teacher_path.is_dir():
        per_fold_teacher = {}
        for fold_file in sorted(teacher_path.glob("fold_*.npz")):
            fold_num = int(fold_file.stem.split("_")[1])
            per_fold_teacher[fold_num] = fold_file
        if not per_fold_teacher:
            raise SystemExit(f"No fold_N.npz files found in {teacher_path}")
        print(f"Per-fold teacher embeddings found: folds {sorted(per_fold_teacher)}")
        _sample = np.load(str(list(per_fold_teacher.values())[0]))
        imu_dim = _sample["embeddings"].shape[1]
        teacher_embeddings = None
        pair_index_to_pos = None
    else:
        print(f"Loading teacher embeddings from {args.teacher_embeddings}...")
        npz = np.load(args.teacher_embeddings)
        teacher_embeddings = npz["embeddings"]
        pair_index_to_pos = {int(pi): pos for pos, pi in enumerate(npz["pair_indices"])}
        imu_dim = teacher_embeddings.shape[1]
        per_fold_teacher = {}
        print(f"Teacher embeddings: {teacher_embeddings.shape}, dim={imu_dim}")

    meta_df = dataset.pairs_metadata_frame().reset_index(drop=True)
    folds = create_folds(meta_df, args.n_splits, args.val_ratio, args.random_seed)
    if args.max_folds:
        folds = folds[: args.max_folds]

    depth_ckpt_dir = Path(args.depth_ckpt_dir)

    config = vars(args)
    config["imu_dim"] = imu_dim
    config["n_pairs"] = dataset.n_pairs_total
    config["variant"] = "grl"
    (output_root / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # ---------------------------------------------------------------------------
    # Optuna tuning — skipped if --best-params-json provided
    # ---------------------------------------------------------------------------
    if args.best_params_json:
        with open(args.best_params_json) as f:
            best_params = json.load(f)
        print(f"\nSkipping Optuna — loaded best params from {args.best_params_json}")
        print(f"Params: {best_params}")
        with open(output_root / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        # Jump straight to fold evaluation
        _skip_optuna = True
    else:
        _skip_optuna = False

    if not _skip_optuna:
        print(f"\nOptuna: tuning on fold {args.tune_fold_id} with {args.optuna_trials} trials...")
        tune_fold = next((f for f in folds if f["fold_id"] == args.tune_fold_id), folds[0])
        tune_ckpt = depth_ckpt_dir / f"fold_{tune_fold['fold_id']}.pt"
        if not tune_ckpt.exists():
            raise SystemExit(f"Depth checkpoint not found: {tune_ckpt}")

        print("Computing FM normalization stats from train fold...")
        fm_mean, fm_std = dataset.compute_fm_normalizer(tune_fold["train"].tolist())

        tune_fold_id = tune_fold["fold_id"]
        if per_fold_teacher:
            _npz_path = per_fold_teacher.get(tune_fold_id, list(per_fold_teacher.values())[0])
            _npz = np.load(str(_npz_path))
            tune_teacher_embeddings = _npz["embeddings"]
            tune_pair_index_to_pos = {int(pi): pos for pos, pi in enumerate(_npz["pair_indices"])}
        else:
            tune_teacher_embeddings = teacher_embeddings
            tune_pair_index_to_pos = pair_index_to_pos

        def make_split_dataset(fold_indices):
            return PairedWithTeacher(
                dataset, tune_teacher_embeddings, tune_pair_index_to_pos,
                fold_indices, fm_mean, fm_std,
            )

        def objective(trial: optuna.Trial) -> float:
            params = sample_params(trial)
            shared_dim = int(params["shared_dim"])
            encoder, _ = load_depth_tcn_encoder(tune_ckpt, device=device)
            model = CrossModalDepthModel(
                encoder=encoder,
                shared_dim=shared_dim,
                imu_dim=imu_dim,
                dropout=float(params["proj_dropout"]),
                use_decoder=False,
            ).to(device)
            disc = ModalityDiscriminator(shared_dim, dropout=float(params["proj_dropout"])).to(device)
            grl = GradientReversalLayer(alpha=1.0)

            train_ds = make_split_dataset(tune_fold["train"])
            val_ds   = make_split_dataset(tune_fold["val"])
            train_labels = np.array([p.label for p in train_ds.pairs])
            sampler = weighted_sampler(train_labels)
            train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), sampler=sampler)
            train_eval_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]), shuffle=False)

            try:
                metric_rows, _ = train_one_fold(
                    model=model,
                    discriminator=disc,
                    grl=grl,
                    train_loader=train_loader,
                    train_eval_loader=train_eval_loader,
                    val_loader=val_loader,
                    test_loader=val_loader,
                    params=params,
                    device=device,
                    fold_id=tune_fold["fold_id"],
                    output_dir=None,
                    random_seed=args.random_seed,
                    paired_dataset=dataset,
                    val_pair_indices=tune_fold["val"],
                    test_pair_indices=tune_fold["val"],
                    pop_mean=fm_mean,
                    pop_std=fm_std,
                )
                val_ba = next((r["balanced_accuracy"] for r in metric_rows if r["split"] == "val"), 0.0)
                return float(val_ba)
            except Exception as e:
                import traceback
                print(f"\n  [Optuna trial {trial.number}] FAILED: {e}")
                traceback.print_exc()
                return 0.0

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=not args.no_progress)
        best_params = study.best_trial.params
        print(f"Best val BA: {study.best_trial.value:.4f}")
        print(f"Best params: {best_params}")
        with open(output_root / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

    # ---------------------------------------------------------------------------
    # Evaluate on all folds with best params
    # ---------------------------------------------------------------------------
    print(f"\nEvaluating on all {len(folds)} folds...")
    all_metric_rows = []

    for fold in folds:
        fold_id = fold["fold_id"]
        ckpt_path = depth_ckpt_dir / f"fold_{fold_id}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found for fold {fold_id}, skipping.")
            continue

        fold_teacher_embeddings = teacher_embeddings
        fold_pair_index_to_pos = pair_index_to_pos
        if per_fold_teacher:
            npz_path = per_fold_teacher.get(fold_id, list(per_fold_teacher.values())[0])
            npz = np.load(str(npz_path))
            fold_teacher_embeddings = npz["embeddings"]
            fold_pair_index_to_pos = {int(pi): pos for pos, pi in enumerate(npz["pair_indices"])}

        print(f"  Fold {fold_id}...")
        fm_mean_f, fm_std_f = dataset.compute_fm_normalizer(fold["train"].tolist())

        def make_ds_fold(indices):
            return PairedWithTeacher(
                dataset, fold_teacher_embeddings, fold_pair_index_to_pos,
                indices, fm_mean_f, fm_std_f,
            )

        bs = int(best_params.get("batch_size", 64))
        train_ds   = make_ds_fold(fold["train"])
        val_ds     = make_ds_fold(fold["val"])
        test_ds    = make_ds_fold(fold["test"])
        train_labels = np.array([p.label for p in train_ds.pairs])
        sampler = weighted_sampler(train_labels)
        train_loader      = DataLoader(train_ds, batch_size=bs, sampler=sampler)
        train_eval_loader = DataLoader(train_ds, batch_size=bs, shuffle=False)
        val_loader        = DataLoader(val_ds,   batch_size=bs, shuffle=False)
        test_loader       = DataLoader(test_ds,  batch_size=bs, shuffle=False)

        shared_dim = int(best_params.get("shared_dim", 128))
        encoder, _ = load_depth_tcn_encoder(ckpt_path, device=device)
        model = CrossModalDepthModel(
            encoder=encoder,
            shared_dim=shared_dim,
            imu_dim=imu_dim,
            dropout=float(best_params.get("proj_dropout", 0.1)),
            use_decoder=False,
        )
        disc = ModalityDiscriminator(shared_dim, dropout=float(best_params.get("proj_dropout", 0.1)))
        grl  = GradientReversalLayer(alpha=1.0)

        fold_dir = output_root / f"fold_{fold_id}"
        metric_rows, _ = train_one_fold(
            model=model,
            discriminator=disc,
            grl=grl,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            params=best_params,
            device=device,
            fold_id=fold_id,
            output_dir=fold_dir,
            random_seed=args.random_seed,
            paired_dataset=dataset,
            val_pair_indices=fold["val"],
            test_pair_indices=fold["test"],
            pop_mean=fm_mean_f,
            pop_std=fm_std_f,
        )
        for row in metric_rows:
            row["run_name"] = run_name
        all_metric_rows.extend(metric_rows)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    results_df = pd.DataFrame(all_metric_rows)
    results_df.to_csv(output_root / "per_fold_metrics.csv", index=False)

    test_df = results_df[results_df["split"] == "test"]
    summary_cols = ["balanced_accuracy", "auc", "sensitivity", "specificity", "mcc"]
    summary = test_df[summary_cols].agg(["mean", "std"])
    print("\n=== Test metrics (mean ± std across folds) ===")
    for col in summary_cols:
        m, s = summary.loc["mean", col], summary.loc["std", col]
        print(f"  {col:25s}: {m:.3f} ± {s:.3f}")

    summary.to_csv(output_root / "test_summary.csv")


if __name__ == "__main__":
    main()
