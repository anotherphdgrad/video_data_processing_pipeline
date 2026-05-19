#!/usr/bin/env python3
"""
Subject-normalized inference for saved cross-modal depth classifiers.

Analogous to ../downstream_search/infer_subject_normalized.py but for
CrossModalDepthModel checkpoints produced by train_cross_modal.py.

Re-runs test inference on each saved fold checkpoint, replacing the
population normalization used at training time with per-subject normalization
(mean/std computed from each test subject's own windows in the FM zarr store).

Both strategies are reported so the delta is directly comparable.

Usage
-----
python infer_subject_normalized_cross_modal.py \
    --run-dir /path/to/outputs_cross_modal/my_run \
    --device cuda

# Smoke: one fold only
python infer_subject_normalized_cross_modal.py \
    --run-dir /path/to/outputs_cross_modal/my_run \
    --max-folds 1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from cross_modal_model import CrossModalDepthModel
from depth_models import load_depth_tcn_encoder
from paired_dataset import PairedDepthIMUDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    classes = np.unique(y_true)
    if len(classes) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Fold reconstruction (must match train_cross_modal.py exactly)
# ---------------------------------------------------------------------------

def create_folds(meta_df: pd.DataFrame, n_splits: int, val_ratio: float, random_seed: int):
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


# ---------------------------------------------------------------------------
# Per-subject normalized inference
# ---------------------------------------------------------------------------

def collect_scores_subject_normalized(
    model: CrossModalDepthModel,
    dataset: PairedDepthIMUDataset,
    pair_indices: np.ndarray,
    pop_mean: np.ndarray,
    pop_std: np.ndarray,
    device: str,
    batch_size: int = 64,
    min_windows: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (y_true, y_score, n_fallback_subjects).
    Normalizes each test subject's FM embeddings by their own mean/std;
    subjects with fewer than min_windows windows fall back to pop stats.
    """
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

    n_fallback = 0
    for subj, positions in subject_positions.items():
        source_indices = np.array([pairs[p].fm_source_index for p in positions], dtype=np.int64)
        sort_order = np.argsort(source_indices)
        X_raw = np.asarray(
            x_array.get_orthogonal_selection(
                (source_indices[sort_order], slice(None), slice(None))
            ),
            dtype=np.float32,
        )
        if len(positions) >= min_windows:
            flat = X_raw.reshape(-1, D).astype(np.float64)
            s_mean = flat.mean(0).astype(np.float32)
            s_std = flat.std(0).astype(np.float32)
            s_std[s_std < 1e-4] = 1.0
        else:
            s_mean, s_std = pop_mean, pop_std
            n_fallback += 1
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

    return np.concatenate(ys), np.concatenate(scores), n_fallback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-run cross-modal test inference with per-subject normalization."
    )
    p.add_argument("--run-dir", required=True,
                   help="Path to the cross-modal run directory (contains config.json, "
                        "per_fold_metrics.csv, fold_*/fold_*.pt).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results. Defaults to <run-dir>/subject_norm/.")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--min-windows", type=int, default=3,
                   help="Min windows per test subject before falling back to pop norm.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "subject_norm"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"{output_dir} already exists and is non-empty. Use --overwrite or --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Load run config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        cfg = json.load(f)

    fm_store = cfg["fm_store"]
    fm_meta_csv = cfg["fm_meta_csv"]
    imu_data_root = cfg["imu_data_root"]
    imu_channel_mode = cfg.get("imu_channel_mode", "raw_absdelta")
    depth_ckpt_dir = Path(cfg["depth_ckpt_dir"])
    imu_dim = int(cfg["imu_dim"])
    n_splits = int(cfg.get("n_splits", 5))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    random_seed = int(cfg.get("random_seed", 42))

    best_params_path = run_dir / "best_params.json"
    if not best_params_path.exists():
        raise SystemExit(f"best_params.json not found in {run_dir}")
    with open(best_params_path) as f:
        best_params = json.load(f)

    print("Loading paired dataset...")
    dataset = PairedDepthIMUDataset(
        fm_store_path=Path(fm_store),
        fm_metadata_csv_path=Path(fm_meta_csv),
        imu_data_root=Path(imu_data_root),
        imu_channel_mode=imu_channel_mode,
        verbose=True,
    )
    print(f"Total paired windows: {dataset.n_pairs_total}")

    meta_df = dataset.pairs_metadata_frame().reset_index(drop=True)
    folds = create_folds(meta_df, n_splits, val_ratio, random_seed)
    if args.max_folds:
        folds = folds[: args.max_folds]

    # Load original test metrics for comparison
    orig_metrics_path = run_dir / "per_fold_metrics.csv"
    orig_df = pd.read_csv(orig_metrics_path) if orig_metrics_path.exists() else pd.DataFrame()

    all_rows = []

    for fold in folds:
        fold_id = fold["fold_id"]
        ckpt_path = run_dir / f"fold_{fold_id}" / f"fold_{fold_id}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found for fold {fold_id} ({ckpt_path}) — skipping")
            continue

        print(f"\n  Fold {fold_id}...")

        # Recompute population stats from this fold's train split
        fm_mean, fm_std = dataset.compute_fm_normalizer(fold["train"].tolist())

        # Load checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        threshold = float(ckpt["threshold"])
        params = ckpt["params"]

        # Rebuild model
        depth_ckpt = depth_ckpt_dir / f"fold_{fold_id}.pt"
        if not depth_ckpt.exists():
            print(f"  WARNING: depth checkpoint not found for fold {fold_id} — skipping")
            continue
        encoder, _ = load_depth_tcn_encoder(depth_ckpt, device=device)
        model = CrossModalDepthModel(
            encoder=encoder,
            shared_dim=int(params.get("shared_dim", best_params.get("shared_dim", 128))),
            imu_dim=imu_dim,
            dropout=float(params.get("proj_dropout", best_params.get("proj_dropout", 0.1))),
            use_decoder=bool(params.get("use_decoder", best_params.get("use_decoder", True))),
        )
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)

        test_pair_indices = fold["test"]

        # Population-norm test metrics (from saved per_fold_metrics.csv if available)
        if not orig_df.empty:
            orig_test = orig_df[(orig_df["fold_id"] == fold_id) & (orig_df["split"] == "test")]
            if not orig_test.empty:
                row = orig_test.iloc[0]
                all_rows.append({
                    "fold_id": fold_id, "strategy": "population_norm",
                    "auc": row.get("auc", float("nan")),
                    "balanced_accuracy": row.get("balanced_accuracy", float("nan")),
                    "sensitivity": row.get("sensitivity", float("nan")),
                    "specificity": row.get("specificity", float("nan")),
                    "mcc": row.get("mcc", float("nan")),
                    "n_fallback_subjects": 0,
                })

        # Subject-norm inference
        y_true, y_score, n_fallback = collect_scores_subject_normalized(
            model=model,
            dataset=dataset,
            pair_indices=test_pair_indices,
            pop_mean=fm_mean,
            pop_std=fm_std,
            device=device,
            batch_size=args.batch_size,
            min_windows=args.min_windows,
        )
        metrics = compute_metrics(y_true, y_score, threshold)
        all_rows.append({
            "fold_id": fold_id, "strategy": "subject_norm",
            "n_fallback_subjects": n_fallback,
            **metrics,
        })
        pop_rows_this_fold = [r for r in all_rows if r["fold_id"] == fold_id and r["strategy"] == "population_norm"]
        if pop_rows_this_fold:
            pr = pop_rows_this_fold[0]
            print(f"    pop_norm  AUC={pr['auc']:.3f} BA={pr['balanced_accuracy']:.3f}")
        print(f"    subj_norm AUC={metrics['auc']:.3f} BA={metrics['balanced_accuracy']:.3f} "
              f"(fallback: {n_fallback} subjects)")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_dir / "per_fold_metrics.csv", index=False)

    # Summary table
    summary_cols = ["auc", "balanced_accuracy", "sensitivity", "specificity", "mcc"]
    rows_summary = []
    for strategy in ["population_norm", "subject_norm"]:
        sub = results_df[results_df["strategy"] == strategy]
        if sub.empty:
            continue
        row = {"strategy": strategy}
        for col in summary_cols:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        rows_summary.append(row)
    summary_df = pd.DataFrame(rows_summary)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Delta table
    if len(rows_summary) == 2:
        pop_row = next(r for r in rows_summary if r["strategy"] == "population_norm")
        sn_row  = next(r for r in rows_summary if r["strategy"] == "subject_norm")
        delta = {col: sn_row[f"{col}_mean"] - pop_row[f"{col}_mean"] for col in summary_cols}
    else:
        delta = {}

    # Print results
    pop_row = next((r for r in rows_summary if r["strategy"] == "population_norm"), {})
    sn_row  = next((r for r in rows_summary if r["strategy"] == "subject_norm"), {})
    print(f"\n{'='*60}")
    print("=== Test metrics (mean ± std across folds) ===")
    print(f"{'Metric':<25} {'population_norm':>20} {'subject_norm':>20} {'delta':>10}")
    print("-" * 75)
    for col in summary_cols:
        pop_m = pop_row.get(f"{col}_mean", float("nan"))
        pop_s = pop_row.get(f"{col}_std", float("nan"))
        sn_m  = sn_row.get(f"{col}_mean", float("nan"))
        sn_s  = sn_row.get(f"{col}_std", float("nan"))
        d = delta.get(col, float("nan"))
        sign = "+" if d > 0 else ""
        print(f"  {col:<23} {pop_m:.3f} ± {pop_s:.3f}       {sn_m:.3f} ± {sn_s:.3f}   {sign}{d:.3f}")

    print(f"\nWrote per-fold metrics : {output_dir / 'per_fold_metrics.csv'}")
    print(f"Wrote summary          : {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
