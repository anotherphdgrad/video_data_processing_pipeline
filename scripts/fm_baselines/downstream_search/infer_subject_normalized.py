#!/usr/bin/env python3
"""
Subject-normalized inference for saved downstream FM classifiers.

Re-runs inference on saved fold checkpoints, replacing the population
normalization (mean/std fit on the training fold) with per-subject
normalization (mean/std computed from each test subject's own windows).

WHY THIS HELPS
--------------
During training, a single population mean/std is computed across all training
participants and applied globally to val and test subjects too.  Test subjects
are held-out people whose embedding distributions may differ from the training
population (different body size, movement style, camera distance, etc.).
Normalizing each test subject by their own statistics re-centres their inputs
to match the space the model was trained in, reducing inter-subject
distributional shift without any retraining.

Non-destructive: reads existing checkpoints and zarr stores, writes to a
new --output-root.  No checkpoints or original fold_predictions.csv files
are modified.

Both strategies are written per fold so the comparison is direct:
  - population_norm   : original inference using training fold mean/std
                        (reproduced from the saved checkpoint stats)
  - subject_norm      : each test subject normalised by their own mean/std

Usage
-----
# Motion features, all encoders, all model families:
python infer_subject_normalized.py \
    --embedding-root /scratch/hsharm62/.../embeddings_zarr2_entropy_selected \
    --input-root     /scratch/hsharm62/.../outputs_downstream_pub \
    --output-root    /scratch/hsharm62/.../outputs_downstream_pub_subjectnorm \
    --device cuda

# Smoke test — one combo, two folds:
python infer_subject_normalized.py \
    --embedding-root /scratch/hsharm62/.../embeddings_zarr2_entropy_selected \
    --input-root     /scratch/hsharm62/.../outputs_downstream_pub \
    --output-root    /scratch/hsharm62/.../outputs_downstream_pub_subjectnorm_smoke \
    --encoders dinov2 \
    --features motion_prev_rgb \
    --model-families transformer_encoder \
    --max-folds 2 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from dataset import EmbeddingStore, load_embedding_store
from models import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_COLS = ["auc", "avg_precision", "balanced_accuracy", "sensitivity", "specificity", "mcc"]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auc": safe_auc(y_true, y_score),
        "avg_precision": safe_avg_precision(y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run inference with per-subject embedding normalization."
    )
    parser.add_argument(
        "--embedding-root",
        required=True,
        help="Root of the entropy-selected zarr stores, e.g. outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root that contains runs/*/*/*/*/fold_predictions.csv and checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="New directory for subject-norm inference outputs. Never overwrites source.",
    )
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run_name filter.")
    parser.add_argument("--encoders", nargs="*", default=["imagebind", "omnivore", "dinov2"])
    parser.add_argument("--features", nargs="*", default=["motion_prev_rgb", "motion_prev_depth"])
    parser.add_argument("--model-families", nargs="*", default=["attn_pool_mlp", "rnn_attn", "tcn", "transformer_encoder"])
    parser.add_argument(
        "--min-windows-for-subject-norm",
        type=int,
        default=3,
        help=(
            "Minimum number of test windows a subject must have before per-subject "
            "normalization is applied. Subjects below this threshold fall back to "
            "the population (training fold) mean/std. Default: 3."
        ),
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Process at most this many folds per combo (useful for smoke tests).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Combo discovery
# ---------------------------------------------------------------------------

def discover_combos(input_root: Path, args: argparse.Namespace) -> list[dict]:
    """Find all (run_name, encoder, feature, model_family) combos that have
    both a fold_predictions.csv and at least one checkpoint."""
    combos = []
    for pred_path in sorted(input_root.glob("runs/*/*/*/*/fold_predictions.csv")):
        parts = pred_path.relative_to(input_root).parts
        if len(parts) != 6:
            continue
        _, run_name, encoder, feature, model_family, _ = parts
        if args.runs is not None and run_name not in set(args.runs):
            continue
        if encoder not in set(args.encoders):
            continue
        if feature not in set(args.features):
            continue
        if model_family not in set(args.model_families):
            continue
        ckpt_dir = pred_path.parent / "checkpoints"
        if not ckpt_dir.exists() or not any(ckpt_dir.glob("fold_*.pt")):
            print(f"  WARNING: no checkpoints found for {run_name}/{encoder}/{feature}/{model_family} — skipping")
            continue
        combos.append({
            "run_name": run_name,
            "encoder": encoder,
            "feature": feature,
            "model_family": model_family,
            "pred_path": pred_path,
            "ckpt_dir": ckpt_dir,
        })
    return combos


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_subject_normalized_inference(
    *,
    model: torch.nn.Module,
    store: EmbeddingStore,
    test_indices: np.ndarray,
    pop_mean: np.ndarray,
    pop_std: np.ndarray,
    device: str,
    batch_size: int,
    min_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Loads raw embeddings from zarr, normalises per test subject, runs model.

    Returns (y_true, y_score, source_indices, n_fallback_subjects) where
    n_fallback_subjects is the number of subjects that fell back to population
    normalization due to having fewer than min_windows test windows.
    """
    import zarr

    test_indices = np.asarray(test_indices, dtype=np.int64)
    N = len(test_indices)
    T, D = store.x_shape[1], store.x_shape[2]

    group = zarr.open_group(str(store.store_path), mode="r")
    x_array = group["X"]

    subject_ids = store.base_subject_id[test_indices]
    X_norm = np.zeros((N, T, D), dtype=np.float32)
    n_fallback = 0

    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        subj_positions = np.where(subj_mask)[0]          # positions within test_indices
        subj_zarr_idxs = test_indices[subj_positions]    # actual zarr row indices

        # zarr orthogonal selection requires sorted indices
        sort_order = np.argsort(subj_zarr_idxs)
        sorted_zarr_idxs = subj_zarr_idxs[sort_order]

        X_raw = np.asarray(
            x_array.get_orthogonal_selection((sorted_zarr_idxs, slice(None), slice(None))),
            dtype=np.float32,
        )

        if len(subj_positions) >= min_windows:
            flat = X_raw.reshape(-1, D).astype(np.float64)
            s_mean = flat.mean(axis=0).astype(np.float32)
            s_std = flat.std(axis=0).astype(np.float32)
            s_std[s_std < 1e-4] = 1.0
        else:
            s_mean, s_std = pop_mean, pop_std
            n_fallback += 1

        X_subj_norm = (X_raw - s_mean[None, None, :]) / s_std[None, None, :]
        # Place back: X_raw[i] came from sorted_zarr_idxs[i], which is at
        # position subj_positions[sort_order[i]] in test_indices.
        X_norm[subj_positions[sort_order]] = X_subj_norm

    # Run model in batches
    model.eval()
    scores_list, labels_list, idx_list = [], [], []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = start + batch_size
            batch = torch.from_numpy(X_norm[start:end]).to(device=device, dtype=torch.float32)
            probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            scores_list.append(probs)
            labels_list.append(store.y[test_indices[start:end]])
            idx_list.append(test_indices[start:end])

    return (
        np.concatenate(labels_list),
        np.concatenate(scores_list),
        np.concatenate(idx_list),
        n_fallback,
    )


# ---------------------------------------------------------------------------
# Process one combo
# ---------------------------------------------------------------------------

def process_combo(
    combo: dict,
    store: EmbeddingStore,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict], list[dict]]:
    pred_path: Path = combo["pred_path"]
    ckpt_dir: Path = combo["ckpt_dir"]
    run_name = combo["run_name"]
    encoder = combo["encoder"]
    feature = combo["feature"]
    model_family = combo["model_family"]

    baseline_preds = pd.read_csv(pred_path)
    fold_ids = sorted(baseline_preds["fold_id"].unique())
    if args.max_folds is not None:
        fold_ids = fold_ids[: int(args.max_folds)]

    pred_rows: list[dict] = []
    metric_rows: list[dict] = []

    for fold_id in fold_ids:
        ckpt_path = ckpt_dir / f"fold_{fold_id}.pt"
        if not ckpt_path.exists():
            print(f"    WARNING: checkpoint missing for fold {fold_id} — skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pop_mean: np.ndarray = np.asarray(ckpt["mean"], dtype=np.float32)
        pop_std: np.ndarray = np.asarray(ckpt["std"], dtype=np.float32)
        threshold: float = float(ckpt["threshold"])
        params: dict = ckpt["params"]
        input_dim: int = int(ckpt["input_dim"])
        seq_len: int = int(ckpt["seq_len"])

        model = build_model(model_family, input_dim=input_dim, seq_len=seq_len, params=params)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()

        fold_baseline = baseline_preds[baseline_preds["fold_id"] == fold_id]
        test_baseline = fold_baseline[fold_baseline["split"] == "test"]

        if test_baseline.empty:
            print(f"    WARNING: no test rows in fold_predictions for fold {fold_id} — skipping")
            continue

        test_indices = test_baseline["source_index"].to_numpy(dtype=np.int64)

        # --- Population-norm baseline (reproduced from checkpoint stats) ---
        for split in ["train", "val", "test"]:
            split_df = fold_baseline[fold_baseline["split"] == split]
            if split_df.empty:
                continue
            y_true = split_df["label"].to_numpy(dtype=np.int64)
            y_score = split_df["score"].to_numpy(dtype=np.float64)
            y_pred = split_df["prediction"].to_numpy(dtype=np.int64)
            saved_thr = float(split_df["threshold"].iloc[0])
            metrics = compute_metrics(y_true, y_pred, y_score)
            metric_rows.append({
                "run_name": run_name, "encoder": encoder, "feature": feature,
                "model_family": model_family, "fold_id": fold_id,
                "split": split, "strategy": "population_norm",
                "n_windows": len(split_df),
                "n_stress": int(y_true.sum()),
                "n_nonstress": int((1 - y_true).sum()),
                "threshold": saved_thr,
                "n_fallback_subjects": 0,
                **metrics,
            })
            for _, row in split_df.iterrows():
                pred_rows.append({
                    "run_name": run_name, "encoder": encoder, "feature": feature,
                    "model_family": model_family, "fold_id": fold_id,
                    "split": split, "strategy": "population_norm",
                    "source_index": int(row["source_index"]),
                    "window_id": row["window_id"],
                    "base_subject_id": str(row["base_subject_id"]),
                    "label": int(row["label"]),
                    "score": float(row["score"]),
                    "prediction": int(row["prediction"]),
                    "threshold": float(row["threshold"]),
                    "n_fallback_subjects": 0,
                })

        # --- Subject-norm inference (test split only) ---
        y_true_sn, y_score_sn, source_idx_sn, n_fallback = run_subject_normalized_inference(
            model=model,
            store=store,
            test_indices=test_indices,
            pop_mean=pop_mean,
            pop_std=pop_std,
            device=device,
            batch_size=args.batch_size,
            min_windows=args.min_windows_for_subject_norm,
        )
        y_pred_sn = (y_score_sn >= threshold).astype(np.int64)
        metrics_sn = compute_metrics(y_true_sn, y_pred_sn, y_score_sn)
        metric_rows.append({
            "run_name": run_name, "encoder": encoder, "feature": feature,
            "model_family": model_family, "fold_id": fold_id,
            "split": "test", "strategy": "subject_norm",
            "n_windows": len(y_true_sn),
            "n_stress": int(y_true_sn.sum()),
            "n_nonstress": int((1 - y_true_sn).sum()),
            "threshold": threshold,
            "n_fallback_subjects": n_fallback,
            **metrics_sn,
        })
        subj_ids_sn = store.base_subject_id[source_idx_sn]
        for i in range(len(source_idx_sn)):
            pred_rows.append({
                "run_name": run_name, "encoder": encoder, "feature": feature,
                "model_family": model_family, "fold_id": fold_id,
                "split": "test", "strategy": "subject_norm",
                "source_index": int(source_idx_sn[i]),
                "window_id": int(store.window_id[source_idx_sn[i]]),
                "base_subject_id": str(subj_ids_sn[i]),
                "label": int(y_true_sn[i]),
                "score": float(y_score_sn[i]),
                "prediction": int(y_pred_sn[i]),
                "threshold": threshold,
                "n_fallback_subjects": n_fallback,
            })

    return pred_rows, metric_rows


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def aggregate_test_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    df = per_fold[per_fold["split"] == "test"].copy()
    group_cols = ["run_name", "encoder", "feature", "model_family", "strategy"]
    parts = []
    for col in METRIC_COLS:
        part = df.groupby(group_cols)[col].agg(
            **{f"{col}_mean": "mean", f"{col}_std": "std"}
        )
        parts.append(part)
    summary = pd.concat(parts, axis=1).reset_index()
    n_folds = df.groupby(group_cols)["fold_id"].nunique().reset_index(name="n_folds")
    summary = summary.merge(n_folds, on=group_cols)
    for col in METRIC_COLS:
        summary[f"{col}_fmt"] = summary.apply(
            lambda r, c=col: f"{r[f'{c}_mean']:.3f}±{r[f'{c}_std']:.3f}"
            if np.isfinite(r[f"{c}_mean"]) else "nan",
            axis=1,
        )
    return summary.sort_values("auc_mean", ascending=False)


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side delta: subject_norm minus population_norm per combo."""
    key_cols = ["run_name", "encoder", "feature", "model_family"]
    pop = summary[summary["strategy"] == "population_norm"].set_index(key_cols)
    sn = summary[summary["strategy"] == "subject_norm"].set_index(key_cols)
    rows = []
    for idx in sn.index:
        if idx not in pop.index:
            continue
        row = {col: idx[i] for i, col in enumerate(key_cols)}
        for col in METRIC_COLS:
            row[f"pop_{col}"] = pop.loc[idx, f"{col}_mean"]
            row[f"sn_{col}"] = sn.loc[idx, f"{col}_mean"]
            row[f"delta_{col}"] = row[f"sn_{col}"] - row[f"pop_{col}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_auc", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    embedding_root = Path(args.embedding_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"{output_root} already exists and is non-empty. "
            "Use --overwrite or choose another --output-root."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        device = "cpu"

    combos = discover_combos(input_root, args)
    if not combos:
        raise SystemExit("No combos found matching the requested filters.")

    print(f"Found {len(combos)} combo(s) to process.")

    iterator = combos
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(combos, desc="Subject-norm inference", unit="combo")

    all_pred_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    failures: list[dict] = []

    for combo in iterator:
        encoder, feature = combo["encoder"], combo["feature"]
        try:
            store = load_embedding_store(embedding_root, encoder, feature)
        except FileNotFoundError as exc:
            failures.append({**combo, "error": str(exc), "pred_path": str(combo["pred_path"])})
            print(f"  SKIPPED {encoder}/{feature}: {exc}")
            continue

        try:
            pred_rows, metric_rows = process_combo(combo, store, args, device)
            all_pred_rows.extend(pred_rows)
            all_metric_rows.extend(metric_rows)
        except Exception as exc:  # noqa: BLE001
            failures.append({**combo, "error": repr(exc), "pred_path": str(combo["pred_path"])})
            print(f"  ERROR {encoder}/{feature}/{combo['model_family']}: {exc}")

    if not all_metric_rows:
        raise SystemExit("No metrics produced. Check --embedding-root and --input-root paths.")

    per_fold = pd.DataFrame(all_metric_rows)
    predictions = pd.DataFrame(all_pred_rows)
    summary = aggregate_test_metrics(per_fold)
    comparison = build_comparison(summary)

    per_fold.to_csv(output_root / "subject_norm_per_fold_metrics.csv", index=False)
    predictions.to_csv(output_root / "subject_norm_fold_predictions.csv", index=False)
    summary.to_csv(output_root / "subject_norm_test_metrics.csv", index=False)
    comparison.to_csv(output_root / "subject_norm_comparison.csv", index=False)

    config = {
        "embedding_root": str(embedding_root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "encoders": args.encoders,
        "features": args.features,
        "model_families": args.model_families,
        "min_windows_for_subject_norm": args.min_windows_for_subject_norm,
        "device": device,
        "n_combos": len(combos),
        "n_failures": len(failures),
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if failures:
        pd.DataFrame([{k: str(v) for k, v in f.items()} for f in failures]).to_csv(
            output_root / "failures.csv", index=False
        )
        print(f"\nWARNING: {len(failures)} combo(s) failed — see failures.csv")

    print(f"\nWrote per-fold metrics : {output_root / 'subject_norm_per_fold_metrics.csv'}")
    print(f"Wrote test summary     : {output_root / 'subject_norm_test_metrics.csv'}")
    print(f"Wrote comparison       : {output_root / 'subject_norm_comparison.csv'}")

    print("\n=== Test AUC — population_norm vs subject_norm ===")
    disp_cols = ["encoder", "feature", "model_family", "strategy", "auc_fmt", "balanced_accuracy_fmt", "mcc_fmt"]
    avail = [c for c in disp_cols if c in summary.columns]
    print(summary[avail].head(20).to_string(index=False))

    if not comparison.empty:
        print("\n=== Mean delta: subject_norm − population_norm (positive = improvement) ===")
        delta_cols = [c for c in comparison.columns if c.startswith("delta_")]
        print(comparison[["encoder", "feature", "model_family"] + delta_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
