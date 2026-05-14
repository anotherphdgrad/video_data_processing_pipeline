#!/usr/bin/env python3
"""
Generate publication-ready metrics from downstream search outputs.

Reads fold_predictions.csv from every combo directory, recomputes per-fold
metrics from raw scores (not from pre-stored balanced_accuracy/auc), and writes:

  publication_test_metrics.csv   — one row per combo, test split only,
                                   mean ± std across folds; ready for paper tables
  all_splits_metrics.csv         — same but includes train and val splits
  per_fold_metrics_full.csv      — one row per combo × fold × split (detailed)

Metrics reported:
  - AUC (ROC-AUC)          threshold-free; primary clinical metric
  - Average Precision       PR-AUC; better than ROC-AUC when classes are imbalanced
  - Balanced Accuracy       already used; kept for consistency with other results
  - Sensitivity             recall for stress class (label=1); clinically important
  - Specificity             recall for non-stress class (label=0)
  - MCC                     Matthews Correlation Coefficient; single balanced
                            metric that does not depend on threshold selection bias

Usage:
    python generate_publication_metrics.py \
        --output-root /scratch/hsharm62/video_data_processing_pipeline/outputs_downstream_pub \
        --split test

    # include all splits (train / val / test)
    python generate_publication_metrics.py \
        --output-root outputs_downstream_pub \
        --split all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


METRIC_COLS = ["auc", "avg_precision", "balanced_accuracy", "sensitivity", "specificity", "mcc"]

METRIC_LABELS = {
    "auc": "AUC (ROC)",
    "avg_precision": "Avg Precision (PR-AUC)",
    "balanced_accuracy": "Balanced Accuracy",
    "sensitivity": "Sensitivity (Stress Recall)",
    "specificity": "Specificity (Non-stress Recall)",
    "mcc": "MCC",
}


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


def parse_combo_path(pred_path: Path) -> dict[str, str]:
    # structure: output_root/runs/{run_name}/{encoder}/{feature}/{model_family}/fold_predictions.csv
    model_family = pred_path.parent.name
    feature = pred_path.parent.parent.name
    encoder = pred_path.parent.parent.parent.name
    run_name = pred_path.parent.parent.parent.parent.name
    return {
        "run_name": run_name,
        "encoder": encoder,
        "feature": feature,
        "model_family": model_family,
    }


def process_predictions(pred_path: Path) -> list[dict]:
    combo = parse_combo_path(pred_path)
    preds = pd.read_csv(pred_path)

    required = {"fold_id", "split", "label", "score", "prediction", "threshold"}
    missing = required - set(preds.columns)
    if missing:
        print(f"  WARNING: skipping {pred_path} — missing columns {missing}")
        return []

    rows = []
    for fold_id, fold_df in preds.groupby("fold_id"):
        for split, split_df in fold_df.groupby("split"):
            y_true = split_df["label"].to_numpy(dtype=np.int64)
            y_pred = split_df["prediction"].to_numpy(dtype=np.int64)
            y_score = split_df["score"].to_numpy(dtype=np.float64)
            threshold = float(split_df["threshold"].iloc[0])

            metrics = compute_metrics(y_true, y_pred, y_score)
            rows.append(
                {
                    **combo,
                    "fold_id": int(fold_id),
                    "split": split,
                    "n_windows": len(split_df),
                    "n_stress": int(y_true.sum()),
                    "n_nonstress": int((1 - y_true).sum()),
                    "threshold": threshold,
                    **metrics,
                }
            )
    return rows


def aggregate(per_fold: pd.DataFrame, split: str) -> pd.DataFrame:
    if split != "all":
        df = per_fold[per_fold["split"] == split].copy()
    else:
        df = per_fold.copy()

    group_cols = ["encoder", "feature", "model_family", "split"] if split == "all" else ["encoder", "feature", "model_family"]

    agg_parts = []
    for col in METRIC_COLS:
        part = df.groupby(group_cols)[col].agg(
            **{f"{col}_mean": "mean", f"{col}_std": "std"}
        ).reset_index()
        agg_parts.append(part.set_index(group_cols))

    summary = pd.concat(agg_parts, axis=1).reset_index()

    n_folds = df.groupby(group_cols)["fold_id"].nunique().reset_index(name="n_folds")
    summary = summary.merge(n_folds, on=group_cols)

    # mean windows per fold
    win = df.groupby(group_cols)["n_windows"].mean().reset_index(name="mean_windows_per_fold")
    summary = summary.merge(win, on=group_cols)

    # formatted mean±std strings for direct copy-paste
    for col in METRIC_COLS:
        summary[f"{col}_fmt"] = summary.apply(
            lambda r, c=col: f"{r[f'{c}_mean']:.3f}±{r[f'{c}_std']:.3f}"
            if not np.isnan(r[f"{c}_mean"]) else "nan",
            axis=1,
        )

    return summary


def print_summary(summary: pd.DataFrame, split: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Test metrics ({split} split)" if split != "all" else "  All splits")
    print(f"{'='*70}")

    cols = ["encoder", "feature", "model_family"] + [f"{c}_fmt" for c in METRIC_COLS]
    if split == "all":
        cols = ["split"] + cols

    avail = [c for c in cols if c in summary.columns]
    print(summary[avail].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication metrics from downstream search outputs.")
    parser.add_argument(
        "--output-root",
        default="outputs_downstream_pub",
        help="Root directory passed to run_optuna_downstream.py as --output-root",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "all"],
        default="test",
        help="Which split to summarise (default: test)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write CSVs (default: same as --output-root)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(output_root.glob("runs/*/*/*/*/fold_predictions.csv"))
    if not pred_paths:
        raise SystemExit(f"No fold_predictions.csv found under {output_root}/runs/")

    print(f"Found {len(pred_paths)} combo prediction files")

    all_rows = []
    for path in pred_paths:
        rows = process_predictions(path)
        if rows:
            all_rows.extend(rows)
        else:
            print(f"  SKIPPED: {path}")

    if not all_rows:
        raise SystemExit("No metrics computed — check that fold_predictions.csv files are non-empty.")

    per_fold = pd.DataFrame(all_rows)

    # write detailed per-fold file
    per_fold_path = out_dir / "per_fold_metrics_full.csv"
    per_fold.to_csv(per_fold_path, index=False)
    print(f"\nWrote per-fold details  → {per_fold_path}")

    # write summary for requested split
    summary = aggregate(per_fold, args.split)
    summary_path = out_dir / ("publication_test_metrics.csv" if args.split == "test" else f"publication_{args.split}_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote publication summary → {summary_path}")

    # always also write the all-splits version
    if args.split != "all":
        all_splits = aggregate(per_fold, "all")
        all_splits_path = out_dir / "all_splits_metrics.csv"
        all_splits.to_csv(all_splits_path, index=False)
        print(f"Wrote all-splits summary  → {all_splits_path}")

    print_summary(summary, args.split)

    # print best test AUC per encoder/feature (useful at-a-glance for paper)
    if args.split == "test" or args.split == "all":
        ref = summary if args.split != "all" else summary[summary["split"] == "test"]
        if not ref.empty:
            best = (
                ref.sort_values("auc_mean", ascending=False)
                .groupby(["encoder", "feature"])
                .first()
                .reset_index()[["encoder", "feature", "model_family", "auc_fmt", "balanced_accuracy_fmt", "mcc_fmt"]]
            )
            print(f"\n{'='*70}")
            print("  Best model per encoder/feature (by test AUC)")
            print(f"{'='*70}")
            print(best.to_string(index=False))


if __name__ == "__main__":
    main()
