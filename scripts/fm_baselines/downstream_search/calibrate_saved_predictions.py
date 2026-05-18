#!/usr/bin/env python3
"""Audit validation-only calibration for saved downstream FM predictions.

This script is intentionally non-destructive: it reads existing
``fold_predictions.csv`` files and writes calibrated audit artifacts to a new
output directory. No neural model checkpoints are modified or retrained.

Compared strategies per fold:

1. ``current_saved_threshold``
   Uses the saved prediction/threshold from the original downstream run.
2. ``val_platt_lr_0p5``
   Fits a one-dimensional logistic-regression/Platt calibrator on that fold's
   validation scores only, then predicts stress at calibrated score >= 0.5.
3. ``val_platt_lr_val_threshold``
   Uses the same validation-only calibrator, then selects a balanced-accuracy
   threshold on calibrated validation scores and applies that threshold to
   train/val/test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


METRIC_COLS = [
    "auc",
    "avg_precision",
    "balanced_accuracy",
    "f1_macro",
    "sensitivity",
    "specificity",
    "mcc",
]

THRESHOLDS = np.linspace(0.05, 0.95, 19)


@dataclass(frozen=True)
class ComboInfo:
    run_name: str
    encoder: str
    feature: str
    model_family: str
    pred_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Platt/LR calibration on saved fold predictions.")
    parser.add_argument(
        "--input-root",
        default="outputs_downstream_pub",
        help="Root containing runs/*/*/*/*/fold_predictions.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_downstream_pub_calibrated",
        help="New root for calibration audit outputs. Existing source results are never overwritten.",
    )
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run_name filter, e.g. run_gpu_families.")
    parser.add_argument("--encoders", nargs="*", default=None, help="Optional encoder filter.")
    parser.add_argument("--features", nargs="*", default=None, help="Optional feature filter.")
    parser.add_argument("--model-families", nargs="*", default=None, help="Optional downstream model family filter.")
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of combo prediction files.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Clipping epsilon before score-to-logit transform.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing calibration CSVs in the output root.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


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
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= float(threshold)).astype(np.int64)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in THRESHOLDS:
        score = float(balanced_accuracy_score(y_true, apply_threshold(y_score, float(threshold))))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def score_to_logit_feature(score: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(score.astype(np.float64), epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    return logits.reshape(-1, 1)


def fit_platt_lr(val_scores: np.ndarray, val_labels: np.ndarray, epsilon: float) -> LogisticRegression | None:
    if len(np.unique(val_labels)) < 2:
        return None
    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(score_to_logit_feature(val_scores, epsilon), val_labels.astype(np.int64))
    return model


def calibrated_scores(model: LogisticRegression, scores: np.ndarray, epsilon: float) -> np.ndarray:
    return model.predict_proba(score_to_logit_feature(scores, epsilon))[:, 1].astype(np.float64)


def parse_combo_path(pred_path: Path, input_root: Path) -> ComboInfo | None:
    try:
        rel = pred_path.relative_to(input_root)
    except ValueError:
        return None
    parts = rel.parts
    # Expected: runs/{run_name}/{encoder}/{feature}/{model_family}/fold_predictions.csv
    if len(parts) != 6 or parts[0] != "runs" or parts[-1] != "fold_predictions.csv":
        return None
    return ComboInfo(
        run_name=parts[1],
        encoder=parts[2],
        feature=parts[3],
        model_family=parts[4],
        pred_path=str(pred_path),
    )


def passes_filters(combo: ComboInfo, args: argparse.Namespace) -> bool:
    if args.runs is not None and combo.run_name not in set(args.runs):
        return False
    if args.encoders is not None and combo.encoder not in set(args.encoders):
        return False
    if args.features is not None and combo.feature not in set(args.features):
        return False
    if args.model_families is not None and combo.model_family not in set(args.model_families):
        return False
    return True


def discover_prediction_files(input_root: Path, args: argparse.Namespace) -> list[tuple[ComboInfo, Path]]:
    pred_paths = sorted(input_root.glob("runs/*/*/*/*/fold_predictions.csv"))
    combos: list[tuple[ComboInfo, Path]] = []
    for path in pred_paths:
        combo = parse_combo_path(path, input_root)
        if combo is None:
            continue
        if passes_filters(combo, args):
            combos.append((combo, path))
    if args.max_combos is not None:
        combos = combos[: int(args.max_combos)]
    return combos


def validate_prediction_frame(preds: pd.DataFrame, pred_path: Path) -> None:
    required = {
        "encoder",
        "feature",
        "model_family",
        "fold_id",
        "split",
        "source_index",
        "window_id",
        "base_subject_id",
        "label",
        "score",
        "prediction",
        "threshold",
    }
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"{pred_path} is missing required columns: {sorted(missing)}")
    splits = set(preds["split"].astype(str).unique())
    if not {"val", "test"}.issubset(splits):
        raise ValueError(f"{pred_path} must contain at least val and test splits; found {sorted(splits)}")


def add_prediction_rows(
    *,
    rows: list[dict],
    combo: ComboInfo,
    fold_id: int,
    split_df: pd.DataFrame,
    strategy: str,
    score: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    calibrator_status: str,
) -> None:
    for base, calibrated_score, calibrated_pred in zip(split_df.to_dict("records"), score, pred):
        rows.append(
            {
                "run_name": combo.run_name,
                "encoder": combo.encoder,
                "feature": combo.feature,
                "model_family": combo.model_family,
                "fold_id": int(fold_id),
                "split": str(base["split"]),
                "strategy": strategy,
                "source_index": int(base["source_index"]),
                "window_id": base["window_id"],
                "base_subject_id": str(base["base_subject_id"]),
                "label": int(base["label"]),
                "original_score": float(base["score"]),
                "score": float(calibrated_score),
                "prediction": int(calibrated_pred),
                "threshold": float(threshold),
                "original_threshold": float(base["threshold"]),
                "calibrator_status": calibrator_status,
            }
        )


def add_metric_row(
    *,
    rows: list[dict],
    combo: ComboInfo,
    fold_id: int,
    split: str,
    split_df: pd.DataFrame,
    strategy: str,
    score: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    calibrator_status: str,
) -> None:
    y_true = split_df["label"].to_numpy(dtype=np.int64)
    metrics = compute_metrics(y_true, pred.astype(np.int64), score.astype(np.float64))
    rows.append(
        {
            "run_name": combo.run_name,
            "encoder": combo.encoder,
            "feature": combo.feature,
            "model_family": combo.model_family,
            "fold_id": int(fold_id),
            "split": split,
            "strategy": strategy,
            "n_windows": int(len(split_df)),
            "n_stress": int(y_true.sum()),
            "n_nonstress": int((1 - y_true).sum()),
            "threshold": float(threshold),
            "calibrator_status": calibrator_status,
            **metrics,
        }
    )


def process_combo(combo: ComboInfo, pred_path: Path, epsilon: float) -> tuple[list[dict], list[dict], list[dict]]:
    preds = pd.read_csv(pred_path)
    validate_prediction_frame(preds, pred_path)

    pred_rows: list[dict] = []
    metric_rows: list[dict] = []
    calibrator_rows: list[dict] = []

    for fold_id, fold_df in preds.groupby("fold_id", sort=True):
        fold_id = int(fold_id)
        val_df = fold_df[fold_df["split"] == "val"].copy()
        val_y = val_df["label"].to_numpy(dtype=np.int64)
        val_scores = val_df["score"].to_numpy(dtype=np.float64)
        calibrator = fit_platt_lr(val_scores, val_y, epsilon=epsilon)

        if calibrator is None:
            calibrator_status = "skipped_val_single_class"
            calibrated_val_threshold = float("nan")
            calibrated_val_ba = float("nan")
            coef = float("nan")
            intercept = float("nan")
        else:
            calibrator_status = "ok"
            calibrated_val_scores = calibrated_scores(calibrator, val_scores, epsilon=epsilon)
            calibrated_val_threshold, calibrated_val_ba = find_best_threshold(val_y, calibrated_val_scores)
            coef = float(calibrator.coef_[0, 0])
            intercept = float(calibrator.intercept_[0])

        calibrator_rows.append(
            {
                "run_name": combo.run_name,
                "encoder": combo.encoder,
                "feature": combo.feature,
                "model_family": combo.model_family,
                "fold_id": fold_id,
                "status": calibrator_status,
                "n_val": int(len(val_df)),
                "n_val_stress": int(val_y.sum()),
                "n_val_nonstress": int((1 - val_y).sum()),
                "coef": coef,
                "intercept": intercept,
                "val_selected_threshold": calibrated_val_threshold,
                "val_selected_balanced_accuracy": calibrated_val_ba,
                "epsilon": float(epsilon),
            }
        )

        for split, split_df in fold_df.groupby("split", sort=True):
            split_df = split_df.copy()

            original_score = split_df["score"].to_numpy(dtype=np.float64)
            original_pred = split_df["prediction"].to_numpy(dtype=np.int64)
            original_threshold = float(split_df["threshold"].iloc[0])
            add_prediction_rows(
                rows=pred_rows,
                combo=combo,
                fold_id=fold_id,
                split_df=split_df,
                strategy="current_saved_threshold",
                score=original_score,
                pred=original_pred,
                threshold=original_threshold,
                calibrator_status="not_applicable",
            )
            add_metric_row(
                rows=metric_rows,
                combo=combo,
                fold_id=fold_id,
                split=split,
                split_df=split_df,
                strategy="current_saved_threshold",
                score=original_score,
                pred=original_pred,
                threshold=original_threshold,
                calibrator_status="not_applicable",
            )

            if calibrator is None:
                continue

            platt_score = calibrated_scores(calibrator, original_score, epsilon=epsilon)

            pred_0p5 = apply_threshold(platt_score, 0.5)
            add_prediction_rows(
                rows=pred_rows,
                combo=combo,
                fold_id=fold_id,
                split_df=split_df,
                strategy="val_platt_lr_0p5",
                score=platt_score,
                pred=pred_0p5,
                threshold=0.5,
                calibrator_status=calibrator_status,
            )
            add_metric_row(
                rows=metric_rows,
                combo=combo,
                fold_id=fold_id,
                split=split,
                split_df=split_df,
                strategy="val_platt_lr_0p5",
                score=platt_score,
                pred=pred_0p5,
                threshold=0.5,
                calibrator_status=calibrator_status,
            )

            pred_val_threshold = apply_threshold(platt_score, calibrated_val_threshold)
            add_prediction_rows(
                rows=pred_rows,
                combo=combo,
                fold_id=fold_id,
                split_df=split_df,
                strategy="val_platt_lr_val_threshold",
                score=platt_score,
                pred=pred_val_threshold,
                threshold=calibrated_val_threshold,
                calibrator_status=calibrator_status,
            )
            add_metric_row(
                rows=metric_rows,
                combo=combo,
                fold_id=fold_id,
                split=split,
                split_df=split_df,
                strategy="val_platt_lr_val_threshold",
                score=platt_score,
                pred=pred_val_threshold,
                threshold=calibrated_val_threshold,
                calibrator_status=calibrator_status,
            )

    return pred_rows, metric_rows, calibrator_rows


def aggregate_metrics(per_fold: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    df = per_fold[per_fold["split"] == split].copy()
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
    n_windows = df.groupby(group_cols)["n_windows"].mean().reset_index(name="mean_windows_per_fold")
    summary = summary.merge(n_windows, on=group_cols)

    for col in METRIC_COLS:
        summary[f"{col}_fmt"] = summary.apply(
            lambda row, c=col: f"{row[f'{c}_mean']:.3f}±{row[f'{c}_std']:.3f}"
            if np.isfinite(row[f"{c}_mean"])
            else "nan",
            axis=1,
        )
    return summary.sort_values(["auc_mean", "balanced_accuracy_mean"], ascending=False)


def build_concise_summary(test_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in test_summary.to_dict("records"):
        model_name = f"{row['encoder']}_{row['model_family']}_{row['strategy']}"
        rows.append(
            {
                "model_name": model_name,
                "module_name": "RGBDepth-FM-SavedPredictionCalibration",
                "window_strategy": "30s window / 15s overlap; saved framewise-FM downstream predictions",
                "input_mode": row["feature"],
                "representation_family": "saved downstream score calibration",
                "representation_equation": f"{row['encoder']}:{row['feature']}:{row['model_family']}:{row['strategy']}",
                "sequence_pooling": "no neural retraining; validation-only Platt/LR audit",
                "sequence_length": 150,
                "optuna_trials": 0,
                "n_folds": int(row["n_folds"]),
                "person_disjoint_setting": "Existing GroupKFold(base_subject_id) folds; calibration fit on validation split only",
                "train_balanced_accuracy_mean": float("nan"),
                "train_auc_mean": float("nan"),
                "val_balanced_accuracy_mean": float("nan"),
                "val_auc_mean": float("nan"),
                "test_balanced_accuracy_mean": row["balanced_accuracy_mean"],
                "test_auc_mean": row["auc_mean"],
                "test_mcc_mean": row["mcc_mean"],
                "test_avg_precision_mean": row["avg_precision_mean"],
                "test_f1_macro_mean": row["f1_macro_mean"],
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    output_root: Path,
    pred_rows: list[dict],
    metric_rows: list[dict],
    calibrator_rows: list[dict],
    config: dict,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    predictions = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)
    calibrators = pd.DataFrame(calibrator_rows)

    predictions.to_csv(output_root / "calibrated_fold_predictions.csv", index=False)
    metrics.to_csv(output_root / "calibration_per_fold_metrics.csv", index=False)
    calibrators.to_csv(output_root / "calibrator_params.csv", index=False)

    all_splits = []
    for split in ["train", "val", "test"]:
        split_summary = aggregate_metrics(metrics, split=split)
        split_summary.insert(4, "split", split)
        all_splits.append(split_summary)
    all_splits_summary = pd.concat(all_splits, ignore_index=True)
    all_splits_summary.to_csv(output_root / "calibration_all_splits_metrics.csv", index=False)

    test_summary = aggregate_metrics(metrics, split="test")
    test_summary.to_csv(output_root / "calibration_test_metrics.csv", index=False)

    concise = build_concise_summary(test_summary)
    concise.to_csv(output_root / "calibration_all_window_mode_results_concise.csv", index=False)

    print(f"Wrote calibrated predictions: {output_root / 'calibrated_fold_predictions.csv'}")
    print(f"Wrote per-fold metrics:        {output_root / 'calibration_per_fold_metrics.csv'}")
    print(f"Wrote test summary:            {output_root / 'calibration_test_metrics.csv'}")
    print(f"Wrote concise summary:         {output_root / 'calibration_all_window_mode_results_concise.csv'}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_root} already exists and is non-empty. Use --overwrite or choose another --output-root.")

    combos = discover_prediction_files(input_root, args)
    if not combos:
        raise SystemExit(f"No combo fold_predictions.csv files found under {input_root} with the requested filters.")

    iterator = combos
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(combos, desc="Calibrating saved prediction files", unit="combo")

    all_pred_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_calibrator_rows: list[dict] = []
    failures: list[dict] = []

    for combo, pred_path in iterator:
        try:
            pred_rows, metric_rows, calibrator_rows = process_combo(combo, pred_path, epsilon=float(args.epsilon))
            all_pred_rows.extend(pred_rows)
            all_metric_rows.extend(metric_rows)
            all_calibrator_rows.extend(calibrator_rows)
        except Exception as exc:  # keep long audits diagnosable
            failures.append({**asdict(combo), "error": repr(exc)})

    if not all_metric_rows:
        raise SystemExit("No calibration metrics were produced. Check failures and input prediction files.")

    config = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "runs": args.runs,
        "encoders": args.encoders,
        "features": args.features,
        "model_families": args.model_families,
        "max_combos": args.max_combos,
        "epsilon": args.epsilon,
        "n_combo_files": len(combos),
        "n_failures": len(failures),
        "strategies": [
            "current_saved_threshold",
            "val_platt_lr_0p5",
            "val_platt_lr_val_threshold",
        ],
        "calibration_note": (
            "Platt/LR calibrators are fit on each outer fold's validation split only. "
            "No neural checkpoints are retrained or overwritten."
        ),
    }
    write_outputs(
        output_root=output_root,
        pred_rows=all_pred_rows,
        metric_rows=all_metric_rows,
        calibrator_rows=all_calibrator_rows,
        config=config,
    )

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_path = output_root / "calibration_failures.csv"
        failure_df.to_csv(failure_path, index=False)
        print(f"WARNING: {len(failures)} combo files failed. Details: {failure_path}")

    test_summary = pd.read_csv(output_root / "calibration_test_metrics.csv")
    display_cols = [
        "run_name",
        "encoder",
        "feature",
        "model_family",
        "strategy",
        "auc_fmt",
        "balanced_accuracy_fmt",
        "mcc_fmt",
    ]
    print("\nTop calibrated audit rows by test AUC:")
    print(test_summary[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
