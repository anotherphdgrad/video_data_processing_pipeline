#!/usr/bin/env python3
"""Optuna downstream search over framewise RGB/depth FM embeddings."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install optuna in the active environment: pip install optuna") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from dataset import (
    balanced_feature_sample_indices,
    discover_embedding_stores,
    load_embedding_store,
)
from models import sample_model_params
from train_eval import (
    create_grouped_folds,
    fold_assignments_frame,
    save_json,
    train_one_fold,
)


DEFAULT_ENCODERS = ["imagebind", "omnivore", "dinov2"]
DEFAULT_FEATURES = ["motion_prev_rgb", "motion_prev_depth"]
DEFAULT_MODEL_FAMILIES = ["attn_pool_mlp", "rnn_attn", "tcn", "transformer_encoder"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune downstream stress classifiers on framewise FM embeddings.")
    parser.add_argument("--embedding-root", default="outputs_rgb_depth_fm/embeddings_zarr2")
    parser.add_argument("--output-root", default="outputs_rgb_depth_fm_downstream_search")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--encoders", nargs="*", default=DEFAULT_ENCODERS)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--model-families", nargs="*", default=DEFAULT_MODEL_FAMILIES)
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--tune-fold-id", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-windows-per-feature", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def selected_indices_for_store(store, max_windows: int | None) -> np.ndarray:
    if max_windows is None:
        return np.arange(len(store.y), dtype=np.int64)
    return balanced_feature_sample_indices(store.y, store.base_subject_id, int(max_windows))


def create_folds_for_selected(store, selected: np.ndarray, args: argparse.Namespace):
    labels = store.y[selected]
    subjects = store.base_subject_id[selected]
    folds = create_grouped_folds(labels, subjects, args.n_splits, args.val_ratio, args.random_seed)
    remapped = []
    from train_eval import FoldSplit

    for fold in folds:
        remapped.append(
            FoldSplit(
                fold_id=fold.fold_id,
                train_indices=selected[fold.train_indices],
                val_indices=selected[fold.val_indices],
                test_indices=selected[fold.test_indices],
            )
        )
    if args.max_folds is not None:
        remapped = remapped[: int(args.max_folds)]
    return remapped


def check_participant_disjoint(assignments: pd.DataFrame) -> None:
    for fold_id, group in assignments.groupby("fold_id"):
        split_subjects = {
            split: set(split_df["base_subject_id"].astype(str))
            for split, split_df in group.groupby("split")
        }
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = split_subjects.get(left, set()) & split_subjects.get(right, set())
            if overlap:
                raise ValueError(f"Participant leakage in fold {fold_id}: {left}/{right} overlap {sorted(overlap)[:5]}")


def tune_combo(store, folds, model_family: str, combo_dir: Path, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    tune_fold = next((fold for fold in folds if fold.fold_id == int(args.tune_fold_id)), None)
    if tune_fold is None:
        raise ValueError(f"Requested tune fold {args.tune_fold_id}, but available folds are {[f.fold_id for f in folds]}")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    def objective(trial: optuna.Trial) -> float:
        params = sample_model_params(trial, model_family)
        trial_dir = combo_dir / "trial_artifacts" / f"trial_{trial.number:04d}"
        try:
            _rows, history, _preds, best_val = train_one_fold(
                store=store,
                fold=tune_fold,
                model_family=model_family,
                params=params,
                device=args.device,
                random_seed=args.random_seed,
                output_dir=None,
                trial=trial,
                no_progress=True,
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
            save_json(trial_dir / "params.json", params)
            return float(best_val)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001
            trial_dir.mkdir(parents=True, exist_ok=True)
            save_json(trial_dir / "failed.json", {"error_type": type(exc).__name__, "error": str(exc), "params": params})
            raise optuna.TrialPruned(str(exc)) from exc

    study.optimize(objective, n_trials=int(args.optuna_trials), show_progress_bar=bool(tqdm and not args.no_progress))
    study_df = study.trials_dataframe()
    combo_dir.mkdir(parents=True, exist_ok=True)
    study_df.to_csv(combo_dir / "optuna_study.csv", index=False)
    if study.best_trial is None:
        raise RuntimeError(f"No completed Optuna trials for {combo_dir}")
    best_params = dict(study.best_trial.params)
    save_json(combo_dir / "best_params.json", best_params)
    save_json(combo_dir / "best_trial_summary.json", {"number": study.best_trial.number, "value": study.best_trial.value, "params": best_params})
    return best_params, study_df


def evaluate_best_config(store, folds, model_family: str, best_params: dict, combo_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    iterator = folds
    if tqdm and not args.no_progress:
        iterator = tqdm(folds, desc=f"eval {store.encoder}/{store.feature}/{model_family}", unit="fold")
    for fold in iterator:
        checkpoint_dir = combo_dir / "checkpoints"
        rows, history, preds, _best_val = train_one_fold(
            store=store,
            fold=fold,
            model_family=model_family,
            params=best_params,
            device=args.device,
            random_seed=args.random_seed,
            output_dir=checkpoint_dir,
            trial=None,
            no_progress=args.no_progress,
        )
        for row in rows:
            metric_rows.append(
                {
                    "encoder": store.encoder,
                    "feature": store.feature,
                    "model_family": model_family,
                    "model_name": f"{store.encoder}_{store.feature}_{model_family}",
                    **row,
                }
            )
        history_path = combo_dir / "histories" / f"fold_{fold.fold_id}_history.csv"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        preds.insert(0, "model_family", model_family)
        preds.insert(0, "feature", store.feature)
        preds.insert(0, "encoder", store.encoder)
        prediction_frames.append(preds)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_metrics(metric_df: pd.DataFrame, run_dir: Path, args: argparse.Namespace, seq_len_by_combo: dict | None = None) -> pd.DataFrame:
    rows = []
    for (encoder, feature, model_family), group in metric_df.groupby(["encoder", "feature", "model_family"]):
        seq_len = (seq_len_by_combo or {}).get((encoder, feature), 150)
        row = {
            "model_name": f"{encoder}_{feature}_{model_family}",
            "module_name": "RGBDepth-FM-Framewise-Downstream",
            "window_strategy": f"30s window / 15s overlap; {seq_len} cached frame embeddings",
            "input_mode": feature,
            "representation_family": "Framewise foundation model embedding downstream search",
            "representation_equation": f"{encoder}:{feature}:framewise_{seq_len}:{model_family}",
            "sequence_pooling": model_family,
            "sequence_length": seq_len,
            "optuna_trials": int(args.optuna_trials),
            "n_folds": int(args.max_folds or args.n_splits),
            "person_disjoint_setting": "GroupKFold(base_subject_id) with GroupShuffleSplit validation",
            "encoder": encoder,
            "feature": feature,
            "model_family": model_family,
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


def aggregate_results(output_root: Path) -> None:
    frames = []
    for path in sorted((output_root / "runs").glob("*/all_window_mode_results_concise.csv")):
        df = pd.read_csv(path)
        df.insert(0, "run_dir", str(path.parent))
        frames.append(df)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_root / "all_window_mode_results_concise.csv", index=False)


def main() -> None:
    args = parse_args()
    embedding_root = Path(args.embedding_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = args.run_name or datetime.now().strftime("fm_downstream_%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", vars(args))

    available = discover_embedding_stores(embedding_root)
    available.to_csv(run_dir / "available_embedding_stores.csv", index=False)
    seq_len_by_combo: dict[tuple[str, str], int] = {
        (row.encoder, row.feature): row.frames_per_window
        for row in available.itertuples()
        if hasattr(row, "frames_per_window")
    }
    metric_frames = []
    pred_frames = []
    assignment_frames = []
    combos = [(encoder, feature, model_family) for encoder in args.encoders for feature in args.features for model_family in args.model_families]
    iterator = combos
    if tqdm and not args.no_progress:
        iterator = tqdm(combos, desc="downstream combos", unit="combo")
    started = time.time()
    for encoder, feature, model_family in iterator:
        combo_dir = run_dir / encoder / feature / model_family
        done_path = combo_dir / "done.json"
        if done_path.exists() and not args.overwrite:
            print(f"Skipping completed combo: {encoder}/{feature}/{model_family}")
            if (combo_dir / "per_fold_metrics.csv").exists():
                metric_frames.append(pd.read_csv(combo_dir / "per_fold_metrics.csv"))
            if (combo_dir / "fold_predictions.csv").exists():
                pred_frames.append(pd.read_csv(combo_dir / "fold_predictions.csv"))
            continue
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_start = time.time()
        store = load_embedding_store(embedding_root, encoder, feature)
        selected = selected_indices_for_store(store, args.max_windows_per_feature)
        folds = create_folds_for_selected(store, selected, args)
        assignments = fold_assignments_frame(store, folds)
        assignments.insert(0, "model_family", model_family)
        assignments.insert(0, "feature", feature)
        assignments.insert(0, "encoder", encoder)
        check_participant_disjoint(assignments)
        assignments.to_csv(combo_dir / "fold_assignments.csv", index=False)
        assignment_frames.append(assignments)
        best_params, _study_df = tune_combo(store, folds, model_family, combo_dir, args)
        metrics, preds = evaluate_best_config(store, folds, model_family, best_params, combo_dir, args)
        metrics.to_csv(combo_dir / "per_fold_metrics.csv", index=False)
        preds.to_csv(combo_dir / "fold_predictions.csv", index=False)
        save_json(
            done_path,
            {
                "encoder": encoder,
                "feature": feature,
                "model_family": model_family,
                "elapsed_seconds": time.time() - combo_start,
                "best_params": best_params,
            },
        )
        metric_frames.append(metrics)
        pred_frames.append(preds)

    if not metric_frames:
        raise SystemExit("No downstream metrics were produced.")
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    all_assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    all_metrics.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    all_preds.to_csv(run_dir / "fold_predictions.csv", index=False)
    all_assignments.to_csv(run_dir / "fold_assignments.csv", index=False)
    summarize_metrics(all_metrics, run_dir, args, seq_len_by_combo=seq_len_by_combo)
    save_json(run_dir / "run_summary.json", {"elapsed_seconds": time.time() - started, "num_metric_rows": int(len(all_metrics))})
    aggregate_results(output_root)
    print(f"Wrote downstream search run: {run_dir}")
    print(f"Aggregate concise summary: {output_root / 'all_window_mode_results_concise.csv'}")


if __name__ == "__main__":
    main()
