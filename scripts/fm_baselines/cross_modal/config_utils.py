#!/usr/bin/env python3
"""
Config loader for cross-modal training scripts.

Loads config.json and merges with argparse namespace so that:
  - Config values provide defaults
  - Explicit CLI args override config values
"""

from __future__ import annotations

import json
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    # Strip comment keys
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def apply_config(args, cfg: dict) -> None:
    """
    Fill in argparse namespace from config where the CLI arg was not set.
    Only fills fields that exist in args and are currently None.
    """
    flat = {
        "fm_store":             cfg.get("fm_store"),
        "fm_meta_csv":          cfg.get("fm_meta_csv"),
        "imu_data_root":        cfg.get("imu_data_root"),
        "imu_channel_mode":     cfg.get("imu_channel_mode", "raw_absdelta"),
        "limu_bert_public_repo": cfg.get("limu_bert_public_repo"),
        "depth_ckpt_dir":       cfg.get("depth_ckpt_dir"),
        "output_root":          cfg.get("output_root"),
    }
    training = cfg.get("training", {})
    flat.update({
        "n_splits":      training.get("n_splits", 5),
        "val_ratio":     training.get("val_ratio", 0.2),
        "random_seed":   training.get("random_seed", 42),
        "optuna_trials": training.get("optuna_trials", 30),
        "tune_fold_id":  training.get("tune_fold_id", 1),
        "device":        training.get("device", "cuda"),
    })
    teacher = cfg.get("teacher", {})
    flat.update({
        "teacher_type":      teacher.get("type", "limu_bert"),
        "imu_seq_len":       teacher.get("imu_seq_len", 12),
        "imu_feature_dim":   teacher.get("imu_feature_dim", 6),
    })
    for key, value in flat.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def teacher_checkpoints(cfg: dict) -> dict[int, str]:
    """Return {fold_id: checkpoint_path} from config teacher.checkpoints."""
    raw = cfg.get("teacher", {}).get("checkpoints", {})
    return {int(k): v for k, v in raw.items()}


def teacher_embeddings_dir(cfg: dict) -> Path:
    """Default directory for extracted teacher embeddings."""
    return Path(cfg.get("output_root", ".")) / "teacher_embeddings"
