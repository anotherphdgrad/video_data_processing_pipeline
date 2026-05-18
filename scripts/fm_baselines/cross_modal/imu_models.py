#!/usr/bin/env python3
"""
Standalone IMU teacher model classes for cross-modal training.

Copied and adapted from IMU-Stress-sensing/baselines_flirt_torch/common.py.
No imports from the IMU pipeline source — self-contained.

Two teacher types are supported:

  FlirtLSTMTeacher
    Input  : FLIRT feature sequences  (B, seq_len, flirt_dim)
    Pooling: attention (as in flirt_lstm_attention)
    Dep    : torch only

  LimuBertTeacher
    Input  : raw ACC sequences segment-pooled to seq_len tokens (B, seq_len, C)
    Pooling: first token (as in flirt_limu_bert_scratch_first)
    Dep    : LIMU-BERT-Public Transformer class — see README for setup

Both expose a get_representation(x, mask) method that returns the
pre-classification pooled embedding.  The classifier head is kept so
full checkpoints can be loaded without key mismatches, but is never
called during cross-modal training.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Shared pooling helper
# ---------------------------------------------------------------------------

class MaskedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(self.proj(inputs))).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(inputs * weights.unsqueeze(-1), dim=1)


# ---------------------------------------------------------------------------
# FLIRT-LSTM teacher
# ---------------------------------------------------------------------------

class FlirtLSTMTeacher(nn.Module):
    """
    LSTM over FLIRT feature sequences.  Matches flirt_lstm_attention checkpoint.

    Parameters
    ----------
    input_dim   : FLIRT feature dimension (depends on input_mode, ~100-200)
    hidden_size : LSTM hidden units (from checkpoint params)
    num_layers  : number of LSTM layers
    dropout     : dropout probability
    pooling     : one of {first, mean, last, attention}
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_pool = MaskedAttentionPool(hidden_size, hidden_size) if pooling == "attention" else None
        self.classifier = nn.Linear(hidden_size, 2)
        self.repr_dim = hidden_size

    def _pool(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "first":
            return outputs[:, 0, :]
        if self.pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (outputs * mask.unsqueeze(-1)).sum(dim=1) / denom
        if self.pooling == "attention":
            return self.attention_pool(outputs, mask)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(outputs.size(0), device=outputs.device)
        return outputs[batch_idx, lengths - 1, :]

    def get_representation(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns pooled representation before the classifier head."""
        outputs, _ = self.lstm(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.float32)
        pooled = self._pool(outputs, mask)
        return self.dropout_layer(pooled)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.get_representation(x, mask))


def load_flirt_lstm_teacher(checkpoint_path: Path, device: str = "cpu") -> FlirtLSTMTeacher:
    """
    Load a FlirtLSTMTeacher from a saved fold checkpoint.

    The checkpoint is expected to match the format saved by
    baselines_flirt_torch/common.py:_train_supervised, i.e.
    {'state_dict': ..., 'params': ...}.

    The input_dim is inferred from the first LSTM weight in the state_dict.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    params = ckpt["params"]

    # Infer input_dim from lstm.weight_ih_l0 shape: (4*hidden, input_dim)
    lstm_weight = state_dict["lstm.weight_ih_l0"]
    input_dim = int(lstm_weight.shape[1])
    hidden_size = int(lstm_weight.shape[0]) // 4
    num_layers = int(params.get("num_layers", 1))
    dropout = float(params.get("dropout", 0.0))
    pooling = str(params.get("sequence_pooling", "attention"))

    model = FlirtLSTMTeacher(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# LIMU-BERT teacher
# ---------------------------------------------------------------------------

def make_limu_cfg(feature_dim: int, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden=72,
        hidden_ff=144,
        feature_num=feature_dim,
        n_layers=4,
        n_heads=4,
        seq_len=seq_len,
        emb_norm=True,
    )


class LimuBertTeacher(nn.Module):
    """
    LIMU-BERT transformer teacher.  Matches flirt_limu_bert_scratch_first checkpoint.

    Requires the LIMU-BERT-Public Transformer class.  See README for setup.

    Parameters
    ----------
    cfg     : SimpleNamespace from make_limu_cfg()
    pooling : one of {first, mean, last, attention}
    dropout : dropout probability
    """

    def __init__(self, cfg: SimpleNamespace, pooling: str = "first", dropout: float = 0.0) -> None:
        super().__init__()
        Transformer = _load_limu_transformer()
        self.transformer = Transformer(cfg)
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_pool = MaskedAttentionPool(cfg.hidden, cfg.hidden) if pooling == "attention" else None
        self.classifier = nn.Linear(cfg.hidden, 2)
        self.repr_dim = cfg.hidden

    def _pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "first":
            return encoded[:, 0, :]
        if self.pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
        if self.pooling == "attention":
            return self.attention_pool(encoded, mask)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        return encoded[batch_idx, lengths - 1, :]

    def get_representation(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded = self.transformer(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.float32)
        pooled = self._pool(encoded, mask)
        return self.dropout_layer(pooled)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.get_representation(x, mask))


def _load_limu_transformer():
    """Load the Transformer class from LIMU-BERT-Public. Raises if not found."""
    import importlib
    import sys

    repo_candidates = [
        Path("modules/LIMU-BERT-Public"),
        Path(__file__).resolve().parents[3] / "IMU-Stress-sensing" / "modules" / "LIMU-BERT-Public",
    ]
    for repo_dir in repo_candidates:
        models_path = repo_dir / "models.py"
        if models_path.exists():
            module_name = "_limu_bert_public_models"
            if module_name in sys.modules:
                return sys.modules[module_name].Transformer
            sys.path.insert(0, str(repo_dir))
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(models_path))
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module.Transformer
            finally:
                if sys.path and sys.path[0] == str(repo_dir):
                    sys.path.pop(0)

    raise ImportError(
        "LIMU-BERT-Public not found. Expected at modules/LIMU-BERT-Public/models.py.\n"
        "See cross_modal/README.md for setup instructions."
    )


def load_limu_bert_teacher(checkpoint_path: Path, feature_dim: int = 6, seq_len: int = 12, device: str = "cpu") -> LimuBertTeacher:
    """
    Load a LimuBertTeacher from a saved fold checkpoint.

    checkpoint_path : path to fold_N_model.pt saved by the IMU pipeline
    feature_dim     : ACC feature channels (6 for raw_absdelta with 3-axis)
    seq_len         : number of tokens (12 for the FLIRT-Torch baseline)
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    params = ckpt["params"]
    pooling = str(params.get("sequence_pooling", "first"))
    dropout = float(params.get("dropout", 0.0))
    cfg = make_limu_cfg(feature_dim, seq_len)
    model = LimuBertTeacher(cfg, pooling=pooling, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sequence preparation helper for LIMU-BERT (no FLIRT needed)
# ---------------------------------------------------------------------------

def segment_mean_pool(sequences: np.ndarray, target_len: int) -> np.ndarray:
    """
    Reduce (N, T, C) raw ACC sequences to (N, target_len, C) by segment mean pooling.
    Matches the preprocessing used in the IMU pipeline for LIMU-BERT input.
    """
    batch = np.asarray(sequences, dtype=np.float32)
    n, seq_len, c = batch.shape
    pooled = np.empty((n, target_len, c), dtype=np.float32)
    boundaries = np.linspace(0, seq_len, target_len + 1, dtype=np.int64)
    for i in range(target_len):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        end = max(end, start + 1)
        pooled[:, i, :] = batch[:, start:end, :].mean(axis=1)
    return pooled
