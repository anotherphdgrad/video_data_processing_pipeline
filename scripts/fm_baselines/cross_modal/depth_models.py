#!/usr/bin/env python3
"""
Standalone depth TCN encoder for cross-modal training.

Copied and adapted from scripts/fm_baselines/downstream_search/models.py.
The TemporalConvNet is split into an encoder (conv stack + attention pool)
and a head (MLP classifier) so the pooled representation can be extracted
and fed to the shared projection layer.

Loading from an existing downstream checkpoint (fold_N.pt) gives a warm
start — the encoder has already learned to aggregate DINOv2 frame embeddings
for stress classification.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks (copied from downstream_search/models.py)
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.net(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Depth TCN encoder — split from TemporalConvNet
# ---------------------------------------------------------------------------

class DepthTCNEncoder(nn.Module):
    """
    TCN encoder: conv stack + attention pool → pooled representation.

    Attributes
    ----------
    repr_dim : int
        Dimensionality of the pooled output (== hidden_dim).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = input_dim
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx
            padding = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPool(hidden_dim, max(32, hidden_dim // 2), dropout)
        self.repr_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D)  depth frame embeddings

        Returns
        -------
        pooled : (B, hidden_dim)  window-level representation
        """
        x = x.transpose(1, 2)              # (B, D, T) for Conv1d
        encoded = self.net(x).transpose(1, 2)  # (B, T, hidden_dim)
        pooled, _ = self.pool(encoded)
        return pooled


class DepthTCNWithHead(nn.Module):
    """Full TCN model (encoder + MLP head) matching the downstream checkpoint format."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = DepthTCNEncoder(input_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.head = MLPHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_depth_tcn_encoder(checkpoint_path: Path, device: str = "cpu") -> tuple[DepthTCNEncoder, dict]:
    """
    Load DepthTCNEncoder weights from a downstream fold checkpoint.

    The checkpoint is the fold_N.pt saved by run_optuna_downstream.py.
    Expected keys: state_dict, params, model_family, input_dim, seq_len,
                   mean, std, threshold, best_epoch.

    Returns
    -------
    encoder : DepthTCNEncoder  (weights loaded, eval mode)
    meta    : dict with input_dim, seq_len, mean, std, threshold, params
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if ckpt.get("model_family") != "tcn":
        raise ValueError(
            f"Checkpoint {checkpoint_path} has model_family={ckpt.get('model_family')}; "
            "expected 'tcn'."
        )

    params = ckpt["params"]
    input_dim = int(ckpt["input_dim"])

    # Build full model to load complete state_dict, then extract encoder
    full_model = DepthTCNWithHead(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
    )
    # The original downstream checkpoint was saved from TemporalConvNet which has
    # flat keys (net.*, pool.*, head.*).  DepthTCNWithHead wraps net+pool under
    # an 'encoder' submodule, so remap keys before loading.
    raw_sd = ckpt["state_dict"]
    remapped = {
        (f"encoder.{k}" if k.startswith("net.") or k.startswith("pool.") else k): v
        for k, v in raw_sd.items()
    }
    full_model.load_state_dict(remapped)

    encoder = full_model.encoder.to(device)
    encoder.eval()

    return encoder, {
        "input_dim": input_dim,
        "seq_len": int(ckpt["seq_len"]),
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "threshold": float(ckpt["threshold"]),
        "params": params,
    }
