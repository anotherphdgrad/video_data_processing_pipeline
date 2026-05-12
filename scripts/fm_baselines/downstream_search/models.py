#!/usr/bin/env python3
"""Torch downstream classifiers for framewise FM embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
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
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
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


class AttentionPoolMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.pool = AttentionPool(input_dim, attn_dim, dropout)
        self.head = MLPHead(input_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _weights = self.pool(x)
        return self.head(pooled)


class RNNAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attn_dim: int,
        dropout: float,
        rnn_type: str,
        num_layers: int,
        bidirectional: bool,
    ):
        super().__init__()
        klass = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = klass(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionPool(out_dim, attn_dim, dropout)
        self.head = MLPHead(out_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _state = self.rnn(x)
        pooled, _weights = self.pool(encoded)
        return self.head(pooled)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = input_dim
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            padding = (kernel_size - 1) * dilation // 2
            layers.extend(
                [
                    nn.Conv1d(in_ch, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPool(hidden_dim, max(32, hidden_dim // 2), dropout)
        self.head = MLPHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.net(x).transpose(1, 2)
        pooled, _weights = self.pool(encoded)
        return self.head(pooled)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        pooling: str,
        max_len: int = 150,
    ):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPool(model_dim, max(32, model_dim // 2), dropout)
        self.head = MLPHead(model_dim, model_dim, dropout)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.proj(x)
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        encoded = self.encoder(x)
        if self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            pooled, _weights = self.attn_pool(encoded[:, 1:])
        return self.head(pooled)


def build_model(model_family: str, input_dim: int, seq_len: int, params: dict) -> nn.Module:
    if model_family == "attn_pool_mlp":
        return AttentionPoolMLP(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            attn_dim=int(params["attn_dim"]),
            dropout=float(params["dropout"]),
        )
    if model_family == "rnn_attn":
        return RNNAttention(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            attn_dim=int(params["attn_dim"]),
            dropout=float(params["dropout"]),
            rnn_type=str(params["rnn_type"]),
            num_layers=int(params["num_layers"]),
            bidirectional=bool(params["bidirectional"]),
        )
    if model_family == "tcn":
        return TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
        )
    if model_family == "transformer_encoder":
        return TransformerClassifier(
            input_dim=input_dim,
            model_dim=int(params["model_dim"]),
            num_heads=int(params["num_heads"]),
            num_layers=int(params["num_layers"]),
            ff_dim=int(params["ff_dim"]),
            dropout=float(params["dropout"]),
            pooling=str(params["pooling"]),
            max_len=seq_len,
        )
    raise ValueError(f"Unknown model family: {model_family}")


def sample_model_params(trial, model_family: str) -> dict:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": trial.suggest_categorical("max_epochs", [25, 50, 75]),
        "patience": trial.suggest_categorical("patience", [5, 8, 12]),
        "loss": trial.suggest_categorical("loss", ["weighted_ce", "focal"]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "grad_clip_norm": trial.suggest_categorical("grad_clip_norm", [0.0, 1.0, 5.0]),
    }
    if params["loss"] == "focal":
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 3.0)
    else:
        params["focal_gamma"] = 0.0
    if model_family == "attn_pool_mlp":
        params.update(
            {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "attn_dim": trial.suggest_categorical("attn_dim", [64, 128, 256]),
            }
        )
    elif model_family == "rnn_attn":
        params.update(
            {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "attn_dim": trial.suggest_categorical("attn_dim", [64, 128, 256]),
                "rnn_type": trial.suggest_categorical("rnn_type", ["gru", "lstm"]),
                "num_layers": trial.suggest_int("num_layers", 1, 2),
                "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            }
        )
    elif model_family == "tcn":
        params.update(
            {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "num_layers": trial.suggest_int("num_layers", 2, 4),
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            }
        )
    elif model_family == "transformer_encoder":
        model_dim = trial.suggest_categorical("model_dim", [128, 256, 384])
        params.update(
            {
                "model_dim": model_dim,
                "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "ff_dim": trial.suggest_categorical("ff_dim", [256, 512, 1024]),
                "pooling": trial.suggest_categorical("pooling", ["cls", "attention"]),
            }
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    return params
