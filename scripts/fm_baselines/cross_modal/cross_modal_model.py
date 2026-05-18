#!/usr/bin/env python3
"""
Cross-modal depth model with shared projection and alignment losses.

Architecture
------------

  depth_x (B, T, D_depth)
       |
  DepthTCNEncoder   ← trainable (warm-started from downstream checkpoint)
       |
  depth_repr (B, D_tcn)
       |
  shared_proj (MLP, same weights for both modalities)
       |
  z_depth (B, D_shared)              z_imu = shared_proj(imu_embed)
       |                                         |
  classifier → L_cls            cosine(z_depth, z_imu) → L_align
       |
  decoder(z_depth) → imu_recon → MSE(imu_embed) → L_recon   [optional]

At inference: only depth path is used.  IMU branch is dropped.

Loss
----
  L = L_cls + λ_align * L_align + λ_recon * L_recon

  L_cls   : cross-entropy on stress label
  L_align : 1 - mean cosine similarity between z_depth and z_imu
  L_recon : MSE between decoder(z_depth) and the frozen imu_embed
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from depth_models import DepthTCNEncoder


# ---------------------------------------------------------------------------
# Shared projection MLP
# ---------------------------------------------------------------------------

class SharedProjectionMLP(nn.Module):
    """
    Single-hidden-layer MLP that projects any D_in → D_shared.
    Same weights used for both depth and IMU modalities.
    """

    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder (depth shared space → IMU embedding space)
# ---------------------------------------------------------------------------

class IMUDecoder(nn.Module):
    """Reconstruct IMU teacher embedding from z_depth."""

    def __init__(self, shared_dim: int, imu_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(shared_dim, imu_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, imu_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class StressClassifier(nn.Module):
    def __init__(self, shared_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, max(32, shared_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, shared_dim // 2), 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Combined cross-modal model
# ---------------------------------------------------------------------------

class CrossModalDepthModel(nn.Module):
    """
    Full cross-modal model for training.

    Parameters
    ----------
    encoder      : DepthTCNEncoder (warm-started from downstream checkpoint)
    shared_dim   : projection output dimension
    imu_dim      : teacher embedding dimension (D_imu)
    dropout      : dropout for projection / classifier / decoder
    use_decoder  : whether to include IMU reconstruction loss
    """

    def __init__(
        self,
        encoder: DepthTCNEncoder,
        shared_dim: int,
        imu_dim: int,
        dropout: float = 0.1,
        use_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.shared_proj = SharedProjectionMLP(encoder.repr_dim, shared_dim, dropout)
        self.classifier = StressClassifier(shared_dim, dropout)
        self.decoder = IMUDecoder(shared_dim, imu_dim, dropout) if use_decoder else None
        self.shared_dim = shared_dim
        self.imu_dim = imu_dim

    def encode_depth(self, depth_x: torch.Tensor) -> torch.Tensor:
        """depth_x: (B, T, D) → z_depth: (B, shared_dim)"""
        depth_repr = self.encoder(depth_x)
        return self.shared_proj(depth_repr)

    def encode_imu(self, imu_embed: torch.Tensor) -> torch.Tensor:
        """imu_embed: (B, imu_dim) → z_imu: (B, shared_dim)"""
        return self.shared_proj(imu_embed)

    def forward(
        self,
        depth_x: torch.Tensor,
        imu_embed: torch.Tensor | None = None,
        lambda_align: float = 0.5,
        lambda_recon: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          logits     : (B, 2) classification logits
          loss_cls   : scalar
          loss_align : scalar  (0 if imu_embed is None)
          loss_recon : scalar  (0 if imu_embed is None or use_decoder=False)
          loss_total : weighted sum
          z_depth    : (B, shared_dim)
        """
        z_depth = self.encode_depth(depth_x)
        logits = self.classifier(z_depth)

        loss_cls = F.cross_entropy(logits, torch.zeros(depth_x.size(0), dtype=torch.long, device=depth_x.device))
        loss_align = depth_x.new_zeros(1).squeeze()
        loss_recon = depth_x.new_zeros(1).squeeze()

        if imu_embed is not None:
            imu_embed_detached = imu_embed.detach()
            z_imu = self.encode_imu(imu_embed_detached)
            loss_align = 1.0 - F.cosine_similarity(z_depth, z_imu, dim=1).mean()
            if self.decoder is not None:
                imu_recon = self.decoder(z_depth)
                loss_recon = F.mse_loss(imu_recon, imu_embed_detached)

        loss_total = loss_cls + lambda_align * loss_align + lambda_recon * loss_recon
        return {
            "logits": logits,
            "loss_cls": loss_cls,
            "loss_align": loss_align,
            "loss_recon": loss_recon,
            "loss_total": loss_total,
            "z_depth": z_depth,
        }

    def predict_scores(self, depth_x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns stress probability scores (B,)."""
        self.eval()
        with torch.no_grad():
            z = self.encode_depth(depth_x)
            logits = self.classifier(z)
            return torch.softmax(logits, dim=1)[:, 1]
