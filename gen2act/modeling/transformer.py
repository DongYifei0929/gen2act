"""Standard transformer encoder stack for sequence fusion and prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceTransformerEncoder(nn.Module):
    """Thin wrapper around nn.TransformerEncoder.

    Shapes:
        input:  [B, N, D]
        output: [B, N, D]
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_mult * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, N, D], got {tuple(x.shape)}")
        return self.encoder(x)
