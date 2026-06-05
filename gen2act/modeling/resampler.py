"""Perceiver-style resampler used to compress token sequences.

This module takes an input token sequence and returns a fixed number of latent
tokens. It is used independently for:
- generated human video tokens
- robot history tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Cross-attend from learned latents to an input token sequence.

    Shapes:
        input:  [B, N, D] or [B, T, P, D]
        output: [B, K, D]
    """

    def __init__(
        self,
        dim: int,
        num_latents: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) / dim**0.5)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_norm": nn.LayerNorm(dim),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "self_norm": nn.LayerNorm(dim),
                        "self_attn": nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ff_norm": nn.LayerNorm(dim),
                        "ff": nn.Sequential(
                            nn.Linear(dim, ff_mult * dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ff_mult * dim, dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_attn_gates = nn.Parameter(torch.ones(num_layers))
        self.self_attn_gates = nn.Parameter(torch.ones(num_layers))
        self.ff_gates = nn.Parameter(torch.ones(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, 2)  # [B, T*P, D]
        if x.dim() != 3:
            raise ValueError(f"Expected [B, N, D] or [B, T, P, D], got {tuple(x.shape)}")

        x = self.in_norm(x)
        batch_size = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]

        for index, block in enumerate(self.blocks):
            cross_out, _ = block["cross_attn"](
                query=block["cross_norm"](latents),
                key=x,
                value=x,
                need_weights=False,
            )
            latents = latents + torch.tanh(self.cross_attn_gates[index]) * cross_out
            normed = block["self_norm"](latents)
            attn_out, _ = block["self_attn"](normed, normed, normed, need_weights=False)
            latents = latents + torch.tanh(self.self_attn_gates[index]) * attn_out
            latents = latents + torch.tanh(self.ff_gates[index]) * block["ff"](block["ff_norm"](latents))

        return latents
