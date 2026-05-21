"""Vision Transformer backbone used for per-frame image encoding.

The backbone now wraps a real torchvision ViT first, with a timm fallback for
environments where torchvision is unavailable or a non-default configuration is
requested. The public API stays the same as the prior local implementation:
it returns patch tokens shaped ``[B, P, D]`` rather than classifier logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16
except Exception:  # pragma: no cover - import-time fallback only
    vit_b_16 = None

try:
    import timm
except Exception:  # pragma: no cover - import-time fallback only
    timm = None


class ViTBackbone(nn.Module):
    """Thin adapter around torchvision/timm Vision Transformer implementations."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        del batch_first

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.backend = self._select_backend(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def _select_backend(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> nn.Module:
        if self._matches_vit_b16(image_size, patch_size, in_channels, hidden_dim, mlp_dim, num_layers, num_heads):
            if vit_b_16 is None:
                if timm is None:
                    raise ImportError("torchvision is unavailable and timm is not installed")
                return self._build_timm(
                    image_size=image_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            return vit_b_16(weights=None, image_size=image_size, num_classes=0, dropout=dropout)

        if timm is None:
            raise ImportError(
                "timm is required for non-default ViT configurations when torchvision's vit_b_16 is not a fit"
            )
        return self._build_timm(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    @staticmethod
    def _matches_vit_b16(
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> bool:
        return (
            image_size == 224
            and patch_size == 16
            and in_channels == 3
            and hidden_dim == 768
            and mlp_dim == 3072
            and num_layers == 12
            and num_heads == 12
        )

    @staticmethod
    def _build_timm(
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> nn.Module:
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_dim / float(hidden_dim),
            drop_rate=dropout,
            num_classes=0,
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into patch tokens."""
        if x.dim() != 4:
            raise ValueError(f"Expected [B, 3, H, W], got shape {tuple(x.shape)}")

        if isinstance(self.backend, nn.Module) and hasattr(self.backend, "conv_proj"):
            return self._forward_torchvision(x, self.backend)
        return self._forward_timm(x, self.backend)

    def _forward_torchvision(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        x = model._process_input(x)
        batch_size = x.shape[0]
        cls_token = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        encoder = model.encoder
        x = encoder.dropout(x)
        for layer in encoder.layers:
            x = layer(x)
        x = encoder.ln(x)
        return x[:, 1:, :]

    def _forward_timm(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        tokens = model.forward_features(x)
        if tokens.dim() != 3:
            raise RuntimeError(
                f"Expected timm forward_features to return [B, N, D], got shape {tuple(tokens.shape)}"
            )
        if tokens.shape[1] > 1:
            return tokens[:, 1:, :]
        return tokens
