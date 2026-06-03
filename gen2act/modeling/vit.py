"""Vision Transformer backbone used for per-frame image encoding.

Supports:
- torchvision ViT-B/16 (random init or ImageNet pretrained)
- DINOv2 ViT-B/14 via torchvision (pretrained)
- timm fallback for custom configurations
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16
except Exception:  # pragma: no cover - import-time fallback only
    vit_b_16 = None

try:
    from torchvision.models import dinov2_vitb14 as _tv_dinov2_vitb14
except Exception:  # pragma: no cover - import-time fallback only
    _tv_dinov2_vitb14 = None

try:
    import timm
except Exception:  # pragma: no cover - import-time fallback only
    timm = None


class ViTBackbone(nn.Module):
    """Thin adapter around torchvision/timm Vision Transformer implementations.

    Set ``pretrained="dinov2"`` to load a DINOv2 ViT-B/14 backbone from torchvision
    (or timm as fallback).  When using DINOv2 the effective patch size is 14 and the
    image size must be 224.
    """

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
        pretrained: Optional[str] = None,
    ) -> None:
        super().__init__()
        del batch_first

        self.pretrained = pretrained
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        if pretrained == "dinov2":
            if image_size != 224 or in_channels != 3:
                raise ValueError(f"DINOv2 requires image_size=224, in_channels=3; got {image_size}, {in_channels}")
            self.patch_size = 14
            self.hidden_dim = 768
            self.image_size = 224
            self.num_patches = (224 // 14) * (224 // 14)

        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

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
        if self.pretrained == "dinov2":
            return self._build_dinov2(image_size=image_size)

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
    def _build_dinov2(*, image_size: int) -> nn.Module:
        """Load a DINOv2 ViT-B/14 backbone (torchvision first, then timm)."""
        if _tv_dinov2_vitb14 is not None:
            try:
                model = _tv_dinov2_vitb14(weights="DEFAULT")
                model.image_size = image_size
                return model
            except Exception as exc:
                warnings.warn(f"torchvision DINOv2 failed ({exc}), trying timm fallback")

        if timm is None:
            raise ImportError(
                "DINOv2 requires torchvision >= 0.16 or timm.  Neither is available."
            )
        model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=True,
            img_size=image_size,
            num_classes=0,
        )
        return model

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

        if self.pretrained == "dinov2":
            return self._forward_dinov2(x, self.backend)

        if isinstance(self.backend, nn.Module) and hasattr(self.backend, "conv_proj"):
            return self._forward_torchvision(x, self.backend)
        return self._forward_timm(x, self.backend)

    def _forward_dinov2(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        # torchvision DINOv2 uses ``blocks`` + ``norm`` internally
        if hasattr(model, "_process_input") and hasattr(model, "blocks"):
            return self._forward_dinov2_torchvision(x, model)
        # timm fallback
        return self._forward_timm(x, model)

    @staticmethod
    def _forward_dinov2_torchvision(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        x = model._process_input(x)
        for block in model.blocks:
            x = block(x)
        x = model.norm(x)
        return x[:, 1:, :]  # strip CLS token

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
