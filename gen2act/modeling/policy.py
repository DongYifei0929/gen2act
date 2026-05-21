"""Gen2Act policy composition.

The policy consumes:
- a generated human video [B, 16, 3, 224, 224]
- a robot observation history [B, 8, 3, 224, 224]

and predicts:
- discretized end-effector actions
- terminate logits
- gripper logits

During training, it optionally predicts auxiliary point tracks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gen2act.modeling.resampler import PerceiverResampler
from gen2act.modeling.track import TrackPredictor
from gen2act.modeling.transformer import SequenceTransformerEncoder
from gen2act.modeling.vit import ViTBackbone


class ActionHead(nn.Module):
    """Predict discretized actions and binary auxiliary decisions."""

    def __init__(self, dim: int, num_action_dims: int, num_bins: int = 256) -> None:
        super().__init__()
        self.num_action_dims = num_action_dims
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.action_proj = nn.Linear(dim, num_action_dims * num_bins)
        self.terminate_proj = nn.Linear(dim, 2)
        self.gripper_proj = nn.Linear(dim, 2)

    def forward(self, ctx: torch.Tensor):
        hidden = self.mlp(ctx)
        action_logits = self.action_proj(hidden).view(-1, self.num_action_dims, self.num_bins)
        terminate_logits = self.terminate_proj(hidden)
        gripper_logits = self.gripper_proj(hidden)
        return action_logits, terminate_logits, gripper_logits


class Gen2ActPolicy(nn.Module):
    """Compose the full Gen2Act translation policy."""

    def __init__(
        self,
        vit_encoder: nn.Module,
        human_resampler: nn.Module,
        robot_resampler: nn.Module,
        fusion_encoder: nn.Module,
        action_head: nn.Module,
        track_predictor: nn.Module,
        num_action_dims: int,
        num_bins: int = 256,
    ) -> None:
        super().__init__()
        self.vit = vit_encoder
        self.human_resampler = human_resampler
        self.robot_resampler = robot_resampler
        self.fusion = fusion_encoder
        self.action_head = action_head
        self.track_predictor = track_predictor
        self.num_action_dims = num_action_dims
        self.num_bins = num_bins

    def encode_video(self, frames: torch.Tensor, resampler: nn.Module) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(frames.shape)}")

        batch_size, time_steps, channels, height, width = frames.shape
        x = frames.reshape(batch_size * time_steps, channels, height, width)
        patch_tokens = self.vit(x)  # [B*T, P, D]
        patch_tokens = patch_tokens.view(batch_size, time_steps, patch_tokens.shape[1], patch_tokens.shape[2])
        return resampler(patch_tokens)

    def forward(
        self,
        scene_img,
        task_prompt_tokens,
        human_video,
        robot_history,
        gt_human_tracks=None,
        gt_robot_tracks=None,
        gt_actions=None,
        gt_terminate=None,
        gt_gripper=None,
    ):
        del scene_img, task_prompt_tokens, gt_actions, gt_terminate, gt_gripper

        human_tokens = self.encode_video(human_video, self.human_resampler)   # [B, Kh, D]
        robot_tokens = self.encode_video(robot_history, self.robot_resampler)  # [B, Kr, D]

        tokens = torch.cat([human_tokens, robot_tokens], dim=1)  # [B, Kh+Kr, D]
        tokens = self.fusion(tokens)
        ctx = tokens.mean(dim=1)  # [B, D]

        action_logits, terminate_logits, gripper_logits = self.action_head(ctx)
        out = {
            "action_logits": action_logits,
            "terminate_logits": terminate_logits,
            "gripper_logits": gripper_logits,
        }

        if self.training and gt_human_tracks is not None and gt_robot_tracks is not None:
            out["human_track_pred"] = self.track_predictor(human_tokens, gt_human_tracks)
            out["robot_track_pred"] = self.track_predictor(robot_tokens, gt_robot_tracks)

        return out


def build_default_policy(
    num_action_dims: int = 7,
    num_bins: int = 256,
    image_size: int = 224,
    patch_size: int = 16,
    hidden_dim: int = 768,
    num_vit_layers: int = 12,
    num_vit_heads: int = 12,
    latent_tokens: int = 16,
) -> Gen2ActPolicy:
    """Build the default Gen2Act model stack.

    This uses a standard ViT-B/16-style backbone and a standard PyTorch
    transformer encoder stack.
    """

    vit = ViTBackbone(
        image_size=image_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_layers=num_vit_layers,
        num_heads=num_vit_heads,
    )
    human_resampler = PerceiverResampler(dim=hidden_dim, num_latents=latent_tokens, num_layers=2, num_heads=8)
    robot_resampler = PerceiverResampler(dim=hidden_dim, num_latents=latent_tokens, num_layers=2, num_heads=8)
    fusion = SequenceTransformerEncoder(dim=hidden_dim, depth=4, heads=8)
    action_head = ActionHead(dim=hidden_dim, num_action_dims=num_action_dims, num_bins=num_bins)
    track_predictor = TrackPredictor(dim=hidden_dim, depth=6, heads=8, track_dim=2, max_steps=64)

    return Gen2ActPolicy(
        vit_encoder=vit,
        human_resampler=human_resampler,
        robot_resampler=robot_resampler,
        fusion_encoder=fusion,
        action_head=action_head,
        track_predictor=track_predictor,
        num_action_dims=num_action_dims,
        num_bins=num_bins,
    )
