"""Gen2Act policy composition.

The policy consumes:
- a generated human video [B, 16, 3, 224, 224]
- a robot observation history [B, 8, 3, 224, 224]

and predicts:
- a 6D end-effector pose action vector, discretized per dimension
- terminate logits
- gripper logits

During training, it optionally predicts auxiliary point tracks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gen2act.modeling.resampler import PerceiverResampler
from gen2act.modeling.track import CoTrackerPointTracker, TrackPredictor
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


class CrossAttentionFusion(nn.Module):
    """Cross-attend robot history queries to human video keys and values."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if query_tokens.dim() != 3:
            raise ValueError(f"Expected query tokens as [B, N, D], got {tuple(query_tokens.shape)}")
        if context_tokens.dim() != 3:
            raise ValueError(f"Expected context tokens as [B, N, D], got {tuple(context_tokens.shape)}")

        attended, _ = self.cross_attn(
            query=self.query_norm(query_tokens),
            key=self.context_norm(context_tokens),
            value=context_tokens,
            need_weights=False,
        )
        fused = query_tokens + attended
        fused = fused + self.ff(fused)
        return fused


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
        image_size: int = 224,
        point_tracker: nn.Module | None = None,
        enable_point_tracking: bool = False,
        track_grid_size: int = 10,
        track_query_frame: int = 0,
        track_backward: bool = False,
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
        self.image_size = image_size
        self.point_tracker = point_tracker
        self.enable_point_tracking = enable_point_tracking
        self.track_grid_size = track_grid_size
        self.track_query_frame = track_query_frame
        self.track_backward = track_backward

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
        debug_isfinite: bool = False,
    ):
        del scene_img, task_prompt_tokens, gt_actions, gt_terminate, gt_gripper

        def _check_finite(name: str, tensor: torch.Tensor) -> None:
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
                first_bad = bad[0].tolist() if bad.numel() > 0 else None
                print(f"[nan] {name} non-finite shape={tuple(tensor.shape)} first_bad={first_bad}")

        human_tokens = self.encode_video(human_video, self.human_resampler)   # [B, Kh, D]
        robot_tokens = self.encode_video(robot_history, self.robot_resampler)  # [B, Kr, D]

        if debug_isfinite:
            _check_finite("human_tokens", human_tokens)
            _check_finite("robot_tokens", robot_tokens)

        robot_tokens = self.fusion(robot_tokens, human_tokens)  # [B, Kr, D]
        ctx = robot_tokens.mean(dim=1)  # [B, D]

        if debug_isfinite:
            _check_finite("fusion_tokens", robot_tokens)
            _check_finite("ctx", ctx)

        action_logits, terminate_logits, gripper_logits = self.action_head(ctx)

        if debug_isfinite:
            _check_finite("action_logits", action_logits)
            _check_finite("terminate_logits", terminate_logits)
            _check_finite("gripper_logits", gripper_logits)
        out = {
            "action_logits": action_logits,
            "terminate_logits": terminate_logits,
            "gripper_logits": gripper_logits,
        }

        human_tracks_for_loss = gt_human_tracks
        robot_tracks_for_loss = gt_robot_tracks
        human_track_vis = None
        robot_track_vis = None

        if (
            self.training
            and self.enable_point_tracking
            and self.point_tracker is not None
            and (human_tracks_for_loss is None or robot_tracks_for_loss is None)
        ):
            if human_tracks_for_loss is None:
                human_tracks_for_loss, human_track_vis = self.point_tracker.track(
                    human_video,
                    grid_size=self.track_grid_size,
                    grid_query_frame=self.track_query_frame,
                    backward_tracking=self.track_backward,
                    output_format="bnt",
                )
                human_tracks_for_loss = human_tracks_for_loss / float(self.image_size)
                out["human_tracks"] = human_tracks_for_loss
                out["human_track_vis"] = human_track_vis
            if robot_tracks_for_loss is None:
                robot_tracks_for_loss, robot_track_vis = self.point_tracker.track(
                    robot_history,
                    grid_size=self.track_grid_size,
                    grid_query_frame=self.track_query_frame,
                    backward_tracking=self.track_backward,
                    output_format="bnt",
                )
                robot_tracks_for_loss = robot_tracks_for_loss / float(self.image_size)
                out["robot_tracks"] = robot_tracks_for_loss
                out["robot_track_vis"] = robot_track_vis

        if self.training and human_tracks_for_loss is not None and robot_tracks_for_loss is not None:
            out["human_track_pred"] = self.track_predictor(human_tokens, human_tracks_for_loss)
            out["robot_track_pred"] = self.track_predictor(robot_tokens, robot_tracks_for_loss)
            if gt_human_tracks is not None:
                out["human_tracks"] = gt_human_tracks / float(self.image_size)
            if gt_robot_tracks is not None:
                out["robot_tracks"] = gt_robot_tracks / float(self.image_size)

        return out


def build_default_policy(
    num_action_dims: int = 6,
    num_bins: int = 256,
    image_size: int = 224,
    patch_size: int = 16,
    hidden_dim: int = 768,
    num_vit_layers: int = 12,
    num_vit_heads: int = 12,
    latent_tokens: int = 16,
    vit_pretrained: str | None = None,
    enable_point_tracking: bool = False,
    point_tracker_checkpoint: str | None = None,
    point_tracker_use_hub: bool = False,
    point_tracker_offline: bool = True,
    point_tracker_v2: bool = False,
    point_tracker_window_len: int = 60,
    track_grid_size: int = 10,
    track_query_frame: int = 0,
    track_backward: bool = False,
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
        pretrained=vit_pretrained,
    )
    human_resampler = PerceiverResampler(dim=hidden_dim, num_latents=latent_tokens, num_layers=2, num_heads=8)
    robot_resampler = PerceiverResampler(dim=hidden_dim, num_latents=latent_tokens, num_layers=2, num_heads=8)
    fusion = CrossAttentionFusion(dim=hidden_dim, heads=8)
    action_head = ActionHead(dim=hidden_dim, num_action_dims=num_action_dims, num_bins=num_bins)
    track_predictor = TrackPredictor(dim=hidden_dim, depth=6, heads=8, track_dim=2, max_steps=64)
    point_tracker = None
    if enable_point_tracking:
        point_tracker = CoTrackerPointTracker(
            checkpoint=point_tracker_checkpoint,
            offline=point_tracker_offline,
            v2=point_tracker_v2,
            window_len=point_tracker_window_len,
            use_hub=point_tracker_use_hub,
        )

    return Gen2ActPolicy(
        vit_encoder=vit,
        human_resampler=human_resampler,
        robot_resampler=robot_resampler,
        fusion_encoder=fusion,
        action_head=action_head,
        track_predictor=track_predictor,
        num_action_dims=num_action_dims,
        num_bins=num_bins,
        image_size=image_size,
        point_tracker=point_tracker,
        enable_point_tracking=enable_point_tracking,
        track_grid_size=track_grid_size,
        track_query_frame=track_query_frame,
        track_backward=track_backward,
    )
