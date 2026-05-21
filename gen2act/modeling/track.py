"""Point tracking utilities and auxiliary track prediction head.

This module provides two related pieces:
- a CoTracker-backed point tracker for extracting tracks from videos
- a lightweight transformer head that predicts track trajectories from latent
  conditioning tokens during Gen2Act training
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from gen2act.modeling.transformer import SequenceTransformerEncoder


def _ensure_cotracker_path() -> None:
    """Make the bundled `co-tracker` package importable.

    The repository keeps the tracker implementation in a sibling directory
    rather than publishing it as an installed dependency, so we add that
    directory to `sys.path` on demand.
    """

    repo_root = Path(__file__).resolve().parents[2]
    cotracker_root = repo_root / "co-tracker"
    cotracker_path = str(cotracker_root)
    if cotracker_root.is_dir() and cotracker_path not in sys.path:
        sys.path.insert(0, cotracker_path)


class CoTrackerPointTracker(nn.Module):
    """Wrapper around the bundled CoTracker predictor.

    This is the point-tracking component referenced by the architecture doc.
    It delegates video tracking to the local `co-tracker` package and exposes a
    simple PyTorch module interface for Gen2Act code.
    """

    def __init__(
        self,
        checkpoint: str = "./checkpoints/scaled_offline.pth",
        offline: bool = True,
        v2: bool = False,
        window_len: int = 60,
    ) -> None:
        super().__init__()
        _ensure_cotracker_path()

        try:
            from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
        except Exception as exc:  # pragma: no cover - import-time environment issue
            raise ImportError(
                "Could not import the bundled co-tracker package. "
                "Make sure the repository checkout includes the `co-tracker/` directory."
            ) from exc

        predictor_cls = CoTrackerPredictor if offline else CoTrackerOnlinePredictor
        self.predictor = predictor_cls(
            checkpoint=checkpoint,
            offline=offline,
            v2=v2,
            window_len=window_len,
        )

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        segm_mask: Optional[torch.Tensor] = None,
        grid_size: int = 0,
        grid_query_frame: int = 0,
        backward_tracking: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track points through a video.

        Args:
            video: [B, T, 3, H, W]
            queries: Optional [B, N, 3] tensor in (t, x, y) format.
            segm_mask: Optional [B, 1, H, W] mask used to filter a grid.
            grid_size: Dense grid size if queries are not provided.
            grid_query_frame: Query frame for dense/grid tracking.
            backward_tracking: Whether to run the tracker in both directions.

        Returns:
            tracks: [B, T, N, 2]
            visibility: [B, T, N]
        """

        return self.predictor(
            video=video,
            queries=queries,
            segm_mask=segm_mask,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=backward_tracking,
        )


class TrackPredictor(nn.Module):
    """Predict per-point trajectories from conditioning tokens.

    Shapes:
        cond_tokens: [B, K, D]
        gt_tracks:   [B, N, T, 2 or 3]
        output:      [B, N, T, 2 or 3]
    """

    def __init__(
        self,
        dim: int,
        depth: int = 6,
        heads: int = 8,
        track_dim: int = 2,
        max_steps: int = 64,
    ) -> None:
        super().__init__()
        self.track_dim = track_dim
        self.max_steps = max_steps

        self.point_embed = nn.Linear(track_dim, dim)
        self.time_embed = nn.Embedding(max_steps, dim)
        self.encoder = SequenceTransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=4,
        )
        self.out = nn.Linear(dim, track_dim)

    def forward(self, cond_tokens: torch.Tensor, gt_tracks: torch.Tensor) -> torch.Tensor:
        if gt_tracks.dim() != 4:
            raise ValueError(f"Expected [B, N, T, C], got {tuple(gt_tracks.shape)}")

        batch_size, num_points, num_steps, track_dim = gt_tracks.shape
        if track_dim != self.track_dim:
            raise ValueError(f"track_dim mismatch: expected {self.track_dim}, got {track_dim}")
        if num_steps > self.max_steps:
            raise ValueError(f"num_steps={num_steps} exceeds max_steps={self.max_steps}")

        start_points = gt_tracks[:, :, 0, :]  # [B, N, C]
        point_tokens = self.point_embed(start_points)  # [B, N, D]

        time_tokens = self.time_embed.weight[:num_steps]  # [T, D]
        query_tokens = point_tokens[:, :, None, :] + time_tokens[None, None, :, :]  # [B, N, T, D]
        query_tokens = query_tokens.reshape(batch_size, num_points * num_steps, -1)  # [B, N*T, D]

        x = torch.cat([cond_tokens, query_tokens], dim=1)
        x = self.encoder(x)
        pred = self.out(x[:, -num_points * num_steps :, :])
        return pred.view(batch_size, num_points, num_steps, self.track_dim)
