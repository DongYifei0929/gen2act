"""LeRobot v3 video-backed dataset adapter for Gen2Act policy training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import imageio
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError("imageio is required to read LeRobot videos") from exc


DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


@dataclass(frozen=True)
class EpisodeVideoInfo:
    chunk_index: int
    file_index: int
    from_timestamp: Optional[float]


class LeRobotVideoPolicyDataset(Dataset):
    """Load fixed-length policy windows from LeRobot v3 videos and parquet data."""

    def __init__(
        self,
        repo_root: str | Path,
        video_key: str = "observation.images.table_cam",
        action_key: str = "action",
        episode_index_key: str = "episode_index",
        frame_index_key: str = "frame_index",
        timestamp_key: str = "timestamp",
        human_video_len: int = 16,
        robot_history_len: int = 8,
        image_size: int = 224,
        num_action_dims: int = 6,
        action_stride: int = 1,
        max_samples: Optional[int] = None,
        max_episodes: Optional[int] = None,
        gripper_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve()
        self.video_key = video_key
        self.action_key = action_key
        self.episode_index_key = episode_index_key
        self.frame_index_key = frame_index_key
        self.timestamp_key = timestamp_key
        self.human_video_len = human_video_len
        self.robot_history_len = robot_history_len
        self.image_size = image_size
        self.num_action_dims = num_action_dims
        self.action_stride = action_stride
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.gripper_threshold = gripper_threshold

        self._info = self._load_info()
        self._fps = float(self._info.get("fps", 20))
        self._video_path_template = self._info.get("video_path") or DEFAULT_VIDEO_PATH

        self._actions, self._episode_indices, self._frame_indices, self._timestamps = self._load_data()
        self._episode_videos = self._load_episode_videos()
        self._episode_to_frames = self._group_frames_by_episode()
        self._samples = self._index_samples()

        self._video_readers: Dict[Path, imageio.core.format.Reader] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_readers"] = {}
        return state

    def _load_info(self) -> dict:
        info_path = self.repo_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found: {info_path}")
        return json_load(info_path)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_dir = self.repo_root / "data"
        parquet_paths = sorted(data_dir.glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        frames: List[pd.DataFrame] = []
        for path in parquet_paths:
            df = pd.read_parquet(path, columns=[
                self.action_key,
                self.episode_index_key,
                self.frame_index_key,
                self.timestamp_key,
                "index",
            ])
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("index").reset_index(drop=True)

        actions = np.stack(merged[self.action_key].to_numpy())
        episode_indices = merged[self.episode_index_key].to_numpy(dtype=np.int64)
        frame_indices = merged[self.frame_index_key].to_numpy(dtype=np.int64)
        timestamps = merged[self.timestamp_key].to_numpy(dtype=np.float32)
        return actions, episode_indices, frame_indices, timestamps

    def _load_episode_videos(self) -> Dict[int, EpisodeVideoInfo]:
        episodes_dir = self.repo_root / "meta" / "episodes"
        parquet_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No episode metadata found under {episodes_dir}")

        key_prefix = f"videos/{self.video_key}"
        chunk_key = f"{key_prefix}/chunk_index"
        file_key = f"{key_prefix}/file_index"
        from_key = f"{key_prefix}/from_timestamp"

        frames: List[pd.DataFrame] = []
        for path in parquet_paths:
            df = pd.read_parquet(path)
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        missing = [k for k in ("episode_index", chunk_key, file_key) if k not in merged.columns]
        if missing:
            raise KeyError(f"Episode metadata missing required columns: {missing}")

        episode_videos: Dict[int, EpisodeVideoInfo] = {}
        for _, row in merged.iterrows():
            episode_idx = int(row["episode_index"])
            chunk_index = int(row[chunk_key])
            file_index = int(row[file_key])
            from_timestamp = row[from_key] if from_key in merged.columns else None
            if isinstance(from_timestamp, float) and np.isnan(from_timestamp):
                from_timestamp = None
            episode_videos[episode_idx] = EpisodeVideoInfo(
                chunk_index=chunk_index,
                file_index=file_index,
                from_timestamp=from_timestamp,
            )
        return episode_videos

    def _group_frames_by_episode(self) -> Dict[int, np.ndarray]:
        episode_ids = np.unique(self._episode_indices)
        if self.max_episodes is not None:
            episode_ids = episode_ids[: self.max_episodes]
        episode_to_frames: Dict[int, np.ndarray] = {}
        for episode_id in episode_ids:
            mask = self._episode_indices == episode_id
            frame_ids = np.nonzero(mask)[0]
            order = np.argsort(self._frame_indices[frame_ids])
            episode_to_frames[int(episode_id)] = frame_ids[order]
        return episode_to_frames

    def _index_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        window_len = max(self.human_video_len, self.robot_history_len)
        for episode_id, frame_ids in self._episode_to_frames.items():
            if len(frame_ids) < window_len:
                continue
            last_start = len(frame_ids) - window_len
            for start_index in range(0, last_start + 1, self.action_stride):
                samples.append((episode_id, start_index))
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _ensure_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim != 3:
            raise ValueError(f"Unexpected image shape for {self.video_key}: {tuple(tensor.shape)}")
        tensor = tensor.float()
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            h, w = tensor.shape[-2], tensor.shape[-1]
            if h < w:
                new_h = self.image_size
                new_w = int(round(w * self.image_size / h))
            else:
                new_w = self.image_size
                new_h = int(round(h * self.image_size / w))
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if new_h > self.image_size:
                start_h = (new_h - self.image_size) // 2
                tensor = tensor[:, start_h:start_h + self.image_size, :]
            if new_w > self.image_size:
                start_w = (new_w - self.image_size) // 2
                tensor = tensor[:, :, start_w:start_w + self.image_size]
        return tensor

    def _get_video_path(self, episode_id: int) -> Path:
        info = self._episode_videos.get(episode_id)
        if info is None:
            raise KeyError(f"Missing video metadata for episode {episode_id}")
        return self.repo_root / self._video_path_template.format(
            video_key=self.video_key,
            chunk_index=info.chunk_index,
            file_index=info.file_index,
        )

    def _get_reader(self, video_path: Path):
        reader = self._video_readers.get(video_path)
        if reader is None:
            reader = imageio.get_reader(str(video_path))
            self._video_readers[video_path] = reader
        return reader

    def _frame_from_timestamp(self, episode_id: int, timestamp: float, frame_index: int) -> int:
        info = self._episode_videos.get(episode_id)
        if info is None or info.from_timestamp is None:
            return int(frame_index)
        return max(0, int(round((timestamp - info.from_timestamp) * self._fps)))

    def _read_video(self, episode_id: int, frame_ids: np.ndarray, length: int) -> torch.Tensor:
        video_path = self._get_video_path(episode_id)
        reader = self._get_reader(video_path)
        frames = []
        for row_idx in frame_ids[:length]:
            frame_idx = self._frame_from_timestamp(
                episode_id,
                float(self._timestamps[row_idx]),
                int(self._frame_indices[row_idx]),
            )
            frame = reader.get_data(frame_idx)
            frames.append(self._ensure_image_tensor(frame))
        return torch.stack(frames, dim=0)

    def _read_targets(self, row_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        action = torch.as_tensor(self._actions[row_idx]).float()
        action_target = action[: self.num_action_dims]
        if action.numel() > self.num_action_dims:
            gripper_value = action[self.num_action_dims]
        else:
            gripper_value = action[-1]
        gripper_target = torch.tensor(int(gripper_value.item() > self.gripper_threshold), dtype=torch.long)
        return action_target, gripper_target

    def sample_window(self, episode_id: int, start_index: int) -> Dict[str, torch.Tensor | int]:
        frame_ids = self._episode_to_frames[episode_id]
        window_frame_ids = frame_ids[start_index : start_index + max(self.human_video_len, self.robot_history_len)]
        human_video = self._read_video(episode_id, window_frame_ids, self.human_video_len)
        robot_history = self._read_video(episode_id, window_frame_ids, self.robot_history_len)

        target_offset = self.robot_history_len - 1
        target_row_idx = int(window_frame_ids[target_offset])
        action_target, gripper_target = self._read_targets(target_row_idx)
        return {
            "demo_name": int(episode_id),
            "start_index": int(start_index),
            "human_video": human_video,
            "robot_history": robot_history,
            "action_target": action_target,
            "gripper_target": gripper_target,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        episode_id, start_index = self._samples[index]
        return self.sample_window(episode_id, start_index)


def json_load(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)
