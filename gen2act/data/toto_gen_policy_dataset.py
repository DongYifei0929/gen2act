"""Toto-Gen dataset adapter for Gen2Act policy training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import imageio
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError("imageio is required to read Toto-Gen videos") from exc


@dataclass(frozen=True)
class TotoEpisode:
    episode_id: str
    data_path: Path
    robot_video_path: Path
    human_video_path: Path
    world_vector: np.ndarray
    rotation_delta: np.ndarray
    open_gripper: np.ndarray
    terminate: np.ndarray


class TotoGenPolicyDataset(Dataset):
    """Load fixed-length policy windows from Toto-Gen mp4 + json episodes."""

    def __init__(
        self,
        dataset_root: str | Path,
        generated_subdir: str = "toto-gen",
        robot_subdir: str = "toto",
        generated_video_name: str = "generated.mp4",
        robot_video_name: str = "image.mp4",
        metadata_name: str = "data.json",
        human_video_len: int = 16,
        robot_history_len: int = 8,
        image_size: int = 224,
        num_action_dims: int = 6,
        action_stride: int = 1,
        max_samples: Optional[int] = None,
        max_episodes: Optional[int] = None,
        gripper_threshold: float = 0.5,
        terminate_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).resolve()
        self.generated_subdir = generated_subdir
        self.robot_subdir = robot_subdir
        self.generated_video_name = generated_video_name
        self.robot_video_name = robot_video_name
        self.metadata_name = metadata_name
        self.human_video_len = human_video_len
        self.robot_history_len = robot_history_len
        self.image_size = image_size
        self.num_action_dims = num_action_dims
        self.action_stride = action_stride
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.gripper_threshold = gripper_threshold
        self.terminate_threshold = terminate_threshold

        self._episodes = self._load_episodes()
        self._samples = self._index_samples()
        self._video_readers: Dict[Path, imageio.core.format.Reader] = {}
        self._human_video_cache: Dict[Path, torch.Tensor] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_readers"] = {}
        state["_human_video_cache"] = {}
        return state

    def _warn_non_finite(self, name: str, tensor: torch.Tensor, episode_id: str, step: int | None = None) -> None:
        if torch.isfinite(tensor).all():
            return
        step_info = f" step={step}" if step is not None else ""
        print(f"[warn] Non-finite {name} in episode={episode_id}{step_info}")

    def _load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def _load_episodes(self) -> List[TotoEpisode]:
        robot_root = self.dataset_root / self.robot_subdir
        human_root = self.dataset_root / self.generated_subdir
        if not robot_root.exists():
            raise FileNotFoundError(f"Missing robot demo folder: {robot_root}")
        if not human_root.exists():
            raise FileNotFoundError(f"Missing generated video folder: {human_root}")

        episodes: List[TotoEpisode] = []
        for robot_dir in sorted(robot_root.glob("episode_*/")):
            episode_id = robot_dir.name
            human_dir = human_root / episode_id
            if not human_dir.exists():
                continue

            data_path = robot_dir / self.metadata_name
            robot_video_path = robot_dir / self.robot_video_name
            human_video_path = human_dir / self.generated_video_name
            if not data_path.exists() or not robot_video_path.exists() or not human_video_path.exists():
                continue

            payload = self._load_json(data_path)
            world_vector = np.asarray(payload.get("world_vector", []), dtype=np.float32)
            rotation_delta = np.asarray(payload.get("rotation_delta", []), dtype=np.float32)
            open_gripper = np.asarray(payload.get("open_gripper", []))
            terminate = np.asarray(payload.get("terminate_episode", []), dtype=np.float32)

            if world_vector.size == 0:
                raise ValueError(f"Empty world_vector in {data_path}")
            if rotation_delta.size == 0:
                raise ValueError(f"Empty rotation_delta in {data_path}")
            if open_gripper.size == 0:
                raise ValueError(f"Empty open_gripper in {data_path}")
            if terminate.size == 0:
                raise ValueError(f"Empty terminate_episode in {data_path}")
            if world_vector.shape[-1] != 3 or rotation_delta.shape[-1] != 3:
                raise ValueError(
                    f"Expected world_vector and rotation_delta as 3D actions in {data_path}; "
                    f"got {world_vector.shape} and {rotation_delta.shape}"
                )

            episodes.append(
                TotoEpisode(
                    episode_id=episode_id,
                    data_path=data_path,
                    robot_video_path=robot_video_path,
                    human_video_path=human_video_path,
                    world_vector=world_vector,
                    rotation_delta=rotation_delta,
                    open_gripper=open_gripper,
                    terminate=terminate,
                )
            )

        if not episodes:
            raise FileNotFoundError(
                f"No valid episodes found under {robot_root} and {human_root}"
            )
        if self.max_episodes is not None:
            episodes = episodes[: self.max_episodes]
        return episodes

    def _index_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        window_len = self.robot_history_len
        for episode_idx, episode in enumerate(self._episodes):
            num_steps = min(
                len(episode.world_vector),
                len(episode.rotation_delta),
                len(episode.open_gripper),
                len(episode.terminate),
            )
            if num_steps < window_len:
                continue
            last_start = num_steps - window_len
            for start_index in range(0, last_start + 1, self.action_stride):
                samples.append((episode_idx, start_index))
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
            raise ValueError(f"Unexpected image shape: {tuple(tensor.shape)}")
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

    def _get_reader(self, video_path: Path):
        reader = self._video_readers.get(video_path)
        if reader is None:
            reader = imageio.get_reader(str(video_path))
            self._video_readers[video_path] = reader
        return reader

    def _get_video_length(self, reader) -> int:
        length = 0
        if hasattr(reader, "count_frames"):
            try:
                length = int(reader.count_frames() or 0)
            except Exception:
                length = 0
        if length <= 0 and hasattr(reader, "get_length"):
            try:
                candidate = float(reader.get_length() or 0)
            except Exception:
                candidate = 0.0
            if candidate > 0.0 and not math.isinf(candidate):
                length = int(candidate)
        return length

    def _read_generated_video(self, episode: TotoEpisode) -> torch.Tensor:
        cached = self._human_video_cache.get(episode.human_video_path)
        if cached is not None:
            return cached
        reader = self._get_reader(episode.human_video_path)
        total_frames = self._get_video_length(reader)
        if total_frames <= 0:
            raise ValueError(f"Unable to determine frame count for {episode.human_video_path}")
        indices = np.linspace(0, total_frames - 1, self.human_video_len)
        frames = []
        for idx in indices:
            frame = reader.get_data(int(round(float(idx))))
            frame_tensor = self._ensure_image_tensor(frame)
            self._warn_non_finite("human_video_frame", frame_tensor, episode.episode_id, int(round(float(idx))))
            frames.append(frame_tensor)
        video = torch.stack(frames, dim=0)
        self._human_video_cache[episode.human_video_path] = video
        return video

    def _read_robot_history(self, episode: TotoEpisode, start_index: int) -> torch.Tensor:
        reader = self._get_reader(episode.robot_video_path)
        frames = []
        for offset in range(self.robot_history_len):
            frame_idx = start_index + offset
            frame = reader.get_data(int(frame_idx))
            frame_tensor = self._ensure_image_tensor(frame)
            self._warn_non_finite("robot_history_frame", frame_tensor, episode.episode_id, frame_idx)
            frames.append(frame_tensor)
        return torch.stack(frames, dim=0)

    def _read_targets(self, episode: TotoEpisode, target_step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = np.concatenate(
            [episode.world_vector[target_step], episode.rotation_delta[target_step]],
            axis=0,
        )
        if action.shape[0] < self.num_action_dims:
            raise ValueError(
                f"TOTO action dim {action.shape[0]} < num_action_dims {self.num_action_dims} in {episode.data_path}"
            )
        action_target = torch.as_tensor(action[: self.num_action_dims]).float()
        gripper_value = float(episode.open_gripper[target_step])
        gripper_target = torch.tensor(int(gripper_value > self.gripper_threshold), dtype=torch.long)
        terminate_value = float(episode.terminate[target_step])
        terminate_target = torch.tensor(int(terminate_value > self.terminate_threshold), dtype=torch.long)
        self._warn_non_finite("action_target", action_target, episode.episode_id, target_step)
        self._warn_non_finite("open_gripper", torch.tensor(gripper_value), episode.episode_id, target_step)
        self._warn_non_finite("terminate", torch.tensor(terminate_value), episode.episode_id, target_step)
        return action_target, terminate_target, gripper_target

    def sample_window(self, episode_idx: int, start_index: int) -> Dict[str, torch.Tensor | str | int]:
        episode = self._episodes[episode_idx]
        human_video = self._read_generated_video(episode)
        robot_history = self._read_robot_history(episode, start_index)
        target_step = start_index + self.robot_history_len - 1
        action_target, terminate_target, gripper_target = self._read_targets(episode, target_step)
        return {
            "demo_name": episode.episode_id,
            "start_index": int(start_index),
            "human_video": human_video,
            "robot_history": robot_history,
            "action_target": action_target,
            "terminate_target": terminate_target,
            "gripper_target": gripper_target,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        episode_idx, start_index = self._samples[index]
        return self.sample_window(episode_idx, start_index)
