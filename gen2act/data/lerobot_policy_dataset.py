"""LeRobot v3 dataset adapter for Gen2Act policy training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LeRobotPolicyDataset(Dataset):
    """Load fixed-length policy windows from a LeRobot v3 dataset."""

    def __init__(
        self,
        repo_id_or_path: str | Path,
        image_key: str = "observation.images.table_cam",
        action_key: str = "action",
        # done_key: str = "next.done",
        episode_index_key: str = "episode_index",
        frame_index_key: str = "frame_index",
        human_video_len: int = 16,
        robot_history_len: int = 8,
        image_size: int = 224,
        num_action_dims: int = 6,
        action_stride: int = 1,
        max_samples: Optional[int] = None,
        max_episodes: Optional[int] = None,
        gripper_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.repo_id_or_path = str(repo_id_or_path)
        self.image_key = image_key
        self.action_key = action_key
        # self.done_key = done_key
        self.episode_index_key = episode_index_key
        self.frame_index_key = frame_index_key
        self.human_video_len = human_video_len
        self.robot_history_len = robot_history_len
        self.image_size = image_size
        self.num_action_dims = num_action_dims
        self.action_stride = action_stride
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.gripper_threshold = gripper_threshold

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(self.repo_id_or_path)
        self.hf_dataset = self._resolve_hf_dataset(self.dataset)
        self._episode_to_frames = self._group_frames_by_episode()
        self._samples = self._index_samples()

    @staticmethod
    def _resolve_hf_dataset(lerobot_dataset):
        for attr in ("dataset", "hf_dataset", "_hf_dataset", "_dataset"):
            value = getattr(lerobot_dataset, attr, None)
            if value is not None:
                return value
        raise AttributeError("LeRobotDataset does not expose a Hugging Face dataset attribute.")

    def _group_frames_by_episode(self) -> Dict[int, np.ndarray]:
        episode_indices = np.asarray(self.hf_dataset[self.episode_index_key])
        frame_indices = np.asarray(self.hf_dataset[self.frame_index_key])
        episode_ids = np.unique(episode_indices)
        if self.max_episodes is not None:
            episode_ids = episode_ids[: self.max_episodes]
        episode_to_frames: Dict[int, np.ndarray] = {}
        for episode_id in episode_ids:
            mask = episode_indices == episode_id
            frame_ids = np.nonzero(mask)[0]
            order = np.argsort(frame_indices[frame_ids])
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

    def _ensure_image_tensor(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(image)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            pass
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape for {self.image_key}: {tuple(tensor.shape)}")
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

    def _read_video(self, frame_ids: np.ndarray, length: int) -> torch.Tensor:
        frames = []
        for frame_id in frame_ids[:length]:
            sample = self.dataset[int(frame_id)]
            if self.image_key not in sample:
                raise KeyError(f"Missing {self.image_key} in LeRobot sample keys: {list(sample.keys())}")
            frames.append(self._ensure_image_tensor(sample[self.image_key]))
        return torch.stack(frames, dim=0)

    def _read_targets(self, frame_id: int, is_terminal_frame: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[int(frame_id)]
        if self.action_key not in sample:
            raise KeyError(f"Missing {self.action_key} in LeRobot sample keys: {list(sample.keys())}")
        action = torch.as_tensor(sample[self.action_key]).float()
        action_target = action[: self.num_action_dims]
        if action.numel() > self.num_action_dims:
            gripper_value = action[self.num_action_dims]
        else:
            gripper_value = action[-1]
        gripper_target = torch.tensor(int(gripper_value.item() > self.gripper_threshold), dtype=torch.long)
        # if self.done_key in sample:
        #     terminate_target = torch.tensor(int(bool(sample[self.done_key])), dtype=torch.long)
        # else:
        #     terminate_target = torch.tensor(int(is_terminal_frame), dtype=torch.long)
        return action_target, gripper_target

    def sample_window(self, episode_id: int, start_index: int) -> Dict[str, torch.Tensor | int]:
        frame_ids = self._episode_to_frames[episode_id]
        window_frame_ids = frame_ids[start_index : start_index + max(self.human_video_len, self.robot_history_len)]
        human_video = self._read_video(window_frame_ids, self.human_video_len)
        robot_history = self._read_video(window_frame_ids, self.robot_history_len)

        target_offset = self.robot_history_len - 1
        target_frame_id = int(window_frame_ids[target_offset])
        is_terminal_frame = start_index + target_offset == len(frame_ids) - 1
        action_target, gripper_target = self._read_targets(target_frame_id, is_terminal_frame)
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
