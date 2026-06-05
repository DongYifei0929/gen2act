"""HDF5 dataset adapter for Gen2Act policy training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PolicyDemoDataset(Dataset):
    """Load fixed-length policy windows from Isaac Lab HDF5 demonstrations."""

    def __init__(
        self,
        hdf5_path: str | Path,
        human_camera: str = "table_cam",
        robot_camera: str = "wrist_cam",
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
        self.hdf5_path = Path(hdf5_path)
        self.human_camera = human_camera
        self.robot_camera = robot_camera
        self.human_video_len = human_video_len
        self.robot_history_len = robot_history_len
        self.image_size = image_size
        self.num_action_dims = num_action_dims
        self.action_stride = action_stride
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.gripper_threshold = gripper_threshold
        self._file: Optional[h5py.File] = None
        self._samples = self._index_samples()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def _index_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        window_len = max(self.human_video_len, self.robot_history_len)

        with h5py.File(self.hdf5_path, "r") as file_handle:
            demo_names = sorted(file_handle["data"].keys())
            if self.max_episodes is not None:
                demo_names = demo_names[: self.max_episodes]
            for demo_name in demo_names:
                num_steps = file_handle[f"data/{demo_name}/actions"].shape[0]
                if num_steps < window_len:
                    continue

                last_start = num_steps - window_len
                for start_index in range(0, last_start + 1, self.action_stride):
                    samples.append((demo_name, start_index))

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _read_video(self, demo_name: str, camera_key: str, start: int, length: int) -> torch.Tensor:
        file_handle = self._open()
        video = np.asarray(file_handle[f"data/{demo_name}/obs/{camera_key}"][start : start + length])
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            h, w = tensor.shape[-2], tensor.shape[-1]
            if h < w:
                new_h = self.image_size
                new_w = int(round(w * self.image_size / h))
            else:
                new_w = self.image_size
                new_h = int(round(h * self.image_size / w))
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
            if new_h > self.image_size:
                start_h = (new_h - self.image_size) // 2
                tensor = tensor[:, :, start_h:start_h + self.image_size, :]
            if new_w > self.image_size:
                start_w = (new_w - self.image_size) // 2
                tensor = tensor[:, :, :, start_w:start_w + self.image_size]
        return tensor

    def _read_actions(self, demo_name: str, start: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_handle = self._open()
        actions = np.asarray(file_handle[f"data/{demo_name}/actions"])
        action_tensor = torch.from_numpy(actions).float()
        target_step = start + self.robot_history_len - 1
        action_target = action_tensor[target_step, : self.num_action_dims]
        gripper_target = torch.tensor(
            int(action_tensor[target_step, -1].item() > self.gripper_threshold),
            dtype=torch.long,
        )
        terminate_target = torch.tensor(int(target_step == action_tensor.shape[0] - 1), dtype=torch.long)
        return action_target, terminate_target, gripper_target

    def sample_window(self, demo_name: str, start_index: int) -> Dict[str, torch.Tensor | str | int]:
        human_video = self._read_video(demo_name, self.human_camera, start_index, self.human_video_len)
        robot_history = self._read_video(demo_name, self.robot_camera, start_index, self.robot_history_len)
        action_target, terminate_target, gripper_target = self._read_actions(demo_name, start_index)
        return {
            "demo_name": demo_name,
            "start_index": start_index,
            "human_video": human_video,
            "robot_history": robot_history,
            "action_target": action_target,
            "terminate_target": terminate_target,
            "gripper_target": gripper_target,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        demo_name, start_index = self._samples[index]
        return self.sample_window(demo_name, start_index)