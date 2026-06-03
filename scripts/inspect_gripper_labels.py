from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 is not expected here
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen2act.data import LeRobotPolicyDataset, LeRobotVideoPolicyDataset, PolicyDemoDataset


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def _iter_indices(num_items: int, num_samples: int, shuffle: bool, seed: int) -> Iterable[int]:
    indices = list(range(num_items))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(num_samples, num_items)]


def _extract_lerobot_video_action(dataset, episode_id: int, start_index: int) -> np.ndarray:
    frame_ids = dataset._episode_to_frames[episode_id]
    target_offset = dataset.robot_history_len - 1
    target_row_idx = int(frame_ids[start_index + target_offset])
    return np.asarray(dataset._actions[target_row_idx])


def _extract_lerobot_action(dataset, episode_id: int, start_index: int) -> np.ndarray:
    frame_ids = dataset._episode_to_frames[episode_id]
    target_offset = dataset.robot_history_len - 1
    target_frame_id = int(frame_ids[start_index + target_offset])
    sample = dataset.dataset[int(target_frame_id)]
    return np.asarray(sample[dataset.action_key])


def _extract_hdf5_action(dataset, demo_name: str, start_index: int) -> np.ndarray:
    file_handle = dataset._open()
    actions = np.asarray(file_handle[f"data/{demo_name}/actions"])
    target_step = start_index + dataset.robot_history_len - 1
    return np.asarray(actions[target_step])


def _gripper_value(action: Sequence[float], num_action_dims: int) -> float:
    if len(action) > num_action_dims:
        return float(action[num_action_dims])
    return float(action[-1])


def _summarize(values: List[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float32)
    return (
        f"min={arr.min():.4f} max={arr.max():.4f} "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect gripper label distribution and raw action values.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gen2act_policy.toml")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset-type", type=str, default=None, choices=["hdf5", "lerobot"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]

    dataset_type = (args.dataset_type or data_cfg.get("dataset_type", "hdf5")).lower()
    raw_dataset_path = str(args.dataset_path or data_cfg["dataset_path"])

    if dataset_type == "lerobot":
        path_candidate = Path(raw_dataset_path)
        dataset_path = path_candidate.resolve() if path_candidate.exists() else raw_dataset_path
        use_video = bool(data_cfg.get("lerobot_use_videos", False))
        if use_video:
            dataset = LeRobotVideoPolicyDataset(
                repo_root=str(dataset_path),
                video_key=str(data_cfg.get("lerobot_video_key", "observation.images.table_cam")),
                action_key=str(data_cfg.get("lerobot_action_key", "action")),
                episode_index_key=str(data_cfg.get("lerobot_episode_key", "episode_index")),
                frame_index_key=str(data_cfg.get("lerobot_frame_key", "frame_index")),
                timestamp_key=str(data_cfg.get("lerobot_timestamp_key", "timestamp")),
                human_video_len=int(data_cfg["human_video_len"]),
                robot_history_len=int(data_cfg["robot_history_len"]),
                image_size=int(data_cfg["image_size"]),
                num_action_dims=int(model_cfg["num_action_dims"]),
                action_stride=int(data_cfg["action_stride"]),
                max_samples=None,
                gripper_threshold=float(data_cfg["gripper_threshold"]),
            )
            extractor = _extract_lerobot_video_action
        else:
            dataset = LeRobotPolicyDataset(
                repo_id_or_path=str(dataset_path),
                image_key=str(data_cfg.get("lerobot_image_key", "observation.images.top")),
                action_key=str(data_cfg.get("lerobot_action_key", "action")),
                episode_index_key=str(data_cfg.get("lerobot_episode_key", "episode_index")),
                frame_index_key=str(data_cfg.get("lerobot_frame_key", "frame_index")),
                human_video_len=int(data_cfg["human_video_len"]),
                robot_history_len=int(data_cfg["robot_history_len"]),
                image_size=int(data_cfg["image_size"]),
                num_action_dims=int(model_cfg["num_action_dims"]),
                action_stride=int(data_cfg["action_stride"]),
                max_samples=None,
                gripper_threshold=float(data_cfg["gripper_threshold"]),
            )
            extractor = _extract_lerobot_action
    else:
        dataset_path = _resolve_path(REPO_ROOT, raw_dataset_path)
        dataset = PolicyDemoDataset(
            hdf5_path=dataset_path,
            human_camera=str(data_cfg["human_camera"]),
            robot_camera=str(data_cfg["robot_camera"]),
            human_video_len=int(data_cfg["human_video_len"]),
            robot_history_len=int(data_cfg["robot_history_len"]),
            image_size=int(data_cfg["image_size"]),
            num_action_dims=int(model_cfg["num_action_dims"]),
            action_stride=int(data_cfg["action_stride"]),
            max_samples=None,
            gripper_threshold=float(data_cfg["gripper_threshold"]),
        )
        extractor = _extract_hdf5_action

    num_action_dims = int(model_cfg["num_action_dims"])
    threshold = float(data_cfg["gripper_threshold"])

    positive = 0
    negative = 0
    raw_values: List[float] = []
    printed = 0

    for idx in _iter_indices(len(dataset), args.num_samples, args.shuffle, args.seed):
        sample_id = dataset._samples[idx]
        if dataset_type == "lerobot":
            episode_id, start_index = sample_id
            action = extractor(dataset, int(episode_id), int(start_index))
        else:
            demo_name, start_index = sample_id
            action = extractor(dataset, str(demo_name), int(start_index))

        grip_value = _gripper_value(action, num_action_dims)
        grip_label = int(grip_value > threshold)
        raw_values.append(grip_value)
        if grip_label == 1:
            positive += 1
        else:
            negative += 1

        if printed < args.print_samples:
            action_list = np.asarray(action, dtype=np.float32).tolist()
            print(
                f"sample={idx} action_len={len(action_list)} "
                f"gripper_value={grip_value:.4f} gripper_label={grip_label} "
                f"action={action_list}"
            )
            printed += 1

    total = max(1, positive + negative)
    print("\n--- summary ---")
    print(f"dataset_type={dataset_type}")
    print(f"dataset_path={dataset_path}")
    print(f"num_action_dims={num_action_dims} gripper_threshold={threshold}")
    print(f"samples={positive + negative} positive={positive} negative={negative}")
    print(f"positive_frac={positive / total:.4f} negative_frac={negative / total:.4f}")
    print(f"gripper_value_stats={_summarize(raw_values)}")


if __name__ == "__main__":
    main()
