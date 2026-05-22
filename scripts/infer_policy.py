from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 is not expected here
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen2act.data import PolicyDemoDataset
from gen2act.modeling import build_default_policy


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_config(config_path: Path):
    with config_path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def bin_to_value(bin_index: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    scale = max(high - low, 1e-6)
    return low + bin_index.float() * scale / max(num_bins - 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gen2Act inference on a demo window.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gen2act_policy.toml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--demo-index", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]
    infer_cfg = config["infer"]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_path = _resolve_path(REPO_ROOT, str(args.dataset_path or data_cfg["dataset_path"]))
    checkpoint_path = _resolve_path(REPO_ROOT, str(args.checkpoint or infer_cfg["checkpoint_path"]))
    save_path = Path(args.save_path or _resolve_path(REPO_ROOT, str(infer_cfg["save_path"])))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = PolicyDemoDataset(
        hdf5_path=dataset_path,
        human_camera=str(data_cfg["human_camera"]),
        robot_camera=str(data_cfg["robot_camera"]),
        human_video_len=int(data_cfg["human_video_len"]),
        robot_history_len=int(data_cfg["robot_history_len"]),
        image_size=int(data_cfg["image_size"]),
        num_action_dims=int(model_cfg["num_action_dims"]),
        action_stride=int(data_cfg["action_stride"]),
        max_samples=None if int(data_cfg["max_samples"]) <= 0 else int(data_cfg["max_samples"]),
        gripper_threshold=float(data_cfg["gripper_threshold"]),
    )

    if args.demo_index is None:
        sample = dataset[int(infer_cfg["demo_index"])]
    else:
        demo_name, _ = dataset._samples[int(args.demo_index)]
        sample = dataset.sample_window(demo_name, int(args.start_index or infer_cfg["start_index"]))

    model = build_default_policy(
        num_action_dims=int(model_cfg["num_action_dims"]),
        num_bins=int(model_cfg["num_bins"]),
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_vit_layers=int(model_cfg["num_vit_layers"]),
        num_vit_heads=int(model_cfg["num_vit_heads"]),
        latent_tokens=int(model_cfg["latent_tokens"]),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(
            scene_img=None,
            task_prompt_tokens=None,
            human_video=sample["human_video"].unsqueeze(0).to(device),
            robot_history=sample["robot_history"].unsqueeze(0).to(device),
        )

    action_logits = outputs["action_logits"][0].cpu()
    terminate_logits = outputs["terminate_logits"][0].cpu()
    gripper_logits = outputs["gripper_logits"][0].cpu()

    action_bins = action_logits.argmax(dim=-1)
    action_values = bin_to_value(action_bins, int(model_cfg["num_bins"]))

    result = {
        "demo_name": sample["demo_name"],
        "start_index": sample["start_index"],
        "action_bins": action_bins.tolist(),
        "action_values": action_values.tolist(),
        "terminate_prob": terminate_logits.softmax(dim=-1).tolist(),
        "gripper_prob": gripper_logits.softmax(dim=-1).tolist(),
        "checkpoint": str(checkpoint_path),
    }

    np.savez_compressed(
        save_path,
        action_bins=np.asarray(result["action_bins"]),
        action_values=np.asarray(result["action_values"]),
        terminate_prob=np.asarray(result["terminate_prob"]),
        gripper_prob=np.asarray(result["gripper_prob"]),
        demo_name=np.asarray(result["demo_name"]),
        start_index=np.asarray(result["start_index"]),
    )
    save_path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()