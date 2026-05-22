from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

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


def load_config(config_path: Path) -> Dict[str, Dict[str, object]]:
    with config_path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def discretize_actions(actions: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    clipped = actions.clamp(low, high)
    scale = max(high - low, 1e-6)
    normalized = (clipped - low) / scale
    bins = torch.round(normalized * (num_bins - 1)).long()
    return bins.clamp_(0, num_bins - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Gen2Act policy on Isaac Lab HDF5 demos.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gen2act_policy.toml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir or _resolve_path(REPO_ROOT, str(train_cfg["output_dir"])))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = _resolve_path(REPO_ROOT, str(args.dataset_path or data_cfg["dataset_path"]))
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

    val_size = max(1, int(len(dataset) * float(train_cfg["val_split"])))
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(train_cfg["seed"])),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    def run_epoch(loader: DataLoader, train: bool) -> Dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_action = 0.0
        total_terminate = 0.0
        total_gripper = 0.0
        steps = 0

        for batch in loader:
            human_video = batch["human_video"].to(device)
            robot_history = batch["robot_history"].to(device)
            action_target = batch["action_target"].to(device)
            terminate_target = batch["terminate_target"].to(device)
            gripper_target = batch["gripper_target"].to(device)

            action_bins = discretize_actions(action_target, int(model_cfg["num_bins"]))

            with torch.set_grad_enabled(train):
                outputs = model(
                    scene_img=None,
                    task_prompt_tokens=None,
                    human_video=human_video,
                    robot_history=robot_history,
                )

                action_logits = outputs["action_logits"]
                terminate_logits = outputs["terminate_logits"]
                gripper_logits = outputs["gripper_logits"]

                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, action_logits.shape[-1]),
                    action_bins.reshape(-1),
                )
                terminate_loss = F.cross_entropy(terminate_logits, terminate_target)
                gripper_loss = F.cross_entropy(gripper_logits, gripper_target)
                loss = action_loss + terminate_loss + gripper_loss

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())
            total_action += float(action_loss.item())
            total_terminate += float(terminate_loss.item())
            total_gripper += float(gripper_loss.item())
            steps += 1

        denom = max(1, steps)
        return {
            "loss": total_loss / denom,
            "action_loss": total_action / denom,
            "terminate_loss": total_terminate / denom,
            "gripper_loss": total_gripper / denom,
        }

    epochs = int(args.epochs or train_cfg["epochs"])
    best_val = float("inf")
    config_snapshot = {
        "config": config,
        "resolved_dataset_path": str(dataset_path),
        "resolved_output_dir": str(output_dir),
    }
    (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_action={train_metrics['action_loss']:.4f} val_action={val_metrics['action_loss']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "latest.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

        if epoch % int(train_cfg["save_every"]) == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()