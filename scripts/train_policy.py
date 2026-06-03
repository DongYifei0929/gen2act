from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict
# sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 is not expected here
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen2act.data import LeRobotPolicyDataset, LeRobotVideoPolicyDataset, PolicyDemoDataset, TotoGenPolicyDataset
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


def _l2_norm(tensors) -> float:
    total_sq = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        total_sq += float(torch.sum(tensor.float() ** 2).item())
    return total_sq ** 0.5


def _tensor_l2_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    total_sq = None
    for tensor in tensors:
        if tensor is None:
            continue
        contrib = torch.sum(tensor.float() ** 2)
        total_sq = contrib if total_sq is None else total_sq + contrib
    if total_sq is None:
        return torch.tensor(0.0)
    return torch.sqrt(total_sq)


def _module_stats(module: torch.nn.Module) -> Dict[str, float | bool]:
    params = list(module.parameters())
    weights = [p.data for p in params if p is not None]
    grads = [p.grad for p in params if p is not None and p.grad is not None]
    weight_finite = all(torch.isfinite(w).all().item() for w in weights) if weights else True
    grad_finite = all(torch.isfinite(g).all().item() for g in grads) if grads else True
    return {
        "weight_norm": _l2_norm(weights),
        "grad_norm": _l2_norm(grads),
        "weight_finite": weight_finite,
        "grad_finite": grad_finite,
    }


def _monitored_modules(model) -> Dict[str, torch.nn.Module]:
    return {
        "vit": model.vit,
        "human_resampler": model.human_resampler,
        "robot_resampler": model.robot_resampler,
        "fusion": model.fusion,
        "action_head": model.action_head,
        "track_predictor": model.track_predictor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Gen2Act policy on Isaac Lab HDF5 demos.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gen2act_policy.toml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset-type", type=str, default=None, choices=["hdf5", "lerobot", "toto_gen"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    track_cfg = config.get("track", {})
    wandb_cfg = config.get("wandb", {})

    wandb_run = None
    wandb_enabled = bool(wandb_cfg.get("enabled", False))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir or _resolve_path(REPO_ROOT, str(train_cfg["output_dir"])))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = (args.dataset_type or data_cfg.get("dataset_type", "hdf5")).lower()
    raw_dataset_path = str(args.dataset_path or data_cfg["dataset_path"])
    if dataset_type == "lerobot":
        path_candidate = Path(raw_dataset_path)
        dataset_path = path_candidate.resolve() if path_candidate.exists() else raw_dataset_path
    else:
        dataset_path = _resolve_path(REPO_ROOT, raw_dataset_path)
    max_samples = None if int(data_cfg["max_samples"]) <= 0 else int(data_cfg["max_samples"])
    max_episodes = None
    if "max_episodes" in data_cfg:
        raw_max_episodes = int(data_cfg["max_episodes"])
        max_episodes = None if raw_max_episodes <= 0 else raw_max_episodes
    if dataset_type == "lerobot":
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
                max_samples=max_samples,
                max_episodes=max_episodes,
                gripper_threshold=float(data_cfg["gripper_threshold"]),
            )
        else:
            dataset = LeRobotPolicyDataset(
                repo_id_or_path=str(dataset_path),
                image_key=str(data_cfg.get("lerobot_image_key", "observation.images.top")),
                action_key=str(data_cfg.get("lerobot_action_key", "action")),
                done_key=str(data_cfg.get("lerobot_done_key", "next.done")),
                episode_index_key=str(data_cfg.get("lerobot_episode_key", "episode_index")),
                frame_index_key=str(data_cfg.get("lerobot_frame_key", "frame_index")),
                human_video_len=int(data_cfg["human_video_len"]),
                robot_history_len=int(data_cfg["robot_history_len"]),
                image_size=int(data_cfg["image_size"]),
                num_action_dims=int(model_cfg["num_action_dims"]),
                action_stride=int(data_cfg["action_stride"]),
                max_samples=max_samples,
                max_episodes=max_episodes,
                gripper_threshold=float(data_cfg["gripper_threshold"]),
            )
    elif dataset_type == "toto_gen":
        dataset = TotoGenPolicyDataset(
            dataset_root=dataset_path,
            generated_subdir=str(data_cfg.get("toto_generated_subdir", "toto-gen")),
            robot_subdir=str(data_cfg.get("toto_robot_subdir", "toto")),
            generated_video_name=str(data_cfg.get("toto_generated_video_name", "generated.mp4")),
            robot_video_name=str(data_cfg.get("toto_robot_video_name", "image.mp4")),
            metadata_name=str(data_cfg.get("toto_metadata_name", "data.json")),
            human_video_len=int(data_cfg["human_video_len"]),
            robot_history_len=int(data_cfg["robot_history_len"]),
            image_size=int(data_cfg["image_size"]),
            num_action_dims=int(model_cfg["num_action_dims"]),
            action_stride=int(data_cfg["action_stride"]),
            max_samples=max_samples,
            max_episodes=max_episodes,
            gripper_threshold=float(data_cfg["gripper_threshold"]),
            terminate_threshold=float(data_cfg.get("terminate_threshold", 0.5)),
        )
    else:
        dataset = PolicyDemoDataset(
            hdf5_path=dataset_path,
            human_camera=str(data_cfg["human_camera"]),
            robot_camera=str(data_cfg["robot_camera"]),
            human_video_len=int(data_cfg["human_video_len"]),
            robot_history_len=int(data_cfg["robot_history_len"]),
            image_size=int(data_cfg["image_size"]),
            num_action_dims=int(model_cfg["num_action_dims"]),
            action_stride=int(data_cfg["action_stride"]),
            max_samples=max_samples,
            max_episodes=max_episodes,
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
        shuffle=False,  # 先定位哪个数据集有问题
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

    track_checkpoint = track_cfg.get("checkpoint") or None

    model = build_default_policy(
        num_action_dims=int(model_cfg["num_action_dims"]),
        num_bins=int(model_cfg["num_bins"]),
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_vit_layers=int(model_cfg["num_vit_layers"]),
        num_vit_heads=int(model_cfg["num_vit_heads"]),
        latent_tokens=int(model_cfg["latent_tokens"]),
        vit_pretrained=model_cfg.get("pretrained"),
        enable_point_tracking=bool(track_cfg.get("enable", False)),
        point_tracker_checkpoint=track_checkpoint,
        point_tracker_use_hub=bool(track_cfg.get("use_hub", False)),
        point_tracker_offline=bool(track_cfg.get("offline", True)),
        point_tracker_v2=bool(track_cfg.get("use_v2_model", False)),
        point_tracker_window_len=int(track_cfg.get("window_len", 60)),
        track_grid_size=int(track_cfg.get("grid_size", 10)),
        track_query_frame=int(track_cfg.get("grid_query_frame", 0)),
        track_backward=bool(track_cfg.get("backward_tracking", False)),
    ).to(device)

    gradnorm_enabled = bool(train_cfg.get("gradnorm", False))
    gradnorm_alpha = float(train_cfg.get("gradnorm_alpha", 1.5))
    gradnorm_weights = None
    gradnorm_state = {"init_losses": None}
    if gradnorm_enabled:
        gradnorm_weights = torch.nn.Parameter(torch.ones(3, device=device))

    optimizer_params = [
        {
            "params": model.parameters(),
            "lr": float(train_cfg["lr"]),
            "weight_decay": float(train_cfg["weight_decay"]),
        }
    ]
    if gradnorm_enabled and gradnorm_weights is not None:
        optimizer_params.append(
            {
                "params": [gradnorm_weights],
                "lr": float(train_cfg.get("gradnorm_lr", train_cfg["lr"])),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_params)

    if wandb_enabled:
        try:
            import wandb
        except ModuleNotFoundError:
            print("[warn] wandb is enabled but not installed; disable wandb or install it.")
            wandb_enabled = False
        else:
            wandb_run = wandb.init(
                project=str(wandb_cfg.get("project", "gen2act")),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("run_name"),
                tags=list(wandb_cfg.get("tags", [])),
                dir=str(output_dir),
                config={
                    "model": dict(model_cfg),
                    "data": dict(data_cfg),
                    "train": dict(train_cfg),
                    "resolved_dataset_path": str(dataset_path),
                    "dataset_type": dataset_type,
                    "output_dir": str(output_dir),
                },
            )

    def run_epoch(
        loader: DataLoader,
        train: bool,
        epoch: int,
        global_step_offset: int,
    ) -> Dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_action = 0.0
        # total_terminate = 0.0
        total_gripper = 0.0
        total_track = 0.0
        steps = 0
        total_steps = len(loader)
        log_every = max(1, int(train_cfg.get("log_every", 50)))
        debug_isfinite = bool(train_cfg.get("debug_isfinite", False))
        debug_norms = bool(train_cfg.get("debug_norms", False))
        debug_norm_every = max(1, int(train_cfg.get("debug_norm_every", log_every)))
        start_time = time.time()

        for batch in loader:
            human_video = batch["human_video"].to(device)
            robot_history = batch["robot_history"].to(device)
            action_target = batch["action_target"].to(device)
            # terminate_target = batch["terminate_target"].to(device)
            gripper_target = batch["gripper_target"].to(device)
            human_tracks = batch.get("human_tracks")
            robot_tracks = batch.get("robot_tracks")
            human_track_vis = None
            robot_track_vis = None
            if human_tracks is not None:
                human_tracks = human_tracks.to(device)
            if robot_tracks is not None:
                robot_tracks = robot_tracks.to(device)

            # ================= check for bad data =================
            print(f"here start index is {batch['start_index']}")
            if not torch.isfinite(human_video).all():
                print("bad human_video", batch["demo_name"], batch["start_index"])
            if not torch.isfinite(robot_history).all():
                print("bad robot_history", batch["demo_name"], batch["start_index"])
            if not torch.isfinite(action_target).all():
                print("bad action_target", batch["demo_name"], batch["start_index"])

            action_bins = discretize_actions(action_target, int(model_cfg["num_bins"]))

            with torch.set_grad_enabled(train):
                outputs = model(
                    scene_img=None,
                    task_prompt_tokens=None,
                    human_video=human_video,
                    robot_history=robot_history,
                    gt_human_tracks=human_tracks,
                    gt_robot_tracks=robot_tracks,
                    debug_isfinite=debug_isfinite,
                )

                action_logits = outputs["action_logits"]
                # terminate_logits = outputs["terminate_logits"]
                gripper_logits = outputs["gripper_logits"]

                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, action_logits.shape[-1]),
                    action_bins.reshape(-1),
                )
                # terminate_loss = F.cross_entropy(terminate_logits, terminate_target)
                gripper_loss = F.cross_entropy(gripper_logits, gripper_target)

                track_loss = torch.tensor(0.0, device=device)
                if human_tracks is None and "human_tracks" in outputs:
                    human_tracks = outputs["human_tracks"]
                    human_track_vis = outputs.get("human_track_vis")
                if robot_tracks is None and "robot_tracks" in outputs:
                    robot_tracks = outputs["robot_tracks"]
                    robot_track_vis = outputs.get("robot_track_vis")

                if (
                    human_tracks is not None
                    and robot_tracks is not None
                    and "human_track_pred" in outputs
                    and "robot_track_pred" in outputs
                ):
                    if human_track_vis is not None:
                        human_mask = human_track_vis.unsqueeze(-1).float()
                        human_track_loss = (
                            (outputs["human_track_pred"] - human_tracks).pow(2) * human_mask
                        ).sum() / human_mask.sum().clamp(min=1.0)
                    else:
                        human_track_loss = F.mse_loss(outputs["human_track_pred"], human_tracks)
                    if robot_track_vis is not None:
                        robot_mask = robot_track_vis.unsqueeze(-1).float()
                        robot_track_loss = (
                            (outputs["robot_track_pred"] - robot_tracks).pow(2) * robot_mask
                        ).sum() / robot_mask.sum().clamp(min=1.0)
                    else:
                        robot_track_loss = F.mse_loss(outputs["robot_track_pred"], robot_tracks)
                    track_loss = human_track_loss + robot_track_loss

                track_weight = float(train_cfg.get("track_loss_weight", 0.001))
                task_losses = [action_loss, gripper_loss, track_loss]
                base_weights = torch.tensor([1.0, 1.0, track_weight], device=device)
                if gradnorm_enabled and gradnorm_weights is not None:
                    weights = F.softplus(gradnorm_weights) * base_weights
                    weighted_losses = [w * loss_i for w, loss_i in zip(weights, task_losses)]
                    loss = sum(weighted_losses)
                else:
                    loss = action_loss + gripper_loss + track_weight * track_loss

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    if gradnorm_enabled and gradnorm_weights is not None:
                        loss.backward(retain_graph=True)

                        shared_params = list(model.fusion.parameters())
                        grad_norms = []
                        for weighted_loss in weighted_losses:
                            grads = torch.autograd.grad(
                                weighted_loss,
                                shared_params,
                                retain_graph=True,
                                create_graph=True,
                            )
                            grad_norms.append(_tensor_l2_norm(list(grads)))

                        if gradnorm_state["init_losses"] is None:
                            gradnorm_state["init_losses"] = [loss_i.detach() for loss_i in task_losses]

                        init_losses = gradnorm_state["init_losses"]
                        loss_ratios = torch.stack(
                            [loss_i / init_loss for loss_i, init_loss in zip(task_losses, init_losses)]
                        )
                        inv_train_rate = loss_ratios / loss_ratios.mean().clamp(min=1e-6)
                        grad_norms_tensor = torch.stack(grad_norms)
                        target = grad_norms_tensor.mean().detach() * (inv_train_rate ** gradnorm_alpha)
                        gradnorm_loss = torch.sum(torch.abs(grad_norms_tensor - target))
                        grad_w = torch.autograd.grad(gradnorm_loss, gradnorm_weights)[0]
                        gradnorm_weights.grad = grad_w
                    else:
                        loss.backward()

                    if debug_norms:
                        should_log = steps % debug_norm_every == 0
                        for name, module in _monitored_modules(model).items():
                            stats = _module_stats(module)
                            if not stats["weight_finite"] or not stats["grad_finite"] or should_log:
                                print(
                                    f"[norm] {name} step={steps} "
                                    f"w_norm={stats['weight_norm']:.3e} "
                                    f"g_norm={stats['grad_norm']:.3e} "
                                    f"w_finite={stats['weight_finite']} g_finite={stats['grad_finite']}"
                                )

                    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            # ================= check for bad data =================
            if not torch.isfinite(action_logits).all() or not torch.isfinite(gripper_logits).all():
                print("bad logits", batch["start_index"])
            if not torch.isfinite(loss):
                print("bad loss", batch["start_index"])
                break

            total_loss += float(loss.item())
            total_action += float(action_loss.item())
            # total_terminate += float(terminate_loss.item())
            total_gripper += float(gripper_loss.item())
            total_track += float(track_loss.item())
            steps += 1

            if train and (steps % log_every == 0 or steps == total_steps):
                elapsed = max(1e-6, time.time() - start_time)
                steps_per_sec = steps / elapsed
                lr = optimizer.param_groups[0]["lr"]
                weight_log = None
                if gradnorm_enabled and gradnorm_weights is not None:
                    weight_values = (F.softplus(gradnorm_weights) * base_weights).detach().cpu().tolist()
                    weight_log = {"action": weight_values[0], "gripper": weight_values[1], "track": weight_values[2]}
                if wandb_enabled and wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": total_loss / steps,
                            "train/action": total_action / steps,
                            "train/gripper": total_gripper / steps,
                            "train/track": total_track / steps,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                                **({"train/gradnorm_action_w": weight_log["action"],
                                   "train/gradnorm_gripper_w": weight_log["gripper"],
                                   "train/gradnorm_track_w": weight_log["track"]} if weight_log else {}),
                            "epoch": epoch,
                        },
                        step=global_step_offset + steps,
                    )
                print(
                    f"[train] epoch={epoch} step={steps}/{total_steps} "
                    f"loss={total_loss / steps:.4f} action={total_action / steps:.4f} "
                    f"gripper={total_gripper / steps:.4f} track={total_track / steps:.4f} "
                    f"lr={lr:.3e} {steps_per_sec:.2f} steps/s"
                )

        denom = max(1, steps)
        return {
            "loss": total_loss / denom,
            "action_loss": total_action / denom,
            # "terminate_loss": total_terminate / denom,
            "gripper_loss": total_gripper / denom,
            "track_loss": total_track / denom,
        }

    epochs = int(args.epochs or train_cfg["epochs"])
    best_val = float("inf")
    config_snapshot = {
        "config": config,
        "resolved_dataset_path": str(dataset_path),
        "dataset_type": dataset_type,
        "resolved_output_dir": str(output_dir),
    }
    (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    print(
        f"dataset_type={dataset_type} train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)} batch_size={int(train_cfg['batch_size'])} "
        f"steps_per_epoch={len(train_loader)}"
    )

    for epoch in range(1, epochs + 1):
        global_step_offset = (epoch - 1) * len(train_loader)
        train_metrics = run_epoch(train_loader, train=True, epoch=epoch, global_step_offset=global_step_offset)
        val_metrics = run_epoch(val_loader, train=False, epoch=epoch, global_step_offset=global_step_offset)

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_action={train_metrics['action_loss']:.4f} val_action={val_metrics['action_loss']:.4f}"
        )

        if wandb_enabled and wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_action": train_metrics["action_loss"],
                    "train/epoch_gripper": train_metrics["gripper_loss"],
                    "train/epoch_track": train_metrics["track_loss"],
                    "val/epoch_loss": val_metrics["loss"],
                    "val/epoch_action": val_metrics["action_loss"],
                    "val/epoch_gripper": val_metrics["gripper_loss"],
                    "val/epoch_track": val_metrics["track_loss"],
                },
                step=global_step_offset + len(train_loader),
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

    if wandb_enabled and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()