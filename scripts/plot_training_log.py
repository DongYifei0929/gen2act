#!/usr/bin/env python3

"""Parse Gen2Act training logs and plot metrics over step and epoch."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"\[train\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<total>\d+)"
    r"\s+loss=(?P<loss>[0-9.]+)\s+action=(?P<action>[0-9.]+)"
    r"\s+gripper=(?P<gripper>[0-9.]+)\s+track=(?P<track>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        nargs="?",
        default="logs/0602_run3.log",
        help="Path to the training log file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/0602_run3_plots",
        help="Directory to write plots and the parsed CSV.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window for step plots.",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def parse_log(log_path: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRAIN_RE.search(line)
            if match is None:
                continue

            epoch = int(match.group("epoch"))
            step = int(match.group("step"))
            total = int(match.group("total"))
            loss = float(match.group("loss"))
            action = float(match.group("action"))
            gripper = float(match.group("gripper"))
            track = float(match.group("track"))

            records.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "total": total,
                    "global_step": (epoch - 1) * total + step,
                    "loss": loss,
                    "action": action,
                    "gripper": gripper,
                    "track": track,
                }
            )

    return records


def write_csv(records: list[dict[str, float | int]], output_path: Path) -> None:
    fieldnames = ["epoch", "step", "total", "global_step", "loss", "action", "gripper", "track"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_step_curves(records: list[dict[str, float | int]], output_path: Path, smooth_window: int) -> None:
    metrics = ["loss", "action", "gripper", "track"]
    step_values = [int(record["global_step"]) for record in records]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes_flat = axes.flatten()

    for axis, metric in zip(axes_flat, metrics, strict=True):
        raw_values = [float(record[metric]) for record in records]
        values = moving_average(raw_values, smooth_window)
        axis.plot(step_values, values, linewidth=1.5)
        axis.set_title(f"{metric} vs global step")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("global step")

    fig.suptitle("Training metrics over global step", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_epoch_curves(records: list[dict[str, float | int]], output_path: Path) -> None:
    metrics = ["loss", "action", "gripper", "track"]
    by_epoch: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        epoch = int(record["epoch"])
        for metric in metrics:
            by_epoch[epoch][metric].append(float(record[metric]))

    epochs = sorted(by_epoch)
    epoch_means: dict[str, list[float]] = {
        metric: [sum(by_epoch[epoch][metric]) / len(by_epoch[epoch][metric]) for epoch in epochs]
        for metric in metrics
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes_flat = axes.flatten()

    for axis, metric in zip(axes_flat, metrics, strict=True):
        axis.plot(epochs, epoch_means[metric], marker="o", linewidth=1.8)
        axis.set_title(f"{metric} vs epoch mean")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("epoch")

    fig.suptitle("Training metrics aggregated by epoch", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(log_path)
    if not records:
        raise SystemExit(f"No training records found in {log_path}")

    csv_path = output_dir / "parsed_training_log.csv"
    step_plot_path = output_dir / "metrics_over_step.png"
    epoch_plot_path = output_dir / "metrics_over_epoch.png"

    write_csv(records, csv_path)
    plot_step_curves(records, step_plot_path, args.smooth_window)
    plot_epoch_curves(records, epoch_plot_path)

    print(f"Parsed records: {len(records)}")
    print(f"CSV: {csv_path}")
    print(f"Step plot: {step_plot_path}")
    print(f"Epoch plot: {epoch_plot_path}")


if __name__ == "__main__":
    main()