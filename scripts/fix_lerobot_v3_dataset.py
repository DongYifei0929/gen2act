#!/usr/bin/env python3
"""Fix legacy LeRobot dataset layout to v3-compatible metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def fix_dataset(root: Path) -> None:
    root = root.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")

    info = _read_json(info_path)

    # Resolve data parquet file name
    data_chunk_dir = root / "data" / "chunk-000"
    legacy_parquet = data_chunk_dir / "0000.parquet"
    target_parquet = data_chunk_dir / "file-000.parquet"
    if legacy_parquet.exists() and not target_parquet.exists():
        legacy_parquet.rename(target_parquet)

    if not target_parquet.exists():
        raise FileNotFoundError(f"data parquet not found: {target_parquet}")

    df = pd.read_parquet(target_parquet)
    if "task" not in df.columns:
        raise ValueError("data parquet missing required column: task")

    # Build tasks table
    tasks = _unique_in_order(df["task"].tolist())
    tasks_df = pd.DataFrame({"task_index": range(len(tasks))}, index=tasks)
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(tasks_path)

    task_to_index = {t: i for i, t in enumerate(tasks)}

    # Add task_index to data if missing
    if "task_index" not in df.columns:
        df = df.copy()
        df["task_index"] = df["task"].map(task_to_index).astype("int64")
        df.to_parquet(target_parquet, index=False)

    # Build episode metadata parquet
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_path = episodes_dir / "file-000.parquet"
    if episodes_jsonl.exists():
        records: list[dict] = []
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        # Map episode -> tasks list
        tasks_by_ep = (
            df.groupby("episode_index")["task"]
            .apply(lambda s: _unique_in_order(s.tolist()))
            .to_dict()
        )

        for rec in records:
            ep_idx = rec.get("episode_index")
            rec["tasks"] = tasks_by_ep.get(ep_idx, [])
            rec["meta/episodes/chunk_index"] = 0
            rec["meta/episodes/file_index"] = 0
            rec["data/chunk_index"] = 0
            rec["data/file_index"] = 0

        episodes_df = pd.DataFrame(records)
        episodes_dir.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)
    else:
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_jsonl}")

    # Update info.json to v3-compatible fields
    total_episodes = int(df["episode_index"].max()) + 1 if len(df) else 0
    total_frames = int(len(df))
    info.setdefault("codebase_version", "v3.0")
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info.setdefault("chunks_size", 1000)
    info.setdefault("data_files_size_in_mb", 100)
    info.setdefault("video_files_size_in_mb", 200)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = DEFAULT_DATA_PATH

    has_video = any(ft.get("dtype") == "video" for ft in info.get("features", {}).values())
    info["video_path"] = DEFAULT_VIDEO_PATH if has_video else None

    # Add task_index feature if missing
    features = info.get("features", {})
    if "task_index" not in features:
        features["task_index"] = {"dtype": "int64", "shape": [1]}

    # Add channel names for image features if missing
    for key, ft in features.items():
        if ft.get("dtype") == "image" and "names" not in ft:
            ft["names"] = ["height", "width", "channels"]

    info["features"] = features
    _write_json(info_path, info)

    # Remove legacy metadata files if they would confuse loading
    legacy_files = [root / "meta" / "episodes.jsonl", root / "data" / "train.jsonl"]
    for lf in legacy_files:
        if lf.exists():
            lf.rename(lf.with_suffix(lf.suffix + ".legacy"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fix LeRobot v3 dataset metadata in-place.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help="Path to dataset root (contains meta/data)",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root_flag",
        type=Path,
        help="Path to dataset root (contains meta/data)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root_flag or args.dataset_root
    if dataset_root is None:
        parser.error("dataset_root is required")

    fix_dataset(dataset_root)


if __name__ == "__main__":
    main()
