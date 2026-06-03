"""Download 10 episodes of the TOTO Benchmark dataset (scooping / pouring
on Franka, Zhou et al. ICRA 2023) from the Open-X GCS bucket and save
locally:

    out_dir/
        episode_000/
            image.mp4               # 480x640 third-person camera (HIGH RES!)
            language_embedding.npy  # (T, 512) float32
            data.json               # action (world_vec, rot_delta, gripper, terminate),
                                    # joint states, rewards, flags, instruction
        ...
"""

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow_datasets as tfds


DATASET = "toto"
VERSION = "0.1.0"
GCS_PATH = f"gs://gresearch/robotics/{DATASET}/{VERSION}"
OUT_DIR = Path("/mnt/afs/dongyifei/DreamFlyWheel/Gen2Act/dataset/toto_gen") / DATASET
N_EPISODES = 101
FPS = 10


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def save_episode(episode, ep_dir: Path) -> int:
    ep_dir.mkdir(parents=True, exist_ok=True)

    images = []
    states = []
    world_vectors, rotation_deltas = [], []
    open_gripper, terminate = [], []
    rewards = []
    is_first, is_last, is_terminal = [], [], []
    instructions, embeddings = [], []

    for step in episode["steps"]:
        obs = step["observation"]
        images.append(obs["image"].numpy())
        states.append(obs["state"].numpy().tolist())
        instructions.append(decode_text(obs["natural_language_instruction"].numpy()))
        embeddings.append(obs["natural_language_embedding"].numpy())

        act = step["action"]
        world_vectors.append(act["world_vector"].numpy().tolist())
        rotation_deltas.append(act["rotation_delta"].numpy().tolist())
        open_gripper.append(bool(act["open_gripper"].numpy()))
        terminate.append(float(act["terminate_episode"].numpy()))

        rewards.append(float(step["reward"].numpy()))
        is_first.append(bool(step["is_first"].numpy()))
        is_last.append(bool(step["is_last"].numpy()))
        is_terminal.append(bool(step["is_terminal"].numpy()))

    imageio.mimsave(ep_dir / "image.mp4", images, format="FFMPEG", fps=FPS, macro_block_size=1)
    np.save(ep_dir / "language_embedding.npy", np.stack(embeddings).astype(np.float32))

    def unique_text(seq):
        uniq = list(dict.fromkeys(seq))
        return uniq[0] if len(uniq) == 1 else uniq

    record = {
        "num_steps": len(rewards),
        "fps": FPS,
        "action_spec": "world_vector(3), rotation_delta(3), open_gripper(bool), terminate_episode(1)",
        "state_spec": "7x robot joint angles (absolute)",
        "natural_language_instruction": unique_text(instructions),
        "natural_language_instruction_per_step": instructions,
        "world_vector": world_vectors,
        "rotation_delta": rotation_deltas,
        "open_gripper": open_gripper,
        "terminate_episode": terminate,
        "states": states,
        "rewards": rewards,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "image_shape": list(images[0].shape),
    }
    with open(ep_dir / "data.json", "w") as fp:
        json.dump(record, fp, indent=2)
    return len(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="first episode index (inclusive)")
    parser.add_argument("--end", type=int, default=N_EPISODES, help="last episode index (exclusive)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {DATASET} from {GCS_PATH}")
    builder = tfds.builder_from_directory(builder_dir=GCS_PATH)
    print("builder loaded")
    print(f"Splits: train={builder.info.splits['train'].num_examples}, "
          f"test={builder.info.splits['test'].num_examples}")
    n = args.end - args.start
    print(f"Taking train[{args.start}:{args.end}] ({n} episodes) -> {OUT_DIR}")

    ds = builder.as_dataset(split=f"train[{args.start}:{args.end}]")
    print("dataset created")
    for offset, episode in enumerate(ds):
        ep_idx = args.start + offset
        ep_dir = OUT_DIR / f"episode_{ep_idx:03d}"
        if (ep_dir / "data.json").exists():
            print(f"[{offset + 1}/{n}] skip episode_{ep_idx:03d} (already exists)")
            continue
        n_steps = save_episode(episode, ep_dir)
        print(f"[{offset + 1}/{n}] saved -> {ep_dir.name} ({n_steps} steps)")

    print("Done.")


if __name__ == "__main__":
    main()
