#!/usr/bin/env python3
"""Create a π0.5 RGB-to-velocity dataset from a Quadloco collection."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import modify_features, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ACTION_KEY = "action"
STATE_KEY = "observation.state"
VELOCITY_KEY = "observation.velocity_command"
DEPTH_KEYS = ("observation.depth.camera1", "observation.depth_scale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument(
        "--keep-depth",
        action="store_true",
        help="Keep depth features in the derived dataset. π0.5 will still use RGB unless modified for depth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if output_root.exists():
        raise FileExistsError(
            f"Output dataset already exists: {output_root}. "
            "Remove it explicitly or choose another --output-root."
        )

    dataset = LeRobotDataset(repo_id=args.source_repo_id, root=source_root)
    required = {ACTION_KEY, VELOCITY_KEY}
    missing = required.difference(dataset.meta.features)
    if missing:
        raise KeyError(f"Source dataset is missing required features: {sorted(missing)}")

    source_action_shape = tuple(dataset.meta.features[ACTION_KEY]["shape"])
    velocity_shape = tuple(dataset.meta.features[VELOCITY_KEY]["shape"])
    if velocity_shape != (3,):
        raise ValueError(f"Expected {VELOCITY_KEY} shape (3,), found {velocity_shape}")

    velocity_matrix = np.asarray(dataset.hf_dataset[VELOCITY_KEY], dtype=np.float32)
    if velocity_matrix.ndim != 2 or velocity_matrix.shape[1] != 3:
        raise ValueError(f"Expected velocity array with shape (frames, 3), found {velocity_matrix.shape}")

    # dataset_tools.modify_features writes through pandas. A numeric (N, 3)
    # matrix is interpreted as three DataFrame columns, while one LeRobot
    # feature must occupy one column whose rows each contain a 3-vector.
    velocity_actions = np.empty(len(velocity_matrix), dtype=object)
    velocity_actions[:] = [row for row in velocity_matrix]

    output_root.parent.mkdir(parents=True, exist_ok=True)
    intermediate_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}_prepare_", dir=output_root.parent)
    )
    shutil.rmtree(intermediate_root)

    try:
        first_pass_removals = [ACTION_KEY]
        if STATE_KEY in dataset.meta.features:
            first_pass_removals.append(STATE_KEY)

        without_action = modify_features(
            dataset,
            remove_features=first_pass_removals,
            output_dir=intermediate_root,
            repo_id=f"{args.output_repo_id}-intermediate",
        )

        remove_features = [VELOCITY_KEY]
        if not args.keep_depth:
            remove_features.extend(key for key in DEPTH_KEYS if key in without_action.meta.features)

        dummy_states = np.zeros((len(velocity_matrix), 1), dtype=np.float32)
        derived = modify_features(
            without_action,
            add_features={
                ACTION_KEY: (
                    velocity_actions,
                    {
                        "dtype": "float32",
                        "shape": (3,),
                        "names": ["vx", "vy", "wz"],
                    },
                ),
                STATE_KEY: (
                    dummy_states,
                    {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["dummy"],
                    },
                ),
            },
            remove_features=remove_features,
            output_dir=output_root,
            repo_id=args.output_repo_id,
        )
        recompute_stats(derived, skip_image_video=True)
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise
    finally:
        if intermediate_root.exists():
            shutil.rmtree(intermediate_root)

    print(f"Created π0.5 velocity dataset: {output_root}")
    print(f"Converted action shape {source_action_shape} -> (3,)")
    print("π0.5 inputs: RGB + task instruction + constant dummy state [0.0]")
    print("π0.5 target: [vx, vy, wz]")


if __name__ == "__main__":
    main()
