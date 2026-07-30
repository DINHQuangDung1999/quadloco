# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate (one avoids cross-environment camera interference).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--collect_data",
    action="store_true",
    default=False,
    help="Collect and save RGB-D navigation trajectories.",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of trajectories to collect.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="datasets/goal_navigation",
    help="Directory in which completed trajectories are saved.",
)
parser.add_argument(
    "--dataset_format",
    choices=("pt", "lerobot"),
    default="pt",
    help="Output format. 'pt' preserves the legacy per-episode files; 'lerobot' writes a LeRobot v3 dataset.",
)
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="quadloco/goal_navigation",
    help="LeRobot dataset identifier stored in metadata and used when pushing to Hugging Face.",
)
parser.add_argument(
    "--dataset_task",
    type=str,
    default="Navigate to the target",
    help="Natural-language task attached to every recorded LeRobot frame.",
)
parser.add_argument(
    "--lerobot_include_depth",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Store metric depth as float32 arrays in Parquet (enabled by default). "
        "Use --no-lerobot_include_depth to omit it and reduce dataset size."
    ),
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push the finalized LeRobot dataset to the Hugging Face Hub.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# Camera rendering is required for RGB-D data collection, even without video recording.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadloco.tasks  # noqa: F401


def _cpu_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    """Detach one recorded sample and return a contiguous NumPy array."""
    array = tensor.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _rgb_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert an Isaac Lab RGB/RGBA observation to a LeRobot RGB frame."""
    frame = _cpu_numpy(tensor)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HWC RGB/RGBA frame, got shape {frame.shape}.")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        # Isaac Lab emits uint8 when RGB normalization is disabled, but accept
        # normalized float images as well to make this boundary robust.
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _depth_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert metric depth to finite float32 values for Parquet storage."""
    frame = _cpu_numpy(tensor, np.float32)
    return np.ascontiguousarray(np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0))


def _create_lerobot_dataset(
    dataset_dir: str,
    repo_id: str,
    fps: int,
    rgb: torch.Tensor,
    base_velocity: torch.Tensor,
    policy_observation: torch.Tensor,
    low_level_action: torch.Tensor,
    velocity_command: torch.Tensor,
    depth: torch.Tensor | None,
):
    """Create a LeRobot v3 dataset using shapes observed from the environment."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is required for --dataset_format lerobot. Install the sibling "
            "repository with `pip install -e ../lerobot` in the Isaac Lab environment."
        ) from exc

    root = os.path.abspath(dataset_dir)
    if os.path.exists(root):
        raise FileExistsError(
            f"LeRobot dataset directory already exists: {root}. "
            "Choose a new --dataset_dir to avoid overwriting data."
        )

    rgb_sample = _rgb_frame(rgb)
    state_sample = _cpu_numpy(base_velocity, np.float32)
    policy_state_sample = _cpu_numpy(policy_observation, np.float32)
    low_level_action_sample = _cpu_numpy(low_level_action, np.float32)
    command_sample = _cpu_numpy(velocity_command, np.float32)
    height, width, channels = rgb_sample.shape
    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": state_sample.shape,
            "names": ["vx", "vy", "wz"] if state_sample.shape == (3,) else None,
        },
        "action": {
            "dtype": "float32",
            "shape": low_level_action_sample.shape,
            "names": None,
        },
        "observation.policy_state": {
            "dtype": "float32",
            "shape": policy_state_sample.shape,
            "names": None,
        },
        "observation.velocity_command": {
            "dtype": "float32",
            "shape": command_sample.shape,
            "names": ["vx", "vy", "wz"] if command_sample.shape == (3,) else None,
        },
    }
    if depth is not None:
        depth_sample = _depth_frame(depth)
        features["observation.depth.camera1"] = {
            "dtype": "float32",
            "shape": depth_sample.shape,
            "names": None,
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="quadloco",
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    num_envs = env.unwrapped.num_envs
    dataset_dir = os.path.abspath(args_cli.dataset_dir)
    trajectories = [
        {"rgb": [], "depth": [], "action": [], "velocity_command": []} for _ in range(num_envs)
    ]
    num_collected = 0
    collect_data = args_cli.collect_data
    lerobot_dataset = None
    if collect_data:
        if args_cli.dataset_format == "lerobot" and num_envs != 1:
            raise ValueError(
                "LeRobot recording currently requires --num_envs 1 because episode frames "
                "must be written sequentially."
            )
        if args_cli.dataset_format == "pt":
            os.makedirs(dataset_dir, exist_ok=True)
        print(
            f"[INFO] Collecting {args_cli.num_episodes} trajectories in "
            f"{args_cli.dataset_format} format: {dataset_dir}"
        )

    # In collection mode, stop after exactly the requested number of episodes.
    while simulation_app.is_running() and (not collect_data or num_collected < args_cli.num_episodes):
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Compute the action from the current observation. Store this
            # observation-action pair before stepping to keep them aligned.
            actions = policy(obs)
            applied_actions = actions
            if agent_cfg.clip_actions is not None:
                applied_actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)

            if collect_data:
                camera_obs = obs["camera"]
                velocity_commands = env.unwrapped.command_manager.get_command("base_velocity")
                robot_data = env.unwrapped.scene["robot"].data
                base_velocities = torch.cat(
                    (robot_data.root_lin_vel_b[:, :2], robot_data.root_ang_vel_b[:, 2:3]), dim=-1
                )
                if args_cli.dataset_format == "lerobot":
                    if lerobot_dataset is None:
                        fps = round(1.0 / dt)
                        lerobot_dataset = _create_lerobot_dataset(
                            dataset_dir=dataset_dir,
                            repo_id=args_cli.dataset_repo_id,
                            fps=fps,
                            rgb=camera_obs["rgb"][0],
                            base_velocity=base_velocities[0],
                            policy_observation=obs["policy"][0],
                            low_level_action=applied_actions[0],
                            velocity_command=velocity_commands[0],
                            depth=camera_obs["depth"][0] if args_cli.lerobot_include_depth else None,
                        )
                        print(f"[INFO] Initialized LeRobot v3 dataset at: {lerobot_dataset.root}")

                    frame = {
                        "observation.images.camera1": _rgb_frame(camera_obs["rgb"][0]),
                        "observation.state": _cpu_numpy(base_velocities[0], np.float32),
                        "observation.policy_state": _cpu_numpy(obs["policy"][0], np.float32),
                        "observation.velocity_command": _cpu_numpy(
                            velocity_commands[0], np.float32
                        ),
                        "action": _cpu_numpy(applied_actions[0], np.float32),
                        "task": args_cli.dataset_task,
                    }
                    if args_cli.lerobot_include_depth:
                        frame["observation.depth.camera1"] = _depth_frame(camera_obs["depth"][0])
                    lerobot_dataset.add_frame(frame)
                else:
                    for env_id in range(num_envs):
                        trajectories[env_id]["rgb"].append(camera_obs["rgb"][env_id].cpu().clone())
                        trajectories[env_id]["depth"].append(camera_obs["depth"][env_id].cpu().clone())
                        trajectories[env_id]["action"].append(applied_actions[env_id].cpu().clone())
                        trajectories[env_id]["velocity_command"].append(
                            velocity_commands[env_id].cpu().clone()
                        )

            obs, _, dones, _ = env.step(actions)

            # Isaac Lab auto-resets completed environments. Save each finished
            # trajectory immediately, then start a fresh buffer for that env.
            done_env_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            for env_id in done_env_ids if collect_data else []:
                if num_collected >= args_cli.num_episodes:
                    break
                if args_cli.dataset_format == "lerobot":
                    episode_steps = lerobot_dataset.episode_buffer["size"]
                    lerobot_dataset.save_episode()
                    output_path = lerobot_dataset.root
                else:
                    trajectory = {
                        key: torch.stack(values) for key, values in trajectories[env_id].items()
                    }
                    trajectory["env_id"] = env_id
                    trajectory["dt"] = dt
                    output_path = os.path.join(dataset_dir, f"trajectory_{num_collected:06d}.pt")
                    torch.save(trajectory, output_path)
                    episode_steps = trajectory["action"].shape[0]
                num_collected += 1
                print(
                    f"[INFO] Saved trajectory {num_collected}/{args_cli.num_episodes} "
                    f"({episode_steps} steps): {output_path}"
                )

            for env_id in done_env_ids if collect_data and args_cli.dataset_format == "pt" else []:
                trajectories[env_id] = {
                    "rgb": [],
                    "depth": [],
                    "action": [],
                    "velocity_command": [],
                }

            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if lerobot_dataset is not None:
        lerobot_dataset.finalize()
        print(f"[INFO] Finalized LeRobot dataset: {lerobot_dataset.root}")
        if args_cli.push_to_hub:
            lerobot_dataset.push_to_hub()
            print(f"[INFO] Pushed dataset to: https://huggingface.co/datasets/{args_cli.dataset_repo_id}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
