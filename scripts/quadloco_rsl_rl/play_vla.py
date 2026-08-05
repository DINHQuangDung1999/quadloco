# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a hierarchical PI0.5 + RSL-RL policy in the Quadloco navigation task.

PI0.5 receives the front RGB camera, a one-dimensional dummy state, and the
goal instruction. It produces ``[vx, vy, wz]``. The pretrained RSL-RL policy
then maps that velocity command and proprioception to the Go2 joint actions.

Launch ``scripts/pi05_velocity_server.py`` from the LeRobot Python 3.12
environment before launching this script from the Isaac Lab environment.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import time
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play PI0.5 velocity commands through an RSL-RL locomotion policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record an Isaac Lab viewport video.")
parser.add_argument("--video_length", type=int, default=500, help="Recorded video length in simulation steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Only one environment is currently supported.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0",
    help="Registered manager-based Quadloco navigation task.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Attempt real-time simulation.")
parser.add_argument("--vla_host", default="127.0.0.1", help="PI0.5 server address.")
parser.add_argument("--vla_port", type=int, default=5555, help="PI0.5 server TCP port.")
parser.add_argument("--vla_timeout", type=float, default=120.0, help="Socket timeout in seconds.")
parser.add_argument("--max_vx", type=float, default=1.5, help="Safety clamp for absolute forward velocity.")
parser.add_argument("--max_vy", type=float, default=1.0, help="Safety clamp for absolute lateral velocity.")
parser.add_argument("--max_wz", type=float, default=1.5, help="Safety clamp for absolute yaw velocity.")
parser.add_argument("--print_freq", type=int, default=25, help="Print VLA commands every N simulation steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    parser.error("play_vla.py currently supports exactly --num_envs 1.")

# The front camera is needed even when viewport video recording is disabled.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata
import os

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadloco.tasks  # noqa: F401


_HEADER = struct.Struct("!Q")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("PI0.5 server disconnected while receiving a message.")
        chunks.extend(chunk)
    return bytes(chunks)


class VelocityVLAClient:
    """Small synchronous client for the Python 3.12 PI0.5 process."""

    def __init__(self, host: str, port: int, timeout: float):
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)

    def close(self) -> None:
        self.connection.close()

    def predict(self, rgb: np.ndarray, task: str, reset: bool) -> tuple[np.ndarray, float]:
        request = {
            "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
            "task": task,
            "reset": reset,
        }
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.connection.sendall(_HEADER.pack(len(payload)))
        self.connection.sendall(payload)

        (size,) = _HEADER.unpack(_recv_exact(self.connection, _HEADER.size))
        response: dict[str, Any] = pickle.loads(  # noqa: S301 - trusted localhost server
            _recv_exact(self.connection, size)
        )
        if "error" in response:
            raise RuntimeError(f"PI0.5 server failed: {response['error']}")
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (3,):
            raise ValueError(f"Expected VLA action [vx, vy, wz], received shape {action.shape}.")
        return action, float(response["inference_s"])


def _rgb_frame(tensor: torch.Tensor) -> np.ndarray:
    frame = tensor.detach().cpu().numpy()
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HWC RGB/RGBA camera frame, received {frame.shape}.")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run hierarchical VLA and locomotion-policy inference."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # PI0.5 currently consumes RGB only. Avoid allocating and computing the
    # depth render product during playback; data-collection configuration is
    # left unchanged.
    env_cfg.scene.front_camera.data_types = ["rgb"]
    env_cfg.observations.camera.depth = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise TypeError("play_vla.py requires the manager-based single-agent navigation environment.")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "play_vla"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading low-level RSL-RL checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    low_level_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # RSL-RL has used several names for the underlying inference network.
    # The PPO version installed with this Isaac Lab workspace exposes
    # ``actor`` even though its package version is newer than 2.3.
    if hasattr(runner.alg, "actor"):
        policy_nn = runner.alg.actor
    elif hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        policy_nn = runner.alg.actor_critic
    else:
        raise AttributeError(
            "Could not find the RSL-RL inference network on runner.alg; "
            "expected actor, policy, or actor_critic."
        )

    print(f"[INFO] Connecting to PI0.5 server at {args_cli.vla_host}:{args_cli.vla_port}")
    vla_client = VelocityVLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    if not hasattr(command_term, "goal_task_names") or not hasattr(command_term, "velocity_command"):
        raise TypeError("The base_velocity term must be UniformGoalVelocityCommand.")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    reset_vla = True
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.perf_counter()
            task = command_term.goal_task_names[0]
            rgb = _rgb_frame(obs["camera"]["rgb"][0])
            velocity_np, inference_s = vla_client.predict(rgb, task, reset_vla)
            reset_vla = False

            limits = np.asarray([args_cli.max_vx, args_cli.max_vy, args_cli.max_wz], dtype=np.float32)
            velocity_np = np.clip(velocity_np, -limits, limits)
            velocity = torch.as_tensor(
                velocity_np,
                dtype=command_term.velocity_command.dtype,
                device=command_term.velocity_command.device,
            )

            with torch.inference_mode():
                # CommandManager generates the expert command during env.step().
                # Replace it with PI0.5's command, then recompute observations so
                # the low-level policy sees the replacement in this same cycle.
                command_term.velocity_command[0].copy_(velocity)
                obs = env.get_observations()
                joint_actions = low_level_policy(obs)
                obs, _, dones, _ = env.step(joint_actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    low_level_policy.reset(dones)
                else:
                    policy_nn.reset(dones)

            if timestep % args_cli.print_freq == 0:
                print(
                    f"[VLA] step={timestep} task={task!r} "
                    f"command=[{velocity_np[0]:+.3f}, {velocity_np[1]:+.3f}, {velocity_np[2]:+.3f}] "
                    f"server_s={inference_s:.3f}"
                )

            if bool(dones[0].item()):
                reset_vla = True

            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break

            sleep_time = dt - (time.perf_counter() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        vla_client.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
