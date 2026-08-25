# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a hierarchical PI0.5 + RSL-RL policy in the Quadloco navigation task.

PI0.5 receives camera input, proprioception, and the goal instruction. It may
produce either a robot-frame waypoint ``[x_forward, y_left]`` or a direct
velocity command ``[vx, vy, wz]`` for the pretrained RSL-RL locomotion policy.

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
from waypoint_adapter import waypoint_to_velocity  # isort: skip


parser = argparse.ArgumentParser(description="Play PI0.5 velocity commands through an RSL-RL locomotion policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record an Isaac Lab viewport video.")
parser.add_argument("--video_length", type=int, default=500, help="Recorded video length in simulation steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Only one environment is currently supported.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-Go2-Quadloco-ManagerBased-Rough-Direct-v0",
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
parser.add_argument(
    "--waypoint_vis",
    choices=("vla", "astar", "none"),
    default="vla",
    help="Display the VLA waypoint, the oracle A* route, or no waypoint markers.",
)
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
import isaaclab.sim as sim_utils
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply, yaw_quat
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


class WaypointVLAClient:
    """Small synchronous client for the Python 3.12 PI0.5 process."""

    def __init__(self, host: str, port: int, timeout: float):
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)

    def close(self) -> None:
        self.connection.close()

    def predict(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray,
        depth_scale: float,
        state: np.ndarray,
        task: str,
        reset: bool,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_z16 = np.ascontiguousarray(depth_z16, dtype=np.uint16)
        state = np.ascontiguousarray(state, dtype=np.float32)
        request = {
            # Send only Python primitives and bytes across environments. NumPy
            # arrays pickled by NumPy 2.x cannot be loaded by NumPy 1.x because
            # their private module paths differ (numpy._core vs numpy.core).
            "rgb_bytes": rgb.tobytes(),
            "rgb_shape": rgb.shape,
            "depth_bytes": depth_z16.tobytes(),
            "depth_shape": depth_z16.shape,
            "depth_scale": float(depth_scale),
            "state_bytes": state.tobytes(),
            "state_shape": state.shape,
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
        if action.shape not in ((2,), (3,)):
            raise ValueError(f"Expected a 2D waypoint or 3D velocity, received {action.shape}.")
        action_plan = np.asarray(response["action_plan"], dtype=np.float32)
        if action_plan.ndim != 2 or action_plan.shape[1] != action.shape[0]:
            raise ValueError(
                f"Expected VLA action plan with shape (N, {action.shape[0]}), "
                f"received {action_plan.shape}."
            )
        action_plan_index = int(response["action_plan_index"])
        if not 0 <= action_plan_index < len(action_plan):
            raise ValueError(
                f"Invalid action plan index {action_plan_index} for plan length {len(action_plan)}."
            )
        return action, action_plan, action_plan_index, float(response["inference_s"])


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


def _depth_z16_frame(
    tensor: torch.Tensor,
    output_size: tuple[int, int] = (96, 128),
    depth_scale: float = 0.001,
) -> np.ndarray:
    """Resize metric simulator depth and encode D435i-style Z16."""
    if depth_scale != 0.001:
        raise ValueError(f"Deployment depth scale must be 0.001, got {depth_scale}.")
    depth = tensor.detach()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W) or (H,W,1), received {tuple(tensor.shape)}.")
    depth = torch.nn.functional.interpolate(
        depth[None, None].float(), size=output_size, mode="nearest"
    )[0, 0]
    valid = torch.isfinite(depth) & (depth > 0)
    encoded = torch.where(valid, torch.round(depth / depth_scale), 0.0)
    encoded = encoded.clamp(0, np.iinfo(np.uint16).max)
    return np.ascontiguousarray(encoded.cpu().numpy().astype(np.uint16)[..., None])


installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run hierarchical VLA and locomotion-policy inference."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # RGB-D PI0.5 requires the same camera products used during collection.
    env_cfg.scene.front_camera.data_types = ["rgb", "distance_to_image_plane"]

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
        task_name = args_cli.task.lower()
        if "objectrelative" in task_name:
            task_slug = "object_relative"
        elif "nearfar" in task_name:
            task_slug = "near_far"
        elif "relational" in task_name:
            task_slug = "relational"
        elif "occluded" in task_name:
            task_slug = "occluded"
        else:
            task_slug = "direct"
        video_folder = os.path.join(
            os.path.dirname(resume_path), "videos", "play_vla", task_slug
        )
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"vla-{task_slug}",
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
    vla_client = WaypointVLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    if not hasattr(command_term, "goal_task_names") or not hasattr(command_term, "velocity_command"):
        raise TypeError("The base_velocity term must be UniformGoalVelocityCommand.")

    # Hide the command term's built-in oracle markers. Playback renders one
    # explicitly selected waypoint source below so A* and VLA cannot be
    # mistaken for each other.
    if hasattr(command_term, "goal_direction_visualizer"):
        command_term.goal_direction_visualizer.set_visibility(False)
    if hasattr(command_term, "velocity_visualizer"):
        command_term.velocity_visualizer.set_visibility(False)
    route_entry_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/OracleRoute/entry",
            markers={
                "entry": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
                ),
            },
        )
    )
    route_exit_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/OracleRoute/exit",
            markers={
                "exit": sim_utils.SphereCfg(
                    radius=0.11,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
                ),
            },
        )
    )
    route_goal_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/OracleRoute/final_goal",
            markers={
                "final_goal": sim_utils.SphereCfg(
                    radius=0.13,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.2)),
                ),
            },
        )
    )
    vla_waypoint_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/VLA/waypoint",
            markers={
                "waypoint": sim_utils.SphereCfg(
                    radius=0.13,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.85, 1.0)
                    ),
                ),
            },
        )
    )
    show_astar = args_cli.waypoint_vis == "astar"
    show_vla = args_cli.waypoint_vis == "vla"
    route_entry_visualizer.set_visibility(show_astar)
    route_exit_visualizer.set_visibility(show_astar)
    route_goal_visualizer.set_visibility(show_astar)
    vla_waypoint_visualizer.set_visibility(show_vla)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    reset_vla = True
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.perf_counter()
            task = command_term.goal_task_names[0]
            rgb = _rgb_frame(obs["camera"]["rgb"][0])
            depth_z16 = _depth_z16_frame(obs["camera"]["depth"][0], depth_scale=0.001)
            state = obs["policy"][0].detach().cpu().numpy().astype(np.float32, copy=False)
            policy_action_np, _action_plan_np, _action_plan_index, inference_s = vla_client.predict(
                rgb, depth_z16, 0.001, state, task, reset_vla
            )
            reset_vla = False

            ground_z = env.unwrapped.scene.env_origins[0, 2] + 0.05
            if show_vla and policy_action_np.shape == (2,):
                waypoint_b = torch.zeros(
                    (1, 3),
                    dtype=command_term.robot.data.root_pos_w.dtype,
                    device=command_term.robot.data.root_pos_w.device,
                )
                waypoint_b[0, :2] = torch.as_tensor(
                    policy_action_np, dtype=waypoint_b.dtype, device=waypoint_b.device
                )
                waypoint_w = command_term.robot.data.root_pos_w[0:1] + quat_apply(
                    yaw_quat(command_term.robot.data.root_quat_w[0:1]), waypoint_b
                )
                waypoint_w[:, 2] = ground_z
                vla_waypoint_visualizer.visualize(translations=waypoint_w)
            elif show_vla:
                # A direct velocity is not a spatial waypoint, so displaying
                # it as a world-frame marker would be misleading.
                vla_waypoint_visualizer.set_visibility(False)

            if show_astar:
                # Red: all intermediate A* points; orange: active point;
                # green: final oracle goal.
                goal_w = command_term.goal_pos_w[0:1].clone()
                goal_w[:, 2] = ground_z
                route_goal_visualizer.visualize(translations=goal_w)
                has_route = hasattr(command_term, "route_waypoints_w") and hasattr(
                    command_term, "route_lengths"
                )
                if has_route:
                    route_length = int(command_term.route_lengths[0].item())
                    final_stage = route_length - 1
                    intermediate_w = command_term.route_waypoints_w[0, :final_stage].clone()
                    has_intermediate = len(intermediate_w) > 0
                    route_entry_visualizer.set_visibility(has_intermediate)
                    if has_intermediate:
                        intermediate_w[:, 2] = ground_z
                        route_entry_visualizer.visualize(translations=intermediate_w)

                    active_stage = int(command_term.route_stage[0].item())
                    has_active_intermediate = active_stage < final_stage
                    route_exit_visualizer.set_visibility(has_active_intermediate)
                    if has_active_intermediate:
                        active_w = command_term.route_waypoints_w[
                            0, active_stage : active_stage + 1
                        ].clone()
                        active_w[:, 2] = ground_z
                        route_exit_visualizer.visualize(translations=active_w)
                else:
                    route_entry_visualizer.set_visibility(False)
                    route_exit_visualizer.set_visibility(False)

            if policy_action_np.shape == (2,):
                velocity_np = waypoint_to_velocity(
                    policy_action_np,
                    forward_velocity=command_term.cfg.forward_velocity,
                    yaw_gain=command_term.cfg.yaw_gain,
                    max_yaw_rate=command_term.cfg.max_yaw_rate,
                    goal_tolerance=command_term.cfg.goal_tolerance,
                    slowdown_distance=command_term.cfg.slowdown_distance,
                    minimum_approach_velocity=command_term.cfg.minimum_approach_velocity,
                    local_waypoint_radius=command_term.cfg.local_waypoint_radius,
                )
                action_text = (
                    f"waypoint=[{policy_action_np[0]:+.3f}, {policy_action_np[1]:+.3f}]"
                )
            else:
                velocity_np = policy_action_np
                action_text = (
                    f"direct_velocity=[{policy_action_np[0]:+.3f}, "
                    f"{policy_action_np[1]:+.3f}, {policy_action_np[2]:+.3f}]"
                )

            limits = np.asarray([args_cli.max_vx, args_cli.max_vy, args_cli.max_wz], dtype=np.float32)
            velocity_np = np.clip(velocity_np, -limits, limits)
            velocity = torch.as_tensor(
                velocity_np,
                dtype=command_term.velocity_command.dtype,
                device=command_term.velocity_command.device,
            )

            # CommandManager generates the expert command during env.step().
            # Replace it with PI0.5's command, then recompute observations so
            # the low-level policy sees the replacement in this same cycle.
            # Isaac Lab environment operations must stay outside inference_mode:
            # otherwise auto-reset state can become an inference tensor and reject
            # subsequent in-place updates.
            command_term.velocity_command[0].copy_(velocity)
            obs = env.get_observations()
            with torch.no_grad():
                joint_actions = low_level_policy(obs)

            obs, _, dones, _ = env.step(joint_actions)
            with torch.no_grad():
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    low_level_policy.reset(dones)
                else:
                    policy_nn.reset(dones)

            if timestep % args_cli.print_freq == 0:
                print(
                    f"[VLA] step={timestep} task={task!r} "
                    f"{action_text} "
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
