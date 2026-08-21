#!/usr/bin/env python

"""Evaluate hierarchical RGB-D PI0.5 waypoint navigation in Isaac Lab.

Run one task family per process. Episode records are written as JSONL and an
aggregate summary is written as JSON. PI0.5 never receives ground-truth goal
state; simulator state is used only by this evaluator for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from evaluation_assets import semantic_candidate_assets  # isort: skip
from waypoint_adapter import waypoint_to_velocity  # isort: skip


TASK_SUFFIXES = {
    "direct": "Rough-Direct-v0",
    "occluded": "Rough-Occluded-v0",
    "relational": "Rough-Relational-v0",
    "near_far": "Rough-NearFar-v0",
    "object_relative": "Rough-ObjectRelative-v0",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--navigation_mode", choices=tuple(TASK_SUFFIXES), required=True)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--output_dir", type=Path, default=Path("eval_results/pi05"))
parser.add_argument("--vla_host", default="127.0.0.1")
parser.add_argument("--vla_port", type=int, default=5555)
parser.add_argument("--vla_timeout", type=float, default=120.0)
parser.add_argument("--stop_hold_s", type=float, default=0.5)
parser.add_argument("--max_stopped_speed", type=float, default=0.15)
parser.add_argument("--max_stopped_yaw_rate", type=float, default=0.20)
parser.add_argument("--semantic_margin", type=float, default=0.25)
parser.add_argument("--collision_force_threshold", type=float, default=1.0)
parser.add_argument("--max_vx", type=float, default=1.5)
parser.add_argument("--max_vy", type=float, default=1.0)
parser.add_argument("--max_wz", type=float, default=1.5)
parser.add_argument("--print_freq", type=int, default=25)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    parser.error("eval_pi05.py supports exactly one environment for camera-safe evaluation.")
if args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be positive.")
if args_cli.task is None:
    args_cli.task = (
        "Unitree-Go2-Quadloco-ManagerBased-" + TASK_SUFFIXES[args_cli.navigation_mode]
    )
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
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
    def __init__(self, host: str, port: int, timeout: float):
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)

    def close(self) -> None:
        self.connection.close()

    def predict(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray,
        state: np.ndarray,
        task: str,
        reset: bool,
    ) -> tuple[np.ndarray, float]:
        request = {
            "rgb_bytes": np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
            "rgb_shape": rgb.shape,
            "depth_bytes": np.ascontiguousarray(depth_z16, dtype=np.uint16).tobytes(),
            "depth_shape": depth_z16.shape,
            "depth_scale": 0.001,
            "state_bytes": np.ascontiguousarray(state, dtype=np.float32).tobytes(),
            "state_shape": state.shape,
            "task": task,
            "reset": reset,
        }
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.connection.sendall(_HEADER.pack(len(payload)))
        self.connection.sendall(payload)
        (size,) = _HEADER.unpack(_recv_exact(self.connection, _HEADER.size))
        response: dict[str, Any] = pickle.loads(_recv_exact(self.connection, size))  # noqa: S301
        if "error" in response:
            raise RuntimeError(f"PI0.5 server failed: {response['error']}")
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape not in ((2,), (3,)):
            raise ValueError(f"Expected a 2D waypoint or 3D velocity, received {action.shape}.")
        return action, float(response["inference_s"])


def _rgb_frame(tensor: torch.Tensor) -> np.ndarray:
    frame = tensor.detach().cpu().numpy()[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _depth_z16_frame(tensor: torch.Tensor) -> np.ndarray:
    depth = tensor.detach()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth HWC/HW input, received {tuple(tensor.shape)}.")
    depth = torch.nn.functional.interpolate(
        depth[None, None].float(), size=(96, 128), mode="nearest"
    )[0, 0]
    valid = torch.isfinite(depth) & (depth > 0)
    encoded = torch.where(valid, torch.round(depth / 0.001), 0.0)
    encoded = encoded.clamp(0, np.iinfo(np.uint16).max)
    return np.ascontiguousarray(encoded.cpu().numpy().astype(np.uint16)[..., None])


def _unwrap_reset(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _non_foot_collision(env: Any, threshold: float) -> bool:
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    body_ids = [
        index
        for index, name in enumerate(sensor.body_names)
        if name == "base" or "thigh" in name or "calf" in name
    ]
    if not body_ids:
        return False
    forces = sensor.data.net_forces_w_history[0, :, body_ids, :]
    return bool((torch.linalg.vector_norm(forces, dim=-1) > threshold).any().item())


def _wrong_target_distance(command_term: Any, robot_xy: torch.Tensor, mode: str) -> float | None:
    goal_xy = command_term.goal_pos_w[0, :2]
    distances: list[float] = []
    unused_height = float(command_term.cfg.unused_candidate_height)
    for asset in semantic_candidate_assets(command_term, mode):
        candidate_xy = asset.data.root_pos_w[0, :2]
        if torch.linalg.vector_norm(candidate_xy - goal_xy) < 0.25:
            continue
        # Inactive variants are parked far below their environment. Use the
        # configured parking height instead of assuming a fixed world Z.
        if float(asset.data.root_pos_w[0, 2].item()) <= unused_height + 1.0:
            continue
        distances.append(float(torch.linalg.vector_norm(candidate_xy - robot_xy).item()))
    return min(distances) if distances else None


def _instruction_stratum(mode: str, instruction: str) -> str | None:
    text = instruction.lower()
    if mode == "near_far":
        if any(word in text for word in ("closest", "closer", "nearest")):
            return "near"
        if "middle" in text:
            return "middle"
        if any(word in text for word in ("farthest", "most distant")):
            return "far"
    if mode == "object_relative":
        relation = next(
            (name for name in ("front", "behind", "left", "right") if name in text),
            None,
        )
        distance = re.search(r"(0\.5|0\.75|1(?:\.0)?|1\.25)\s*m", text)
        if relation and distance:
            return f"{relation}_{float(distance.group(1)):.2f}m"
    return None


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _write_outputs(
    output_dir: Path,
    mode: str,
    records: list[dict[str, Any]],
    *,
    verbose: bool = True,
) -> None:
    """Atomically persist all completed episodes and their current summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{mode}_episodes.jsonl"
    records_tmp = output_dir / f".{mode}_episodes.jsonl.tmp"
    with records_tmp.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    records_tmp.replace(records_path)

    strata = sorted({record["stratum"] for record in records if record["stratum"]})
    stratified = {
        stratum: {
            "num_episodes": len(group := [r for r in records if r["stratum"] == stratum]),
            "success_rate": _mean(group, "success"),
            "semantic_accuracy": _mean(group, "semantic_correct"),
            "mean_final_goal_error_m": _mean(group, "final_goal_error_m"),
        }
        for stratum in strata
    }
    summary = {
        "navigation_mode": mode,
        "num_episodes": len(records),
        "success_rate": _mean(records, "success"),
        "semantic_accuracy": _mean(records, "semantic_correct"),
        "geometric_success_rate": _mean(records, "geometric_success"),
        "collision_rate": _mean(records, "collision"),
        "fall_rate": _mean(records, "fall"),
        "timeout_rate": _mean(records, "timeout"),
        "mean_final_goal_error_m": _mean(records, "final_goal_error_m"),
        "mean_time_to_success_s": _mean(
            [record for record in records if record["success"]], "time_to_success_s"
        ),
        "mean_spl": _mean(records, "spl"),
        "mean_path_length_m": _mean(records, "path_length_m"),
        "stratified": stratified,
        "criteria": {
            "stop_hold_s": args_cli.stop_hold_s,
            "max_stopped_speed_mps": args_cli.max_stopped_speed,
            "max_stopped_yaw_rate_radps": args_cli.max_stopped_yaw_rate,
            "semantic_margin_m": args_cli.semantic_margin,
            "depth_scale_m_per_unit": 0.001,
        },
    }
    summary_path = output_dir / f"{mode}_summary.json"
    summary_tmp = output_dir / f".{mode}_summary.json.tmp"
    with summary_tmp.open("w") as file:
        file.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    summary_tmp.replace(summary_path)
    if verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[INFO] Episode records: {records_path}")
        print(f"[INFO] Summary: {summary_path}")


installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.front_camera.data_types = ["rgb", "distance_to_image_plane"]
    env_cfg.commands.base_velocity.debug_vis = False
    # Evaluation applies its own, stricter stable-stop criterion (0.5 s by
    # default). The environment's normal 0.2 s goal termination would reset
    # an episode before that criterion can be satisfied.
    if hasattr(env_cfg.terminations, "goal_reached"):
        env_cfg.terminations.goal_reached = None
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise TypeError("eval_pi05.py requires a manager-based single-agent environment.")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    low_level_policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = next(
        (
            getattr(runner.alg, name)
            for name in ("actor", "policy", "actor_critic")
            if hasattr(runner.alg, name)
        ),
        None,
    )
    if policy_nn is None:
        raise AttributeError("Could not find the RSL-RL inference network.")

    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    dt = float(env.unwrapped.step_dt)
    hold_steps_required = max(1, math.ceil(args_cli.stop_hold_s / dt))
    episode_length_s = float(env.unwrapped.cfg.episode_length_s)
    limits = np.asarray([args_cli.max_vx, args_cli.max_vy, args_cli.max_wz], dtype=np.float32)

    client = WaypointVLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
    obs = env.get_observations()
    records: list[dict[str, Any]] = []
    reset_vla = True
    episode_index = 0
    episode_steps = 0
    hold_steps = 0
    collision = False
    path_length = 0.0
    previous_xy = robot.data.root_pos_w[0, :2].clone()
    initial_distance = float(
        torch.linalg.vector_norm(command_term.goal_pos_w[0, :2] - previous_xy).item()
    )

    try:
        while simulation_app.is_running() and episode_index < args_cli.num_episodes:
            loop_start = time.perf_counter()
            task = str(command_term.goal_task_names[0])
            rgb = _rgb_frame(obs["camera"]["rgb"][0])
            depth = _depth_z16_frame(obs["camera"]["depth"][0])
            state = obs["policy"][0].detach().cpu().numpy().astype(np.float32, copy=False)
            policy_action, inference_s = client.predict(rgb, depth, state, task, reset_vla)
            reset_vla = False

            if policy_action.shape == (2,):
                velocity_np = waypoint_to_velocity(
                    policy_action,
                    forward_velocity=command_term.cfg.forward_velocity,
                    yaw_gain=command_term.cfg.yaw_gain,
                    max_yaw_rate=command_term.cfg.max_yaw_rate,
                    goal_tolerance=command_term.cfg.goal_tolerance,
                    slowdown_distance=command_term.cfg.slowdown_distance,
                    minimum_approach_velocity=command_term.cfg.minimum_approach_velocity,
                )
            else:
                velocity_np = policy_action
            velocity_np = np.clip(velocity_np, -limits, limits)
            velocity = torch.as_tensor(
                velocity_np,
                dtype=command_term.velocity_command.dtype,
                device=command_term.velocity_command.device,
            )

            robot_xy = robot.data.root_pos_w[0, :2].clone()
            path_length += float(torch.linalg.vector_norm(robot_xy - previous_xy).item())
            previous_xy = robot_xy
            goal_error = float(
                torch.linalg.vector_norm(command_term.goal_pos_w[0, :2] - robot_xy).item()
            )
            planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0, :2]).item())
            yaw_rate = abs(float(robot.data.root_ang_vel_b[0, 2].item()))
            collision = collision or _non_foot_collision(env, args_cli.collision_force_threshold)
            geometric = goal_error <= float(command_term.cfg.goal_tolerance)
            stopped = (
                planar_speed < args_cli.max_stopped_speed
                and yaw_rate < args_cli.max_stopped_yaw_rate
            )
            hold_steps = hold_steps + 1 if geometric and stopped else 0
            wrong_distance = _wrong_target_distance(command_term, robot_xy, args_cli.navigation_mode)
            semantic_correct = wrong_distance is None or (
                goal_error + args_cli.semantic_margin <= wrong_distance
            )
            stable_goal = hold_steps >= hold_steps_required

            # Keep Isaac Lab environment operations outside inference_mode.
            # Tensors created while stepping/resetting under inference_mode cannot
            # later be updated in-place by Isaac Lab during an episode reset.
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

            episode_steps += 1
            done = bool(dones[0].item())
            success = stable_goal and semantic_correct and not collision
            if episode_steps % args_cli.print_freq == 0:
                print(
                    f"[EVAL] mode={args_cli.navigation_mode} episode={episode_index + 1}/"
                    f"{args_cli.num_episodes} step={episode_steps} goal_err={goal_error:.3f} "
                    f"collision={collision} action={policy_action.tolist()} server_s={inference_s:.3f}"
                )

            if success or done:
                elapsed_s = episode_steps * dt
                timeout = done and elapsed_s >= episode_length_s - 2.0 * dt
                fall = done and not timeout
                spl = (
                    initial_distance / max(initial_distance, path_length, 1e-6)
                    if success
                    else 0.0
                )
                record = {
                    "episode_index": episode_index,
                    "navigation_mode": args_cli.navigation_mode,
                    "instruction": task,
                    "stratum": _instruction_stratum(args_cli.navigation_mode, task),
                    "success": success,
                    "semantic_correct": semantic_correct,
                    "geometric_success": stable_goal,
                    "collision": collision,
                    "fall": fall,
                    "timeout": timeout,
                    "final_goal_error_m": goal_error,
                    "wrong_target_distance_m": wrong_distance,
                    "time_to_success_s": elapsed_s if success else None,
                    "episode_duration_s": elapsed_s,
                    "path_length_m": path_length,
                    "shortest_path_m": initial_distance,
                    "spl": spl,
                    "final_planar_speed_mps": planar_speed,
                    "final_yaw_rate_radps": yaw_rate,
                }
                records.append(record)
                # Make every completed episode durable immediately. Atomic
                # replacement keeps the previous checkpoint intact if the
                # evaluator is interrupted during a write.
                _write_outputs(
                    args_cli.output_dir,
                    args_cli.navigation_mode,
                    records,
                    verbose=False,
                )
                print(f"[EPISODE] {json.dumps(record, sort_keys=True)}")
                episode_index += 1
                if episode_index >= args_cli.num_episodes:
                    break

                # ManagerBasedRLEnv auto-resets terminated/timed-out environments
                # inside env.step(). Only reset explicitly when evaluation ended
                # early because the success criterion was reached.
                if not done:
                    obs = _unwrap_reset(env.reset())
                    reset_mask = torch.ones(1, dtype=torch.bool, device=env.unwrapped.device)
                    with torch.no_grad():
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            low_level_policy.reset(reset_mask)
                        else:
                            policy_nn.reset(reset_mask)
                reset_vla = True
                episode_steps = 0
                hold_steps = 0
                collision = False
                path_length = 0.0
                previous_xy = robot.data.root_pos_w[0, :2].clone()
                initial_distance = float(
                    torch.linalg.vector_norm(
                        command_term.goal_pos_w[0, :2] - previous_xy
                    ).item()
                )

            sleep_s = dt - (time.perf_counter() - loop_start)
            if args_cli.real_time and sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        client.close()
        env.close()

    _write_outputs(args_cli.output_dir, args_cli.navigation_mode, records)


if __name__ == "__main__":
    main()
    simulation_app.close()
