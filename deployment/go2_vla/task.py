#!/usr/bin/env python3
"""Run one instructed Go2 task using a waiting PI0.5 inference server."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from typing import Any

import numpy as np

from .camera import RosRgbdReader
from .socket_policy import SocketVelocityPolicy
from .udp_velocity import STATE_DIM, UdpPolicyStateReader, UdpVelocityOutput
from .velocity_output import DryRunOutput, RosTwistOutput


class RosPolicyStateReader:
    """Read the exact locomotion observation exported by rl_sar."""

    def __init__(self, topic: str, expected_dim: int, timeout: float) -> None:
        import rclpy
        from std_msgs.msg import Float32MultiArray

        self._rclpy = rclpy
        self._expected_dim = expected_dim
        self._timeout = timeout
        self._lock = threading.Lock()
        self._state: np.ndarray | None = None
        self._received_at = -np.inf
        self._node = rclpy.create_node("quadloco_vla_policy_state_reader")
        self._subscription = self._node.create_subscription(
            Float32MultiArray, topic, self._callback, 5
        )
        print(f"[STATE] Waiting for {expected_dim}D locomotion state on {topic}")

    def _callback(self, message: Any) -> None:
        state = np.asarray(message.data, dtype=np.float32)
        if state.shape != (self._expected_dim,) or not np.isfinite(state).all():
            return
        with self._lock:
            self._state = np.ascontiguousarray(state)
            self._received_at = time.monotonic()

    def read(self) -> np.ndarray:
        deadline = time.monotonic() + self._timeout
        while True:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            with self._lock:
                state = None if self._state is None else self._state.copy()
                age = time.monotonic() - self._received_at
            if state is not None and age <= self._timeout:
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No fresh {self._expected_dim}D policy state received within {self._timeout:.1f}s."
                )

    def close(self) -> None:
        self._node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--server-timeout", type=float, default=120.0)
    parser.add_argument("--rgb-topic", default="/d435i/color/image_raw")
    parser.add_argument("--depth-topic", default="/d435i/aligned_depth_to_color/image_raw")
    parser.add_argument("--cmd-vel-topic", default="/quadloco/cmd_vel")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--sync-queue-size", type=int, default=8)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--state-source", choices=("zeros", "ros", "udp"), default="zeros")
    parser.add_argument("--output", choices=("ros", "udp"), default="udp")
    parser.add_argument("--robot-host", default="192.168.0.192")
    parser.add_argument("--velocity-port", type=int, default=5560)
    parser.add_argument("--state-port", type=int, default=5561)
    parser.add_argument("--policy-state-topic", default="/quadloco/policy_state")
    parser.add_argument("--state-timeout", type=float, default=0.5)
    parser.add_argument("--max-vx", type=float, default=0.15)
    parser.add_argument("--max-vy", type=float, default=0.10)
    parser.add_argument("--max-wz", type=float, default=0.20)
    parser.add_argument("--max-inference-age", type=float, default=120.0)
    parser.add_argument("--enable-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0 or args.server_timeout <= 0 or args.max_inference_age <= 0:
        raise ValueError("Rate and timeout values must be positive.")
    if args.enable_motion and args.state_source == "zeros":
        raise ValueError("Motion requires a live --state-source (udp or ros).")

    policy = SocketVelocityPolicy(args.server_host, args.server_port, args.server_timeout)
    camera = RosRgbdReader(
        args.rgb_topic,
        args.depth_topic,
        args.camera_timeout,
        args.sync_slop,
        args.sync_queue_size,
        (96, 128),
    )
    udp_output = None
    if args.enable_motion and args.output == "udp":
        udp_output = UdpVelocityOutput(args.robot_host, args.velocity_port)
        output = udp_output
    elif args.enable_motion:
        output = RosTwistOutput(args.cmd_vel_topic)
    else:
        output = DryRunOutput()
    if args.state_source == "ros":
        state_reader = RosPolicyStateReader(
            args.policy_state_topic, policy.state_dim, args.state_timeout
        )
    elif args.state_source == "udp":
        selected_indices = policy.state_feature_indices
        if policy.state_dim < STATE_DIM or (
            selected_indices is not None and max(selected_indices, default=-1) >= STATE_DIM
        ) or (
            selected_indices is None
            and (policy.state_token_dim is None or policy.state_token_dim > STATE_DIM)
        ):
            raise ValueError(
                f"UDP state bridge supplies recorded indices 0..{STATE_DIM - 1}; server "
                f"reports state_dim={policy.state_dim}, selected_indices={selected_indices}."
            )
        state_reader = UdpPolicyStateReader(
            args.state_port, args.state_timeout, args.robot_host
        )
        # Establish the return address before the first state is requested.
        if udp_output is not None:
            udp_output.publish(np.zeros(3, dtype=np.float32))
        else:
            handshake = UdpVelocityOutput(args.robot_host, args.velocity_port)
            handshake.publish(np.zeros(3, dtype=np.float32))
            handshake.close()
    else:
        state_reader = None
    zero_state = np.zeros(policy.state_dim, dtype=np.float32)
    limits = np.asarray([args.max_vx, args.max_vy, args.max_wz], dtype=np.float32)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    policy.reset()
    print(f"[TASK] {args.instruction!r}")
    print(f"[SAFETY] {'MOTION ENABLED' if args.enable_motion else 'DRY RUN'}; limits={limits.tolist()}")

    try:
        while not stop_requested:
            started_at = time.monotonic()
            rgb, depth = camera.read()
            if state_reader is None:
                state = zero_state
            elif args.state_source == "udp":
                state = zero_state.copy()
                state[:STATE_DIM] = state_reader.read()
            else:
                state = state_reader.read()
            velocity = policy.predict(rgb, depth, args.depth_scale, args.instruction, state)
            inference_age = time.monotonic() - started_at
            if inference_age > args.max_inference_age:
                raise RuntimeError(
                    f"Inference exceeded deadline: {inference_age:.3f}s > {args.max_inference_age:.3f}s"
                )
            velocity = np.clip(velocity, -limits, limits)
            print(
                f"[VLA] vx={velocity[0]:+.3f} vy={velocity[1]:+.3f} "
                f"wz={velocity[2]:+.3f} inference={inference_age:.3f}s"
            )
            output.publish(velocity)
            remaining = 1.0 / args.rate - (time.monotonic() - started_at)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        output.close()
        if state_reader is not None:
            state_reader.close()
        camera.close()
        policy.close()
        print("[SAFETY] Zero velocity published; task stopped.")


if __name__ == "__main__":
    main()
