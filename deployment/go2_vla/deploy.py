#!/usr/bin/env python3
"""Run local PI0.5 inference and optionally command a real Unitree Go2."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from .camera import OpenCvRgbReader, RecordedRgbdReader, RosRgbdReader
from .policy import Pi05VelocityPolicy
from .unitree_interface import UnitreeInterface, UnitreeState
from .velocity_output import DryRunOutput, RosTwistOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--camera-source", choices=("ros", "opencv", "recorded"), default="ros")
    parser.add_argument("--camera", default="0", help="OpenCV camera index, path, or stream URL.")
    parser.add_argument("--recorded-rgb", help="RGB PNG/JPEG used by --camera-source recorded.")
    parser.add_argument("--recorded-depth", help="Optional uint16 Z16 PNG paired with --recorded-rgb.")
    parser.add_argument("--one-shot", action="store_true", help="Run one inference and exit.")
    parser.add_argument("--rgb-topic", default="/d435i/color/image_raw")
    parser.add_argument("--depth-topic", default="/d435i/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--sync-queue-size", type=int, default=8)
    parser.add_argument("--depth-height", type=int, default=96)
    parser.add_argument("--depth-width", type=int, default=128)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rate", type=float, default=5.0, help="Maximum command rate in Hz.")
    parser.add_argument("--max-vx", type=float, default=0.15)
    parser.add_argument("--max-vy", type=float, default=0.10)
    parser.add_argument("--max-wz", type=float, default=0.20)
    parser.add_argument("--max-inference-age", type=float, default=1.0)
    parser.add_argument("--state-source", choices=("zeros", "unitree"), default="zeros")
    parser.add_argument("--network-interface", help="Interface connected to the Go2, e.g. eth0.")
    parser.add_argument("--cmd-vel-topic", default="/quadloco/cmd_vel", help="rl_sar Twist input topic.")
    parser.add_argument(
        "--output",
        choices=("dry-run", "ros", "sport"),
        default="dry-run",
        help="Command transport. The recommended real-robot path is ros -> rl_sar.",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Arm command output. Required in addition to selecting ros or sport.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0 or args.depth_scale <= 0.0:
        raise ValueError("--rate and --depth-scale must be positive.")
    if args.output != "dry-run" and not args.enable_motion:
        raise ValueError("Non-dry-run output also requires --enable-motion.")
    if (args.output == "sport" or args.state_source == "unitree") and not args.network_interface:
        raise ValueError("Unitree state or Sport output requires --network-interface.")

    policy = Pi05VelocityPolicy(args.checkpoint, args.device)
    state_reader = UnitreeState() if args.state_source == "unitree" else None
    needs_sdk = args.output == "sport" or state_reader is not None
    unitree = UnitreeInterface(args.network_interface, state_reader, args.output == "sport") if needs_sdk else None
    if args.output == "ros":
        output = RosTwistOutput(args.cmd_vel_topic)
    elif args.output == "sport":
        output = unitree
    else:
        output = DryRunOutput()

    if args.camera_source == "ros":
        camera = RosRgbdReader(
            args.rgb_topic,
            args.depth_topic,
            args.camera_timeout,
            args.sync_slop,
            args.sync_queue_size,
            (args.depth_height, args.depth_width),
        )
    elif args.camera_source == "opencv":
        if policy.depth_enabled:
            raise ValueError("A depth-enabled checkpoint requires ROS or recorded RGB-D input.")
        camera = OpenCvRgbReader(args.camera)
    else:
        if not args.recorded_rgb:
            raise ValueError("--camera-source recorded requires --recorded-rgb.")
        if policy.depth_enabled and not args.recorded_depth:
            raise ValueError("This depth-enabled checkpoint also requires --recorded-depth.")
        camera = RecordedRgbdReader(
            args.recorded_rgb,
            args.recorded_depth,
            (args.depth_height, args.depth_width),
        )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    limits = np.asarray([args.max_vx, args.max_vy, args.max_wz], dtype=np.float32)
    period = 1.0 / args.rate
    policy.reset()
    mode = f"MOTION ENABLED ({args.output})" if args.enable_motion else "DRY RUN"
    print(f"[SAFETY] {mode}; limits=[{limits[0]}, {limits[1]}, {limits[2]}]")

    try:
        while not stop_requested:
            started_at = time.monotonic()
            rgb, depth_z16 = camera.read()
            state = None
            if policy.state_dim:
                state = (
                    np.zeros(policy.state_dim, dtype=np.float32)
                    if state_reader is None
                    else state_reader.vector(policy.state_dim)
                )
            velocity = policy.predict(rgb, depth_z16, args.depth_scale, args.instruction, state)
            inference_age = time.monotonic() - started_at
            if inference_age > args.max_inference_age:
                raise RuntimeError(
                    f"Inference exceeded safety deadline: {inference_age:.3f}s > "
                    f"{args.max_inference_age:.3f}s."
                )
            velocity = np.clip(velocity, -limits, limits)
            print(
                f"[VLA] vx={velocity[0]:+.3f} vy={velocity[1]:+.3f} "
                f"wz={velocity[2]:+.3f} inference={inference_age:.3f}s"
            )
            if args.enable_motion:
                if args.output == "sport":
                    output.move(velocity)
                else:
                    output.publish(velocity)
            if args.one_shot:
                break
            remaining = period - (time.monotonic() - started_at)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        camera.close()
        if args.output == "ros":
            output.close()
        if unitree is not None:
            unitree.stop()
        print("[SAFETY] Zero velocity sent; deployment stopped.")


if __name__ == "__main__":
    main()
