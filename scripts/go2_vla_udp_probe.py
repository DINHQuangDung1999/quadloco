#!/usr/bin/env python3
"""Print-only Go2 VLA UDP receiver; never publishes ROS or Unitree commands."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

QUADLOCO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUADLOCO_ROOT))

from deployment.go2_vla.udp_velocity import DEFAULT_PORT, PACKET, decode_velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allowed-host", default="192.168.0.88")
    parser.add_argument("--timeout", type=float, default=0.5)
    args = parser.parse_args()

    limits = np.asarray([0.15, 0.10, 0.20], dtype=np.float32)
    last_received = -np.inf
    last_sequence: int | None = None
    timed_out = False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind((args.bind, args.port))
        receiver.settimeout(0.1)
        print(
            f"[PROBE] Listening on {args.bind}:{args.port}; allowed={args.allowed_host}; "
            f"packet_size={PACKET.size}. No robot command output exists."
        )
        while True:
            try:
                payload, address = receiver.recvfrom(256)
            except socket.timeout:
                if time.monotonic() - last_received > args.timeout and not timed_out:
                    print("[WATCHDOG] No fresh packet; effective command=[0, 0, 0]")
                    timed_out = True
                continue
            if address[0] != args.allowed_host:
                print(f"[REJECT] Unexpected source {address[0]}:{address[1]}")
                continue
            try:
                packet = decode_velocity(payload)
            except ValueError as error:
                print(f"[REJECT] {error}")
                continue
            if last_sequence is not None and packet.sequence <= last_sequence:
                print(f"[REJECT] Non-increasing sequence {packet.sequence} after {last_sequence}")
                continue
            if np.any(np.abs(packet.velocity) > limits):
                print(f"[REJECT] Out-of-bounds velocity {packet.velocity.tolist()}")
                continue
            last_sequence = packet.sequence
            last_received = time.monotonic()
            timed_out = False
            print(f"[RX] seq={packet.sequence} velocity={packet.velocity.tolist()}")


if __name__ == "__main__":
    main()
