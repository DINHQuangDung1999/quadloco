#!/usr/bin/env python3
"""Send zero-only VLA velocity packets for transport validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

QUADLOCO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUADLOCO_ROOT))

from deployment.go2_vla.udp_velocity import DEFAULT_PORT, UdpVelocityOutput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.0.192")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()
    if args.count <= 0 or args.rate <= 0:
        raise ValueError("--count and --rate must be positive.")

    output = UdpVelocityOutput(args.host, args.port)
    try:
        for index in range(args.count):
            output.publish(np.zeros(3, dtype=np.float32))
            print(f"[TX] zero packet {index + 1}/{args.count}")
            time.sleep(1.0 / args.rate)
    finally:
        output.close()


if __name__ == "__main__":
    main()
