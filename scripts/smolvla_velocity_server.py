#!/usr/bin/env python3
"""SmolVLA entry point for the shared VLA socket server."""

import sys

from vla_velocity_server import main


if __name__ == "__main__":
    if "--expected_policy_type" not in sys.argv:
        sys.argv.extend(("--expected_policy_type", "smolvla"))
    main()
