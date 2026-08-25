#!/usr/bin/env python3
"""Compatibility entry point for deployment.go2_vla.deploy."""

from __future__ import annotations

import sys
from pathlib import Path

QUADLOCO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUADLOCO_ROOT))

from deployment.go2_vla.deploy import main  # noqa: E402


if __name__ == "__main__":
    main()
