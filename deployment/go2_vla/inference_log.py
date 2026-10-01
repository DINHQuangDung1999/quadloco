"""Durable JSONL logging for VLA inference timing."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def predicted_action_fields(action: np.ndarray) -> dict[str, Any]:
    """Describe either a 2D waypoint or a 3D base-velocity prediction."""
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size == 2:
        action_type = "waypoint"
    elif values.size == 3:
        action_type = "velocity"
    else:
        action_type = "unknown"
    return {
        "predicted_action": values.tolist(),
        "predicted_action_dim": int(values.size),
        "predicted_action_type": action_type,
    }


class InferenceLogger:
    """Append one durable, machine-readable record per forward pass."""

    def __init__(self, path: Path | None, deployment: str) -> None:
        if path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = Path("logs/deployment") / f"{deployment}_{timestamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8")
        self._sequence = 0
        print(f"[LOG] Inference timings: {path.resolve()}")

    def write(self, **values: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sequence": self._sequence,
            **values,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._sequence += 1

    def close(self) -> None:
        self._file.close()
