"""Transport-only client for the Quadloco PI0.5 socket server."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np


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
    """Synchronous NumPy client that is independent of ROS and Isaac Lab."""

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
            # Byte payloads keep the protocol compatible across NumPy versions.
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
        response: dict[str, Any] = pickle.loads(  # noqa: S301 - the configured PI0.5 peer is trusted
            _recv_exact(self.connection, size)
        )
        if "error" in response:
            raise RuntimeError(f"PI0.5 server failed: {response['error']}")

        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected VLA waypoint [x_forward, y_left], received shape {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError("VLA waypoint contains NaN or infinity.")

        action_plan = np.asarray(response["action_plan"], dtype=np.float32)
        if action_plan.ndim != 2 or action_plan.shape[1] != 2:
            raise ValueError(f"Expected VLA action plan with shape (N, 2), received {action_plan.shape}.")
        if not np.isfinite(action_plan).all():
            raise ValueError("VLA action plan contains NaN or infinity.")

        action_plan_index = int(response["action_plan_index"])
        if not 0 <= action_plan_index < len(action_plan):
            raise ValueError(f"Invalid action plan index {action_plan_index} for plan length {len(action_plan)}.")
        return action, action_plan, action_plan_index, float(response["inference_s"])
