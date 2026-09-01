"""Client for a long-lived PI0.5 direct-velocity inference server."""

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


class SocketVelocityPolicy:
    """Expose the remote direct-velocity policy with the local policy API."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._connection = socket.create_connection((host, port), timeout=timeout)
        self._connection.settimeout(timeout)
        self._reset = True
        description = self._request({"op": "describe"})
        if description.get("action_mode") != "direct_velocity" or description.get("action_dim") != 3:
            raise ValueError(f"Server does not expose a 3D direct-velocity policy: {description}")
        self.state_dim = int(description["state_dim"])
        self.state_token_dim = description.get("state_token_dim")
        indices = description.get("state_feature_indices")
        self.state_feature_indices = None if indices is None else tuple(indices)
        self.depth_enabled = bool(description["depth_enabled"])
        print(
            f"[POLICY] Connected to {host}:{port}; state={self.state_dim}, "
            f"state_token_dim={self.state_token_dim}, depth={self.depth_enabled}"
        )

    def _request(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._connection.sendall(_HEADER.pack(len(payload)))
        self._connection.sendall(payload)
        (size,) = _HEADER.unpack(_recv_exact(self._connection, _HEADER.size))
        response = pickle.loads(_recv_exact(self._connection, size))  # noqa: S301 - trusted peer
        if "error" in response:
            raise RuntimeError(f"PI0.5 server failed: {response['error']}")
        return response

    def reset(self) -> None:
        self._reset = True

    def predict(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray | None,
        depth_scale: float,
        instruction: str,
        state: np.ndarray | None,
    ) -> np.ndarray:
        if self.depth_enabled and depth_z16 is None:
            raise ValueError("The server policy requires aligned depth input.")
        if state is None or state.shape != (self.state_dim,):
            shape = None if state is None else state.shape
            raise ValueError(f"Server policy expects state ({self.state_dim},), got {shape}.")
        depth = np.empty((0, 0, 1), dtype=np.uint16) if depth_z16 is None else depth_z16
        response = self._request(
            {
                "rgb_bytes": np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
                "rgb_shape": rgb.shape,
                "depth_bytes": np.ascontiguousarray(depth, dtype=np.uint16).tobytes(),
                "depth_shape": depth.shape,
                "depth_scale": float(depth_scale),
                "state_bytes": np.ascontiguousarray(state, dtype=np.float32).tobytes(),
                "state_shape": state.shape,
                "task": instruction,
                "reset": self._reset,
            }
        )
        self._reset = False
        velocity = np.asarray(response["action"], dtype=np.float32)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError(f"Expected finite [vx, vy, wz], received {velocity}.")
        return velocity

    def close(self) -> None:
        self._connection.close()
