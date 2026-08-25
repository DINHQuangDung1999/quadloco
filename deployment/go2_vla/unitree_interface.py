"""Optional Unitree SDK2 state and high-level motion integration."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import numpy as np


class UnitreeState:
    """Thread-safe q/dq snapshot from the Go2 LowState DDS topic."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._q = np.zeros(12, dtype=np.float32)
        self._dq = np.zeros(12, dtype=np.float32)
        self._updated_at = 0.0

    def callback(self, message: Any) -> None:
        q = np.asarray([message.motor_state[i].q for i in range(12)], dtype=np.float32)
        dq = np.asarray([message.motor_state[i].dq for i in range(12)], dtype=np.float32)
        with self._lock:
            self._q = q
            self._dq = dq
            self._updated_at = time.monotonic()

    def vector(self, expected_dim: int, max_age: float = 0.25) -> np.ndarray:
        with self._lock:
            age = time.monotonic() - self._updated_at
            q = self._q.copy()
            dq = self._dq.copy()
        if self._updated_at == 0.0 or age > max_age:
            raise RuntimeError(f"Go2 LowState is stale (age={age:.3f}s).")
        q_dq = np.concatenate((q, dq))
        if expected_dim < len(q_dq):
            raise ValueError(f"Checkpoint state dimension {expected_dim} is smaller than q+dq (24).")
        return np.pad(q_dq, (0, expected_dim - len(q_dq))).astype(np.float32, copy=False)


class UnitreeInterface:
    """Optional Go2 state subscriber and high-level velocity publisher."""

    def __init__(self, interface: str, state: UnitreeState | None, enable_motion: bool) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "unitree_sdk2py is required for --enable-motion or --state-source unitree."
            ) from error

        ChannelFactoryInitialize(0, interface)
        self._client = None
        if enable_motion:
            self._client = SportClient()
            self._client.SetTimeout(1.0)
            self._client.Init()
        self._subscriber = None
        if state is not None:
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(state.callback, 10)

    def move(self, velocity: np.ndarray) -> None:
        if self._client is None:
            raise RuntimeError("Motion client is disabled.")
        self._client.Move(float(velocity[0]), float(velocity[1]), float(velocity[2]))

    def stop(self) -> None:
        if self._client is not None:
            self._client.Move(0.0, 0.0, 0.0)
            self._client.StopMove()
