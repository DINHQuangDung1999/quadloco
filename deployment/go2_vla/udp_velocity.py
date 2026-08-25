"""Binary UDP transport for bounded VLA base-velocity commands."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np


MAGIC = b"QVLA"
VERSION = 1
DEFAULT_PORT = 5560
PACKET = struct.Struct("!4sBIfff")
STATE_MAGIC = b"QVST"
DEFAULT_STATE_PORT = 5561
STATE_DIM = 42
STATE_PACKET = struct.Struct(f"!4sBI{STATE_DIM}f")


@dataclass(frozen=True)
class VelocityPacket:
    sequence: int
    velocity: np.ndarray


@dataclass(frozen=True)
class StatePacket:
    sequence: int
    state: np.ndarray


def encode_velocity(sequence: int, velocity: np.ndarray) -> bytes:
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError(f"Velocity must be a finite [vx, vy, wz] vector; got {velocity}.")
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("Sequence must fit in an unsigned 32-bit integer.")
    return PACKET.pack(MAGIC, VERSION, sequence, *(float(value) for value in velocity))


def decode_velocity(payload: bytes) -> VelocityPacket:
    if len(payload) != PACKET.size:
        raise ValueError(f"Expected {PACKET.size} bytes, received {len(payload)}.")
    magic, version, sequence, vx, vy, wz = PACKET.unpack(payload)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"Unsupported packet magic/version: {magic!r}/{version}.")
    velocity = np.asarray([vx, vy, wz], dtype=np.float32)
    if not np.isfinite(velocity).all():
        raise ValueError("Velocity packet contains NaN or infinity.")
    return VelocityPacket(sequence=sequence, velocity=velocity)


def decode_state(payload: bytes) -> StatePacket:
    if len(payload) != STATE_PACKET.size:
        raise ValueError(f"Expected {STATE_PACKET.size} bytes, received {len(payload)}.")
    magic, version, sequence, *values = STATE_PACKET.unpack(payload)
    if magic != STATE_MAGIC or version != VERSION:
        raise ValueError(f"Unsupported state packet magic/version: {magic!r}/{version}.")
    state = np.asarray(values, dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("State packet contains NaN or infinity.")
    return StatePacket(sequence=sequence, state=state)


class UdpVelocityOutput:
    """Send bounded velocity datagrams to an onboard receiver."""

    def __init__(self, host: str, port: int = DEFAULT_PORT) -> None:
        self._target = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0

    def publish(self, velocity: np.ndarray) -> None:
        self._socket.sendto(encode_velocity(self._sequence, velocity), self._target)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF

    def close(self) -> None:
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(3):
            self.publish(zero)
        self._socket.close()


class UdpPolicyStateReader:
    """Receive the 42D locomotion observation exported by onboard rl_sar."""

    def __init__(
        self,
        port: int = DEFAULT_STATE_PORT,
        timeout: float = 0.5,
        expected_host: str | None = None,
    ) -> None:
        self._timeout = timeout
        self._expected_host = expected_host
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", port))
        self._socket.settimeout(timeout)
        self._last_sequence: int | None = None
        print(f"[STATE] Waiting for {STATE_DIM}D onboard policy state on UDP port {port}")

    def read(self) -> np.ndarray:
        while True:
            try:
                payload, source = self._socket.recvfrom(65535)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"No fresh {STATE_DIM}D onboard policy state received within "
                    f"{self._timeout:.1f}s."
                ) from exc
            if self._expected_host is not None and source[0] != self._expected_host:
                continue
            try:
                packet = decode_state(payload)
            except ValueError:
                continue
            if self._last_sequence is not None:
                delta = (packet.sequence - self._last_sequence) & 0xFFFFFFFF
                if delta == 0 or delta >= 0x80000000:
                    continue
            self._last_sequence = packet.sequence
            return packet.state

    def close(self) -> None:
        self._socket.close()
