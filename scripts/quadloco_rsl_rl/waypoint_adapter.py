"""Deterministic robot-frame waypoint to locomotion-command adapter."""

from __future__ import annotations

import numpy as np


def waypoint_to_velocity(
    waypoint: np.ndarray,
    *,
    forward_velocity: float,
    yaw_gain: float,
    max_yaw_rate: float,
    goal_tolerance: float,
    slowdown_distance: float,
    minimum_approach_velocity: float = 0.0,
) -> np.ndarray:
    """Convert ``[x_forward, y_left]`` into ``[vx, vy, wz]``.

    This mirrors ``UniformGoalVelocityCommandDirect._waypoint_to_velocity``
    and its distance slowdown so data collection and deployment use the same
    low-level command semantics.
    """
    waypoint = np.asarray(waypoint, dtype=np.float32)
    if waypoint.shape != (2,):
        raise ValueError(f"Expected waypoint shape (2,), received {waypoint.shape}.")
    if not np.isfinite(waypoint).all():
        raise ValueError("Waypoint must contain finite values.")
    if not 0.0 <= goal_tolerance < slowdown_distance:
        raise ValueError("Expected 0 <= goal_tolerance < slowdown_distance.")
    if not 0.0 <= minimum_approach_velocity <= forward_velocity:
        raise ValueError(
            "Expected 0 <= minimum_approach_velocity <= forward_velocity."
        )

    distance = float(np.linalg.norm(waypoint))
    if distance < goal_tolerance:
        return np.zeros(3, dtype=np.float32)

    distance_scale = np.clip(
        (distance - goal_tolerance) / (slowdown_distance - goal_tolerance),
        0.0,
        1.0,
    )
    heading_error = float(np.arctan2(waypoint[1], waypoint[0]))
    heading_scale = max(float(np.cos(heading_error)), 0.0)
    approach_velocity = max(
        forward_velocity * distance_scale, minimum_approach_velocity
    )
    return np.asarray(
        [
            approach_velocity * heading_scale,
            0.0,
            np.clip(yaw_gain * heading_error, -max_yaw_rate, max_yaw_rate),
        ],
        dtype=np.float32,
    )
