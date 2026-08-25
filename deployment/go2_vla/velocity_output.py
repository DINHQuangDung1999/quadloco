"""Bounded velocity-command outputs for the VLA deployment."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class RosTwistOutput:
    """Publish VLA base commands for the rl_sar locomotion controller."""

    def __init__(self, topic: str) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.qos import QoSProfile, ReliabilityPolicy
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ROS velocity output requires rclpy and geometry_msgs. "
                "Source the ROS environment and use its compatible Python version."
            ) from error

        self._rclpy = rclpy
        self._twist_type = Twist
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=[])
        self._node = rclpy.create_node("quadloco_vla_velocity_output")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self._node.create_publisher(Twist, topic, qos)
        self._closed = False
        print(f"[OUTPUT] Publishing bounded VLA commands to {topic}")

    def publish(self, velocity: np.ndarray) -> None:
        velocity = np.asarray(velocity, dtype=np.float32)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError(f"Velocity must be a finite [vx, vy, wz] vector; got {velocity}.")
        message = self._twist_type()
        message.linear.x = float(velocity[0])
        message.linear.y = float(velocity[1])
        message.angular.z = float(velocity[2])
        self._publisher.publish(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        zero = np.zeros(3, dtype=np.float32)
        # Several samples make shutdown robust to a single lossy Wi-Fi packet.
        for _ in range(3):
            self.publish(zero)
            time.sleep(0.02)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


class DryRunOutput:
    """No-op output implementing the same interface as RosTwistOutput."""

    def publish(self, _velocity: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass
