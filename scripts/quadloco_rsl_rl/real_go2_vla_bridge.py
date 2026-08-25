#!/usr/bin/env python3
"""Bridge real Go2 D435i ROS topics to the Quadloco PI0.5 server.

The node is deliberately shadow-only: it publishes camera readiness and VLA
waypoints, but never publishes ``/cmd_vel`` or any motor command.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String

QUADLOCO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QUADLOCO_ROOT))

from deployment.go2_vla.camera import (
    image_bytes_to_depth_z16,
    image_bytes_to_rgb,
    resize_depth_nearest,
    validate_policy_state,
)
from waypoint_vla_client import WaypointVLAClient


@dataclass(frozen=True)
class InferenceInput:
    rgb: np.ndarray
    depth_z16: np.ndarray
    state: np.ndarray
    task: str
    rgb_depth_delta_s: float


@dataclass(frozen=True)
class CapturePair:
    rgb: np.ndarray
    depth_z16: np.ndarray
    rgb_stamp_ns: int
    depth_stamp_ns: int
    index: int


def _stamp_seconds(message: Image) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


class RealGo2VLABridge(Node):
    """Validate synchronized D435i input and optionally run shadow PI0.5 inference."""

    def __init__(self) -> None:
        super().__init__("quadloco_real_go2_vla_bridge")
        self.declare_parameter("rgb_topic", "/d435i/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/aligned_depth_to_color/image_raw")
        self.declare_parameter("policy_state_topic", "/quadloco/policy_state")
        self.declare_parameter("task_topic", "/quadloco/task")
        self.declare_parameter("task", "Walk toward the instructed target while avoiding obstacles.")
        self.declare_parameter("camera_only", True)
        self.declare_parameter("vla_host", "127.0.0.1")
        self.declare_parameter("vla_port", 5555)
        self.declare_parameter("vla_timeout_s", 120.0)
        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("state_timeout_s", 0.25)
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("sync_queue_size", 8)
        self.declare_parameter("expected_rgb_height", 480)
        self.declare_parameter("expected_rgb_width", 640)
        self.declare_parameter("depth_output_height", 96)
        self.declare_parameter("depth_output_width", 128)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("save_images", False)
        self.declare_parameter("save_interval_s", 0.5)
        self.declare_parameter("save_directory", "/home/ws/quadloco/camera_captures")
        self.declare_parameter("save_depth", True)
        self.declare_parameter("save_max_pairs", 120)

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._camera_only = bool(self.get_parameter("camera_only").value)
        self._vla_host = str(self.get_parameter("vla_host").value)
        self._vla_port = int(self.get_parameter("vla_port").value)
        self._vla_timeout_s = float(self.get_parameter("vla_timeout_s").value)
        inference_rate_hz = float(self.get_parameter("inference_rate_hz").value)
        if inference_rate_hz <= 0.0:
            raise ValueError("inference_rate_hz must be positive.")
        self._minimum_inference_period_s = 1.0 / inference_rate_hz
        self._state_timeout_s = float(self.get_parameter("state_timeout_s").value)
        self._expected_rgb_shape = (
            int(self.get_parameter("expected_rgb_height").value),
            int(self.get_parameter("expected_rgb_width").value),
            3,
        )
        self._depth_output_size = (
            int(self.get_parameter("depth_output_height").value),
            int(self.get_parameter("depth_output_width").value),
        )
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        if not np.isclose(self._depth_scale, 0.001):
            raise ValueError("The current Quadloco PI0.5 checkpoint requires depth_scale=0.001.")
        self._save_images = bool(self.get_parameter("save_images").value)
        self._save_interval_s = float(self.get_parameter("save_interval_s").value)
        if self._save_interval_s <= 0.0:
            raise ValueError("save_interval_s must be positive.")
        self._save_directory = Path(str(self.get_parameter("save_directory").value)).expanduser()
        self._save_depth = bool(self.get_parameter("save_depth").value)
        self._save_max_pairs = int(self.get_parameter("save_max_pairs").value)
        if self._save_max_pairs < 0:
            raise ValueError("save_max_pairs must be zero (unlimited) or positive.")

        self._state_lock = threading.Lock()
        self._policy_state: np.ndarray | None = None
        self._policy_state_received_s = -np.inf
        self._task = str(self.get_parameter("task").value)
        self._work_queue: queue.Queue[InferenceInput] = queue.Queue(maxsize=1)
        self._save_queue: queue.Queue[CapturePair] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._client: WaypointVLAClient | None = None
        self._reset_vla = True
        self._last_enqueued_s = -np.inf
        self._last_status_log_s = -np.inf
        self._last_save_s = -np.inf
        self._valid_pair_count = 0
        self._saved_pair_count = 0
        self._scheduled_capture_count = 0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_subscriber = message_filters.Subscriber(self, Image, self._rgb_topic, qos_profile=sensor_qos)
        self._depth_subscriber = message_filters.Subscriber(
            self, Image, self._depth_topic, qos_profile=sensor_qos
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_subscriber, self._depth_subscriber],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_s").value),
        )
        self._synchronizer.registerCallback(self._camera_callback)

        self._policy_state_subscription = self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("policy_state_topic").value),
            self._policy_state_callback,
            5,
        )
        self._task_subscription = self.create_subscription(
            String,
            str(self.get_parameter("task_topic").value),
            self._task_callback,
            5,
        )
        self._camera_ready_publisher = self.create_publisher(Bool, "/quadloco/camera_ready", 1)
        self._waypoint_publisher = self.create_publisher(Float32MultiArray, "/quadloco/vla_waypoint", 1)
        self.create_timer(5.0, self._status_timer)

        self._worker = threading.Thread(target=self._inference_worker, name="quadloco-vla", daemon=True)
        self._worker.start()
        self._save_worker = threading.Thread(target=self._image_save_worker, name="quadloco-save", daemon=True)
        self._save_worker.start()
        mode = "camera validation only" if self._camera_only else "shadow PI0.5 inference"
        self.get_logger().info(
            f"Started in {mode}; RGB={self._rgb_topic}, depth={self._depth_topic}. "
            "This node does not publish /cmd_vel."
        )
        if self._save_images:
            self._save_directory.mkdir(parents=True, exist_ok=True)
            limit = "unlimited" if self._save_max_pairs == 0 else str(self._save_max_pairs)
            self.get_logger().info(
                f"Saving one synchronized image pair every {self._save_interval_s:.3f}s to "
                f"{self._save_directory} (maximum pairs: {limit})."
            )

    def _policy_state_callback(self, message: Float32MultiArray) -> None:
        try:
            state = validate_policy_state(np.asarray(message.data, dtype=np.float32))
        except ValueError as error:
            self.get_logger().error(f"Rejected policy state: {error}")
            return
        with self._state_lock:
            self._policy_state = state
            self._policy_state_received_s = time.monotonic()

    def _task_callback(self, message: String) -> None:
        task = message.data.strip()
        if task:
            self._task = task
            self._reset_vla = True
        else:
            self.get_logger().warning("Ignored an empty /quadloco/task message.")

    def _camera_callback(self, rgb_message: Image, depth_message: Image) -> None:
        try:
            rgb = image_bytes_to_rgb(
                rgb_message.data,
                height=rgb_message.height,
                width=rgb_message.width,
                step=rgb_message.step,
                encoding=rgb_message.encoding,
            )
            if rgb.shape != self._expected_rgb_shape:
                raise ValueError(f"Expected RGB shape {self._expected_rgb_shape}, received {rgb.shape}.")
            depth = image_bytes_to_depth_z16(
                depth_message.data,
                height=depth_message.height,
                width=depth_message.width,
                step=depth_message.step,
                encoding=depth_message.encoding,
                is_bigendian=bool(depth_message.is_bigendian),
            )
            depth_z16 = resize_depth_nearest(depth, self._depth_output_size)
            timestamp_delta_s = abs(_stamp_seconds(rgb_message) - _stamp_seconds(depth_message))
        except ValueError as error:
            self._camera_ready_publisher.publish(Bool(data=False))
            self._log_status("error", f"Rejected RGB-D pair: {error}")
            return

        self._valid_pair_count += 1
        self._camera_ready_publisher.publish(Bool(data=True))
        if self._valid_pair_count == 1:
            self.get_logger().info(
                f"Validated first RGB-D pair: rgb={rgb.shape}/{rgb.dtype}, "
                f"depth={depth_z16.shape}/{depth_z16.dtype}, stamp_delta={timestamp_delta_s:.4f}s."
            )
        self._enqueue_capture(rgb, depth, rgb_message, depth_message)
        if self._camera_only:
            return

        now = time.monotonic()
        if now - self._last_enqueued_s < self._minimum_inference_period_s:
            return
        with self._state_lock:
            state = None if self._policy_state is None else self._policy_state.copy()
            state_age_s = now - self._policy_state_received_s
        if state is None or state_age_s > self._state_timeout_s:
            self._log_status(
                "warning",
                f"Camera is ready, but PI0.5 input is blocked by missing/stale 45D policy state "
                f"(age={state_age_s:.3f}s).",
            )
            return

        item = InferenceInput(rgb, depth_z16, state, self._task, timestamp_delta_s)
        try:
            self._work_queue.put_nowait(item)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                pass
            self._work_queue.put_nowait(item)
        self._last_enqueued_s = now

    def _inference_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._client is None:
                    self._client = WaypointVLAClient(self._vla_host, self._vla_port, self._vla_timeout_s)
                    self._reset_vla = True
                    self.get_logger().info(f"Connected to PI0.5 server at {self._vla_host}:{self._vla_port}.")
                waypoint, _, plan_index, inference_s = self._client.predict(
                    item.rgb,
                    item.depth_z16,
                    self._depth_scale,
                    item.state,
                    item.task,
                    self._reset_vla,
                )
                self._reset_vla = False
                self._waypoint_publisher.publish(Float32MultiArray(data=waypoint.tolist()))
                self.get_logger().info(
                    f"Shadow VLA waypoint=[{waypoint[0]:+.3f}, {waypoint[1]:+.3f}], "
                    f"plan_index={plan_index}, inference={inference_s:.3f}s, "
                    f"rgb_depth_delta={item.rgb_depth_delta_s:.4f}s."
                )
            except Exception as error:
                self.get_logger().error(f"PI0.5 shadow inference failed: {error}")
                if self._client is not None:
                    self._client.close()
                    self._client = None
                self._reset_vla = True

    def _enqueue_capture(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray,
        rgb_message: Image,
        depth_message: Image,
    ) -> None:
        if not self._save_images:
            return
        if self._save_max_pairs and self._scheduled_capture_count >= self._save_max_pairs:
            return
        now = time.monotonic()
        if now - self._last_save_s < self._save_interval_s:
            return
        self._last_save_s = now
        capture = CapturePair(
            rgb=rgb.copy(),
            depth_z16=depth_z16.copy(),
            rgb_stamp_ns=int(_stamp_seconds(rgb_message) * 1e9),
            depth_stamp_ns=int(_stamp_seconds(depth_message) * 1e9),
            index=self._scheduled_capture_count,
        )
        try:
            self._save_queue.put_nowait(capture)
            self._scheduled_capture_count += 1
        except queue.Full:
            self.get_logger().warning("Image writer is busy; dropping one scheduled capture pair.")

    def _image_save_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                capture = self._save_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            stem = f"frame_{capture.index:06d}_{capture.rgb_stamp_ns}"
            rgb_path = self._save_directory / f"{stem}_rgb.png"
            rgb_bgr = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(rgb_path), rgb_bgr):
                self.get_logger().error(f"Failed to save RGB image: {rgb_path}")
                continue
            if self._save_depth:
                depth_path = self._save_directory / f"{stem}_depth_z16.png"
                if not cv2.imwrite(str(depth_path), capture.depth_z16):
                    self.get_logger().error(f"Failed to save depth image: {depth_path}")
                    continue
            self._saved_pair_count += 1
            if self._saved_pair_count == 1 or self._saved_pair_count % 10 == 0:
                delta_s = abs(capture.rgb_stamp_ns - capture.depth_stamp_ns) * 1e-9
                self.get_logger().info(
                    f"Saved {self._saved_pair_count} synchronized image pair(s); "
                    f"latest RGB-depth delta={delta_s:.4f}s."
                )

    def _status_timer(self) -> None:
        if self._valid_pair_count == 0:
            rgb_publishers = self.count_publishers(self._rgb_topic)
            depth_publishers = self.count_publishers(self._depth_topic)
            if rgb_publishers == 0 or depth_publishers == 0:
                self.get_logger().warning(
                    "Camera input unavailable: "
                    f"{self._rgb_topic} has {rgb_publishers} publisher(s), "
                    f"{self._depth_topic} has {depth_publishers} publisher(s). "
                    "Start the D435i driver and verify the configured topic names."
                )
            else:
                self.get_logger().warning(
                    f"RGB and depth publishers are visible, but no synchronized pair was received from "
                    f"{self._rgb_topic} and {self._depth_topic}. Check DDS QoS and timestamp separation."
                )
        else:
            self.get_logger().info(f"Camera bridge healthy: {self._valid_pair_count} valid RGB-D pairs received.")

    def _log_status(self, level: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_status_log_s < 2.0:
            return
        self._last_status_log_s = now
        getattr(self.get_logger(), level)(message)

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=2.0)
        self._save_worker.join(timeout=2.0)
        if self._client is not None:
            self._client.close()
            self._client = None


def main() -> None:
    rclpy.init()
    node: RealGo2VLABridge | None = None
    try:
        node = RealGo2VLABridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
