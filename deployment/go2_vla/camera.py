"""Camera decoding and synchronized ROS 2 RGB-D acquisition."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


RGB_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8"}
DEPTH_ENCODINGS = {"16uc1", "mono16"}


def image_bytes_to_rgb(
    data: bytes | bytearray | memoryview,
    *,
    height: int,
    width: int,
    step: int,
    encoding: str,
) -> np.ndarray:
    """Decode a ROS Image byte buffer into contiguous HWC RGB uint8."""
    encoding = encoding.lower()
    if encoding not in RGB_ENCODINGS:
        raise ValueError(f"Unsupported RGB encoding {encoding!r}; expected one of {sorted(RGB_ENCODINGS)}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, received {width}x{height}.")

    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    packed_step = width * channels
    if step < packed_step:
        raise ValueError(f"RGB row step {step} is smaller than packed width {packed_step}.")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected_size = height * step
    if raw.size < expected_size:
        raise ValueError(f"RGB buffer has {raw.size} bytes; expected at least {expected_size}.")
    frame = raw[:expected_size].reshape(height, step)[:, :packed_step].reshape(height, width, channels)
    frame = frame[..., :3]
    if encoding in {"bgr8", "bgra8"}:
        frame = frame[..., ::-1]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def image_bytes_to_depth_z16(
    data: bytes | bytearray | memoryview,
    *,
    height: int,
    width: int,
    step: int,
    encoding: str,
    is_bigendian: bool,
) -> np.ndarray:
    """Decode a ROS Image byte buffer into contiguous HW native-endian uint16."""
    encoding = encoding.lower()
    if encoding not in DEPTH_ENCODINGS:
        raise ValueError(
            f"Unsupported depth encoding {encoding!r}; expected one of {sorted(DEPTH_ENCODINGS)}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Depth dimensions must be positive, received {width}x{height}.")
    packed_step = width * np.dtype(np.uint16).itemsize
    if step < packed_step:
        raise ValueError(f"Depth row step {step} is smaller than packed width {packed_step}.")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected_size = height * step
    if raw.size < expected_size:
        raise ValueError(f"Depth buffer has {raw.size} bytes; expected at least {expected_size}.")
    packed = np.ascontiguousarray(raw[:expected_size].reshape(height, step)[:, :packed_step])
    source_dtype = np.dtype(">u2" if is_bigendian else "<u2")
    return np.ascontiguousarray(packed.view(source_dtype).reshape(height, width), dtype=np.uint16)


def resize_depth_nearest(depth: np.ndarray, output_size: tuple[int, int] = (96, 128)) -> np.ndarray:
    """Resize a Z16 depth map with nearest-neighbor sampling and append a channel axis."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"Expected a two-dimensional uint16 depth map, received {depth.shape} {depth.dtype}.")
    output_height, output_width = output_size
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"Output dimensions must be positive, received {output_size}.")
    input_height, input_width = depth.shape
    rows = np.floor(np.arange(output_height) * input_height / output_height).astype(np.intp)
    columns = np.floor(np.arange(output_width) * input_width / output_width).astype(np.intp)
    resized = depth[rows[:, None], columns[None, :]]
    return np.ascontiguousarray(resized[..., None], dtype=np.uint16)


def validate_policy_state(values: np.ndarray, expected_size: int = 45) -> np.ndarray:
    """Return a contiguous policy state after enforcing the PI0.5 contract."""
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (expected_size,):
        raise ValueError(f"Expected policy state shape ({expected_size},), received {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError("Policy state contains NaN or infinity.")
    return np.ascontiguousarray(state)


class RosRgbdReader:
    """Blocking synchronized ROS 2 RGB-D reader with no VLA dependency."""

    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        timeout: float,
        sync_slop: float,
        sync_queue_size: int,
        depth_size: tuple[int, int],
    ) -> None:
        try:
            import message_filters
            import rclpy
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ROS camera input requires rclpy, message_filters, and sensor_msgs. "
                "Source /opt/ros/humble/setup.bash and use Python 3.10."
            ) from error
        if timeout <= 0.0 or sync_slop <= 0.0 or sync_queue_size <= 0:
            raise ValueError("Camera timeout, sync slop, and sync queue size must be positive.")

        self._rclpy = rclpy
        self._timeout = timeout
        self._depth_size = depth_size
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=[])
        self._node = rclpy.create_node("quadloco_deploy_rgbd_reader")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_subscriber = message_filters.Subscriber(self._node, Image, rgb_topic, qos_profile=qos)
        self._depth_subscriber = message_filters.Subscriber(self._node, Image, depth_topic, qos_profile=qos)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_subscriber, self._depth_subscriber], queue_size=sync_queue_size, slop=sync_slop
        )
        self._synchronizer.registerCallback(self._callback)
        print(f"[CAMERA] Waiting for synchronized RGB={rgb_topic}, depth={depth_topic}")

    def _callback(self, rgb_message: Any, depth_message: Any) -> None:
        rgb = image_bytes_to_rgb(
            rgb_message.data,
            height=rgb_message.height,
            width=rgb_message.width,
            step=rgb_message.step,
            encoding=rgb_message.encoding,
        )
        depth = image_bytes_to_depth_z16(
            depth_message.data,
            height=depth_message.height,
            width=depth_message.width,
            step=depth_message.step,
            encoding=depth_message.encoding,
            is_bigendian=bool(depth_message.is_bigendian),
        )
        self._latest = (rgb, resize_depth_nearest(depth, self._depth_size))

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + self._timeout
        self._latest = None
        while self._latest is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"No synchronized RGB-D pair received within {self._timeout:.1f}s.")
            self._rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))
        return self._latest

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


class OpenCvRgbReader:
    """Compatibility reader for RGB-only checkpoints."""

    def __init__(self, source: str) -> None:
        camera_source: int | str = int(source) if source.isdecimal() else source
        self._capture = cv2.VideoCapture(camera_source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera source {source!r}.")

    def read(self) -> tuple[np.ndarray, None]:
        ok, bgr = self._capture.read()
        if not ok:
            raise RuntimeError("Camera frame acquisition failed.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None

    def close(self) -> None:
        self._capture.release()


class RecordedRgbdReader:
    """Return a recorded RGB image and optional Z16 depth frame."""

    def __init__(
        self,
        rgb_path: str,
        depth_path: str | None,
        depth_size: tuple[int, int],
    ) -> None:
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read recorded RGB image: {rgb_path}")
        self._rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        self._depth: np.ndarray | None = None
        if depth_path is not None:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Could not read recorded depth image: {depth_path}")
            if depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError(
                    f"Recorded depth must be a single-channel uint16 PNG; got {depth.shape} {depth.dtype}."
                )
            self._depth = resize_depth_nearest(depth, depth_size)
        print(
            f"[CAMERA] Loaded recorded RGB={rgb_path}, "
            f"depth={depth_path if depth_path is not None else 'not provided'}"
        )

    def read(self) -> tuple[np.ndarray, np.ndarray | None]:
        return self._rgb.copy(), None if self._depth is None else self._depth.copy()

    def close(self) -> None:
        pass
