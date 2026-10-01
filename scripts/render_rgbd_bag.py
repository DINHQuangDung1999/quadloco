#!/usr/bin/env python3
"""Render timestamp-aligned RGB and depth topics from a ROS 2 bag to MP4."""

import argparse
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rgb-topic", default="/d435i/color/image_raw")
    parser.add_argument("--depth-topic", default="/d435i/aligned_depth_to_color/image_raw")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-depth-mm", type=float, default=5000.0)
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.rgb_topic, args.depth_topic]))
    bridge = CvBridge()
    rgb_frames = []
    depth_frames = []
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        image = deserialize_message(data, Image)
        if topic == args.rgb_topic:
            rgb = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            rgb_frames.append((timestamp, rgb.copy()))
        elif topic == args.depth_topic:
            depth = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            depth_frames.append((timestamp, depth.copy()))

    if not rgb_frames or not depth_frames:
        raise RuntimeError(f"Need both RGB and depth frames; found {len(rgb_frames)} and {len(depth_frames)}")

    depth_times = [item[0] for item in depth_frames]
    height, width = rgb_frames[0][1].shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    for timestamp, rgb in rgb_frames:
        index = bisect_left(depth_times, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(depth_frames)]
        nearest = min(candidates, key=lambda i: abs(depth_times[i] - timestamp))
        depth = depth_frames[nearest][1].astype(np.float32)
        valid = depth > 0
        scaled = np.clip(depth, 0, args.max_depth_mm) * (255.0 / args.max_depth_mm)
        depth_u8 = scaled.astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        if depth_color.shape[:2] != (height, width):
            depth_color = cv2.resize(depth_color, (width, height), interpolation=cv2.INTER_NEAREST)
        writer.write(np.hstack((rgb, depth_color)))
    writer.release()
    print(f"Wrote {len(rgb_frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
