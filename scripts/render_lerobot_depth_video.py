#!/usr/bin/env python3
"""Render one episode of Z16 depth from a LeRobot dataset as an H.264 MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEPTH_KEY = "observation.depth.camera1"
DEPTH_SCALE_KEY = "observation.depth_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root of the LeRobot dataset.")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index.")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=None, help="Override the dataset FPS.")
    parser.add_argument("--min-depth", type=float, default=0.3, help="Near visualization limit in metres.")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Far visualization limit in metres.")
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="Nearest-neighbour display enlargement. Four turns 128x96 into 512x384.",
    )
    return parser.parse_args()


def load_episode(dataset: Path, episode_index: int) -> tuple[list[np.ndarray], float]:
    parquet_files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data Parquet files found below {dataset / 'data'}")

    rows: list[tuple[int, np.ndarray, float]] = []
    columns = ["index", "episode_index", DEPTH_KEY, DEPTH_SCALE_KEY]
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=columns)
        table = table.filter(pc.equal(table["episode_index"], episode_index))
        for row in table.to_pylist():
            depth = np.asarray(row[DEPTH_KEY], dtype=np.uint16)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                raise ValueError(f"Expected a 2-D Z16 depth image, got {depth.shape}")
            scale = row[DEPTH_SCALE_KEY]
            if isinstance(scale, list):
                scale = scale[0]
            rows.append((int(row["index"]), depth, float(scale)))

    if not rows:
        raise ValueError(f"Episode {episode_index} was not found in {dataset}")

    rows.sort(key=lambda item: item[0])
    scales = np.asarray([row[2] for row in rows], dtype=np.float64)
    if not np.allclose(scales, scales[0]):
        raise ValueError("Depth scale changes within the episode; this renderer expects one canonical scale.")
    return [row[1] for row in rows], float(scales[0])


def colorize_depth(depth_z16: np.ndarray, scale: float, near: float, far: float) -> np.ndarray:
    depth_m = depth_z16.astype(np.float32) * scale
    valid = depth_z16 != 0
    normalized = np.clip((depth_m - near) / (far - near), 0.0, 1.0)

    # Invert TURBO so nearby geometry is warm and distant geometry is cool.
    display = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth must be smaller than --max-depth")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")

    info = json.loads((dataset / "meta" / "info.json").read_text())
    fps = args.fps if args.fps is not None else float(info["fps"])
    depth_frames, depth_scale = load_episode(dataset, args.episode)

    sample_height, sample_width = depth_frames[0].shape
    output_width = sample_width * args.upscale
    output_height = sample_height * args.upscale
    output = args.output
    if output is None:
        output = dataset / f"depth_episode_{args.episode:03d}_h264.mp4"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{output_width}x{output_height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for depth_z16 in depth_frames:
            frame = colorize_depth(depth_z16, depth_scale, args.min_depth, args.max_depth)
            if args.upscale != 1:
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed to encode the depth video.")

    print(
        f"Rendered episode {args.episode} ({len(depth_frames)} frames, scale={depth_scale:g} m/unit) "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
