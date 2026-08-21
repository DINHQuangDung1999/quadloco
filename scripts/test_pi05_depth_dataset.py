#!/usr/bin/env python

"""Integration test for PI0.5 depth encoding with a real QuadLoco sample.

Run from the QuadLoco repository with:

    PYTHONPATH=third_party/lerobot/src pytest -q scripts/test_pi05_depth_dataset.py

Set ``QUADLOCO_VLA_DATASET`` to test a dataset at a non-default location.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.policies.pi05.depth_encoder import PI05DepthEncoder, PI05DepthEncoderConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


DEPTH_KEY = "observation.depth.camera1"
DEPTH_SCALE_KEY = "observation.depth_scale"
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "Dataset"
    / "DinhQuangDung"
    / "quadloco-vla-all-rgbd"
)


def _dataset_path() -> Path:
    return Path(os.environ.get("QUADLOCO_VLA_DATASET", DEFAULT_DATASET)).expanduser().resolve()


def _load_first_depth_frame(
    dataset: Path,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")

    info = json.loads(info_path.read_text())
    feature = info["features"][DEPTH_KEY]
    expected_shape = tuple(feature["shape"])
    if feature["dtype"] != "uint16":
        raise ValueError(
            f"{DEPTH_KEY} must contain Z16 uint16 depth, received {feature['dtype']!r}"
        )
    scale_feature = info["features"][DEPTH_SCALE_KEY]
    if scale_feature["dtype"] != "float32" or tuple(scale_feature["shape"]) != (1,):
        raise ValueError(f"Unexpected {DEPTH_SCALE_KEY} feature: {scale_feature}")

    parquet_files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet samples found below {dataset / 'data'}")

    table = pq.read_table(parquet_files[0], columns=[DEPTH_KEY, DEPTH_SCALE_KEY]).slice(0, 1)
    depth_hwc = np.asarray(table.column(DEPTH_KEY)[0].as_py(), dtype=np.uint16)
    depth_scale = np.asarray(table.column(DEPTH_SCALE_KEY)[0].as_py(), dtype=np.float32)
    if depth_hwc.shape != expected_shape:
        raise ValueError(
            f"Metadata declares depth shape {expected_shape}, but sample has {depth_hwc.shape}"
        )
    if depth_hwc.ndim != 3 or depth_hwc.shape[-1] != 1:
        raise ValueError(f"Expected HWC single-channel depth, received {depth_hwc.shape}")

    return torch.from_numpy(depth_hwc).unsqueeze(0), torch.from_numpy(depth_scale).unsqueeze(0), expected_shape


def test_real_quadloco_depth_forward_and_backward():
    depth_z16, depth_scale, dataset_shape = _load_first_depth_frame(_dataset_path())
    assert tuple(depth_z16.shape) == (1, dataset_shape[0], dataset_shape[1], 1)
    assert depth_z16.dtype == torch.uint16
    assert depth_scale.numel() == 1
    torch.testing.assert_close(depth_scale.reshape(-1), torch.tensor([0.001]))

    # Exercise the real policy boundary, including Z16 -> metric conversion
    # and HWC -> CHW layout normalization.
    policy = object.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("_depth_test_parameter", torch.nn.Parameter(torch.zeros(())))
    policy.config = SimpleNamespace(
        depth_enabled=True,
        depth_feature_key=DEPTH_KEY,
        depth_scale_feature_key=DEPTH_SCALE_KEY,
        depth_default_scale=0.001,
        image_resolution=(224, 224),
        depth_resize_with_rgb=True,
    )
    depth, depth_batch_mask = policy._preprocess_depth(
        {DEPTH_KEY: depth_z16, DEPTH_SCALE_KEY: depth_scale.reshape(1, 1)}
    )
    assert tuple(depth.shape) == (1, 1, 224, 224)
    assert depth_batch_mask.tolist() == [True]
    assert torch.isfinite(depth).all()
    assert (depth > 0).any()
    assert (depth == 0).any()  # Invalid letterbox padding.

    config = PI05DepthEncoderConfig(
        depths=(1, 1, 2),
        dims=(32, 64, 128),
        output_grid_size=(8, 8),
        min_depth=0.05,
        max_depth=20.0,
    )
    encoder = PI05DepthEncoder(output_dim=256, config=config)

    # Open the residual gate so this integration test exercises the ConvNeXt
    # encoder and projection gradients, rather than only the gate parameter.
    with torch.no_grad():
        encoder.output_gate.fill_(0.1)

    tokens, token_mask = encoder(depth)

    assert tokens.shape == (1, 64, 256)
    assert token_mask.shape == (1, 64)
    assert token_mask.dtype == torch.bool
    assert token_mask.all()
    assert torch.isfinite(tokens).all()
    assert tokens.abs().sum() > 0

    tokens.square().mean().backward()
    assert encoder.stem[0].weight.grad is not None
    assert torch.isfinite(encoder.stem[0].weight.grad).all()
    assert encoder.output_projection.weight.grad is not None
