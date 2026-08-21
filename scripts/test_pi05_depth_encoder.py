#!/usr/bin/env python

"""Focused tests for the optional PI0.5 depth encoder."""

import pytest
import torch
from types import SimpleNamespace

from lerobot.policies.pi05.depth_encoder import (
    PI05DepthCrossAttention,
    PI05DepthEncoder,
    PI05DepthEncoderConfig,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def test_depth_encoder_returns_prefix_tokens_and_mask():
    config = PI05DepthEncoderConfig(
        depths=(1, 1),
        dims=(16, 32),
        output_grid_size=(4, 5),
    )
    encoder = PI05DepthEncoder(output_dim=48, config=config)
    depth = torch.rand(2, 1, 64, 80) * config.max_depth

    tokens, mask = encoder(depth)

    assert tokens.shape == (2, 20, 48)
    assert mask.shape == (2, 20)
    assert mask.dtype == torch.bool
    assert mask.all()
    # The zero gate preserves pretrained PI0.5 behavior at initialization.
    torch.testing.assert_close(tokens, torch.zeros_like(tokens))

    # Pairwise fusion uses these ungated tokens and applies the gate once at
    # the RGB fusion boundary.
    raw_tokens, raw_mask = encoder.encode_tokens(depth)
    assert raw_tokens.shape == tokens.shape
    assert raw_mask.equal(mask)
    assert raw_tokens.abs().sum() > 0
    rgb_tokens = torch.randn_like(raw_tokens)
    fused_at_zero = rgb_tokens + torch.tanh(encoder.output_gate) * raw_tokens
    torch.testing.assert_close(fused_at_zero, rgb_tokens)

    encoder.output_gate.data.fill_(0.2)
    fused = rgb_tokens + torch.tanh(encoder.output_gate) * raw_tokens
    torch.testing.assert_close(fused - rgb_tokens, torch.tanh(encoder.output_gate) * raw_tokens)


def test_depth_encoder_handles_invalid_depth_and_backpropagates():
    config = PI05DepthEncoderConfig(depths=(1,), dims=(8,), output_grid_size=(2, 2))
    encoder = PI05DepthEncoder(output_dim=12, config=config)
    encoder.output_gate.data.fill_(0.1)
    depth = torch.tensor([[[[float("nan"), 0.0], [float("inf"), 2.0]]]]).repeat(1, 1, 8, 8)

    tokens, _ = encoder(depth)
    tokens.square().mean().backward()

    assert torch.isfinite(tokens).all()
    assert encoder.output_projection.weight.grad is not None


def test_depth_encoder_rejects_rgb_input():
    encoder = PI05DepthEncoder(output_dim=16)

    with pytest.raises(ValueError, match="Expected depth shape"):
        encoder(torch.zeros(1, 3, 64, 64))


def test_depth_cross_attention_supports_different_token_counts():
    adapter = PI05DepthCrossAttention(embed_dim=32, num_heads=4)
    rgb = torch.randn(2, 16, 32)
    depth = torch.randn(2, 6, 32)
    depth_mask = torch.ones(2, 6, dtype=torch.bool)

    context = adapter(rgb, depth, depth_mask)

    assert context.shape == rgb.shape
    torch.testing.assert_close(
        context.float().square().mean(dim=-1).sqrt().mean(),
        torch.tensor(1.0),
        atol=1e-4,
        rtol=1e-4,
    )
    context.square().mean().backward()
    assert adapter.attention.in_proj_weight.grad is not None


def _preprocessing_probe() -> PI05Policy:
    policy = object.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("_depth_test_parameter", torch.nn.Parameter(torch.zeros(())))
    policy.config = SimpleNamespace(
        depth_enabled=True,
        depth_feature_key="observation.depth.camera1",
        depth_scale_feature_key="observation.depth_scale",
        depth_default_scale=0.001,
        image_resolution=(224, 224),
        depth_resize_with_rgb=True,
    )
    return policy


def test_z16_depth_is_decoded_with_per_sample_scale():
    policy = _preprocessing_probe()
    z16 = torch.tensor(
        [[[[1000], [2500]]], [[[200], [400]]]],
        dtype=torch.uint16,
    )
    scale = torch.tensor([[0.001], [0.01]], dtype=torch.float32)

    depth, mask = policy._preprocess_depth(
        {
            "observation.depth.camera1": z16,
            "observation.depth_scale": scale,
        }
    )

    assert depth.shape == (2, 1, 224, 224)
    torch.testing.assert_close(
        depth[0, 0, 112], torch.cat((torch.full((112,), 1.0), torch.full((112,), 2.5)))
    )
    torch.testing.assert_close(
        depth[1, 0, 112], torch.cat((torch.full((112,), 2.0), torch.full((112,), 4.0)))
    )
    assert mask.tolist() == [True, True]


def test_metric_float_depth_without_scale_is_not_rescaled():
    policy = _preprocessing_probe()
    metric = torch.tensor([[[1.25, 2.0]]], dtype=torch.float32)

    depth, _ = policy._preprocess_depth({"observation.depth.camera1": metric})

    assert depth.shape == (1, 1, 224, 224)
    assert depth.max() == 2.0
    assert depth.min() == 0.0
