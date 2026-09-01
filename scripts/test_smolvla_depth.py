#!/usr/bin/env python

"""Focused tests for metric-depth fusion in SmolVLA."""

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.pi05.depth_encoder import PI05DepthEncoder, PI05DepthEncoderConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching


DEPTH_KEY = "observation.depth.camera1"
DEPTH_SCALE_KEY = "observation.depth_scale"


def _preprocessing_probe() -> SmolVLAPolicy:
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.register_parameter("_depth_test_parameter", nn.Parameter(torch.zeros(())))
    policy.config = SimpleNamespace(
        depth_enabled=True,
        depth_feature_key=DEPTH_KEY,
        depth_scale_feature_key=DEPTH_SCALE_KEY,
        depth_default_scale=0.001,
        depth_resize_with_rgb=True,
        resize_imgs_with_padding=(16, 16),
    )
    return policy


def test_smolvla_decodes_z16_depth_and_preserves_invalid_padding():
    policy = _preprocessing_probe()
    z16 = torch.tensor([[[[1000], [2500]]]], dtype=torch.uint16)

    depth, mask = policy._preprocess_depth(
        {DEPTH_KEY: z16, DEPTH_SCALE_KEY: torch.tensor([[0.001]])}
    )

    assert depth.shape == (1, 1, 16, 16)
    assert mask.tolist() == [True]
    assert depth.max() == 2.5
    assert (depth == 0).any()


def test_smolvla_depth_uses_latest_observation_step():
    policy = _preprocessing_probe()
    z16 = torch.tensor([[[[[1000]]], [[[2500]]]]], dtype=torch.uint16)

    depth, _ = policy._preprocess_depth(
        {DEPTH_KEY: z16, DEPTH_SCALE_KEY: torch.tensor([[0.001]])}
    )

    assert depth.max() == 2.5
    assert not torch.isclose(depth, torch.tensor(1.0)).any()


class _FakeVLM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.hidden_size = hidden_size

    def embed_image(self, image):
        batch = image.shape[0]
        return torch.ones(batch, 4, self.hidden_size, device=image.device)

    def embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.hidden_size, device=tokens.device)


def _fusion_probe(gate: float) -> VLAFlowMatching:
    model = object.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(depth_fusion_mode="pairwise_add")
    model.vlm_with_expert = _FakeVLM(hidden_size=8)
    model.state_proj = nn.Linear(3, 8)
    model.depth_encoder = PI05DepthEncoder(
        output_dim=8,
        config=PI05DepthEncoderConfig(
            depths=(1,), dims=(8,), patch_size=2, output_grid_size=(2, 2)
        ),
    )
    model.depth_encoder.output_gate.data.fill_(gate)
    model.depth_cross_attention = None
    model._last_depth_fusion_metrics = {}
    model.add_image_special_tokens = False
    model.prefix_length = -1
    return model


def _embed(model):
    return model.embed_prefix(
        images=[torch.zeros(2, 3, 8, 8)],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        lang_tokens=torch.ones(2, 2, dtype=torch.long),
        lang_masks=torch.ones(2, 2, dtype=torch.bool),
        state=torch.zeros(2, 3),
        depth=torch.rand(2, 1, 8, 8) + 0.1,
        depth_mask=torch.ones(2, dtype=torch.bool),
    )[0]


def test_zero_gate_preserves_smolvla_rgb_tokens():
    model = _fusion_probe(gate=0.0)
    prefix = _embed(model)

    # SmolVLA scales its 8-dimensional image embedding by sqrt(8).
    torch.testing.assert_close(prefix[:, :4], torch.full_like(prefix[:, :4], 8**0.5))
    assert model._last_depth_fusion_metrics["depth/effective_gate"] == 0


def test_pairwise_depth_fusion_backpropagates_to_gate_and_encoder():
    model = _fusion_probe(gate=0.1)
    prefix = _embed(model)
    prefix[:, :4].square().mean().backward()

    assert model.depth_encoder.output_gate.grad is not None
    assert model.depth_encoder.output_projection.weight.grad is not None
    assert torch.isfinite(model.depth_encoder.output_projection.weight.grad).all()


def test_state_preparation_selects_30d_proprioception():
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    indices = (*range(24), *range(36, 42))
    policy.config = SimpleNamespace(
        state_feature_indices=indices, state_token_dim=30, max_state_dim=32
    )
    state = torch.arange(45, dtype=torch.float32).reshape(1, 45)

    prepared = policy.prepare_state({"observation.state": state})

    assert prepared.shape == (1, 32)
    torch.testing.assert_close(prepared[:, :30], state[:, list(indices)])
    torch.testing.assert_close(prepared[:, 30:], torch.zeros(1, 2))


def test_pretrained_state_projection_can_be_widened(tmp_path):
    class _CheckpointProbe(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.model = nn.Module()
            self.model.state_proj = nn.Linear(width, 3)

    source = _CheckpointProbe(width=32)
    destination = _CheckpointProbe(width=42)
    with torch.no_grad():
        source.model.state_proj.weight.copy_(
            torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)
        )
    checkpoint = tmp_path / "model.safetensors"
    save_file(source.state_dict(), checkpoint)

    SmolVLAPolicy._load_as_safetensor(destination, str(checkpoint), "cpu", strict=False)

    torch.testing.assert_close(
        destination.model.state_proj.weight[:, :32], source.model.state_proj.weight
    )
    assert destination.model.state_proj.weight.shape == (3, 42)
