#!/usr/bin/env python
"""Focused tests for QuadLoco's hybrid SigLIP LoRA implementation."""

import torch
from torch import nn

from lerobot.policies.smolvla.smolvlm_with_expert import (
    VisionLoRALinear,
    inject_siglip_attention_lora,
)


def _fake_vision_model(num_layers: int = 2, hidden_size: int = 8) -> nn.Module:
    layers = []
    for _ in range(num_layers):
        attention = nn.Module()
        attention.q_proj = nn.Linear(hidden_size, hidden_size)
        attention.k_proj = nn.Linear(hidden_size, hidden_size)
        attention.v_proj = nn.Linear(hidden_size, hidden_size)
        attention.out_proj = nn.Linear(hidden_size, hidden_size)
        layer = nn.Module()
        layer.self_attn = attention
        layers.append(layer)
    encoder = nn.Module()
    encoder.layers = nn.ModuleList(layers)
    model = nn.Module()
    model.encoder = encoder
    return model


def test_zero_initialized_lora_preserves_base_output() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    inputs = torch.randn(2, 3, 8)
    expected = base(inputs)
    adapted = VisionLoRALinear.from_linear(base, rank=2, alpha=2.0, dropout=0.0)
    torch.testing.assert_close(adapted(inputs), expected)


def test_injection_count_parameters_and_gradients() -> None:
    model = _fake_vision_model()
    projections, parameters = inject_siglip_attention_lora(
        model,
        rank=2,
        alpha=2.0,
        dropout=0.0,
        targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    )
    assert projections == 2 * 4
    assert parameters == 2 * 4 * 2 * (8 + 8)

    for module in model.modules():
        if isinstance(module, VisionLoRALinear):
            assert not module.weight.requires_grad
            assert module.bias is None or not module.bias.requires_grad
            assert module.lora_A.weight.requires_grad
            assert module.lora_B.weight.requires_grad


def test_legacy_projection_keys_are_preserved() -> None:
    model = _fake_vision_model(num_layers=1)
    inject_siglip_attention_lora(
        model,
        rank=2,
        alpha=2.0,
        dropout=0.0,
        targets=("q_proj",),
    )
    keys = set(model.state_dict())
    assert "encoder.layers.0.self_attn.q_proj.weight" in keys
    assert "encoder.layers.0.self_attn.q_proj.bias" in keys
    assert "encoder.layers.0.self_attn.q_proj.lora_A.weight" in keys
    assert "encoder.layers.0.self_attn.q_proj.lora_B.weight" in keys


if __name__ == "__main__":
    test_zero_initialized_lora_preserves_base_output()
    test_injection_count_parameters_and_gradients()
    test_legacy_projection_keys_are_preserved()
    print("SmolVLA vision LoRA tests passed.")
