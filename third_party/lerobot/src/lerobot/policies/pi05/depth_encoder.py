#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ConvNeXt-style depth token encoder for PI0.5.

The encoder turns a metric depth map into a compact sequence of tokens in the
PaliGemma embedding space. These tokens are intended to be appended to the
PI0.5 prefix alongside the existing SigLIP RGB tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PI05DepthEncoderConfig:
    """Configuration for :class:`PI05DepthEncoder`."""

    input_channels: int = 1
    depths: tuple[int, ...] = (2, 2, 4)
    dims: tuple[int, ...] = (64, 128, 256)
    patch_size: int = 4
    output_grid_size: tuple[int, int] = (8, 8)
    max_depth: float = 10.0
    min_depth: float = 0.05
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.depths) != len(self.dims) or not self.depths:
            raise ValueError("depths and dims must have the same non-zero length")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.min_depth <= 0 or self.max_depth <= self.min_depth:
            raise ValueError("Expected 0 < min_depth < max_depth")


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_probability)
        return x * mask / keep_probability


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels of a channels-first feature map."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt block operating on a depth feature map."""

    def __init__(
        self,
        dim: int,
        drop_path: float,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise_conv(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = F.gelu(x)
        x = self.pointwise_conv2(x)
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class PI05DepthEncoder(nn.Module):
    """Encode ``[B, 1, H, W]`` metric depth maps as PI0.5 prefix tokens.

    Invalid values (NaN, infinity, non-positive depth) are replaced by
    ``max_depth``. Valid depths are clipped, converted to inverse-depth, and
    normalized to ``[0, 1]``. The returned mask has one value per token and can
    be concatenated directly with PI0.5's prefix padding mask.
    """

    def __init__(self, output_dim: int, config: PI05DepthEncoderConfig | None = None):
        super().__init__()
        self.config = config or PI05DepthEncoderConfig()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                self.config.dims[0],
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            ),
            LayerNorm2d(self.config.dims[0], eps=1e-6),
        )

        total_blocks = sum(self.config.depths)
        drop_rates = torch.linspace(0, self.config.drop_path_rate, total_blocks).tolist()
        stages = []
        block_index = 0
        for stage_index, (depth, dim) in enumerate(
            zip(self.config.depths, self.config.dims, strict=True)
        ):
            if stage_index > 0:
                stages.append(
                    nn.Sequential(
                        LayerNorm2d(self.config.dims[stage_index - 1], eps=1e-6),
                        nn.Conv2d(self.config.dims[stage_index - 1], dim, kernel_size=2, stride=2),
                    )
                )
            stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dim,
                            drop_path=drop_rates[block_index + offset],
                            layer_scale_init_value=self.config.layer_scale_init_value,
                        )
                        for offset in range(depth)
                    ]
                )
            )
            block_index += depth
        self.stages = nn.Sequential(*stages)

        self.output_norm = nn.LayerNorm(self.config.dims[-1], eps=1e-6)
        self.output_projection = nn.Linear(self.config.dims[-1], output_dim)
        self.output_gate = nn.Parameter(torch.tensor(0.0))
        self.row_position = nn.Parameter(
            torch.zeros(1, self.config.output_grid_size[0], 1, output_dim)
        )
        self.column_position = nn.Parameter(
            torch.zeros(1, 1, self.config.output_grid_size[1], output_dim)
        )
        nn.init.trunc_normal_(self.row_position, std=0.02)
        nn.init.trunc_normal_(self.column_position, std=0.02)

    def _normalize_depth(self, depth: Tensor) -> Tensor:
        valid = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid, depth, torch.full_like(depth, self.config.max_depth))
        depth = depth.clamp(self.config.min_depth, self.config.max_depth)
        inverse_depth = depth.reciprocal()
        inverse_min = 1.0 / self.config.max_depth
        inverse_max = 1.0 / self.config.min_depth
        return (inverse_depth - inverse_min) / (inverse_max - inverse_min)

    def encode_tokens(self, depth: Tensor) -> tuple[Tensor, Tensor]:
        """Encode depth into ungated tokens, for either concatenation or pairwise fusion."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected depth shape [B, {self.config.input_channels}, H, W], "
                f"received {tuple(depth.shape)}"
            )

        depth = self._normalize_depth(depth.float())
        parameter_dtype = self.stem[0].weight.dtype
        depth = depth.to(dtype=parameter_dtype)
        features = self.stages(self.stem(depth))
        features = F.adaptive_avg_pool2d(features, self.config.output_grid_size)
        features = features.permute(0, 2, 3, 1)
        tokens = self.output_projection(self.output_norm(features))
        tokens = tokens + self.row_position + self.column_position
        tokens = tokens.flatten(1, 2)
        token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, token_mask

    def forward(self, depth: Tensor) -> tuple[Tensor, Tensor]:
        """Return gated tokens, retaining the original standalone encoder API."""
        tokens, token_mask = self.encode_tokens(depth)
        return torch.tanh(self.output_gate) * tokens, token_mask


class PI05DepthCrossAttention(nn.Module):
    """Let RGB queries select context from independently encoded depth tokens."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.rgb_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.depth_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        # Fixed output normalization prevents the attention value/output
        # projections from growing while a small residual gate compensates.
        self.context_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

    def forward(
        self,
        rgb_tokens: Tensor,
        depth_tokens: Tensor,
        depth_token_mask: Tensor,
    ) -> Tensor:
        output_dtype = rgb_tokens.dtype
        attention_dtype = self.attention.in_proj_weight.dtype
        rgb_tokens = rgb_tokens.to(dtype=attention_dtype)
        depth_tokens = depth_tokens.to(dtype=attention_dtype)
        context, _ = self.attention(
            query=self.rgb_norm(rgb_tokens),
            key=self.depth_norm(depth_tokens),
            value=self.depth_norm(depth_tokens),
            key_padding_mask=~depth_token_mask,
            need_weights=False,
        )
        context = self.context_norm(context)
        return context.to(dtype=output_dtype)
