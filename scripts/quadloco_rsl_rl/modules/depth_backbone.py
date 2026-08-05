"""Depth encoders adapted from yang-zj1026/legged-loco.

The upstream fully-connected head assumes a 24 x 32 input.  Adaptive pooling
keeps the same 4 x 6 feature size while allowing the environment to use the
requested 50 x 60 depth image.
"""

import torch
import torch.nn as nn


class DepthBackbone(nn.Module):
    def __init__(self, base_backbone, rnn_hidden_dim, output_dim) -> None:
        super().__init__()
        self.base_backbone = base_backbone
        self.rnn = nn.GRU(
            input_size=base_backbone.output_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
        )
        self.output_mlp = nn.Sequential(nn.Linear(rnn_hidden_dim, output_dim), nn.Tanh())
        self.hidden_states = None

    def forward(self, depth_input, original_shape, masks=None, hidden_states=None):
        depth_latent = self.base_backbone(depth_input)
        if masks is not None:
            if hidden_states is None:
                raise ValueError("Hidden states are required during a recurrent policy update.")
            depth_latent = depth_latent.view(*original_shape[:2], -1)
            output, _ = self.rnn(depth_latent, hidden_states)
        else:
            output, self.hidden_states = self.rnn(depth_latent.unsqueeze(0), self.hidden_states)
            output = output.squeeze(0)
        return self.output_mlp(output.squeeze(1))

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[..., dones, :] = 0.0


class DepthOnlyFCBackbone(nn.Module):
    """Small CNN that accepts arbitrary images of at least 16 x 16 pixels."""

    def __init__(self, output_dim, hidden_dim, activation, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 6, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation,
        )

    def forward(self, images: torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(1)
        return self.image_compression(images)
