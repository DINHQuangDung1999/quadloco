"""Depth CNN actor-critic adapted from yang-zj1026/legged-loco."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.actor_critic_recurrent import Memory

from .depth_backbone import DepthOnlyFCBackbone


class ActorDepthCNN(nn.Module):
    def __init__(
        self,
        num_obs_proprio,
        obs_depth_shape,
        num_actions,
        activation,
        hidden_dims=(256, 256, 128),
    ):
        super().__init__()
        self.prop_mlp = nn.Sequential(
            nn.Linear(num_obs_proprio, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation,
        )
        self.depth_backbone = DepthOnlyFCBackbone(
            output_dim=hidden_dims[2],
            hidden_dim=hidden_dims[1],
            activation=activation,
            num_frames=1,
        )
        self.action_head = nn.Linear(2 * hidden_dims[2], num_actions)
        self.num_obs_proprio = num_obs_proprio
        self.obs_depth_shape = tuple(obs_depth_shape)

    def forward(self, observations):
        proprio = observations[..., : self.num_obs_proprio]
        depth = observations[..., self.num_obs_proprio :].reshape(-1, *self.obs_depth_shape)
        latent = torch.cat((self.prop_mlp(proprio), self.depth_backbone(depth)), dim=-1)
        return self.action_head(latent)

    def encode(self, observations):
        original_shape = observations.shape
        if observations.ndim == 3:
            observations = observations.reshape(-1, original_shape[-1])
        proprio = observations[..., : self.num_obs_proprio]
        depth = observations[..., self.num_obs_proprio :].reshape(-1, *self.obs_depth_shape)
        latent = torch.cat((self.prop_mlp(proprio), self.depth_backbone(depth)), dim=-1)
        if len(original_shape) == 3:
            latent = latent.reshape(*original_shape[:-1], -1)
        return latent

    def reset(self, dones=None):
        if dones is not None:
            self.depth_backbone.reset(dones)


class ActorCriticDepthCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_actor_obs_prop=45,
        obs_depth_shape=(50, 60),
        actor_hidden_dims=(256, 256, 128),
        critic_hidden_dims=(256, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__()
        if num_actor_obs != num_actor_obs_prop + int(torch.tensor(obs_depth_shape).prod()):
            raise ValueError(
                "Actor observation size does not match proprioception + depth: "
                f"{num_actor_obs} != {num_actor_obs_prop} + {tuple(obs_depth_shape)}."
            )
        if kwargs:
            print(f"Ignoring unsupported ActorCriticDepthCNN arguments: {tuple(kwargs)}")

        activation_module = get_activation(activation)
        self.actor = ActorDepthCNN(
            num_actor_obs_prop,
            obs_depth_shape,
            num_actions,
            activation_module,
            actor_hidden_dims,
        )

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_module]
        for index, hidden_dim in enumerate(critic_hidden_dims):
            output_dim = 1 if index == len(critic_hidden_dims) - 1 else critic_hidden_dims[index + 1]
            critic_layers.append(nn.Linear(hidden_dim, output_dim))
            if index != len(critic_hidden_dims) - 1:
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        self.actor.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


class ActorCriticDepthCNNRecurrent(ActorCriticDepthCNN):
    """Recurrent variant retained for compatibility with the upstream module."""

    is_recurrent = True

    def __init__(
        self,
        *args,
        rnn_type="lstm",
        rnn_input_size=256,
        rnn_hidden_size=256,
        rnn_num_layers=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_a = Memory(
            rnn_input_size,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        encoded = self.actor.encode(observations)
        hidden = self.memory_a(encoded, masks, hidden_states)
        self.distribution = Normal(self.actor.action_head(hidden.squeeze(0)), self.std)
        return self.distribution.sample()

    def act_inference(self, observations):
        encoded = self.actor.encode(observations)
        hidden = self.memory_a(encoded)
        return self.actor.action_head(hidden.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, None
