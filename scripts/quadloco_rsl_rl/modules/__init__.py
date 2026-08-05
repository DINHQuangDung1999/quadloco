"""Locally vendored vision modules from yang-zj1026/legged-loco.

Only the custom depth-policy files are vendored.  Standard actor-critic,
recurrent-memory, and utility implementations continue to come from the
installed ``rsl_rl`` package.
"""

from .actor_critic_depth_cnn import ActorCriticDepthCNN, ActorCriticDepthCNNRecurrent
from .register import register_vision_modules

__all__ = [
    "ActorCriticDepthCNN",
    "ActorCriticDepthCNNRecurrent",
    "register_vision_modules",
]
