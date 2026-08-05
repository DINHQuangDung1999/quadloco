"""Register locally vendored policy classes with the RSL-RL v3 runner."""

from __future__ import annotations

import importlib

from .actor_critic_depth_cnn import ActorCriticDepthCNN, ActorCriticDepthCNNRecurrent


def register_vision_modules() -> None:
    """Expose the custom classes where RSL-RL's runner resolves class names."""

    modules_package = importlib.import_module("rsl_rl.modules")
    runner_module = importlib.import_module("rsl_rl.runners.on_policy_runner")

    for module in (modules_package, runner_module):
        module.ActorCriticDepthCNN = ActorCriticDepthCNN
        module.ActorCriticDepthCNNRecurrent = ActorCriticDepthCNNRecurrent
