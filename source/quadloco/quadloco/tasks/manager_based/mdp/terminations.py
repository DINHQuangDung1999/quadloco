from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class goal_reached_for_duration(ManagerTermBase):
    """Terminate a fixed duration after the command's goal is first reached."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        duration_s = cfg.params.get("duration_s", 0.0)
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, received {duration_s}.")
        self.time_at_goal = torch.zeros(env.num_envs, device=env.device)
        self.reached_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.goal_generation = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.time_at_goal[env_ids] = 0.0
        self.reached_once[env_ids] = False
        command_name = self.cfg.params["command_name"]
        command_term = self._env.command_manager.get_term(command_name)
        self.goal_generation[env_ids] = command_term.goal_generation[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        duration_s: float,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        reached = command_term.goal_reached

        # A command may also resample during an episode. Never carry dwell time
        # from the previous goal into the newly sampled one.
        new_goal = self.goal_generation != command_term.goal_generation
        self.time_at_goal[new_goal] = 0.0
        self.reached_once[new_goal] = False
        self.goal_generation[:] = command_term.goal_generation

        # Require continuous residence in the goal region. Latching the first
        # crossing can save an episode after the robot has drifted back out,
        # which creates contradictory stopping demonstrations.
        self.reached_once[:] = reached
        self.time_at_goal[:] = torch.where(
            reached,
            self.time_at_goal + env.step_dt,
            torch.zeros_like(self.time_at_goal),
        )
        return self.time_at_goal >= duration_s
