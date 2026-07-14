# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from .rough_env_cfg import UnitreeGo2RoughEnvCfg, UnitreeGo2RoughPINNEnvCfg, UnitreeGo2RoughFLEXEnvCfg
from ..velocity_env_cfg import LocomotionVelocityRoughEnvCfg

@configclass
class UnitreeGo2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None