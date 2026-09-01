# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from ..navigation_env_cfg import (
    GoalNavigationEnvCfg,
    NearFarGoalNavigationEnvCfg,
    TwoObjectNearFarGoalNavigationEnvCfg,
    ObjectRelativeGoalNavigationEnvCfg,
    OccludedGoalNavigationEnvCfg,
    RelationalGoalNavigationEnvCfg,
)

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 1.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False

@configclass
class UnitreeGo2RoughNavEnvCfg_PLAY(GoalNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0,0.7)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 1.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RoughOccludedNavEnvCfg_PLAY(OccludedGoalNavigationEnvCfg):
    """Occluded-object navigation with the same velocity-command interface."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RoughRelationalNavEnvCfg_PLAY(RelationalGoalNavigationEnvCfg):
    """Relational target-selection environment with deterministic robot reset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RoughNearFarNavEnvCfg_PLAY(NearFarGoalNavigationEnvCfg):
    """Near/far relational environment with deterministic robot reset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RoughTwoObjectNearFarNavEnvCfg_PLAY(
    TwoObjectNearFarGoalNavigationEnvCfg
):
    """Two-object near/far environment with deterministic robot reset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RoughObjectRelativeNavEnvCfg_PLAY(
    ObjectRelativeGoalNavigationEnvCfg
):
    """Metric object-relative environment with deterministic robot reset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
