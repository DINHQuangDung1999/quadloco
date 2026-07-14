

from __future__ import annotations

from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING, Literal
import omni.usd
from isaaclab.assets import RigidObject,Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.actuators import DCMotor
from isaaclab_quadlocofault.actuators import CustomDCMotor
from isaaclab.sensors import RayCasterCamera
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import  ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


def randomize_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    strength_distribution_params: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scale joint effort limits to approximate motor-strength randomization.

    This updates both the articulation effort limits in simulation and the actuator-side
    effort clipping tensor used by the DC motor model.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
        num_joints = asset.num_joints
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)
        num_joints = len(joint_ids)

    scales = torch.empty((len(env_ids), num_joints), device=asset.device).uniform_(
        strength_distribution_params[0], strength_distribution_params[1]
    )
    default_limits = asset.data.joint_effort_limits[env_ids].clone()
    scaled_limits = default_limits * scales
    asset.write_joint_effort_limit_to_sim(scaled_limits, joint_ids=joint_ids, env_ids=env_ids)

    # Keep actuator-side clipping consistent with the randomized effort limits.
    for actuator in asset.actuators.values():
        actuator_joint_ids = actuator.joint_indices
        actuator.effort_limit[env_ids] = asset.data.joint_effort_limits[env_ids][:, actuator_joint_ids]
        if hasattr(actuator, "effort_limit_sim"):
            actuator.effort_limit_sim[env_ids] = asset.data.joint_effort_limits[env_ids][:, actuator_joint_ids]
