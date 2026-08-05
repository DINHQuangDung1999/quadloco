# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth-image velocity locomotion configuration for Unitree Go2.

The actor receives the existing 45-dimensional proprioceptive observation
followed by a normalized 50 x 60 perspective-depth image.  The critic retains
the privileged observations and terrain height scan from ``velocity_env_cfg``.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import quadloco.tasks.manager_based.mdp as mdp

from .velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg as VelocityObservationsCfg,
)


DEPTH_IMAGE_HEIGHT = 50
DEPTH_IMAGE_WIDTH = 60
DEPTH_NEAR_CLIP = 0.1
DEPTH_FAR_CLIP = 5.0
DEPTH_UPDATE_PERIOD = 0.05  # 20 Hz camera for a 50 Hz locomotion policy.


def normalized_depth_image(
    env,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
) -> torch.Tensor:
    """Return flattened depth in [-0.5, 0.5], with invalid pixels set to far."""

    depth = env.scene.sensors[sensor_cfg.name].data.output[data_type].clone()
    if depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    depth = torch.nan_to_num(
        depth,
        nan=DEPTH_FAR_CLIP,
        posinf=DEPTH_FAR_CLIP,
        neginf=DEPTH_NEAR_CLIP,
    )
    depth = depth.clamp(DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP)
    depth = (depth - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP) - 0.5
    return depth.flatten(start_dim=1)


@configclass
class VisionSceneCfg(MySceneCfg):
    """Rough-terrain scene augmented with a forward-facing D435i-like camera."""

    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/D435i",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.10),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=[
            "distance_to_image_plane",
            # "rgb",  # Optional: uncomment to render RGB from the same camera.
        ],
        spawn=sim_utils.PinholeCameraCfg(
            # Approximate Intel D435i color-camera horizontal field of view.
            focal_length=1.93,
            horizontal_aperture=2.65,
            clipping_range=(DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP),
        ),
        width=DEPTH_IMAGE_WIDTH,
        height=DEPTH_IMAGE_HEIGHT,
        update_period=DEPTH_UPDATE_PERIOD,
        depth_clipping_behavior="max",
    )


@configclass
class VisionObservationsCfg:
    """Actor, proprioceptive, and privileged critic observations."""

    @configclass
    class PolicyCfg(VelocityObservationsCfg.PolicyCfg):
        # Keep depth last: ActorCriticDepthCNN splits the first 45 values as
        # proprioception and reshapes the remaining 3000 values to (50, 60).
        depth = ObsTerm(
            func=normalized_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
            clip=(-0.5, 0.5),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Optional RGB observation (requires an RGB encoder, not the depth CNN):
        # rgb = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "rgb", "normalize": True},
        # )

    @configclass
    class ProprioCfg(VelocityObservationsCfg.PolicyCfg):
        """Actor observation without depth; used to verify the 45-value split."""

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: VelocityObservationsCfg.CriticCfg = VelocityObservationsCfg.CriticCfg()


@configclass
class VisionLocomotionVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Training configuration for depth-based Go2 velocity locomotion."""

    scene: VisionSceneCfg = VisionSceneCfg(num_envs=512, env_spacing=2.5)
    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.scene.depth_camera.update_period = DEPTH_UPDATE_PERIOD


@configclass
class VisionLocomotionVelocityRoughEnvCfg_PLAY(VisionLocomotionVelocityRoughEnvCfg):
    """Smaller deterministic scene for evaluating a trained depth policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
