"""Reusable static obstacle assets for Quadloco navigation tasks."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@dataclass(frozen=True)
class OfficeObstacleSpec:
    """Visual USD and conservative cuboid planning/collision footprint."""

    name: str
    instruction_name: str
    usd_path: str
    footprint: tuple[float, float, float]


OFFICE_OBSTACLE_SPECS: tuple[OfficeObstacleSpec, ...] = (
    OfficeObstacleSpec(
        "side_table",
        "table",
        "/Environments/Hospital/Props/SM_SideTable_02a.usd",
        (0.80, 0.55, 0.75),
    ),
    OfficeObstacleSpec(
        "bedside_table",
        "table",
        "/Environments/Hospital/Props/SM_BedSideTable_01b.usd",
        (0.85, 0.55, 0.80),
    ),
    OfficeObstacleSpec(
        "desk_chair_01",
        "desk chair",
        "/Environments/Hospital/Props/SM_Chair_01a.usd",
        (0.65, 0.65, 1.00),
    ),
    OfficeObstacleSpec(
        "desk_chair_04",
        "desk chair",
        "/Environments/Hospital/Props/SM_Chair_04a.usd",
        (0.65, 0.65, 1.00),
    ),
    OfficeObstacleSpec(
        "cardboard_box_a",
        "cardboard box",
        "/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
        (0.55, 0.55, 0.55),
    ),
    OfficeObstacleSpec(
        "cardboard_box_c",
        "cardboard box",
        "/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
        (0.60, 0.60, 0.60),
    ),
    # Computer intentionally excluded from active obstacle sampling.
    # OfficeObstacleSpec(
    #     "computer",
    #     "computer",
    #     "/Environments/Hospital/Props/SM_Computer_02a.usd",
    #     (0.65, 0.35, 0.55),
    # ),
)


def add_office_obstacles_to_scene(
    scene,
    *,
    asset_prefix: str = "occlusion_obstacle",
    prim_prefix: str = "OcclusionObstacle",
    unused_height: float = -100.0,
    specs: tuple[OfficeObstacleSpec, ...] = OFFICE_OBSTACLE_SPECS,
) -> None:
    """Add independently movable office obstacles to a scene config.

    The imported environment USDs are visual-only because they do not expose a
    single rigid-body root. Each USD is parented beneath an invisible kinematic
    cuboid whose dimensions are also used by the 2-D planner. Moving the proxy
    therefore moves the visual and collision geometry together.
    """
    for spec in specs:
        asset_name = f"{asset_prefix}_{spec.name}"
        setattr(
            scene,
            f"{asset_name}_collision",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{prim_prefix}_{spec.name}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, unused_height),
                ),
                spawn=sim_utils.CuboidCfg(
                    size=spec.footprint,
                    visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                ),
            ),
        )
        setattr(
            scene,
            asset_name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{prim_prefix}_{spec.name}/Visual",
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Hospital props are authored with their origin at ground
                    # level, while the proxy origin is at its geometric center.
                    pos=(0.0, 0.0, -0.5 * spec.footprint[2]),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}{spec.usd_path}",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False,
                    ),
                ),
            ),
        )
