from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import trimesh

from .astar_planner import RectangleObstacle, plan_astar
from ...obstacle_assets import OFFICE_OBSTACLE_SPECS

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



from .direct import (
    UniformGoalVelocityCommandCfg,
    UniformGoalVelocityCommandDirect,
    goal_object_center_height,
)

class UniformGoalVelocityCommandNearFar(UniformGoalVelocityCommandDirect):
    """Select the nearest or farthest of several identical objects."""

    cfg: NearFarGoalVelocityCommandCfg

    def __init__(self, cfg: NearFarGoalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.distance_assets = [
            [
                self._env.scene[
                    f"{cfg.distance_asset_prefix}_{slot}_{shape}_{color}"
                ]
                for shape in cfg.marker_shapes
                for color in cfg.marker_colors
            ]
            for slot in range(len(cfg.nominal_distances))
        ]

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        num_colors = len(self.cfg.marker_colors)
        num_variants = len(self.cfg.marker_shapes) * num_colors
        nominal_distances = torch.tensor(
            self.cfg.nominal_distances, device=self.device
        )

        for env_id in env_ids.tolist():
            variant = int(torch.randint(num_variants, (1,), device=self.device).item())
            group_x_offset = torch.empty(1, device=self.device).uniform_(
                *self.cfg.group_x_offset_range
            )
            x_positions = nominal_distances + group_x_offset + torch.empty(
                len(self.cfg.nominal_distances), device=self.device
            ).uniform_(-self.cfg.distance_jitter, self.cfg.distance_jitter)
            x_positions, _ = torch.sort(x_positions)
            lateral_spacing = float(
                torch.empty(1, device=self.device)
                .uniform_(*self.cfg.lateral_spacing_range)
                .item()
            )
            lane_center = 0.5 * (len(x_positions) - 1)
            lateral_lanes = (
                torch.arange(len(x_positions), device=self.device) - lane_center
            ) * lateral_spacing
            lateral_lanes = lateral_lanes[
                torch.randperm(len(x_positions), device=self.device)
            ]
            group_lateral_shift = torch.empty(
                1, device=self.device
            ).uniform_(-self.cfg.lateral_group_jitter, self.cfg.lateral_group_jitter)
            y_positions = lateral_lanes + group_lateral_shift

            origin = self._env.scene.env_origins[env_id]
            positions = torch.zeros(
                len(self.cfg.nominal_distances), 3, device=self.device
            )
            positions[:, 0] = origin[0] + x_positions
            positions[:, 1] = origin[1] + y_positions
            shape_index = variant // num_colors
            positions[:, 2] = (
                origin[2]
                + goal_object_center_height(self.cfg.marker_shapes[shape_index])
            )
            planar_distances = torch.linalg.vector_norm(
                positions[:, :2] - origin[:2], dim=-1
            )

            selection_type = ("nearest", "middle", "farthest")[
                int(torch.randint(3, (1,), device=self.device).item())
            ]
            sorted_slots = torch.argsort(planar_distances)
            selected_rank = {
                "nearest": 0,
                "middle": len(sorted_slots) // 2,
                "farthest": len(sorted_slots) - 1,
            }[selection_type]
            selected_slot = int(sorted_slots[selected_rank].item())
            self.goal_pos_w[env_id] = positions[selected_slot]
            self.marker_indices[env_id] = variant

            shape_index = variant // num_colors
            color_index = variant % num_colors
            shape = self.cfg.shape_instruction_names[shape_index]
            color = self.cfg.marker_colors[color_index]
            templates = {
                "nearest": self.cfg.nearest_task_templates,
                "middle": self.cfg.middle_task_templates,
                "farthest": self.cfg.farthest_task_templates,
            }[selection_type]
            template_index = int(
                torch.randint(len(templates), (1,), device=self.device).item()
            )
            self.goal_task_names[env_id] = templates[template_index].format(
                color=color,
                shape=shape,
            )

            for slot, slot_assets in enumerate(self.distance_assets):
                for variant_index, asset in enumerate(slot_assets):
                    root_pose = asset.data.default_root_state[
                        env_id : env_id + 1, :7
                    ].clone()
                    root_pose[:, :3] = origin
                    root_pose[:, 2] = self.cfg.unused_candidate_height
                    if variant_index == variant:
                        root_pose[0, :3] = positions[slot]
                    asset.write_root_pose_to_sim(
                        root_pose,
                        env_ids=torch.tensor([env_id], device=self.device),
                    )

        self.goal_reached[env_ids] = False
        self.goal_pos_b[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0
        self.velocity_command[env_ids] = 0.0
        self.waypoint_command[env_ids] = 0.0
        self.goal_generation[env_ids] += 1




@configclass
class NearFarGoalVelocityCommandCfg(UniformGoalVelocityCommandCfg):
    """Configuration for :class:`UniformGoalVelocityCommandNearFar`."""

    class_type: type = UniformGoalVelocityCommandNearFar

    distance_asset_prefix: str = "distance_object"
    shape_instruction_names: tuple[str, ...] = ("pyramid", "box", "ball")
    nominal_distances: tuple[float, ...] = (3.0, 4.5, 6.0)
    # One shared offset is added to every object's X position per reset.
    group_x_offset_range: tuple[float, float] = (0.0, 1.0)
    distance_jitter: float = 0.25
    lateral_spacing_range: tuple[float, float] = (0.5, 1.5)
    lateral_group_jitter: float = 0.2
    nearest_task_templates: tuple[str, ...] = (
        "Go to the closest {color} {shape}",
        "Approach the nearest {color} {shape}",
        "Navigate to the {color} {shape} closest to you",
    )
    middle_task_templates: tuple[str, ...] = (
        "Go to the {color} {shape} at the middle distance",
        "Approach the middle-distance {color} {shape}",
        "Navigate to the {color} {shape} between the other two in depth",
    )
    farthest_task_templates: tuple[str, ...] = (
        "Go to the farthest {color} {shape}",
        "Approach the {color} {shape} farthest from you",
        "Navigate to the most distant {color} {shape}",
    )

    def __post_init__(self):
        super().__post_init__()
        if len(self.shape_instruction_names) != len(self.marker_shapes):
            raise ValueError(
                "shape_instruction_names must align with marker_shapes."
            )
        if len(self.nominal_distances) < 2:
            raise ValueError("Near/far selection requires at least two objects.")
        if any(
            right <= left
            for left, right in zip(
                self.nominal_distances, self.nominal_distances[1:]
            )
        ):
            raise ValueError("nominal_distances must be strictly increasing.")
        if self.distance_jitter < 0.0 or self.lateral_group_jitter < 0.0:
            raise ValueError("Distance and lateral jitter must be nonnegative.")
        min_x_offset, max_x_offset = self.group_x_offset_range
        if max_x_offset < min_x_offset:
            raise ValueError("group_x_offset_range must be an ordered range.")
        min_spacing, max_spacing = self.lateral_spacing_range
        if min_spacing <= 0.0 or max_spacing < min_spacing:
            raise ValueError(
                "lateral_spacing_range must be positive and ordered."
            )
        minimum_gap = min(
            right - left
            for left, right in zip(
                self.nominal_distances, self.nominal_distances[1:]
            )
        )
        if 2.0 * self.distance_jitter >= minimum_gap:
            raise ValueError(
                "distance_jitter must preserve ordering between distance bands."
            )
        if (
            not self.nearest_task_templates
            or not self.middle_task_templates
            or not self.farthest_task_templates
        ):
            raise ValueError(
                "Nearest, middle, and farthest templates cannot be empty."
            )
        try:
            for template in (
                *self.nearest_task_templates,
                *self.middle_task_templates,
                *self.farthest_task_templates,
            ):
                template.format(color="red", shape="box")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "Near/far templates must use {color} and {shape}."
            ) from exc

