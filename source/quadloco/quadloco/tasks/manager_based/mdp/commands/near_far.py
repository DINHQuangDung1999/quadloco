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
        self.collision_proxies = [
            self._env.scene[f"{cfg.distance_asset_prefix}_{slot}_collision"]
            for slot in range(len(cfg.nominal_distances))
        ]
        self.candidate_pos_w = torch.zeros(
            self.num_envs, len(cfg.nominal_distances), 3, device=self.device
        )
        self.selected_slots = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.route_waypoints_w = torch.zeros(
            self.num_envs, cfg.astar_max_waypoints, 3, device=self.device
        )
        self.route_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.route_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._selection_cycle_index = 0

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
            forced = self._forced_task_specs[env_id]
            variant = (
                int(forced["variant"])
                if forced is not None
                else int(torch.randint(num_variants, (1,), device=self.device).item())
            )
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

            if forced is not None:
                selection_type = str(forced["selection_type"])
            elif self.cfg.cycle_selection_types:
                selection_type = self.cfg.selection_types[
                    self._selection_cycle_index % len(self.cfg.selection_types)
                ]
                self._selection_cycle_index += 1
            else:
                selection_type = self.cfg.selection_types[
                    int(torch.randint(len(self.cfg.selection_types), (1,), device=self.device).item())
                ]
            sorted_slots = torch.argsort(planar_distances)
            selected_rank = {
                "nearest": 0,
                "middle": len(sorted_slots) // 2,
                "farthest": len(sorted_slots) - 1,
            }[selection_type]
            selected_slot = int(sorted_slots[selected_rank].item())
            self.goal_pos_w[env_id] = positions[selected_slot]
            self.candidate_pos_w[env_id] = positions
            self.selected_slots[env_id] = selected_slot
            self.marker_indices[env_id] = variant

            start_xy = self.robot.data.root_pos_w[env_id, :2]
            goal_xy = self.goal_pos_w[env_id, :2]
            approach_direction = start_xy - goal_xy
            approach_direction /= torch.linalg.vector_norm(approach_direction).clamp_min(1.0e-6)

            if self.cfg.route_planner == "direct":
                # Match direct navigation: continuously steer toward the
                # selected object center and rely on the shared 1 m tolerance
                # and continuous slowdown to stop before contact. Keeping one
                # route entry only satisfies the near--far route bookkeeping.
                route = goal_xy.unsqueeze(0)
            else:
                obstacles = [
                    RectangleObstacle(
                        center=tuple(positions[slot, :2].cpu().tolist()),
                        size=(
                            self.cfg.candidate_obstacle_footprint,
                            self.cfg.candidate_obstacle_footprint,
                        ),
                    )
                    for slot in range(len(positions))
                    if slot != selected_slot
                ]
                path = None
                for angle in (0.0, 0.7854, -0.7854, 1.5708, -1.5708, 3.1416):
                    cosine = torch.cos(torch.tensor(angle, device=self.device))
                    sine = torch.sin(torch.tensor(angle, device=self.device))
                    rotated_direction = torch.stack(
                        (
                            cosine * approach_direction[0] - sine * approach_direction[1],
                            sine * approach_direction[0] + cosine * approach_direction[1],
                        )
                    )
                    route_goal_xy = (
                        goal_xy + self.cfg.goal_standoff_distance * rotated_direction
                    )
                    try:
                        path = plan_astar(
                            start_xy.cpu().numpy(),
                            route_goal_xy.cpu().numpy(),
                            obstacles,
                            resolution=self.cfg.astar_resolution,
                            inflation_radius=self.cfg.robot_radius + self.cfg.safety_margin,
                            planning_margin=self.cfg.astar_planning_margin,
                        )
                        break
                    except RuntimeError:
                        continue
                if path is None:
                    raise RuntimeError("A* could not find a collision-free goal stand-off route")
                route = path[1:]
            if len(route) > self.cfg.astar_max_waypoints:
                raise RuntimeError(
                    f"A* produced {len(route)} waypoints, exceeding "
                    f"astar_max_waypoints={self.cfg.astar_max_waypoints}"
                )
            route_length = max(1, len(route))
            self.route_lengths[env_id] = route_length
            self.route_waypoints_w[env_id].zero_()
            if len(route):
                self.route_waypoints_w[env_id, : len(route), :2] = torch.as_tensor(
                    route, device=self.device
                )
            else:
                self.route_waypoints_w[env_id, 0, :2] = self.goal_pos_w[env_id, :2]
            self.route_waypoints_w[env_id, :route_length, 2] = self.robot.data.root_pos_w[
                env_id, 2
            ]
            self.route_stage[env_id] = 0

            shape_index = variant // num_colors
            color_index = variant % num_colors
            shape = self.cfg.shape_instruction_names[shape_index]
            color = self.cfg.marker_colors[color_index]
            templates = {
                "nearest": self.cfg.nearest_task_templates,
                "middle": self.cfg.middle_task_templates,
                "farthest": self.cfg.farthest_task_templates,
            }[selection_type]
            template_index = (
                int(forced["template_index"])
                if forced is not None
                else int(torch.randint(len(templates), (1,), device=self.device).item())
            )
            self.goal_task_names[env_id] = templates[template_index].format(
                color=color,
                shape=shape,
            )
            self.task_specs[env_id] = {
                "variant": variant,
                "selection_type": selection_type,
                "template_index": template_index,
            }
            self._forced_task_specs[env_id] = None

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
                proxy = self.collision_proxies[slot]
                proxy_pose = proxy.data.default_root_state[env_id : env_id + 1, :7].clone()
                proxy_pose[0, :3] = origin
                proxy_pose[0, 2] = self.cfg.unused_candidate_height
                proxy_pose[0, :2] = positions[slot, :2]
                proxy_pose[0, 2] = origin[2] + 0.5 * self.cfg.candidate_obstacle_height
                proxy.write_root_pose_to_sim(
                    proxy_pose,
                    env_ids=torch.tensor([env_id], device=self.device),
                )

        self.goal_reached[env_ids] = False
        self.goal_pos_b[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0
        self.velocity_command[env_ids] = 0.0
        self.waypoint_command[env_ids] = 0.0
        self.route_stage[env_ids] = 0
        self.goal_generation[env_ids] += 1

    def _update_command(self):
        """Follow the collision-free route while retaining the selected final goal."""
        robot_xy = self.robot.data.root_pos_w[:, :2]
        env_indices = torch.arange(self.num_envs, device=self.device)
        final_stage = self.route_lengths - 1
        active_waypoint_w = self.route_waypoints_w[env_indices, self.route_stage]
        active_distance = torch.linalg.vector_norm(
            active_waypoint_w[:, :2] - robot_xy, dim=-1
        )
        advance = (self.route_stage < final_stage) & (
            active_distance < self.cfg.waypoint_tolerance
        )
        self.route_stage[advance] += 1

        active_waypoint_w = self.route_waypoints_w[env_indices, self.route_stage]
        waypoint_vec_w = active_waypoint_w - self.robot.data.root_pos_w
        waypoint_vec_b = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), waypoint_vec_w
        )
        self.waypoint_command[:] = waypoint_vec_b[:, :2]

        goal_vec_w = self.goal_pos_w - self.robot.data.root_pos_w
        goal_vec_b = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), goal_vec_w
        )
        self.goal_pos_b[:] = goal_vec_b[:, :2]
        goal_distance = torch.linalg.vector_norm(self.goal_pos_b, dim=-1)
        self._update_goal_reached(goal_distance, self.route_stage == final_stage)

        final_scale = torch.clamp(
            (goal_distance - self.cfg.goal_tolerance)
            / (self.cfg.slowdown_distance - self.cfg.goal_tolerance),
            min=0.0,
            max=1.0,
        )
        distance_scale = torch.where(
            self.route_stage < final_stage,
            torch.ones_like(final_scale),
            final_scale,
        )
        self._waypoint_to_velocity(distance_scale)
        self._stop_at_reached_goals()




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
    selection_types: tuple[str, ...] = ("nearest", "middle", "farthest")
    cycle_selection_types: bool = False
    route_planner: str = "astar"
    minimum_distance_gap: float = 0.0
    lateral_spacing_range: tuple[float, float] = (0.5, 1.5)
    lateral_group_jitter: float = 0.2
    candidate_obstacle_footprint: float = 0.36
    candidate_obstacle_height: float = 0.4
    # A* targets a safe robot-base position on the near side of the selected
    # object; goal_pos_w remains the object center used for semantic success.
    goal_standoff_distance: float = 0.9
    robot_radius: float = 0.35
    safety_margin: float = 0.15
    waypoint_tolerance: float = 0.3
    astar_resolution: float = 0.1
    astar_planning_margin: float = 1.0
    astar_max_waypoints: int = 64
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
        if min(
            self.candidate_obstacle_footprint,
            self.candidate_obstacle_height,
            self.goal_standoff_distance,
            self.robot_radius,
            self.waypoint_tolerance,
            self.astar_resolution,
            self.astar_planning_margin,
        ) <= 0.0:
            raise ValueError("Near--far planning dimensions must be positive.")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be nonnegative.")
        if self.astar_max_waypoints <= 0:
            raise ValueError("astar_max_waypoints must be positive.")
        if self.goal_standoff_distance >= self.goal_tolerance:
            raise ValueError(
                "goal_standoff_distance must be smaller than goal_tolerance."
            )
        if len(self.shape_instruction_names) != len(self.marker_shapes):
            raise ValueError(
                "shape_instruction_names must align with marker_shapes."
            )
        if len(self.nominal_distances) < 2:
            raise ValueError("Near/far selection requires at least two objects.")
        supported_selection_types = {"nearest", "middle", "farthest"}
        if not self.selection_types or set(self.selection_types) - supported_selection_types:
            raise ValueError(
                "selection_types may contain only nearest, middle, and farthest."
            )
        if "middle" in self.selection_types and len(self.nominal_distances) < 3:
            raise ValueError("Middle-distance selection requires at least three objects.")
        if self.route_planner not in {"astar", "direct"}:
            raise ValueError("route_planner must be either 'astar' or 'direct'.")
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
        if self.minimum_distance_gap < 0.0:
            raise ValueError("minimum_distance_gap must be nonnegative.")
        guaranteed_gap = minimum_gap - 2.0 * self.distance_jitter
        if guaranteed_gap < self.minimum_distance_gap:
            raise ValueError(
                "nominal_distances and distance_jitter guarantee only "
                f"{guaranteed_gap:g} m separation, below minimum_distance_gap="
                f"{self.minimum_distance_gap:g} m."
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
