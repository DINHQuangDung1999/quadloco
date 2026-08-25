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

class UniformGoalVelocityCommandOccluded(UniformGoalVelocityCommandDirect):
    """Generate velocity commands through detour waypoints around an occluder."""

    cfg: OccludedGoalVelocityCommandCfg

    def __init__(self, cfg: OccludedGoalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.detour_entry_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.detour_exit_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_waypoints_w = torch.zeros(
            self.num_envs, cfg.astar_max_waypoints, 3, device=self.device
        )
        self.route_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.route_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obstacle_variants = list(range(len(cfg.obstacle_types)))
        self.occlusion_collision_proxies = [
            self._env.scene[
                f"{cfg.occlusion_obstacle_asset_prefix}_{cfg.obstacle_types[variant_index]}_collision"
            ]
            for variant_index in self.obstacle_variants
        ]
        self.obstacle_variant_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._occlusion_obstacle_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._occlusion_obstacle_yaw = torch.zeros(
            self.num_envs, device=self.device
        )

    def set_goal(
        self,
        goal_pos_w: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ):
        super().set_goal(goal_pos_w, env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.waypoint_command[env_ids] = 0.0
        self.route_stage[env_ids] = 0

    def _debug_direction_b(self) -> torch.Tensor:
        """Place the dot at the currently active detour waypoint."""
        return self.waypoint_command

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.waypoint_command[env_ids] = 0.0
        self.route_stage[env_ids] = 0

        # The goal is fully sampled by the direct implementation above. Only
        # now do we derive the obstacle pose and detour from that fixed goal.
        self._sample_occlusion_routes(env_ids)

        num_colors = len(self.cfg.marker_colors)
        for env_id in env_ids.tolist():
            marker_index = int(self.marker_indices[env_id].item())
            shape = self.cfg.marker_shapes[marker_index // num_colors]
            color = self.cfg.marker_colors[marker_index % num_colors]
            obstacle_type = self.cfg.obstacle_instruction_names[
                int(self.obstacle_variant_indices[env_id].item())
            ]
            self.goal_task_names[env_id] = self.cfg.occluded_task_template.format(
                color=color,
                shape=shape,
                obstacle_type=obstacle_type,
            )
            self.task_specs[env_id] = {
                "shape_index": marker_index // num_colors,
                "color_index": marker_index % num_colors,
                "obstacle_variant": int(self.obstacle_variant_indices[env_id].item()),
            }
            self._forced_task_specs[env_id] = None

        self._show_only_goal_asset(env_ids)
        self._update_occlusion_obstacle_poses(env_ids)

    def _sample_occlusion_routes(self, env_ids: torch.Tensor) -> None:
        """Place the occluder and plan a pruned A* route around it."""
        self._sample_occlusion_obstacle_geometry(env_ids)

        route_height = self.robot.data.root_pos_w[env_ids, 2]
        for row, env_id in enumerate(env_ids.tolist()):
            variant_index = int(self.obstacle_variant_indices[env_id].item())
            obstacle_size = self.cfg.obstacle_footprints[variant_index]
            obstacle = RectangleObstacle(
                center=tuple(self._occlusion_obstacle_pos_w[env_id, :2].cpu().tolist()),
                size=(obstacle_size[0], obstacle_size[1]),
                yaw=float(self._occlusion_obstacle_yaw[env_id].item()),
            )
            path = plan_astar(
                self._env.scene.env_origins[env_id, :2].cpu().numpy(),
                self.goal_pos_w[env_id, :2].cpu().numpy(),
                [obstacle],
                resolution=self.cfg.astar_resolution,
                inflation_radius=self.cfg.robot_radius + self.cfg.safety_margin,
                planning_margin=self.cfg.astar_planning_margin,
            )
            # The first point is the robot start; command only subsequent path
            # points. The exact final point is always the sampled goal.
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
            self.route_waypoints_w[env_id, :route_length, 2] = route_height[row]

            # Compatibility aliases used by older debug tools. They are no
            # longer the source of the active command.
            self.detour_entry_pos_w[env_id] = self.route_waypoints_w[env_id, 0]
            penultimate = max(0, route_length - 2)
            self.detour_exit_pos_w[env_id] = self.route_waypoints_w[env_id, penultimate]
            self.route_stage[env_id] = 0

    def _sample_occlusion_obstacle_geometry(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the obstacle poses shared by A* and the legacy planner."""
        start_pos_w = self._env.scene.env_origins[env_ids, :2]
        target_vector = self.goal_pos_w[env_ids, :2] - start_pos_w
        target_distance = torch.linalg.vector_norm(
            target_vector, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        forward = target_vector / target_distance
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        self.obstacle_variant_indices[env_ids] = torch.randint(
            len(self.obstacle_variants), size=(len(env_ids),), device=self.device
        )
        for env_id in env_ids.tolist():
            forced = self._forced_task_specs[env_id]
            if forced is not None:
                self.obstacle_variant_indices[env_id] = int(forced["obstacle_variant"])
        obstacle_sizes = torch.tensor(
            self.cfg.obstacle_footprints, device=self.device
        )[self.obstacle_variant_indices[env_ids]]
        obstacle_depth = obstacle_sizes[:, 0]
        obstacle_width = obstacle_sizes[:, 1]
        obstacle_height = obstacle_sizes[:, 2]
        path_fraction = torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(*self.cfg.occlusion_obstacle_path_fraction)
        obstacle_center_on_ray = start_pos_w + path_fraction * target_vector
        offset_fraction = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.obstacle_lateral_offset_fraction
        )
        offset_sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        obstacle_center = (
            obstacle_center_on_ray
            + (offset_sign * offset_fraction * obstacle_width)[:, None] * left
        )
        self._occlusion_obstacle_pos_w[env_ids, :2] = obstacle_center
        self._occlusion_obstacle_pos_w[env_ids, 2] = (
            self._env.scene.env_origins[env_ids, 2] + 0.5 * obstacle_height
        )
        self._occlusion_obstacle_yaw[env_ids] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(
            *self.cfg.obstacle_yaw_range
        )
        return forward, left, obstacle_depth, obstacle_width, obstacle_height, obstacle_center

    def _sample_occlusion_routes_geometric_legacy(self, env_ids: torch.Tensor) -> None:
        """Legacy single-obstacle entry/exit generator (kept for comparison)."""
        # Previous implementation. To restore it, call this method directly
        # from _resample_command and use the legacy _update_command logic.
        (
            forward,
            left,
            obstacle_depth,
            obstacle_width,
            obstacle_height,
            obstacle_center,
        ) = self._sample_occlusion_obstacle_geometry(env_ids)
        side = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        lateral_clearance = (
            0.5 * obstacle_width
            + self.cfg.robot_radius
            + self.cfg.safety_margin
        )
        longitudinal_clearance = (
            0.5 * obstacle_depth
            + self.cfg.robot_radius
            + self.cfg.safety_margin
        )
        lateral_offset = side[:, None] * lateral_clearance * left
        self.detour_entry_pos_w[env_ids, :2] = (
            obstacle_center - longitudinal_clearance * forward + lateral_offset
        )
        self.detour_exit_pos_w[env_ids, :2] = (
            obstacle_center + longitudinal_clearance * forward + lateral_offset
        )
        route_height = self.robot.data.root_pos_w[env_ids, 2]
        self.detour_entry_pos_w[env_ids, 2] = route_height
        self.detour_exit_pos_w[env_ids, 2] = route_height

    def _show_only_goal_asset(self, env_ids: torch.Tensor) -> None:
        """Park every distractor and leave only the selected target visible."""
        for marker_index, asset in enumerate(self.candidate_assets):
            root_pose = asset.data.default_root_state[env_ids, :7].clone()
            root_pose[:, :3] = self._env.scene.env_origins[env_ids]
            root_pose[:, 2] = self.cfg.unused_candidate_height
            selected_rows = torch.where(
                self.marker_indices[env_ids] == marker_index
            )[0]
            if len(selected_rows) > 0:
                root_pose[selected_rows, :3] = self.goal_pos_w[
                    env_ids[selected_rows]
                ]
                shape_index = marker_index // len(self.cfg.marker_colors)
                root_pose[selected_rows, 2] = (
                    self._env.scene.env_origins[env_ids[selected_rows], 2]
                    + goal_object_center_height(self.cfg.marker_shapes[shape_index])
                )
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    def _update_occlusion_obstacle_poses(self, env_ids: torch.Tensor) -> None:
        for variant_index, collision_proxy in enumerate(
            self.occlusion_collision_proxies
        ):
            root_pose = torch.zeros(len(env_ids), 7, device=self.device)
            root_pose[:, 3] = 1.0
            root_pose[:, :3] = self._env.scene.env_origins[env_ids]
            root_pose[:, 2] = self.cfg.unused_candidate_height
            selected_rows = torch.where(
                self.obstacle_variant_indices[env_ids] == variant_index
            )[0]
            if len(selected_rows) > 0:
                selected_env_ids = env_ids[selected_rows]
                root_pose[selected_rows, :3] = self._occlusion_obstacle_pos_w[
                    selected_env_ids
                ]
                yaw = self._occlusion_obstacle_yaw[selected_env_ids]
                zeros = torch.zeros_like(yaw)
                root_pose[selected_rows, 3:7] = quat_from_euler_xyz(
                    zeros, zeros, yaw
                )
            collision_root_pose = collision_proxy.data.default_root_state[
                env_ids, :7
            ].clone()
            collision_root_pose.copy_(root_pose)
            collision_proxy.write_root_pose_to_sim(
                collision_root_pose, env_ids=env_ids
            )

    def _update_command(self):
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
        self.goal_reached[:] = (
            (self.route_stage == final_stage) & (goal_distance < self.cfg.goal_tolerance)
        )

        # Do not apply the final-goal stopping tolerance to intermediate
        # waypoints. Otherwise, when goal_tolerance > waypoint_tolerance, the
        # robot stops before it can trigger the next route stage.
        final_distance_scale = torch.clamp(
            (goal_distance - self.cfg.goal_tolerance)
            / (self.cfg.slowdown_distance - self.cfg.goal_tolerance),
            min=0.0,
            max=1.0,
        )
        distance_scale = torch.where(
            self.route_stage < final_stage,
            torch.ones_like(final_distance_scale),
            final_distance_scale,
        )
        self._waypoint_to_velocity(distance_scale)
        self._stop_at_reached_goals()




@configclass
class OccludedGoalVelocityCommandCfg(UniformGoalVelocityCommandCfg):
    """Configuration for :class:`UniformGoalVelocityCommandOccluded`."""

    class_type: type = UniformGoalVelocityCommandOccluded

    occluded_task_template: str = (
        "Navigate to the {color} {shape} behind the {obstacle_type}"
    )
    occlusion_obstacle_asset_prefix: str = "occlusion_obstacle"
    obstacle_types: tuple[str, ...] = tuple(spec.name for spec in OFFICE_OBSTACLE_SPECS)
    # Human-readable category. Multiple visual variants may share one label.
    obstacle_instruction_names: tuple[str, ...] = tuple(
        spec.instruction_name for spec in OFFICE_OBSTACLE_SPECS
    )
    # Conservative planning footprints (depth, width, height) in metres.
    obstacle_footprints: tuple[tuple[float, float, float], ...] = tuple(
        spec.footprint for spec in OFFICE_OBSTACLE_SPECS
    )
    # Uniform range along the complete robot-to-goal segment.
    occlusion_obstacle_path_fraction: tuple[float, float] = (0.4, 0.6)
    # Random lateral displacement as a fraction of the selected obstacle width.
    obstacle_lateral_offset_fraction: tuple[float, float] = (0.4, 0.6)
    # Independent world-frame yaw sampled for every obstacle at every reset.
    obstacle_yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    robot_radius: float = 0.35
    safety_margin: float = 0.25
    waypoint_tolerance: float = 0.45
    astar_resolution: float = 0.10
    astar_planning_margin: float = 1.0
    astar_max_waypoints: int = 64

    def __post_init__(self):
        super().__post_init__()
        try:
            self.occluded_task_template.format(
                color="red", shape="cube", obstacle_type="chair"
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "occluded_task_template must use {color}, {shape}, and "
                "{obstacle_type}."
            ) from exc
        if not self.obstacle_types:
            raise ValueError("At least one obstacle type must be configured.")
        if len(set(self.obstacle_types)) != len(self.obstacle_types):
            raise ValueError("obstacle_types must be unique.")
        min_path_fraction, max_path_fraction = self.occlusion_obstacle_path_fraction
        if not 0.0 < min_path_fraction <= max_path_fraction < 1.0:
            raise ValueError(
                "occlusion_obstacle_path_fraction must be an ordered range inside (0, 1)."
            )
        min_offset, max_offset = self.obstacle_lateral_offset_fraction
        if min_offset < 0.0 or max_offset < min_offset:
            raise ValueError(
                "obstacle_lateral_offset_fraction must be a nonnegative "
                "(minimum, maximum) range."
            )
        if len(self.obstacle_footprints) != len(self.obstacle_types):
            raise ValueError("Each obstacle type must have one planning footprint.")
        if len(self.obstacle_instruction_names) != len(self.obstacle_types):
            raise ValueError("Each obstacle type must have one instruction name.")
        min_yaw, max_yaw = self.obstacle_yaw_range
        if max_yaw < min_yaw:
            raise ValueError("obstacle_yaw_range must be an ordered (min, max) pair.")
        if any(
            len(size) != 3 or min(size) <= 0.0
            for size in self.obstacle_footprints
        ):
            raise ValueError(
                "Each obstacle footprint must contain positive (depth, width, height)."
            )
        if self.robot_radius < 0.0 or self.safety_margin < 0.0:
            raise ValueError("Occlusion clearance values must be nonnegative.")
        if self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint_tolerance must be positive.")
        if self.astar_resolution <= 0.0:
            raise ValueError("astar_resolution must be positive.")
        if self.astar_planning_margin <= 0.0:
            raise ValueError("astar_planning_margin must be positive.")
        if self.astar_max_waypoints <= 0:
            raise ValueError("astar_max_waypoints must be positive.")
