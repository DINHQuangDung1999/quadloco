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

class ObjectRelativeGoalWaypointCommand(UniformGoalVelocityCommandDirect):
    """Generate metric standing waypoints relative to a selected object."""

    cfg: ObjectRelativeGoalWaypointCommandCfg

    def __init__(
        self, cfg: ObjectRelativeGoalWaypointCommandCfg, env: ManagerBasedEnv
    ):
        super().__init__(cfg, env)
        self.object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.detour_entry_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.detour_exit_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.detour_align_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_stage = torch.full(
            (self.num_envs,), 3, dtype=torch.long, device=self.device
        )
        self.is_behind = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Clear all route state before constructing the new command. This is
        # deliberately explicit so a non-behind reset cannot retain a prior
        # episode's behind/alignment marker.
        self.detour_entry_pos_w[env_ids] = 0.0
        self.detour_exit_pos_w[env_ids] = 0.0
        self.detour_align_pos_w[env_ids] = 0.0
        self.route_stage[env_ids] = 3
        self.is_behind[env_ids] = False

        self.object_pos_w[env_ids] = self.goal_pos_w[env_ids]
        object_xy = self.object_pos_w[env_ids, :2]
        start_xy = self._env.scene.env_origins[env_ids, :2]
        object_vector = object_xy - start_xy
        behind_forward = object_vector / torch.linalg.vector_norm(
            object_vector, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        behind_left = torch.stack(
            (-behind_forward[:, 1], behind_forward[:, 0]), dim=-1
        )
        # Front/left/right use the fixed scene frame so those standing
        # positions remain axis-aligned. Behind intentionally retains the
        # original robot-to-object route frame below.
        forward = torch.zeros_like(object_xy)
        forward[:, 0] = 1.0
        left = torch.zeros_like(object_xy)
        left[:, 1] = 1.0
        relation_indices = torch.randint(
            len(self.cfg.relations), (len(env_ids),), device=self.device
        )
        offset_indices = torch.randint(
            len(self.cfg.metric_offsets), (len(env_ids),), device=self.device
        )
        for row, env_id in enumerate(env_ids.tolist()):
            forced = self._forced_task_specs[env_id]
            if forced is not None:
                relation_indices[row] = int(forced["relation_index"])
                offset_indices[row] = int(forced["offset_index"])
        offsets = torch.tensor(
            self.cfg.metric_offsets, device=self.device
        )[offset_indices]
        center_offsets = offsets + self.cfg.object_radius
        final_xy = object_xy.clone()
        behind_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        route_forward = forward.clone()
        route_left = left.clone()

        for row, env_id in enumerate(env_ids.tolist()):
            relation = self.cfg.relations[int(relation_indices[row].item())]
            offset = center_offsets[row]
            relation_offsets = {
                "front": -offset * forward[row],
                # Preserve the original behind semantics: "behind" lies on
                # the ray continuing from the robot through the object. This
                # also keeps the final detour segment geometrically natural.
                "behind": offset * behind_forward[row],
                "left": offset * left[row],
                "right": -offset * left[row],
            }
            final_xy[row] = object_xy[row] + relation_offsets[relation]
            behind_mask[row] = relation == "behind"
            if relation == "behind":
                route_forward[row] = behind_forward[row]
                route_left[row] = behind_left[row]

            marker_index = int(self.marker_indices[env_id].item())
            num_colors = len(self.cfg.marker_colors)
            shape = self.cfg.shape_instruction_names[marker_index // num_colors]
            color = self.cfg.marker_colors[marker_index % num_colors]
            self.goal_task_names[env_id] = self.cfg.task_templates[relation].format(
                distance=float(offsets[row].item()), color=color, shape=shape
            )
            self.task_specs[env_id] = {
                "shape_index": marker_index // num_colors,
                "color_index": marker_index % num_colors,
                "relation_index": int(relation_indices[row].item()),
                "offset_index": int(offset_indices[row].item()),
            }
            self._forced_task_specs[env_id] = None

        self.goal_pos_w[env_ids, :2] = final_xy
        self.goal_pos_w[env_ids, 2] = self.robot.data.root_pos_w[env_ids, 2]
        self.route_stage[env_ids] = torch.where(
            behind_mask,
            torch.zeros(len(env_ids), dtype=torch.long, device=self.device),
            # Every relation first reaches an outward alignment point, then
            # approaches the final goal directly toward the object.
            torch.full((len(env_ids),), 2, dtype=torch.long, device=self.device),
        )
        self.is_behind[env_ids] = behind_mask
        self._sample_behind_detours(
            env_ids, route_forward, route_left, behind_mask
        )
        self._show_only_reference_object(env_ids)
        self.waypoint_command[env_ids] = 0.0

    def _sample_behind_detours(
        self,
        env_ids: torch.Tensor,
        forward: torch.Tensor,
        left: torch.Tensor,
        behind_mask: torch.Tensor,
    ) -> None:
        """Route around the selected object before approaching a behind goal."""
        clearance = (
            self.cfg.object_radius
            + self.cfg.robot_radius
            + self.cfg.safety_margin
        )
        # Use the original behind-route sampling. Scene spawn ranges are the
        # appropriate place to keep these routes away from fences.
        side = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        lateral_offset = side[:, None] * clearance * left
        object_xy = self.object_pos_w[env_ids, :2]
        self.detour_entry_pos_w[env_ids, :2] = (
            object_xy - clearance * forward + lateral_offset
        )
        self.detour_exit_pos_w[env_ids, :2] = (
            object_xy + clearance * forward + lateral_offset
        )
        # This alignment point is used only by non-behind relations. Behind
        # retains its original entry -> exit -> final route.
        self.detour_align_pos_w[env_ids, :2] = (
            self.goal_pos_w[env_ids, :2]
            - self.cfg.final_approach_distance * forward
        )
        route_height = self.robot.data.root_pos_w[env_ids, 2]
        self.detour_entry_pos_w[env_ids, 2] = route_height
        self.detour_exit_pos_w[env_ids, 2] = route_height
        self.detour_align_pos_w[env_ids, 2] = route_height
        direct_env_ids = env_ids[~behind_mask]
        self.detour_entry_pos_w[direct_env_ids] = self.goal_pos_w[direct_env_ids]
        self.detour_exit_pos_w[direct_env_ids] = self.goal_pos_w[direct_env_ids]

    def _show_only_reference_object(self, env_ids: torch.Tensor) -> None:
        for marker_index, asset in enumerate(self.candidate_assets):
            root_pose = asset.data.default_root_state[env_ids, :7].clone()
            root_pose[:, :3] = self._env.scene.env_origins[env_ids]
            root_pose[:, 2] = self.cfg.unused_candidate_height
            selected_rows = torch.where(
                self.marker_indices[env_ids] == marker_index
            )[0]
            if len(selected_rows) > 0:
                root_pose[selected_rows, :3] = self.object_pos_w[
                    env_ids[selected_rows]
                ]
                shape_index = marker_index // len(self.cfg.marker_colors)
                root_pose[selected_rows, 2] = (
                    self._env.scene.env_origins[env_ids[selected_rows], 2]
                    + goal_object_center_height(self.cfg.marker_shapes[shape_index])
                )
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    def _update_command(self):
        robot_xy = self.robot.data.root_pos_w[:, :2]
        entry_distance = torch.linalg.vector_norm(
            self.detour_entry_pos_w[:, :2] - robot_xy, dim=-1
        )
        self.route_stage[
            (self.route_stage == 0)
            & (entry_distance < self.cfg.waypoint_tolerance)
        ] = 1
        exit_distance = torch.linalg.vector_norm(
            self.detour_exit_pos_w[:, :2] - robot_xy, dim=-1
        )
        exit_reached = (self.route_stage == 1) & (
            exit_distance < self.cfg.waypoint_tolerance
        )
        # Behind routes have already cleared the object at the exit waypoint.
        # Go straight to the final position instead of adding an orientation
        # maneuver merely to make the robot face +X.
        self.route_stage[exit_reached & self.is_behind] = 3
        self.route_stage[exit_reached & ~self.is_behind] = 2
        align_distance = torch.linalg.vector_norm(
            self.detour_align_pos_w[:, :2] - robot_xy, dim=-1
        )
        self.route_stage[
            (self.route_stage == 2)
            & (align_distance < self.cfg.waypoint_tolerance)
        ] = 3

        active_waypoint_w = self.goal_pos_w.clone()
        entry_mask = self.route_stage == 0
        exit_mask = self.route_stage == 1
        align_mask = self.route_stage == 2
        active_waypoint_w[entry_mask] = self.detour_entry_pos_w[entry_mask]
        active_waypoint_w[exit_mask] = self.detour_exit_pos_w[exit_mask]
        active_waypoint_w[align_mask] = self.detour_align_pos_w[align_mask]
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
        self._update_goal_reached(goal_distance, self.route_stage == 3)
        final_scale = torch.clamp(
            (goal_distance - self.cfg.goal_tolerance)
            / (self.cfg.slowdown_distance - self.cfg.goal_tolerance),
            min=0.0,
            max=1.0,
        )
        distance_scale = torch.where(
            self.route_stage < 3,
            torch.ones_like(final_scale),
            final_scale,
        )
        self._waypoint_to_velocity(distance_scale)
        self._stop_at_reached_goals()


# Backward-compatible name for external imports. New code should use the
# explicit Direct or Occluded class.
UniformGoalVelocityCommand = UniformGoalVelocityCommandDirect




@configclass
class ObjectRelativeGoalWaypointCommandCfg(UniformGoalVelocityCommandCfg):
    """Configuration for :class:`ObjectRelativeGoalWaypointCommand`."""

    class_type: type = ObjectRelativeGoalWaypointCommand

    relations: tuple[str, ...] = ("front", "behind", "left", "right")
    metric_offsets: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    shape_instruction_names: tuple[str, ...] = ("pyramid", "box", "ball")
    task_templates: dict[str, str] = {
        "front": "Stand {distance:g} m in front of the {color} {shape}",
        "behind": "Stand {distance:g} m behind the {color} {shape}",
        "left": "Stand {distance:g} m to the left of the {color} {shape}",
        "right": "Stand {distance:g} m to the right of the {color} {shape}",
    }
    object_radius: float = 0.2
    robot_radius: float = 0.35
    safety_margin: float = 0.15
    waypoint_tolerance: float = 0.3
    goal_release_tolerance: float | None = 0.3
    final_approach_distance: float = 0.4

    def __post_init__(self):
        super().__post_init__()
        supported_relations = {"front", "behind", "left", "right"}
        if not self.relations or set(self.relations) - supported_relations:
            raise ValueError(
                "relations may contain only front, behind, left, and right."
            )
        if set(self.relations) - set(self.task_templates):
            raise ValueError("Every enabled relation needs a task template.")
        if not self.metric_offsets or any(
            offset <= 0.0 for offset in self.metric_offsets
        ):
            raise ValueError("metric_offsets must contain positive distances.")
        if len(self.shape_instruction_names) != len(self.marker_shapes):
            raise ValueError(
                "shape_instruction_names must align with marker_shapes."
            )
        if min(self.object_radius, self.robot_radius) <= 0.0:
            raise ValueError("Object and robot radii must be positive.")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be nonnegative.")
        if self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint_tolerance must be positive.")
        if self.final_approach_distance <= self.waypoint_tolerance:
            raise ValueError(
                "final_approach_distance must be greater than waypoint_tolerance."
            )
        try:
            for relation in self.relations:
                self.task_templates[relation].format(
                    distance=0.5, color="blue", shape="box"
                )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "Object-relative templates must use distance, color, and shape."
            ) from exc
