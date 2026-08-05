from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@clone
def spawn_pyramid(
    prim_path: str,
    cfg: PyramidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a square pyramid mesh for goal visualization."""
    pyramid = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, sections=4)
    stage = get_current_stage()
    _spawn_mesh_geom_from_mesh(prim_path, cfg, pyramid, translation, orientation, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class PyramidCfg(sim_utils.MeshCfg):
    """Configuration for a square pyramid marker."""

    func = spawn_pyramid
    radius: float = MISSING
    height: float = MISSING


GOAL_OBJECT_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def make_goal_object_spawn_cfg(shape: str, color: str):
    """Create non-colliding kinematic goal geometry visible to RGB-D cameras."""
    if color not in GOAL_OBJECT_COLORS:
        raise ValueError(f"Unsupported goal object color: {color}")

    material = sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_OBJECT_COLORS[color])
    common = {
        "visual_material": material,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
    }
    if shape == "pyramid":
        return PyramidCfg(radius=0.18, height=0.3, **common)
    if shape == "cube":
        return sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common)
    if shape == "sphere":
        return sim_utils.SphereCfg(radius=0.15, **common)
    raise ValueError(f"Unsupported goal object shape: {shape}")


class UniformGoalVelocityCommand(CommandTerm):
    """Sample planar goals and generate base velocity commands toward them.

    The command has the locomotion-policy-compatible form ``[vx, vy, wz]``.
    Goal positions are stored in world coordinates and transformed into the
    robot's yaw-aligned base frame every environment step.
    """

    cfg: UniformGoalVelocityCommandCfg

    def __init__(self, cfg: UniformGoalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_generation = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.marker_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.candidate_pos_w = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.candidate_marker_indices = torch.zeros(
            self.num_envs, 1, dtype=torch.int32, device=self.device
        )
        self.goal_task_names = [""] * self.num_envs
        self.candidate_assets = [
            self._env.scene[f"{self.cfg.candidate_asset_prefix}_{shape}_{color}"]
            for shape in self.cfg.marker_shapes
            for color in self.cfg.marker_colors
        ]

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the scripted ``[vx, vy, wz]`` command for each environment."""
        return self.velocity_command

    def set_goal(self, goal_pos_w: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Set externally supplied world-frame goals, for example from an object detector."""
        if env_ids is None:
            env_ids = slice(None)
        self.goal_pos_w[env_ids] = goal_pos_w.to(device=self.device, dtype=self.goal_pos_w.dtype)
        self.goal_reached[env_ids] = False
        self.velocity_command[env_ids] = 0.0
        self.goal_generation[env_ids] += 1

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        samples = torch.empty(len(env_ids), 2, device=self.device)
        samples[:, 0].uniform_(*self.cfg.ranges.pos_x)
        samples[:, 1].uniform_(*self.cfg.ranges.pos_y)
        center_pos_w = self._env.scene.env_origins[env_ids].clone()
        center_pos_w[:, :2] += samples
        center_pos_w[:, 2] += self.cfg.marker_height

        candidate_y_offsets = self.cfg.candidate_y_offsets or (0.0,)
        num_candidates = len(candidate_y_offsets)
        if self.candidate_pos_w.shape[1] != num_candidates:
            self.candidate_pos_w = torch.zeros(
                self.num_envs, num_candidates, 3, device=self.device
            )
            self.candidate_marker_indices = torch.zeros(
                self.num_envs, num_candidates, dtype=torch.int32, device=self.device
            )
        self.candidate_pos_w[env_ids] = center_pos_w[:, None, :]
        offsets = torch.tensor(candidate_y_offsets, device=self.device)

        # Jitter every candidate independently along x so that the objects do
        # not always lie on a perfectly straight lateral line.
        if self.cfg.candidate_x_error > 0.0:
            x_jitter = torch.empty(
                len(env_ids), num_candidates, device=self.device
            ).uniform_(-self.cfg.candidate_x_error, self.cfg.candidate_x_error)
            self.candidate_pos_w[env_ids, :, 0] += x_jitter

        # Preserve each nominal adjacent y gap and add an independent positive
        # random spacing. Re-center the resulting group around the y position
        # sampled from ranges.pos_y.
        if num_candidates > 1:
            nominal_gaps = offsets[1:] - offsets[:-1]
            extra_gaps = torch.empty(
                len(env_ids), num_candidates - 1, device=self.device
            ).uniform_(0.0, self.cfg.candidate_y_spacing_error)
            y_offsets = torch.cat(
                (
                    torch.zeros(len(env_ids), 1, device=self.device),
                    torch.cumsum(nominal_gaps[None, :] + extra_gaps, dim=1),
                ),
                dim=1,
            )
            y_offsets -= 0.5 * (y_offsets[:, :1] + y_offsets[:, -1:])
        else:
            y_offsets = torch.zeros(len(env_ids), 1, device=self.device)
        self.candidate_pos_w[env_ids, :, 1] += y_offsets
        goal_candidate_indices = torch.randint(
            num_candidates, size=(len(env_ids),), device=self.device
        )
        self.goal_pos_w[env_ids] = self.candidate_pos_w[
            env_ids, goal_candidate_indices
        ]
        # Clear state associated with the previous goal immediately. CommandTerm
        # calls this method on both timed resampling and episode reset.
        self.goal_reached[env_ids] = False
        self.goal_pos_b[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0
        self.velocity_command[env_ids] = 0.0
        self.goal_generation[env_ids] += 1

        # Match the VLA scene semantics: show one instance of every configured
        # shape, assign each a unique shuffled color, and select one visible
        # object as the navigation goal. Marker insertion order in the config is
        # shape-major and color-minor, hence ``shape * num_colors + color``.
        num_shapes = len(self.cfg.marker_shapes)
        num_colors = len(self.cfg.marker_colors)
        if num_candidates != num_shapes:
            raise ValueError(
                "VLA candidate mode requires one candidate per marker shape: "
                f"got {num_candidates} candidates and {num_shapes} shapes."
            )
        if num_colors < num_candidates:
            raise ValueError(
                "VLA candidate mode requires at least one unique color per candidate: "
                f"got {num_colors} colors and {num_candidates} candidates."
            )

        for row, env_id in enumerate(env_ids.tolist()):
            color_indices = torch.randperm(num_colors, device=self.device)[:num_candidates]
            shape_indices = torch.randperm(num_shapes, device=self.device)[:num_candidates]
            marker_indices = shape_indices * num_colors + color_indices
            self.candidate_marker_indices[env_id] = marker_indices.to(torch.int32)

            goal_candidate = int(goal_candidate_indices[row].item())
            goal_marker = int(marker_indices[goal_candidate].item())
            self.marker_indices[env_id] = goal_marker
            goal_shape = self.cfg.marker_shapes[int(shape_indices[goal_candidate].item())]
            goal_color = self.cfg.marker_colors[int(color_indices[goal_candidate].item())]
            self.goal_task_names[env_id] = self.cfg.task_template.format(
                color=goal_color,
                shape=goal_shape,
            )

        self._update_candidate_asset_poses(env_ids)

    def _update_candidate_asset_poses(self, env_ids: torch.Tensor) -> None:
        """Show the three selected scene objects and park unused variants below the stage."""
        for marker_index, asset in enumerate(self.candidate_assets):
            root_pose = asset.data.default_root_state[env_ids, :7].clone()
            root_pose[:, :3] = self._env.scene.env_origins[env_ids]
            root_pose[:, 2] = self.cfg.unused_candidate_height

            matches = self.candidate_marker_indices[env_ids] == marker_index
            match_rows, candidate_indices = torch.where(matches)
            if len(match_rows) > 0:
                root_pose[match_rows, :3] = self.candidate_pos_w[
                    env_ids[match_rows], candidate_indices
                ]

            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    def _update_command(self):
        target_vec_w = self.goal_pos_w - self.robot.data.root_pos_w
        target_vec_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)

        self.goal_pos_b[:] = target_vec_b[:, :2]
        self.heading_error[:] = torch.atan2(self.goal_pos_b[:, 1], self.goal_pos_b[:, 0])
        distance = torch.linalg.vector_norm(self.goal_pos_b, dim=-1)
        self.goal_reached[:] = distance < self.cfg.goal_tolerance

        heading_scale = torch.clamp(torch.cos(self.heading_error), min=0.0)
        distance_scale = torch.clamp(
            (distance - self.cfg.goal_tolerance) / (self.cfg.slowdown_distance - self.cfg.goal_tolerance),
            min=0.0,
            max=1.0,
        )
        self.velocity_command[:, 0] = self.cfg.forward_velocity * heading_scale * distance_scale
        self.velocity_command[:, 1] = 0.0
        self.velocity_command[:, 2] = torch.clamp(
            self.cfg.yaw_gain * self.heading_error,
            min=-self.cfg.max_yaw_rate,
            max=self.cfg.max_yaw_rate,
        )
        self.velocity_command[self.goal_reached] = 0.0

    def _update_metrics(self):
        self.metrics["position_error"][:] = torch.linalg.vector_norm(self.goal_pos_b, dim=-1)
        self.metrics["heading_error"][:] = torch.abs(self.heading_error)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "velocity_visualizer"):
                self.velocity_visualizer = VisualizationMarkers(self.cfg.velocity_visualizer_cfg)
            if not hasattr(self, "goal_direction_visualizer"):
                self.goal_direction_visualizer = VisualizationMarkers(self.cfg.goal_direction_visualizer_cfg)
            self.velocity_visualizer.set_visibility(True)
            self.goal_direction_visualizer.set_visibility(True)
        elif hasattr(self, "velocity_visualizer"):
            self.velocity_visualizer.set_visibility(False)
            if hasattr(self, "goal_direction_visualizer"):
                self.goal_direction_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        heading = torch.atan2(self.velocity_command[:, 1], self.velocity_command[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat_b = quat_from_euler_xyz(zeros, zeros, heading)
        arrow_quat_w = quat_mul(yaw_quat(self.robot.data.root_quat_w), arrow_quat_b)

        default_scale = self.cfg.velocity_visualizer_cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        arrow_scale[:, 0] *= torch.linalg.vector_norm(self.velocity_command[:, :2], dim=-1) * 3.0
        self.velocity_visualizer.visualize(
            translations=base_pos_w,
            orientations=arrow_quat_w,
            scales=arrow_scale,
        )

        # Show the goal bearing separately because wz cannot be represented by a linear arrow.
        goal_arrow_quat_b = quat_from_euler_xyz(zeros, zeros, self.heading_error)
        goal_arrow_quat_w = quat_mul(yaw_quat(self.robot.data.root_quat_w), goal_arrow_quat_b)
        goal_arrow_scale = torch.tensor(
            self.cfg.goal_direction_visualizer_cfg.markers["arrow"].scale, device=self.device
        ).repeat(self.num_envs, 1)
        self.goal_direction_visualizer.visualize(
            translations=base_pos_w,
            orientations=goal_arrow_quat_w,
            scales=goal_arrow_scale,
        )


def generated_goal(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return ``[goal_x_b, goal_y_b, heading_error]`` from a goal command term."""
    term: UniformGoalVelocityCommand = env.command_manager.get_term(command_name)
    return torch.cat((term.goal_pos_b, term.heading_error.unsqueeze(-1)), dim=-1)


@configclass
class UniformGoalVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformGoalVelocityCommand`."""

    class_type: type = UniformGoalVelocityCommand

    asset_name: str = MISSING
    goal_tolerance: float = 0.2
    marker_height: float = 0.15
    forward_velocity: float = 0.4
    yaw_gain: float = 1.5
    max_yaw_rate: float = 0.8
    slowdown_distance: float = 0.75
    marker_shapes: tuple[str, ...] = ("pyramid", "cube", "sphere")
    marker_colors: tuple[str, ...] = ("red", "green", "blue")
    task_template: str = "Navigate to the {color} {shape}"
    candidate_asset_prefix: str = "goal_object"
    unused_candidate_height: float = -100.0
    # Show the three VLA candidates together on a line parallel to the y-axis.
    # Exactly one candidate is sampled as the navigation goal.
    candidate_y_offsets: tuple[float, ...] | None = (-1.0, 0.0, 1.0)
    # Independent per-object x displacement sampled from [-error, error].
    candidate_x_error: float = 0.0
    # Independent nonnegative addition to every adjacent nominal y gap.
    candidate_y_spacing_error: float = 0.0

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    velocity_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/navigation_velocity"
    )
    velocity_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_direction_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/navigation_goal_direction"
    )
    goal_direction_visualizer_cfg.markers["arrow"].scale = (1.0, 0.5, 0.5)

    def __post_init__(self):
        supported_shapes = {"pyramid", "cube", "sphere"}
        invalid_shapes = set(self.marker_shapes) - supported_shapes
        invalid_colors = set(self.marker_colors) - set(GOAL_OBJECT_COLORS)
        if invalid_shapes:
            raise ValueError(f"Unsupported goal marker shapes: {sorted(invalid_shapes)}")
        if invalid_colors:
            raise ValueError(f"Unsupported goal marker colors: {sorted(invalid_colors)}")
        if not self.marker_shapes or not self.marker_colors:
            raise ValueError("At least one goal marker shape and color must be configured.")
        try:
            self.task_template.format(color="red", shape="cube")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "task_template must be a valid format string using {color} and {shape}."
            ) from exc
        if self.candidate_y_offsets is not None and not self.candidate_y_offsets:
            raise ValueError("candidate_y_offsets must contain at least one offset.")
        if self.candidate_y_offsets is not None and any(
            right <= left
            for left, right in zip(self.candidate_y_offsets, self.candidate_y_offsets[1:])
        ):
            raise ValueError("candidate_y_offsets must be strictly increasing.")
        if self.candidate_x_error < 0.0:
            raise ValueError("candidate_x_error must be nonnegative.")
        if self.candidate_y_spacing_error < 0.0:
            raise ValueError("candidate_y_spacing_error must be nonnegative.")
        if self.slowdown_distance <= self.goal_tolerance:
            raise ValueError("slowdown_distance must be greater than goal_tolerance.")
