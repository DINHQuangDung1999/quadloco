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
        self.candidate_pos_w[env_ids, :, 1] += offsets[None, :]
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
        self.marker_indices[env_ids] = torch.randint(
            len(self.cfg.goal_visualizer_cfg.markers),
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.int32,
        )
        num_marker_styles = len(self.cfg.goal_visualizer_cfg.markers)
        if num_candidates > 1 and num_marker_styles < 2:
            raise ValueError("Multiple candidate objects require at least two marker styles.")
        # Style 0 is the consistent visual cue for the true goal. Distractors
        # use any of the remaining styles, so a vision policy can identify the
        # selected object from the image.
        if num_candidates > 1:
            self.candidate_marker_indices[env_ids] = torch.randint(
                1,
                num_marker_styles,
                size=(len(env_ids), num_candidates),
                device=self.device,
                dtype=torch.int32,
            )
        else:
            self.candidate_marker_indices[env_ids] = 0
        self.candidate_marker_indices[env_ids, goal_candidate_indices] = 0

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
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            if not hasattr(self, "velocity_visualizer"):
                self.velocity_visualizer = VisualizationMarkers(self.cfg.velocity_visualizer_cfg)
            if not hasattr(self, "goal_direction_visualizer"):
                self.goal_direction_visualizer = VisualizationMarkers(self.cfg.goal_direction_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.velocity_visualizer.set_visibility(True)
            self.goal_direction_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            if hasattr(self, "velocity_visualizer"):
                self.velocity_visualizer.set_visibility(False)
            if hasattr(self, "goal_direction_visualizer"):
                self.goal_direction_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_visualizer.visualize(
            translations=self.candidate_pos_w.reshape(-1, 3),
            marker_indices=self.candidate_marker_indices.reshape(-1),
        )

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
    # When set, show one candidate object at each lateral offset from the
    # sampled center. Exactly one candidate is sampled as the navigation goal.
    candidate_y_offsets: tuple[float, ...] | None = None

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_visualizer_cfg: VisualizationMarkersCfg = MISSING
    velocity_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/navigation_velocity"
    )
    velocity_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_direction_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/navigation_goal_direction"
    )
    goal_direction_visualizer_cfg.markers["arrow"].scale = (1.0, 0.5, 0.5)

    def __post_init__(self):
        shape_factories = {
            "pyramid": lambda material: PyramidCfg(radius=0.18, height=0.3, visual_material=material),
            "cube": lambda material: sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.25), visual_material=material
            ),
            "sphere": lambda material: sim_utils.SphereCfg(radius=0.15, visual_material=material),
        }
        color_values = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        invalid_shapes = set(self.marker_shapes) - set(shape_factories)
        invalid_colors = set(self.marker_colors) - set(color_values)
        if invalid_shapes:
            raise ValueError(f"Unsupported goal marker shapes: {sorted(invalid_shapes)}")
        if invalid_colors:
            raise ValueError(f"Unsupported goal marker colors: {sorted(invalid_colors)}")
        if not self.marker_shapes or not self.marker_colors:
            raise ValueError("At least one goal marker shape and color must be configured.")
        if self.candidate_y_offsets is not None and not self.candidate_y_offsets:
            raise ValueError("candidate_y_offsets must contain at least one offset.")
        if self.slowdown_distance <= self.goal_tolerance:
            raise ValueError("slowdown_distance must be greater than goal_tolerance.")

        markers = {}
        for shape_name in self.marker_shapes:
            for color_name in self.marker_colors:
                material = sim_utils.PreviewSurfaceCfg(diffuse_color=color_values[color_name])
                markers[f"{shape_name}_{color_name}"] = shape_factories[shape_name](material)

        self.goal_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/navigation_goal",
            markers=markers,
        )
