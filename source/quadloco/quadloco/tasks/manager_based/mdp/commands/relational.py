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

class UniformGoalVelocityCommandRelational(UniformGoalVelocityCommandDirect):
    """Select one of two identical targets through an asset-based reference."""

    cfg: RelationalGoalVelocityCommandCfg

    def __init__(self, cfg: RelationalGoalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.relational_target_assets = [
            [
                self._env.scene[
                    f"{cfg.relational_asset_prefix}_{slot}_{shape}_{color}_{size_name}"
                ]
                for shape in cfg.marker_shapes
                for color in cfg.marker_colors
                for size_name in cfg.object_size_names
            ]
            for slot in range(2)
        ]
        self.relational_reference_assets = [
            [
                self._env.scene[
                    f"{cfg.relational_reference_asset_prefix}_{slot}_{asset_name}_collision"
                ]
                for asset_name in cfg.reference_asset_names
            ]
            for slot in range(2)
        ]
        # Reference models may have multiple visual variants under one spoken
        # category (for example, two tables). Draw different categories for
        # the two pairs so the language instruction is never ambiguous.
        grouped_variants: dict[str, list[int]] = {}
        for variant_index, label in enumerate(cfg.reference_instruction_names):
            grouped_variants.setdefault(label, []).append(variant_index)
        self.reference_variant_groups = list(grouped_variants.values())

    def _decode_variant(self, variant: int) -> tuple[int, int, int]:
        num_colors = len(self.cfg.marker_colors)
        num_sizes = len(self.cfg.object_size_names)
        shape_index = variant // (num_colors * num_sizes)
        remainder = variant % (num_colors * num_sizes)
        return shape_index, remainder // num_sizes, remainder % num_sizes

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        num_shapes = len(self.cfg.marker_shapes)
        num_colors = len(self.cfg.marker_colors)
        num_sizes = len(self.cfg.object_size_names)
        num_variants = num_shapes * num_colors * num_sizes

        pair_x = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.pos_x
        )
        pair_center_y = torch.tensor(
            self.cfg.pair_center_y, device=self.device
        ).repeat(len(env_ids), 1)
        if self.cfg.pair_center_y_jitter > 0.0:
            pair_center_y += torch.empty_like(pair_center_y).uniform_(
                -self.cfg.pair_center_y_jitter,
                self.cfg.pair_center_y_jitter,
            )

        pair_spacing = torch.empty(
            len(env_ids), 2, device=self.device
        ).uniform_(*self.cfg.within_pair_spacing_range)
        target_y_jitter = torch.empty(
            len(env_ids), 2, device=self.device
        ).uniform_(-self.cfg.target_y_jitter, self.cfg.target_y_jitter)

        for row, env_id in enumerate(env_ids.tolist()):
            shared_variant = int(
                torch.randint(num_variants, (1,), device=self.device).item()
            )
            selected_groups = torch.randperm(
                len(self.reference_variant_groups), device=self.device
            )[:2].tolist()
            reference_variants = [
                group[
                    int(torch.randint(len(group), (1,), device=self.device).item())
                ]
                for group in (
                    self.reference_variant_groups[index] for index in selected_groups
                )
            ]
            selected_pair = int(torch.randint(2, (1,), device=self.device).item())

            origin = self._env.scene.env_origins[env_id]
            target_positions = torch.zeros(2, 3, device=self.device)
            reference_positions = torch.zeros(2, 3, device=self.device)
            target_positions[:, 0] = origin[0] + pair_x[row]
            reference_positions[:, 0] = origin[0] + pair_x[row]
            target_positions[:, 2] = origin[2]
            reference_positions[:, 2] = origin[2]
            for pair_index in range(2):
                reference_height = self.cfg.reference_footprints[
                    reference_variants[pair_index]
                ][2]
                shape_index, _, size_index = self._decode_variant(shared_variant)
                target_positions[pair_index, 2] += goal_object_center_height(
                    self.cfg.marker_shapes[shape_index],
                    self.cfg.object_size_scales[size_index],
                )
                reference_positions[pair_index, 2] += 0.5 * reference_height
                center_y = origin[1] + pair_center_y[row, pair_index]
                side = -1.0 if bool(torch.randint(2, (1,), device=self.device).item()) else 1.0
                reference_positions[pair_index, 1] = center_y
                target_positions[pair_index, 1] = (
                    center_y
                    + side * pair_spacing[row, pair_index]
                    + target_y_jitter[row, pair_index]
                )

            self.goal_pos_w[env_id] = target_positions[selected_pair]
            self.marker_indices[env_id] = shared_variant

            target_shape_index, target_color_index, target_size_index = (
                self._decode_variant(shared_variant)
            )
            target_shape = self.cfg.shape_instruction_names[target_shape_index]
            target_color = self.cfg.marker_colors[target_color_index]
            target_size = self.cfg.object_size_names[target_size_index]
            selected_reference = reference_variants[selected_pair]
            self.goal_task_names[env_id] = self.cfg.next_to_task_template.format(
                size=target_size,
                color=target_color,
                shape=target_shape,
                reference_object=self.cfg.reference_instruction_names[selected_reference],
            )

            single_env_id = torch.tensor([env_id], device=self.device)
            for slot, slot_assets in enumerate(self.relational_target_assets):
                for variant_index, asset in enumerate(slot_assets):
                    root_pose = asset.data.default_root_state[env_id : env_id + 1, :7].clone()
                    root_pose[:, :3] = self._env.scene.env_origins[env_id]
                    root_pose[:, 2] = self.cfg.unused_candidate_height
                    if shared_variant == variant_index:
                        root_pose[0, :3] = target_positions[slot]
                    asset.write_root_pose_to_sim(
                        root_pose,
                        env_ids=single_env_id,
                    )

            for slot, slot_assets in enumerate(self.relational_reference_assets):
                for variant_index, asset in enumerate(slot_assets):
                    root_pose = asset.data.default_root_state[env_id : env_id + 1, :7].clone()
                    root_pose[0, :3] = origin
                    root_pose[0, 2] = self.cfg.unused_candidate_height
                    if reference_variants[slot] == variant_index:
                        root_pose[0, :3] = reference_positions[slot]
                        yaw = torch.empty(1, device=self.device).uniform_(
                            *self.cfg.reference_yaw_range
                        )
                        zeros = torch.zeros_like(yaw)
                        root_pose[0, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)[0]
                    asset.write_root_pose_to_sim(root_pose, env_ids=single_env_id)

        self.goal_reached[env_ids] = False
        self.goal_pos_b[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0
        self.waypoint_command[env_ids] = 0.0
        self.velocity_command[env_ids] = 0.0
        self.goal_generation[env_ids] += 1




@configclass
class RelationalGoalVelocityCommandCfg(UniformGoalVelocityCommandCfg):
    """Configuration for :class:`UniformGoalVelocityCommandRelational`."""

    class_type: type = UniformGoalVelocityCommandRelational

    relational_asset_prefix: str = "relational_object"
    relational_reference_asset_prefix: str = "relational_reference"
    reference_asset_names: tuple[str, ...] = tuple(
        spec.name for spec in OFFICE_OBSTACLE_SPECS
    )
    reference_instruction_names: tuple[str, ...] = tuple(
        spec.instruction_name for spec in OFFICE_OBSTACLE_SPECS
    )
    reference_footprints: tuple[tuple[float, float, float], ...] = tuple(
        spec.footprint for spec in OFFICE_OBSTACLE_SPECS
    )
    reference_yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    next_to_task_template: str = (
        "Navigate to the {size} {color} {shape} next to the {reference_object}"
    )
    relation_types: tuple[str, ...] = ("next_to",)
    shape_instruction_names: tuple[str, ...] = ("pyramid", "box", "ball")
    object_size_names: tuple[str, ...] = ("small", "big")
    object_size_scales: tuple[float, ...] = (0.75, 1.25)
    pair_center_y: tuple[float, float] = (-1.5, 1.5)
    # Target distance from its reference along Y, plus a small independent
    # perturbation that avoids identical layouts across resets.
    within_pair_spacing_range: tuple[float, float] = (0.75, 0.85)
    target_y_jitter: float = 0.05
    pair_center_y_jitter: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        try:
            template_values = dict(
                size="small",
                color="red",
                shape="cube",
                reference_object="high-back chair",
            )
            self.next_to_task_template.format(**template_values)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "next_to_task_template must use target fields and "
                "{reference_object}."
            ) from exc
        if self.relation_types != ("next_to",):
            raise ValueError("Asset-based relational navigation currently uses 'next_to'.")
        if len(self.reference_asset_names) < 2:
            raise ValueError("Relational navigation requires at least two reference assets.")
        if len(set(self.reference_instruction_names)) < 2:
            raise ValueError(
                "Relational navigation requires at least two distinct reference labels."
            )
        if len(set(self.reference_asset_names)) != len(self.reference_asset_names):
            raise ValueError("reference_asset_names must be unique.")
        if not (
            len(self.reference_asset_names)
            == len(self.reference_instruction_names)
            == len(self.reference_footprints)
        ):
            raise ValueError("Reference asset names, labels, and footprints must align.")
        if any(
            len(size) != 3 or min(size) <= 0.0 for size in self.reference_footprints
        ):
            raise ValueError("Reference footprints must contain positive 3-D sizes.")
        min_yaw, max_yaw = self.reference_yaw_range
        if max_yaw < min_yaw:
            raise ValueError("reference_yaw_range must be ordered.")
        if (
            not self.object_size_names
            or len(self.object_size_names) != len(self.object_size_scales)
        ):
            raise ValueError("Object size names and scales must have equal nonzero length.")
        if len(self.shape_instruction_names) != len(self.marker_shapes):
            raise ValueError(
                "shape_instruction_names must align with marker_shapes."
            )
        if any(scale <= 0.0 for scale in self.object_size_scales):
            raise ValueError("Object size scales must be positive.")
        if len(self.pair_center_y) != 2:
            raise ValueError("pair_center_y must contain exactly two pair centers.")
        if self.pair_center_y[0] >= self.pair_center_y[1]:
            raise ValueError("pair_center_y must be strictly increasing.")
        min_spacing, max_spacing = self.within_pair_spacing_range
        if min_spacing <= 0.0 or max_spacing < min_spacing:
            raise ValueError("within_pair_spacing_range must be positive and ordered.")
        if self.pair_center_y_jitter < 0.0:
            raise ValueError("pair_center_y_jitter must be nonnegative.")
        if self.target_y_jitter < 0.0:
            raise ValueError("target_y_jitter must be nonnegative.")

