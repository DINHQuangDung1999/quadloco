import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from quadloco.tasks.manager_based.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import quadloco.tasks.manager_based.mdp as mdp
from quadloco.tasks.manager_based.obstacle_assets import add_office_obstacles_to_scene


@configclass
class CameraObservationsCfg(ObsGroup):
    """Raw RGB-D observations kept separate from the locomotion policy."""

    rgb = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("front_camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    depth = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("front_camera"),
            "data_type": "distance_to_image_plane",
            "normalize": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class GoalNavigationEnvCfg(LocomotionVelocityRoughEnvCfg):

    # # Keep the interactive viewport centered on the robot in environment 0.
    # ### Side view
    # viewer = ViewerCfg(
    #     eye=(0.0, 2.6, 1.6),
    #     asset_name="robot",
    #     origin_type="asset_root",
    #     env_index=0,
    # )
    # ### Behind and above
    # viewer = ViewerCfg(
    #     eye=(-2.6, 0.0, 1.6),
    #     lookat=(0.0, 0.0, 0.3),
    #     asset_name="robot",
    #     origin_type="asset_root",
    # )
    # ### Intel D435i view
    # viewer = ViewerCfg(
    #     cam_prim_path="/World/envs/env_0/Robot/base/D435i",
    #     resolution=(1920, 1080),
    # )
    def __post_init__(self):
        super().__post_init__()

        # Match the action scaling used to train UnitreeGo2RoughEnvCfg.
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # Forward-facing RGB-D camera mounted to the Go2 base. The camera is a
        # scene sensor only; it is intentionally not added to the locomotion
        # policy observations, preserving compatibility with the trained policy.
        self.scene.front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/D435i",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.30, 0.0, 0.10),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                # Approximate the D435i color-camera horizontal field of view.
                focal_length=1.93,
                horizontal_aperture=2.65,
                clipping_range=(0.1, 20.0),
            ),
            width=640,
            height=480,
            # Collection runs at 50 Hz. Update every control step so RGB-D
            # frames are not silently duplicated in a dataset labeled 50 Hz.
            update_period=1.0 / 50.0,
            depth_clipping_behavior="max",
        )

        # Adds obs["camera"]["rgb"] and obs["camera"]["depth"] without
        # changing obs["policy"] or the trained locomotion policy input size.
        self.observations.camera = CameraObservationsCfg()

        self.commands.base_velocity = mdp.UniformGoalVelocityCommandCfg(
            asset_name="robot",
            # Navigation commands are episode-scoped. Keep command resampling
            # beyond the environment time limit to avoid a new instruction at
            # the terminal frame.
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            goal_tolerance=1.0,
            slowdown_distance=1.5,
            forward_velocity=1.0,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            task_template="Navigate to the {color} {shape}",
            candidate_y_offsets=(-1.0, 0.0, 1.0),
            # Keep the actual target x uniform on ranges.pos_x. Adding
            # independent x jitter here would broaden it beyond [4, 8] m.
            candidate_x_error=0.0,
            candidate_y_spacing_error=0.75,
            ranges=mdp.UniformGoalVelocityCommandCfg.Ranges(
                pos_x=(4.0, 8.0),
                pos_y=(-2.0, 2.0),
            ),
        )

        # Spawn every shape/color variant as a real, non-colliding kinematic
        # scene object. The command term moves the three selected variants into
        # view and parks the unused variants below the stage on every reset.
        command_cfg = self.commands.base_velocity
        for shape in command_cfg.marker_shapes:
            for color in command_cfg.marker_colors:
                asset_name = f"{command_cfg.candidate_asset_prefix}_{shape}_{color}"
                setattr(
                    self.scene,
                    asset_name,
                    RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/GoalObject_{shape}_{color}",
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(0.0, 0.0, command_cfg.unused_candidate_height),
                        ),
                        spawn=mdp.make_goal_object_spawn_cfg(shape, color),
                    ),
                )

        # Alternative three-object navigation setting (uncomment to enable):
        #
        # - The robot starts close to its environment origin, with small pose
        #   randomization and faces approximately toward the objects.
        # - Three candidate objects share x=5 m. Their y coordinates are
        #   -1, 0, and +1 m, so the middle object is centered at y=0.
        # - All three objects are visible, but one is randomly selected as the
        #   true goal on every reset. The goal uses the first configured marker
        #   style (pyramid/red); distractors use the remaining styles.
        #
        # self.events.reset_base.params["pose_range"].update(
        #     {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.10, 0.10)}
        # )
        # self.commands.base_velocity.ranges.pos_x = (5.0, 5.0)
        # self.commands.base_velocity.ranges.pos_y = (0.0, 0.0)
        # self.commands.base_velocity.candidate_y_offsets = (-1.0, 0.0, 1.0)

        # Match evaluation's 0.5 s stable-stop requirement and retain enough
        # zero-command frames to supervise stopping behavior.
        self.terminations.goal_reached = DoneTerm(
            func=mdp.goal_reached_for_duration,
            params={"command_name": "base_velocity", "duration_s": 0.5},
        )


        self.scene.ratlab = AssetBaseCfg(
            prim_path="/World/ground",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(3.5, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=(
                    "/home/summerschool/summerschool_ws/"
                    "assets/rat_lab/multicorridor/lab_sense.usd"
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
        )
        self.scene.terrain = None

        # self.scene.kitchen = AssetBaseCfg(
        #     prim_path="/World/ground",
        #     init_state=AssetBaseCfg.InitialStateCfg(
        #         pos=(0.0, 0.0, 0.0),
        #     ),
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=(
        #             "/home/summerschool/summerschool_ws/"
        #             "assets/kitchen/kitchen.usdc"
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=True,
        #         ),
        #     ),
        # )
        # # Replace the inherited generated rough terrain with a flat support
        # # plane so it does not visually occlude the RATLab USD.
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None


        self.curriculum.terrain_levels = None


@configclass
class OccludedGoalNavigationEnvCfg(GoalNavigationEnvCfg):
    """Goal navigation with a solid occluder and A*-generated waypoints."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.OccludedGoalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            goal_tolerance=1.0,
            slowdown_distance=1.5,
            forward_velocity=1.0,
            # Keep the locomotion policy above its low-speed dead zone until
            # the robot actually enters the 1 m success region.
            minimum_approach_velocity=0.15,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            task_template="Navigate to the {color} {shape}",
            occluded_task_template=(
                "Navigate to the {color} {shape} behind the {obstacle_type}"
            ),
            obstacle_lateral_offset_fraction=(0.4, 0.6),
            candidate_y_offsets=(-1.0, 0.0, 1.0),
            candidate_x_error=0.0,
            candidate_y_spacing_error=0.5,
            ranges=mdp.OccludedGoalVelocityCommandCfg.Ranges(
                pos_x=(4.0, 8.0),
                pos_y=(-2.0, 2.0),
            ),
        )

        command_cfg = self.commands.base_velocity
        add_office_obstacles_to_scene(
            self.scene,
            asset_prefix=command_cfg.occlusion_obstacle_asset_prefix,
            unused_height=command_cfg.unused_candidate_height,
        )


@configclass
class RelationalGoalNavigationEnvCfg(GoalNavigationEnvCfg):
    """Two-pair relational target-selection navigation."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.RelationalGoalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            goal_tolerance=1.0,
            slowdown_distance=1.5,
            forward_velocity=1.0,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            shape_instruction_names=("pyramid", "box", "ball"),
            object_size_names=("small", "big"),
            object_size_scales=(0.75, 1.25),
            relation_types=("next_to",),
            pair_center_y=(-1.5, 1.5),
            within_pair_spacing_range=(0.75, 0.85),
            target_y_jitter=0.05,
            pair_center_y_jitter=0.2,
            ranges=mdp.RelationalGoalVelocityCommandCfg.Ranges(
                pos_x=(4.0, 8.0),
                pos_y=(0.0, 0.0),
            ),
        )

        command_cfg = self.commands.base_velocity
        # Each pair has one colored target primitive. The paired reference is
        # selected from the reusable office/warehouse asset catalog below.
        for slot in range(2):
            for shape in command_cfg.marker_shapes:
                for color in command_cfg.marker_colors:
                    for size_name, size_scale in zip(
                        command_cfg.object_size_names,
                        command_cfg.object_size_scales,
                    ):
                        asset_name = (
                            f"{command_cfg.relational_asset_prefix}_"
                            f"{slot}_{shape}_{color}_{size_name}"
                        )
                        setattr(
                            self.scene,
                            asset_name,
                            RigidObjectCfg(
                                prim_path=(
                                    f"{{ENV_REGEX_NS}}/RelationalObject_"
                                    f"{slot}_{shape}_{color}_{size_name}"
                                ),
                                init_state=RigidObjectCfg.InitialStateCfg(
                                    pos=(
                                        0.0,
                                        0.0,
                                        command_cfg.unused_candidate_height,
                                    ),
                                ),
                                spawn=mdp.make_goal_object_spawn_cfg(
                                    shape, color, size_scale
                                ),
                            ),
                        )

        for slot in range(2):
            add_office_obstacles_to_scene(
                self.scene,
                asset_prefix=f"{command_cfg.relational_reference_asset_prefix}_{slot}",
                prim_prefix=f"RelationalReference_{slot}",
                unused_height=command_cfg.unused_candidate_height,
            )


@configclass
class NearFarGoalNavigationEnvCfg(GoalNavigationEnvCfg):
    """Select the nearest or farthest of three identical objects."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.NearFarGoalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            goal_tolerance=1.0,
            slowdown_distance=1.5,
            forward_velocity=1.0,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            shape_instruction_names=("pyramid", "box", "ball"),
            nominal_distances=(4.0, 5.5, 7.0),
            group_x_offset_range=(-2.0, 1.0),
            distance_jitter=0.5,
            lateral_spacing_range=(0.75, 1.5),
            lateral_group_jitter=0.5,
            ranges=mdp.NearFarGoalVelocityCommandCfg.Ranges(
                pos_x=(4.0, 8.0),
                pos_y=(-2.0, 2.0),
            ),
        )

        command_cfg = self.commands.base_velocity
        for slot in range(len(command_cfg.nominal_distances)):
            for shape in command_cfg.marker_shapes:
                for color in command_cfg.marker_colors:
                    asset_name = (
                        f"{command_cfg.distance_asset_prefix}_"
                        f"{slot}_{shape}_{color}"
                    )
                    setattr(
                        self.scene,
                        asset_name,
                        RigidObjectCfg(
                            prim_path=(
                                f"{{ENV_REGEX_NS}}/DistanceObject_"
                                f"{slot}_{shape}_{color}"
                            ),
                            init_state=RigidObjectCfg.InitialStateCfg(
                                pos=(
                                    0.0,
                                    0.0,
                                    command_cfg.unused_candidate_height,
                                ),
                            ),
                            spawn=mdp.make_goal_object_spawn_cfg(shape, color),
                        ),
                    )


@configclass
class ObjectRelativeGoalNavigationEnvCfg(GoalNavigationEnvCfg):
    """Metric standing positions relative to a single visible object."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = mdp.ObjectRelativeGoalWaypointCommandCfg(
            asset_name="robot",
            # Commands are episode-scoped. This must exceed episode_length_s;
            # matching the 20 s timeout allowed the next command to be sampled
            # immediately before termination and mislabeled the final frame.
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            goal_tolerance=0.2,
            slowdown_distance=0.8,
            forward_velocity=1.0,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            shape_instruction_names=("pyramid", "box", "ball"),
            relations=("front", "behind", "left", "right"),
            metric_offsets=(0.5, 0.75, 1.0, 1.25),
            candidate_y_offsets=(-1.0, 0.0, 1.0),
            candidate_x_error=0.0,
            candidate_y_spacing_error=0.3,
            ranges=mdp.ObjectRelativeGoalWaypointCommandCfg.Ranges(
                pos_x=(4.0, 8.0),
                pos_y=(-2.0, 2.0),
            ),
        )
