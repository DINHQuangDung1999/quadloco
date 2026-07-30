import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from quadloco.tasks.manager_based.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import quadloco.tasks.manager_based.mdp as mdp


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
    viewer = ViewerCfg(
        eye=(-2.6, 0.0, 1.6),
        lookat=(0.0, 0.0, 0.3),
        asset_name="robot",
        origin_type="asset_root",
    )
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
            update_period=1.0 / 30.0,
            depth_clipping_behavior="max",
        )

        # Adds obs["camera"]["rgb"] and obs["camera"]["depth"] without
        # changing obs["policy"] or the trained locomotion policy input size.
        self.observations.camera = CameraObservationsCfg()

        self.commands.base_velocity = mdp.UniformGoalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            debug_vis=True,
            goal_tolerance=0.5,
            slowdown_distance=1.0,
            forward_velocity=1.0,
            yaw_gain=1.5,
            max_yaw_rate=0.8,
            marker_height=0.4,
            marker_shapes=("pyramid", "cube", "sphere"),
            marker_colors=("red", "green", "blue"),
            ranges=mdp.UniformGoalVelocityCommandCfg.Ranges(
                pos_x=(5.0, 5.0),
                pos_y=(-4.0, 4.0),
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

        # End an episode after the robot stays within the goal tolerance for 2 seconds.
        self.terminations.goal_reached = DoneTerm(
            func=mdp.goal_reached_for_duration,
            params={"command_name": "base_velocity", "duration_s": 0.2},
        )

        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/dung-admin/quadloco_ws/assets/kitchen/kitchen.usdc",
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
        )
        self.scene.terrain = None