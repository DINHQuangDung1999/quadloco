"""RSL-RL configuration for the 50 x 60 depth locomotion policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlDepthCNNActorCriticCfg(RslRlPpoActorCriticCfg):
    """Extra constructor arguments consumed by the vendored depth policy."""

    class_name = "ActorCriticDepthCNN"
    num_actor_obs_prop: int = 45
    obs_depth_shape: tuple[int, int] = (50, 60)


@configclass
class UnitreeGo2VisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for depth-based Unitree Go2 locomotion."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go2_vision_loco"
    empirical_normalization = False
    clip_actions = 100.0
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlDepthCNNActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
