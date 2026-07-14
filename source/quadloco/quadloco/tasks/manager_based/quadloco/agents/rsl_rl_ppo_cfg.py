# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

@configclass
class UnitreeGo2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_go2_loco"

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

@configclass
class UnitreeGo2RoughPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0,
                                                                    std_type="log")
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPO"
        self.experiment_name = "unitree_go2_rough_loco"

@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0,
                                                                    std_type="log")
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg = None
    )
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "PPO"
        self.experiment_name = "unitree_go2_flat_loco"