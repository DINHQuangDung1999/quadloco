# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Optionally select and reorder recorded state features before padding.
    # If unset, state_token_dim retains a leading subset for compatibility.
    state_feature_indices: tuple[int, ...] | None = None
    state_token_dim: int | None = None
    action_mode: str = "auto"

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Optional metric-depth token encoder. The encoder is shared with PI0.5 so
    # RGB-D ablations use the same ConvNeXt architecture and depth semantics.
    depth_enabled: bool = False
    depth_feature_key: str = "observation.depth.camera1"
    depth_scale_feature_key: str = "observation.depth_scale"
    depth_default_scale: float = 0.001
    depth_min: float = 0.05
    depth_max: float = 20.0
    # Match the compact encoder used by the reported PI0.5 RGB-D experiments.
    depth_stage_depths: tuple[int, ...] = (1, 1, 2)
    depth_stage_dims: tuple[int, ...] = (32, 64, 128)
    depth_patch_size: int = 4
    depth_token_grid: tuple[int, int] = (8, 8)
    depth_drop_path_rate: float = 0.0
    depth_resize_with_rgb: bool = True
    depth_cross_attention_heads: int = 8
    depth_fusion_mode: str = "pairwise_add"

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    gradient_checkpointing: bool = False

    # Hybrid visual adaptation: keep the pretrained SigLIP weights frozen and
    # train low-rank residuals inside its self-attention projections. The
    # action expert and task-specific projectors remain ordinarily trainable.
    vision_lora_enabled: bool = False
    vision_lora_rank: int = 16
    vision_lora_alpha: float = 16.0
    vision_lora_dropout: float = 0.05
    vision_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.state_token_dim is not None:
            if self.state_token_dim <= 0:
                raise ValueError("state_token_dim must be positive when provided")
            if self.state_token_dim > self.max_state_dim:
                raise ValueError("state_token_dim cannot exceed max_state_dim")
            if OBS_STATE in (self.input_features or {}):
                recorded_state_dim = self.input_features[OBS_STATE].shape[0]
                if self.state_token_dim > recorded_state_dim:
                    raise ValueError(
                        f"state_token_dim={self.state_token_dim} exceeds recorded state dimension "
                        f"{recorded_state_dim}"
                    )
        if self.state_feature_indices is not None:
            if not self.state_feature_indices:
                raise ValueError("state_feature_indices cannot be empty")
            if len(set(self.state_feature_indices)) != len(self.state_feature_indices):
                raise ValueError("state_feature_indices cannot contain duplicates")
            if min(self.state_feature_indices) < 0:
                raise ValueError("state_feature_indices cannot contain negative indices")
            if len(self.state_feature_indices) > self.max_state_dim:
                raise ValueError("state_feature_indices length cannot exceed max_state_dim")
            if self.state_token_dim is not None and len(self.state_feature_indices) != self.state_token_dim:
                raise ValueError("state_feature_indices length must equal state_token_dim")
            if OBS_STATE in (self.input_features or {}) and max(self.state_feature_indices) >= self.input_features[OBS_STATE].shape[0]:
                raise ValueError("state_feature_indices exceeds recorded state dimension")
        if self.action_mode not in ("auto", "waypoint", "direct_velocity"):
            raise ValueError(
                "action_mode must be 'auto', 'waypoint', or 'direct_velocity', "
                f"got {self.action_mode!r}"
            )
        if self.vision_lora_enabled:
            supported_targets = {"q_proj", "k_proj", "v_proj", "out_proj"}
            unknown_targets = set(self.vision_lora_targets) - supported_targets
            if self.vision_lora_rank <= 0:
                raise ValueError("vision_lora_rank must be positive")
            if self.vision_lora_alpha <= 0:
                raise ValueError("vision_lora_alpha must be positive")
            if not 0.0 <= self.vision_lora_dropout < 1.0:
                raise ValueError("vision_lora_dropout must be in [0, 1)")
            if not self.vision_lora_targets:
                raise ValueError("vision_lora_targets cannot be empty")
            if unknown_targets:
                raise ValueError(
                    "Unsupported SigLIP LoRA targets: "
                    f"{sorted(unknown_targets)}; supported targets are {sorted(supported_targets)}"
                )
            if not self.freeze_vision_encoder:
                raise ValueError(
                    "vision_lora_enabled=True requires freeze_vision_encoder=True; "
                    "otherwise the base SigLIP weights would also be updated"
                )
        if self.depth_enabled:
            if self.depth_fusion_mode not in ("concatenate", "pairwise_add", "cross_attention"):
                raise ValueError(
                    "depth_fusion_mode must be 'concatenate', 'pairwise_add', or "
                    f"'cross_attention', got {self.depth_fusion_mode!r}"
                )
            if self.depth_default_scale <= 0:
                raise ValueError("depth_default_scale must be positive")
            if self.depth_cross_attention_heads <= 0:
                raise ValueError("depth_cross_attention_heads must be positive")
            if self.depth_feature_key not in (self.input_features or {}):
                raise ValueError(
                    f"depth_enabled=True requires {self.depth_feature_key!r} in input_features"
                )
            depth_feature = self.input_features[self.depth_feature_key]
            if depth_feature.type is not FeatureType.VISUAL:
                raise ValueError(
                    f"{self.depth_feature_key!r} must use FeatureType.VISUAL to bypass state normalization"
                )
            if len(depth_feature.shape) != 3 or 1 not in (
                depth_feature.shape[0],
                depth_feature.shape[-1],
            ):
                raise ValueError(
                    f"{self.depth_feature_key!r} must have shape [1,H,W] or [H,W,1], "
                    f"received {depth_feature.shape}"
                )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
