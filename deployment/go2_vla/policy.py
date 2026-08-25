"""Local PI0.5 direct-velocity inference."""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import copy
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.policies.utils import prepare_observation_for_inference


class Pi05VelocityPolicy:
    """Load a patched LeRobot PI0.5 checkpoint and return [vx, vy, wz]."""

    def __init__(self, checkpoint: Path, device_name: str) -> None:
        checkpoint = checkpoint.expanduser().resolve()
        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
        with config_path.open() as file:
            raw_config = json.load(file)
        action_shape = raw_config["output_features"]["action"]["shape"]
        if action_shape != [3]:
            raise ValueError(f"Expected direct velocity action shape [3], got {action_shape}.")

        state_feature = raw_config.get("input_features", {}).get("observation.state")
        self.state_dim = 0 if state_feature is None else int(state_feature["shape"][0])
        self.depth_enabled = bool(raw_config.get("depth_enabled", False))
        self.device = torch.device(device_name)
        self.config = PreTrainedConfig.from_pretrained(checkpoint)
        self.config.device = str(self.device)
        policy_class = get_policy_class(self.config.type)
        print(f"[INFO] Loading {self.config.type} policy from {checkpoint}")
        self.policy = policy_class.from_pretrained(checkpoint, config=self.config).to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self) -> None:
        self.policy.reset()

    def predict(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray | None,
        depth_scale: float,
        instruction: str,
        state: np.ndarray | None,
    ) -> np.ndarray:
        observation: dict[str, np.ndarray] = {
            "observation.images.camera1": np.ascontiguousarray(rgb, dtype=np.uint8)
        }
        if self.state_dim:
            if state is None or state.shape != (self.state_dim,):
                shape = None if state is None else state.shape
                raise ValueError(f"Checkpoint expects state ({self.state_dim},), got {shape}.")
            observation["observation.state"] = np.ascontiguousarray(state, dtype=np.float32)
        if self.depth_enabled:
            if depth_z16 is None:
                raise ValueError("This checkpoint requires aligned depth, but the camera is RGB-only.")
            observation["observation.depth.camera1"] = np.ascontiguousarray(depth_z16, dtype=np.uint16)
            observation["observation.depth_scale"] = np.asarray([depth_scale], dtype=np.float32)

        amp = torch.autocast(device_type="cuda") if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), amp:
            prepared = prepare_observation_for_inference(copy(observation), self.device, instruction)
            action = self.policy.select_action(self.preprocessor(prepared))
            action = self.postprocessor(action)
        result = action.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if result.shape != (3,):
            raise ValueError(f"Expected [vx, vy, wz], received {result.shape}.")
        return result
