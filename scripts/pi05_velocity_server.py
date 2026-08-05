#!/usr/bin/env python3
"""Serve a LeRobot PI0.5 velocity policy to the Isaac Lab play process.

This server targets the LeRobot 0.4.4 checkout installed in the Isaac Lab
environment. The client in ``quadloco_rsl_rl/play_vla.py`` deliberately uses
only the Python standard library, so it may also run in a separate environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import pickle
import socket
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# LeRobot moved this helper between 0.4 and 0.6.
try:
    from lerobot.common.control_utils import predict_action
except ModuleNotFoundError:
    from lerobot.utils.control_utils import predict_action

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config


_HEADER = struct.Struct("!Q")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Client disconnected while receiving a message.")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_message(connection: socket.socket) -> Any:
    (size,) = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
    return pickle.loads(_recv_exact(connection, size))  # noqa: S301 - trusted localhost peer


def _send_message(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PI0.5 velocity actions over a local TCP socket.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/summerschool/summerschool_ws/quadloco/outputs/"
            "pi05_quadloco_velocity/checkpoints/last/pretrained_model"
        ),
        help="LeRobot pretrained_model directory.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address on which to listen.")
    parser.add_argument("--port", type=int, default=5555, help="TCP port on which to listen.")
    parser.add_argument("--device", default="cuda", help="Torch device used for PI0.5 inference.")
    parser.add_argument(
        "--precision",
        choices=("fp16", "native"),
        default="fp16",
        help="Inference precision. fp16 converts all floating model tensors after loading.",
    )
    parser.add_argument("--use_amp", action="store_true", help="Use CUDA automatic mixed precision.")
    return parser.parse_args()


def _cuda_memory(label: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"[MEM] {label}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")


def _convert_floating_tensors_to_fp16(policy: torch.nn.Module, device: torch.device) -> None:
    """Convert model parameters/buffers and release cached source allocations."""
    if device.type != "cuda":
        raise ValueError("--precision fp16 currently requires a CUDA inference device.")

    _cuda_memory("before FP16 conversion", device)
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float16:
                parameter.data = parameter.data.to(dtype=torch.float16)
        for buffer in policy.buffers():
            if buffer.is_floating_point() and buffer.dtype != torch.float16:
                buffer.data = buffer.data.to(dtype=torch.float16)

    # Tensor replacement releases the old objects, but PyTorch normally keeps
    # their CUDA blocks in its process-local caching allocator. Explicitly
    # return those unused blocks to the driver before Isaac connects.
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    _cuda_memory("after FP16 conversion and empty_cache", device)


def _processor_config_without_disabled_relative_actions(
    source: Path,
    destination: Path,
) -> None:
    """Copy a processor JSON while removing newer disabled action steps."""
    with source.open() as file:
        config = json.load(file)

    unsupported_steps = {"relative_actions_processor", "absolute_actions_processor"}
    filtered_steps = []
    for step in config["steps"]:
        registry_name = step.get("registry_name")
        if registry_name not in unsupported_steps:
            filtered_steps.append(step)
            continue
        if step.get("config", {}).get("enabled", False):
            raise RuntimeError(
                f"Cannot remove enabled processor step {registry_name!r}. "
                "LeRobot 0.4 compatibility is only safe when relative actions are disabled."
            )
        print(f"[COMPAT] Removing disabled LeRobot 0.6 processor step: {registry_name}")

    config["steps"] = filtered_steps
    with destination.open("w") as file:
        json.dump(config, file, indent=2)


def _load_policy_config_compat(checkpoint: Path) -> PreTrainedConfig:
    """Load a PI0.5 config, filtering newer fields for LeRobot 0.4."""
    try:
        return PreTrainedConfig.from_pretrained(checkpoint)
    except Exception as error:
        if "not valid for PI05Config" not in str(error):
            raise

        with (checkpoint / "config.json").open() as file:
            config = json.load(file)
        # ``type`` is the draccus subclass discriminator rather than a
        # dataclass field, so it must be retained explicitly.
        valid_fields = {field.name for field in dataclasses.fields(PI05Config)} | {"type"}
        removed_fields = sorted(set(config) - valid_fields)
        if not removed_fields:
            raise
        print(
            "[COMPAT] Removing config fields unsupported by LeRobot 0.4: "
            + ", ".join(removed_fields)
        )
        filtered_config = {key: value for key, value in config.items() if key in valid_fields}

        with tempfile.TemporaryDirectory(prefix="pi05_lerobot04_config_") as temporary_directory:
            compatibility_path = Path(temporary_directory)
            with (compatibility_path / "config.json").open("w") as file:
                json.dump(filtered_config, file, indent=2)
            return PreTrainedConfig.from_pretrained(compatibility_path)


def _load_pre_post_processors_compat(
    policy_cfg: PreTrainedConfig,
    checkpoint: Path,
    device: torch.device,
):
    """Load processors normally, with a safe LeRobot 0.4 fallback."""
    preprocessor_overrides = {"device_processor": {"device": str(device)}}
    postprocessor_overrides = {"device_processor": {"device": "cpu"}}
    try:
        return make_pre_post_processors(
            policy_cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )
    except ImportError as error:
        error_text = str(error)
        if (
            "relative_actions_processor" not in error_text
            and "absolute_actions_processor" not in error_text
        ):
            raise

        # LeRobot 0.4 cannot resolve the newer, disabled action processors.
        # Construct a temporary processor-only checkpoint that omits them.
        # Model weights are not copied.
        print("[COMPAT] Retrying processor loading with LeRobot 0.4-compatible configs.")
        with tempfile.TemporaryDirectory(prefix="pi05_lerobot04_") as temporary_directory:
            compatibility_path = Path(temporary_directory)
            for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
                _processor_config_without_disabled_relative_actions(
                    checkpoint / filename,
                    compatibility_path / filename,
                )

            # Stateful normalization steps resolve their safetensors relative
            # to the processor JSON directory.
            for state_file in checkpoint.glob("*processor*.safetensors"):
                os.symlink(state_file, compatibility_path / state_file.name)

            return make_pre_post_processors(
                policy_cfg,
                pretrained_path=str(compatibility_path),
                preprocessor_overrides=preprocessor_overrides,
                postprocessor_overrides=postprocessor_overrides,
            )


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"PI0.5 config.json not found in checkpoint: {checkpoint}")

    device = torch.device(args.device)
    policy_cfg = _load_policy_config_compat(checkpoint)
    policy_cfg.device = str(device)
    policy_class = get_policy_class(policy_cfg.type)

    print(f"[INFO] Loading {policy_cfg.type} policy from: {checkpoint}")
    policy = policy_class.from_pretrained(checkpoint, config=policy_cfg)
    policy.to(device)
    _cuda_memory("after native checkpoint load", device)
    if args.precision == "fp16":
        _convert_floating_tensors_to_fp16(policy, device)
    policy.eval()
    use_amp = args.use_amp or args.precision == "fp16"
    preprocessor, postprocessor = _load_pre_post_processors_compat(policy_cfg, checkpoint, device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[INFO] PI0.5 velocity server ready at {args.host}:{args.port}")

        while True:
            connection, address = server.accept()
            print(f"[INFO] Isaac client connected from {address[0]}:{address[1]}")
            with connection:
                try:
                    while True:
                        request = _recv_message(connection)
                        if request.get("reset", False):
                            policy.reset()

                        rgb = np.asarray(request["rgb"])
                        if rgb.ndim != 3 or rgb.shape[-1] != 3:
                            raise ValueError(f"Expected HWC RGB input, received {rgb.shape}.")

                        observation = {
                            "observation.images.camera1": np.ascontiguousarray(rgb, dtype=np.uint8),
                            "observation.state": np.zeros((1,), dtype=np.float32),
                        }
                        start = time.perf_counter()
                        action = predict_action(
                            observation=observation,
                            policy=policy,
                            device=device,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            use_amp=use_amp,
                            task=str(request["task"]),
                        )
                        action_np = action.detach().cpu().numpy().astype(np.float32, copy=False)
                        # LeRobot 0.4 keeps the singleton inference batch
                        # dimension, whereas 0.6 returns the action squeezed.
                        if action_np.shape == (1, 3):
                            action_np = action_np[0]
                        if action_np.shape != (3,):
                            raise ValueError(
                                f"Expected PI0.5 velocity action shape (3,), received {action_np.shape}."
                            )
                        _send_message(
                            connection,
                            {
                                "action": np.ascontiguousarray(action_np),
                                "inference_s": time.perf_counter() - start,
                            },
                        )
                except (ConnectionError, EOFError, BrokenPipeError):
                    print("[INFO] Isaac client disconnected; waiting for another client.")
                except Exception as error:
                    try:
                        _send_message(connection, {"error": f"{type(error).__name__}: {error}"})
                    finally:
                        print(f"[ERROR] Closing client connection: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
