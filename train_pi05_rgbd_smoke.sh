#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-all-rgbd}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-all-rgbd}"
BASE_MODEL="${BASE_MODEL:-${QUADLOCO_ROOT}/checkpoints/pi05_base_lerobot_0.4.4}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUADLOCO_ROOT}/outputs/pi05_quadloco_depth_smoke}"
STEPS="${STEPS:-10}"

if [[ ! -x "${LEROBOT_PYTHON}" ]]; then
    echo "LeRobot Python executable not found: ${LEROBOT_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset not found: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output already exists: ${OUTPUT_DIR}" >&2
    echo "Remove it explicitly or set OUTPUT_DIR to a new path." >&2
    exit 1
fi

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# This is a local smoke test. Prevent all Hugging Face network access so a
# malformed or inherited push_to_hub setting cannot upload checkpoints.
export HF_HUB_OFFLINE=1
exec "${LEROBOT_PYTHON}" "${LEROBOT_ROOT}/src/lerobot/scripts/lerobot_train.py" \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.push_to_hub=false \
    --policy.input_features='{"observation.state":{"type":"STATE","shape":[45]},"observation.images.camera1":{"type":"VISUAL","shape":[3,480,640]},"observation.depth.camera1":{"type":"VISUAL","shape":[1,96,128]}}' \
    --policy.output_features='{"action":{"type":"ACTION","shape":[2]}}' \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.depth_enabled=true \
    --policy.depth_feature_key=observation.depth.camera1 \
    --policy.depth_scale_feature_key=observation.depth_scale \
    --policy.depth_default_scale=0.001 \
    --policy.depth_min=0.05 \
    --policy.depth_max=20.0 \
    --policy.depth_stage_depths='[1,1,2]' \
    --policy.depth_stage_dims='[32,64,128]' \
    --policy.depth_token_grid='[8,8]' \
    --policy.depth_resize_with_rgb=false \
    --policy.depth_cross_attention_heads=8 \
    --policy.depth_fusion_mode=cross_attention \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=pi05_quadloco_depth_smoke \
    --batch_size=3 \
    --num_workers=0 \
    --steps="${STEPS}" \
    --log_freq=1 \
    --eval_freq=0 \
    --save_checkpoint=false \
    --wandb.enable=false
