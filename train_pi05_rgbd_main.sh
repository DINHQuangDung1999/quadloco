#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-near_far-rgbd-small}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-all-rgbd}"
BASE_MODEL="${BASE_MODEL:-${QUADLOCO_ROOT}/checkpoints/pi05_base_lerobot_0.4.4}"
MODEL_REPO="${MODEL_REPO:-DinhQuangDung/pi05-quadloco-rgbd-waypoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUADLOCO_ROOT}/outputs/pi05_quadloco_rgbd_near_far_50k_16x16}"

STEPS="${STEPS:-18750}"
TRAIN_SEED="${TRAIN_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID:-[8,8]}"
DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE:-cross_attention}"
DEPTH_MAX="${DEPTH_MAX:-20.0}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-96}"
DEPTH_WIDTH="${DEPTH_WIDTH:-128}"
PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
ACTION_MODE="${ACTION_MODE:-${ACTION_REPRESENTATION:-waypoint}}"
STATE_MODE="${STATE_MODE:-vision_language_only}"

case "${STATE_MODE}" in
    vision_language_only)
        INPUT_FEATURES="{\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,480,640]},\"observation.depth.camera1\":{\"type\":\"VISUAL\",\"shape\":[1,${DEPTH_HEIGHT},${DEPTH_WIDTH}]}}"
        STATE_ENABLED=false
        STATE_TOKEN_DIM=null
        ;;
    proprioceptive_42d)
        INPUT_FEATURES="{\"observation.state\":{\"type\":\"STATE\",\"shape\":[45]},\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,480,640]},\"observation.depth.camera1\":{\"type\":\"VISUAL\",\"shape\":[1,${DEPTH_HEIGHT},${DEPTH_WIDTH}]}}"
        STATE_ENABLED=true
        STATE_TOKEN_DIM=42
        ;;
    *)
        echo "STATE_MODE must be vision_language_only or proprioceptive_42d" >&2
        exit 1
        ;;
esac

case "${ACTION_MODE}" in
    waypoint)
        OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[2]}}'
        RENAME_MAP='{}'
        JOB_NAME="pi05_quadloco_rgbd_waypoint"
        ;;
    direct_velocity)
        OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[3]}}'
        RENAME_MAP='{"observation.velocity_command":"action"}'
        JOB_NAME="pi05_quadloco_rgbd_direct_velocity"
        ;;
    *)
        echo "ACTION_MODE must be waypoint or direct_velocity" >&2
        exit 1
        ;;
esac

if [[ ! -x "${LEROBOT_PYTHON}" ]]; then
    echo "LeRobot Python executable not found: ${LEROBOT_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${LEROBOT_ROOT}/src/lerobot/scripts/lerobot_train.py" ]]; then
    echo "Vendored LeRobot training script not found: ${LEROBOT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Merged RGB-D dataset not found: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "PI0.5 base checkpoint not found: ${BASE_MODEL}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Training output already exists: ${OUTPUT_DIR}" >&2
    echo "Set OUTPUT_DIR to a new path, or use LeRobot's explicit resume workflow." >&2
    exit 1
fi

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${LEROBOT_PYTHON}" "${LEROBOT_ROOT}/src/lerobot/scripts/lerobot_train.py" \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.repo_id="${MODEL_REPO}" \
    --policy.push_to_hub="${PUSH_MODEL_TO_HUB}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.state_enabled="${STATE_ENABLED}" \
    --policy.state_token_dim="${STATE_TOKEN_DIM}" \
    --policy.action_mode="${ACTION_MODE}" \
    --policy.output_features="${OUTPUT_FEATURES}" \
    --rename_map="${RENAME_MAP}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --policy.depth_enabled=true \
    --policy.depth_feature_key=observation.depth.camera1 \
    --policy.depth_scale_feature_key=observation.depth_scale \
    --policy.depth_default_scale=0.001 \
    --policy.depth_min=0.05 \
    --policy.depth_max="${DEPTH_MAX}" \
    --policy.depth_stage_depths='[1,1,2]' \
    --policy.depth_stage_dims='[32,64,128]' \
    --policy.depth_patch_size=4 \
    --policy.depth_token_grid="${DEPTH_TOKEN_GRID}" \
    --policy.depth_drop_path_rate=0.0 \
    --policy.depth_resize_with_rgb=false \
    --policy.depth_cross_attention_heads=8 \
    --policy.depth_fusion_mode="${DEPTH_FUSION_MODE}" \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --seed="${TRAIN_SEED}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --log_freq=10 \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.disable_artifact=true
