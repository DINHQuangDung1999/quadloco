#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-near_far-rgbd}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-all-rgbd}"
BASE_MODEL="${BASE_MODEL:-${QUADLOCO_ROOT}/checkpoints/pi05_base_lerobot_0.4.4}"
MODEL_REPO="${MODEL_REPO:-DinhQuangDung/pi05-quadloco-rgb-waypoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUADLOCO_ROOT}/outputs/pi05_quadloco_rgb_near_far_50k}"

STEPS="${STEPS:-12500}"
TRAIN_SEED="${TRAIN_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-300000}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
ACTION_MODE="${ACTION_MODE:-${ACTION_REPRESENTATION:-waypoint}}"
STATE_MODE="${STATE_MODE:-vision_language_only}"

case "${STATE_MODE}" in
    vision_language_only)
        INPUT_FEATURES='{"observation.images.camera1":{"type":"VISUAL","shape":[3,480,640]}}'
        STATE_ENABLED=false
        STATE_TOKEN_DIM=null
        ;;
    proprioceptive_42d)
        # The dataset stores 45D policy observations. Only the first 42 are
        # tokenized; the final oracle velocity command [vx, vy, wz] is purged.
        INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[45]},"observation.images.camera1":{"type":"VISUAL","shape":[3,480,640]}}'
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
        JOB_NAME="pi05_quadloco_rgb_waypoint"
        ;;
    direct_velocity)
        OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[3]}}'
        RENAME_MAP='{"observation.velocity_command":"action"}'
        JOB_NAME="pi05_quadloco_rgb_direct_velocity"
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
    --policy.depth_enabled=false \
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
