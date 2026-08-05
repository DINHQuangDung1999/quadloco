#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ENV="${LEROBOT_ENV:-/home/summerschool/miniconda3/envs/env_lerobot}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/quadloco-vla-lerobot_no_depth}"
SOURCE_DATASET_REPO="${SOURCE_DATASET_REPO:-DinhQuangDung/quadloco-object-navigation}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/quadloco-vla-velocity-pi05}"
TRAIN_DATASET_REPO="${TRAIN_DATASET_REPO:-DinhQuangDung/quadloco-object-navigation-velocity}"
BASE_MODEL="${BASE_MODEL:-lerobot/pi05_base}"
MODEL_REPO="${MODEL_REPO:-DinhQuangDung/pi05-quadloco-velocity}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/summerschool/summerschool_ws/quadloco/outputs/pi05_quadloco_velocity}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if [[ ! -x "${LEROBOT_ENV}/bin/lerobot-train" ]]; then
    echo "LeRobot training executable not found: ${LEROBOT_ENV}/bin/lerobot-train" >&2
    exit 1
fi

if [[ ! -d "${TRAIN_DATASET_ROOT}/meta" ]]; then
    "${LEROBOT_ENV}/bin/python" \
        "${QUADLOCO_ROOT}/scripts/prepare_pi05_velocity_dataset.py" \
        --source-root "${SOURCE_DATASET_ROOT}" \
        --source-repo-id "${SOURCE_DATASET_REPO}" \
        --output-root "${TRAIN_DATASET_ROOT}" \
        --output-repo-id "${TRAIN_DATASET_REPO}"
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Training output already exists: ${OUTPUT_DIR}" >&2
    echo "Set OUTPUT_DIR to a new path, or use LeRobot's explicit resume workflow." >&2
    exit 1
fi

exec "${LEROBOT_ENV}/bin/lerobot-train" \
    --dataset.repo_id="${TRAIN_DATASET_REPO}" \
    --dataset.root="${TRAIN_DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.repo_id="${MODEL_REPO}" \
    --policy.push_to_hub="${PUSH_MODEL_TO_HUB}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=pi05_quadloco_velocity \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --log_freq=10 \
    --save_freq="${SAVE_FREQ}" \
    --env_eval_freq=0 \
    --wandb.enable="${WANDB_ENABLE}"
