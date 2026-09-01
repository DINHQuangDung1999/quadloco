#!/usr/bin/env bash
set -euo pipefail

# Unified SmolVLA training/evaluation launcher.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$(command -v python)}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/checkpoints/smolvla_base_lerobot_0.4.4}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-direct_1000_40k_2epoch}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-direct-rgbd-1000}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-direct-rgbd-1000}"
TASKS="${TASKS:-direct}"
MODALITIES="${MODALITIES:-rgb rgbd}"
ACTION_MODE="${ACTION_MODE:-direct_velocity}"
RUN_PHASE="${RUN_PHASE:-train}"
DRY_RUN="${DRY_RUN:-false}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ_OVERRIDE="${SAVE_FREQ_OVERRIDE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

VISION_LORA_ENABLED="${VISION_LORA_ENABLED:-false}"
VISION_LORA_RANK="${VISION_LORA_RANK:-16}"
VISION_LORA_ALPHA="${VISION_LORA_ALPHA:-16.0}"
VISION_LORA_DROPOUT="${VISION_LORA_DROPOUT:-0.05}"
VISION_LORA_TARGETS="${VISION_LORA_TARGETS:-[\"q_proj\",\"k_proj\",\"v_proj\",\"out_proj\"]}"
DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID:-[16,16]}"
DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE:-pairwise_add}"
DEPTH_MAX="${DEPTH_MAX:-20.0}"

EPISODES="${EPISODES:-50}"
EVAL_EPISODE_LENGTH_S="${EVAL_EPISODE_LENGTH_S:-10.0}"
EVAL_SEED="${EVAL_SEED:-42}"
LOCOMOTION_CHECKPOINT="${LOCOMOTION_CHECKPOINT:-${ROOT}/ckpt/model_999.pt}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-5555}"
VLA_DEVICE="${VLA_DEVICE:-cuda}"
VLA_PRECISION="${VLA_PRECISION:-fp16}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"

bool() { case "$2" in true|false) ;; *) echo "$1 must be true or false" >&2; exit 2 ;; esac; }
for name in DRY_RUN GRADIENT_CHECKPOINTING TRAIN_EXPERT_ONLY FREEZE_VISION_ENCODER VISION_LORA_ENABLED PUSH_MODEL_TO_HUB WANDB_ENABLE; do bool "$name" "${!name}"; done
case "${RUN_PHASE}" in train|eval|all) ;; *) echo "RUN_PHASE must be train, eval, or all" >&2; exit 2 ;; esac
case "${ACTION_MODE}" in waypoint|direct_velocity) ;; *) echo "ACTION_MODE must be waypoint or direct_velocity" >&2; exit 2 ;; esac
(( BATCH_SIZE > 0 && STEPS > 0 && NUM_WORKERS >= 0 )) || { echo "Invalid batch, step, or worker count" >&2; exit 2; }
[[ "${VISION_LORA_ENABLED}" != true || "${FREEZE_VISION_ENCODER}" == true ]] || { echo "Vision LoRA requires a frozen vision encoder" >&2; exit 2; }
[[ -x "${LEROBOT_PYTHON}" && -x "${ISAACLAB_PYTHON}" ]] || { echo "Invalid Python interpreter" >&2; exit 2; }
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "Base checkpoint not found: ${BASE_MODEL}" >&2; exit 2; }
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

train_model() {
    local modality="$1" dataset_root="$2" dataset_repo="$3" output_dir="$4" model_repo="$5"
    env MODALITY="${modality}" DATASET_ROOT="${dataset_root}" DATASET_REPO="${dataset_repo}" \
      OUTPUT_DIR="${output_dir}" MODEL_REPO="${model_repo}" BASE_MODEL="${BASE_MODEL}" \
      ACTION_MODE="${ACTION_MODE}" STEPS="${STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-${STEPS}}" TRAIN_SEED="${TRAIN_SEED}" \
      GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY}" \
      FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER}" PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB}" \
      WANDB_ENABLE="${WANDB_ENABLE}" VISION_LORA_ENABLED="${VISION_LORA_ENABLED}" \
      VISION_LORA_RANK="${VISION_LORA_RANK}" VISION_LORA_ALPHA="${VISION_LORA_ALPHA}" \
      VISION_LORA_DROPOUT="${VISION_LORA_DROPOUT}" VISION_LORA_TARGETS="${VISION_LORA_TARGETS}" \
      DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID}" DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE}" DEPTH_MAX="${DEPTH_MAX}" \
      bash "${ROOT}/train_smolvla.sh"
}

evaluate_model() {
    local task="$1" checkpoint="$2" output_dir="$3"
    VLA_CHECKPOINT="${checkpoint}" OUTPUT_DIR="${output_dir}" TASKS="${task}" \
      LOCOMOTION_CHECKPOINT="${LOCOMOTION_CHECKPOINT}" EPISODES="${EPISODES}" \
      EPISODE_LENGTH_S="${EVAL_EPISODE_LENGTH_S}" EVAL_SEED="${EVAL_SEED}" \
      VLA_HOST="${VLA_HOST}" VLA_PORT="${VLA_PORT}" VLA_DEVICE="${VLA_DEVICE}" \
      VLA_PRECISION="${VLA_PRECISION}" SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT}" \
      bash "${ROOT}/eval_smolvla.sh"
}

cd "${ROOT}"
for task in ${TASKS}; do
  case "${task}" in direct|near_far|object_relative|occluded|relational) ;; *) echo "Unsupported task: ${task}" >&2; exit 2 ;; esac
  dataset_root="${DATASET_ROOT}"
  dataset_repo="${DATASET_REPO}"
  info="${dataset_root}/meta/info.json"; [[ -f "${info}" ]] || { echo "Dataset metadata not found: ${info}" >&2; exit 2; }
  frames="$(${LEROBOT_PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_frames"])' "${info}")"
  epochs="$(${LEROBOT_PYTHON} -c 'import sys; print(f"{int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3]):.3f}")' "${STEPS}" "${BATCH_SIZE}" "${frames}")"
  for modality in ${MODALITIES}; do
    case "${modality}" in rgb) fusion=clean_rgb ;; rgbd) fusion="${DEPTH_FUSION_MODE}_$(tr -d '[]' <<<"${DEPTH_TOKEN_GRID}" | tr ',' 'x')" ;; *) echo "Unsupported modality: ${modality}" >&2; exit 2 ;; esac
    tag="${fusion}_${ACTION_MODE}_30d-state"
    [[ "${VISION_LORA_ENABLED}" == false ]] || tag="${tag}_siglip_lora_r${VISION_LORA_RANK}"
    generated="${ROOT}/outputs/smolvla_quadloco_${modality}_${EXPERIMENT_NAME}_${tag}"
    output="${OUTPUT_DIR:-${generated}}"
    repo="${MODEL_REPO:-DinhQuangDung/smolvla-quadloco-${modality}-${EXPERIMENT_NAME//_/-}-${tag//_/-}}"
    checkpoint="${VLA_CHECKPOINT:-${output}/checkpoints/last/pretrained_model}"
    eval_output="${EVAL_OUTPUT_DIR:-${ROOT}/eval_results/smolvla_${modality}_${EXPERIMENT_NAME}_${tag}}"
    if [[ "${RUN_PHASE}" == train || "${RUN_PHASE}" == all ]]; then echo "[INFO] train ${task}/${modality}: ${STEPS} steps, ~${epochs} epochs, output=${output}"; [[ "${DRY_RUN}" == true ]] || train_model "${modality}" "${dataset_root}" "${dataset_repo}" "${output}" "${repo}"; fi
    if [[ "${RUN_PHASE}" == eval || "${RUN_PHASE}" == all ]]; then echo "[INFO] eval ${task}/${modality}: ${EPISODES} episodes, checkpoint=${checkpoint}"; [[ "${DRY_RUN}" == true ]] || evaluate_model "${task}" "${checkpoint}" "${eval_output}"; fi
  done
done
