#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Experiment template: edit this block for a new setup, or override any value
# from the shell. No task-specific wrapper or preset entry is required.
# ---------------------------------------------------------------------------
EXPERIMENT_NAME="${EXPERIMENT_NAME:-direct_200_40k_bs2_local_repro}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-direct-rgbd-small}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-direct-rgbd-small}"
EVAL_TASKS="${EVAL_TASKS:-direct}"
MODALITIES="${MODALITIES:-rgbd}"
ACTION_MODE="${ACTION_MODE:-direct_velocity}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-96}"
DEPTH_WIDTH="${DEPTH_WIDTH:-128}"
DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID:-[16,16]}"
DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE:-pairwise_add}"
DEPTH_GATE_MODE="${DEPTH_GATE_MODE:-fixed_one}"

# Shared execution settings rarely need editing.
RUN_PHASE="${RUN_PHASE:-all}"
DRY_RUN="${DRY_RUN:-false}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
LOG_FREQ="${LOG_FREQ:-10}"
EPISODES="${EPISODES:-100}"
EVAL_EPISODE_LENGTH_S="${EVAL_EPISODE_LENGTH_S:-10.0}"
EVAL_SEEDS="${EVAL_SEEDS:-${EVAL_SEED:-42 43 44 45 46}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAMESPACE="${MODEL_NAMESPACE:-DinhQuangDung}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/checkpoints/pi05_base_lerobot_0.4.4}"

case "${RUN_PHASE}" in train|eval|all) ;; *) echo "RUN_PHASE must be train, eval, or all" >&2; exit 2 ;; esac
case "${DRY_RUN}" in true|false) ;; *) echo "DRY_RUN must be true or false" >&2; exit 2 ;; esac
case "${ACTION_MODE}" in waypoint|direct_velocity) ;; *) echo "ACTION_MODE must be waypoint or direct_velocity" >&2; exit 2 ;; esac
case "${DEPTH_GATE_MODE}" in learned|fixed_one) ;; *) echo "DEPTH_GATE_MODE must be learned or fixed_one" >&2; exit 2 ;; esac
if [[ "${DEPTH_GATE_MODE}" == "fixed_one" && "${DEPTH_FUSION_MODE}" != "pairwise_add" ]]; then
    echo "DEPTH_GATE_MODE=fixed_one requires DEPTH_FUSION_MODE=pairwise_add" >&2
    exit 2
fi
(( STEPS > 0 && BATCH_SIZE > 0 && NUM_WORKERS >= 0 )) || { echo "Invalid numeric training setting" >&2; exit 2; }
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "Dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2; exit 2; }

for modality in ${MODALITIES}; do
    case "${modality}" in
        rgb) modality_tag=clean_rgb ;;
        rgbd)
            modality_tag="${DEPTH_FUSION_MODE}_$(tr -d '[]' <<<"${DEPTH_TOKEN_GRID}" | tr ',' 'x')"
            if [[ "${DEPTH_GATE_MODE}" == "fixed_one" ]]; then
                modality_tag="fixed_depth_scale_${modality_tag}"
            fi
            ;;
        *) echo "MODALITIES accepts only rgb and rgbd" >&2; exit 2 ;;
    esac

    run_name="pi05_quadloco_${modality}_${EXPERIMENT_NAME}_${modality_tag}_${ACTION_MODE}_42d-state"
    output_dir="${OUTPUT_BASE:-${ROOT}/outputs}/${run_name}"
    model_repo="${MODEL_REPO:-${MODEL_NAMESPACE}/${run_name//_/-}}"
    checkpoint="${VLA_CHECKPOINT:-${output_dir}/checkpoints/last/pretrained_model}"
    eval_dir="${EVAL_OUTPUT_BASE:-${ROOT}/eval_results}/${run_name}"

    if [[ "${RUN_PHASE}" == train || "${RUN_PHASE}" == all ]]; then
        echo "[INFO] train ${modality}: ${STEPS} steps, batch ${BATCH_SIZE}, output=${output_dir}"
        if [[ "${DRY_RUN}" == false ]]; then
            env MODALITY="${modality}" DATASET_ROOT="${DATASET_ROOT}" DATASET_REPO="${DATASET_REPO}" \
                BASE_MODEL="${BASE_MODEL}" MODEL_REPO="${model_repo}" OUTPUT_DIR="${output_dir}" \
                ACTION_MODE="${ACTION_MODE}" STEPS="${STEPS}" \
                BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" SAVE_FREQ="${SAVE_FREQ}" \
                SAVE_CHECKPOINT="${SAVE_CHECKPOINT}" LOG_FREQ="${LOG_FREQ}" TRAIN_SEED="${TRAIN_SEED}" \
                TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY}" GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
                PUSH_MODEL_TO_HUB="${PUSH_MODEL_TO_HUB}" WANDB_ENABLE="${WANDB_ENABLE}" \
                DEPTH_HEIGHT="${DEPTH_HEIGHT}" DEPTH_WIDTH="${DEPTH_WIDTH}" \
                DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID}" DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE}" \
                DEPTH_GATE_MODE="${DEPTH_GATE_MODE}" \
                bash "${ROOT}/train_pi05.sh"
        fi
    fi

    if [[ "${RUN_PHASE}" == eval || "${RUN_PHASE}" == all ]]; then
        for eval_seed in ${EVAL_SEEDS}; do
            [[ "${eval_seed}" =~ ^[0-9]+$ ]] || { echo "EVAL_SEEDS must contain integers" >&2; exit 2; }
            seed_eval_dir="${eval_dir}/seed-${eval_seed}"
            echo "[INFO] eval ${modality}: seed=${eval_seed}, tasks=${EVAL_TASKS}, checkpoint=${checkpoint}"
            if [[ "${DRY_RUN}" == false ]]; then
                VLA_CHECKPOINT="${checkpoint}" OUTPUT_DIR="${seed_eval_dir}" TASKS="${EVAL_TASKS}" \
                    EPISODES="${EPISODES}" EPISODE_LENGTH_S="${EVAL_EPISODE_LENGTH_S}" EVAL_SEED="${eval_seed}" \
                    bash "${ROOT}/eval_pi05.sh"
            fi
        done
    fi
done
