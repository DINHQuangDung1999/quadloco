#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE="${DATASET_BASE:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung}"
EPISODES="${EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
RUN_PHASE="${RUN_PHASE:-all}"
MODALITIES="${MODALITIES:-rgb rgbd}"
TASKS="${TASKS:-direct near_far}"
DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID:-[16,16]}"
DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE:-pairwise_add}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-}"
ACTION_MODE="${ACTION_MODE:-${ACTION_REPRESENTATION:-waypoint}}"
STATE_MODE="${STATE_MODE:-proprioceptive_42d}"
SAVE_FREQ_OVERRIDE="${SAVE_FREQ_OVERRIDE:-}"
ALL_EVAL_TASKS="direct occluded relational near_far object_relative"

case "${GRADIENT_CHECKPOINTING}" in
    true|false) ;;
    *)
        echo "GRADIENT_CHECKPOINTING must be true or false" >&2
        exit 1
        ;;
esac

# Preserve legacy batch-2/non-checkpointed paths while keeping controlled
# optimizer/memory experiments in distinct output directories.
if [[ "${BATCH_SIZE}" == "2" && "${GRADIENT_CHECKPOINTING}" == "false" ]]; then
    TRAINING_SUFFIX=""
elif [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
    TRAINING_SUFFIX="_bs${BATCH_SIZE}_gradckpt"
else
    TRAINING_SUFFIX="_bs${BATCH_SIZE}_no_gradckpt"
fi

case "${RUN_PHASE}" in
    train|eval|all) ;;
    *)
        echo "RUN_PHASE must be one of: train, eval, all" >&2
        exit 1
        ;;
esac

train_task() {
    local task="$1"
    local steps="$2"
    local modality train_script experiment_tag output_dir model_repo dataset_root dataset_repo state_tag

    case "${STATE_MODE}" in
        vision_language_only) state_tag="vision_language_only" ;;
        proprioceptive_42d) state_tag="state42_no_velocity_command" ;;
        *)
            echo "STATE_MODE must be vision_language_only or proprioceptive_42d" >&2
            exit 1
            ;;
    esac

    case "${task}" in
        object_relative)
            dataset_root="${DATASET_BASE}/quadloco-vla-object_relative-rgbd-small"
            dataset_repo="DinhQuangDung/quadloco-vla-object_relative-rgbd-small"
            ;;
        all_1000)
            dataset_root="${DATASET_BASE}/quadloco-vla-all-rgbd-small"
            dataset_repo="DinhQuangDung/quadloco-vla-all-rgbd-small"
            ;;
        all_1500)
            dataset_root="${DATASET_BASE}/quadloco-vla-all-rgbd-clean"
            dataset_repo="DinhQuangDung/quadloco-vla-all-rgbd-clean"
            ;;
        *)
            dataset_root="${DATASET_BASE}/quadloco-vla-${task}-rgbd-small"
            dataset_repo="DinhQuangDung/quadloco-vla-${task}-rgbd-small"
            ;;
    esac

    for modality in ${MODALITIES}; do
        case "${modality}" in
            rgb)
                train_script="train_pi05_rgb_main.sh"
                if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                    experiment_tag="clean_rgb_direct_velocity_${state_tag}"
                else
                    experiment_tag="clean_rgb_waypoint_${state_tag}"
                fi
                ;;
            rgbd)
                train_script="train_pi05_rgbd_main.sh"
                if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity_${state_tag}"
                else
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_waypoint_${state_tag}"
                fi
                ;;
            *)
                echo "Unsupported modality: ${modality}; expected rgb or rgbd" >&2
                exit 1
                ;;
        esac
        experiment_tag="${experiment_tag}${TRAINING_SUFFIX}"

        if [[ "${task}" == "all_1000" || "${task}" == "all_1500" ]]; then
            output_dir="${QUADLOCO_ROOT}/outputs/pi05_quadloco_${modality}_${task}_1epoch_${experiment_tag}"
            model_repo="DinhQuangDung/pi05-quadloco-${modality}-${task//_/-}-1epoch-${experiment_tag//_/-}"
        else
            output_dir="${QUADLOCO_ROOT}/outputs/pi05_quadloco_${modality}_${task}_small_1epoch_${experiment_tag}"
            model_repo="DinhQuangDung/pi05-quadloco-${modality}-${task//_/-}-small-1epoch-${experiment_tag//_/-}"
        fi
        echo "[INFO] Training ${task} ${modality} for one epoch (${steps} optimizer steps)"
        # Current experiment: RGB-only baseline. RGB-D alternatives retained
        # for controlled comparisons:
        #   DEPTH_TOKEN_GRID='[16,16]' DEPTH_FUSION_MODE=concatenate
        #   DEPTH_TOKEN_GRID='[16,16]' DEPTH_FUSION_MODE=pairwise_add
        #   DEPTH_TOKEN_GRID='[8,8]' DEPTH_FUSION_MODE=cross_attention
        DATASET_ROOT="${dataset_root}" \
        DATASET_REPO="${dataset_repo}" \
        OUTPUT_DIR="${output_dir}" \
        MODEL_REPO="${model_repo}" \
        STEPS="${steps}" \
        SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-${steps}}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
        ACTION_MODE="${ACTION_MODE}" \
        STATE_MODE="${STATE_MODE}" \
        DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID}" \
        DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE}" \
        DEPTH_MAX="${DEPTH_MAX}" \
        bash "${QUADLOCO_ROOT}/${train_script}"
    done
}

evaluate_task() {
    local task="$1"
    local modality experiment_tag checkpoint output_dir state_tag

    case "${STATE_MODE}" in
        vision_language_only) state_tag="vision_language_only" ;;
        proprioceptive_42d) state_tag="state42_no_velocity_command" ;;
        *)
            echo "STATE_MODE must be vision_language_only or proprioceptive_42d" >&2
            exit 1
            ;;
    esac

    for modality in ${MODALITIES}; do
        case "${modality}" in
            rgb)
                if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                    experiment_tag="clean_rgb_direct_velocity_${state_tag}"
                else
                    experiment_tag="clean_rgb_waypoint_${state_tag}"
                fi
                ;;
            rgbd)
                if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity_${state_tag}"
                else
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_waypoint_${state_tag}"
                fi
                ;;
            *)
                echo "Unsupported modality: ${modality}; expected rgb or rgbd" >&2
                exit 1
                ;;
        esac
        experiment_tag="${experiment_tag}${TRAINING_SUFFIX}"
        checkpoint="${QUADLOCO_ROOT}/outputs/pi05_quadloco_${modality}_${task}_small_1epoch_${experiment_tag}/checkpoints/last/pretrained_model"
        output_dir="${QUADLOCO_ROOT}/eval_results/${modality}_${task}_small_1epoch_${experiment_tag}"
        echo "[INFO] Evaluating ${task} ${modality}"
        TASKS="${task}" \
        VLA_CHECKPOINT="${checkpoint}" \
        OUTPUT_DIR="${output_dir}" \
        EPISODES="${EPISODES}" \
        EVAL_SEED="${EVAL_SEED}" \
        bash "${QUADLOCO_ROOT}/eval_pi05.sh"
    done
}

evaluate_all() {
    local dataset_tag="$1"
    local modality experiment_tag checkpoint output_dir state_tag
    case "${STATE_MODE}" in
        vision_language_only) state_tag="vision_language_only" ;;
        proprioceptive_42d) state_tag="state42_no_velocity_command" ;;
        *)
            echo "STATE_MODE must be vision_language_only or proprioceptive_42d" >&2
            exit 1
            ;;
    esac
    modality="${MODALITIES}"
    case "${modality}" in
        rgb)
            if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                experiment_tag="clean_rgb_direct_velocity_${state_tag}"
            else
                experiment_tag="clean_rgb_waypoint_${state_tag}"
            fi
            ;;
        rgbd)
            if [[ "${ACTION_MODE}" == "direct_velocity" ]]; then
                experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity_${state_tag}"
            else
                experiment_tag="${DEPTH_FUSION_MODE}_16x16_waypoint_${state_tag}"
            fi
            ;;
        *)
            echo "Full-dataset evaluation expects exactly one modality: rgb or rgbd" >&2
            exit 1
            ;;
    esac
    experiment_tag="${experiment_tag}${TRAINING_SUFFIX}"
    checkpoint="${QUADLOCO_ROOT}/outputs/pi05_quadloco_${modality}_${dataset_tag}_1epoch_${experiment_tag}/checkpoints/last/pretrained_model"
    output_dir="${QUADLOCO_ROOT}/eval_results/${modality}_${dataset_tag}_1epoch_${experiment_tag}"

    if [[ ! -f "${checkpoint}/config.json" ]]; then
        echo "Full-dataset checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Evaluation output already exists: ${output_dir}" >&2
        echo "Choose a new output path or remove the old run explicitly." >&2
        exit 1
    fi

    echo "[INFO] Evaluating the full-dataset ${modality} model on: ${ALL_EVAL_TASKS}"
    TASKS="${ALL_EVAL_TASKS}" \
    VLA_CHECKPOINT="${checkpoint}" \
    OUTPUT_DIR="${output_dir}" \
    EPISODES="${EPISODES}" \
    EVAL_SEED="${EVAL_SEED}" \
    bash "${QUADLOCO_ROOT}/eval_pi05.sh"
}

if [[ "${RUN_PHASE}" == "train" || "${RUN_PHASE}" == "all" ]]; then
    for task in ${TASKS}; do
        case "${task}" in
            direct|near_far|occluded|object_relative)
                task_info="${DATASET_BASE}/quadloco-vla-${task}-rgbd-small/meta/info.json"
                if [[ ! -f "${task_info}" ]]; then
                    echo "Dataset metadata not found: ${task_info}" >&2
                    exit 1
                fi
                task_episodes="$(jq -r '.total_episodes' "${task_info}")"
                task_frames="$(jq -r '.total_frames' "${task_info}")"
                if [[ "${task_episodes}" -ne 200 ]]; then
                    echo "${task} dataset has ${task_episodes} episodes; expected 200" >&2
                    exit 1
                fi
                if [[ "${BATCH_SIZE}" -le 0 ]]; then
                    echo "BATCH_SIZE must be positive" >&2
                    exit 1
                fi
                task_steps="$(( (task_frames + BATCH_SIZE - 1) / BATCH_SIZE ))"
                train_task "${task}" "${STEPS_OVERRIDE:-${task_steps}}"
                ;;
            all_1000|all_1500)
                if [[ -z "${STEPS_OVERRIDE}" ]]; then
                    if [[ "${task}" == "all_1000" ]]; then
                        merged_info="${DATASET_BASE}/quadloco-vla-all-rgbd-small/meta/info.json"
                        expected_episodes=1000
                    else
                        merged_info="${DATASET_BASE}/quadloco-vla-all-rgbd-clean/meta/info.json"
                        expected_episodes=1500
                    fi
                    if [[ ! -f "${merged_info}" ]]; then
                        echo "Merged dataset metadata not found: ${merged_info}" >&2
                        exit 1
                    fi
                    merged_episodes="$(jq -r '.total_episodes' "${merged_info}")"
                    merged_frames="$(jq -r '.total_frames' "${merged_info}")"
                    if [[ "${merged_episodes}" -ne "${expected_episodes}" ]]; then
                        echo "Merged dataset has ${merged_episodes} episodes; expected ${expected_episodes}" >&2
                        exit 1
                    fi
                    if [[ "${BATCH_SIZE}" -le 0 ]]; then
                        echo "BATCH_SIZE must be positive" >&2
                        exit 1
                    fi
                    STEPS_OVERRIDE="$(( (merged_frames + BATCH_SIZE - 1) / BATCH_SIZE ))"
                fi
                train_task "${task}" "${STEPS_OVERRIDE}"
                ;;
            *)
                echo "Unsupported training task: ${task}" >&2
                exit 1
                ;;
        esac
    done
fi

if [[ "${RUN_PHASE}" == "eval" || "${RUN_PHASE}" == "all" ]]; then
    for task in ${TASKS}; do
        if [[ "${task}" == "all_1000" || "${task}" == "all_1500" ]]; then
            evaluate_all "${task}"
        else
            evaluate_task "${task}"
        fi
    done
fi
