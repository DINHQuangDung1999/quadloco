#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE="${DATASET_BASE:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung}"
EPISODES="${EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RUN_PHASE="${RUN_PHASE:-all}"
MODALITIES="${MODALITIES:-rgbd}"
TASKS="${TASKS:-all_1500}"
DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID:-[16,16]}"
DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE:-pairwise_add}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-direct_velocity}"
SAVE_FREQ_OVERRIDE="${SAVE_FREQ_OVERRIDE:-}"
ALL_EVAL_TASKS="direct occluded relational near_far object_relative"

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
    local modality train_script experiment_tag output_dir model_repo dataset_root dataset_repo

    case "${task}" in
        object_relative)
            dataset_root="${DATASET_BASE}/quadloco-vla-object_relative-rgbd-small-clean"
            dataset_repo="DinhQuangDung/quadloco-vla-object_relative-rgbd-small-clean"
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
                if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                    experiment_tag="clean_rgb_direct_velocity"
                else
                    experiment_tag="clean_rgb_baseline"
                fi
                ;;
            rgbd)
                train_script="train_pi05_rgbd_main.sh"
                if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity"
                else
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16"
                fi
                ;;
            *)
                echo "Unsupported modality: ${modality}; expected rgb or rgbd" >&2
                exit 1
                ;;
        esac

        if [[ "${task}" == "all_1500" ]]; then
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
        ACTION_REPRESENTATION="${ACTION_REPRESENTATION}" \
        DEPTH_TOKEN_GRID="${DEPTH_TOKEN_GRID}" \
        DEPTH_FUSION_MODE="${DEPTH_FUSION_MODE}" \
        DEPTH_MAX="${DEPTH_MAX}" \
        bash "${QUADLOCO_ROOT}/${train_script}"
    done
}

evaluate_task() {
    local task="$1"
    local modality experiment_tag checkpoint output_dir

    for modality in ${MODALITIES}; do
        case "${modality}" in
            rgb)
                if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                    experiment_tag="clean_rgb_direct_velocity"
                else
                    experiment_tag="clean_rgb_baseline"
                fi
                ;;
            rgbd)
                if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity"
                else
                    experiment_tag="${DEPTH_FUSION_MODE}_16x16"
                fi
                ;;
            *)
                echo "Unsupported modality: ${modality}; expected rgb or rgbd" >&2
                exit 1
                ;;
        esac
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

evaluate_all_1500() {
    local modality experiment_tag checkpoint output_dir
    modality="${MODALITIES}"
    case "${modality}" in
        rgb)
            if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                experiment_tag="clean_rgb_direct_velocity"
            else
                experiment_tag="clean_rgb_baseline"
            fi
            ;;
        rgbd)
            if [[ "${ACTION_REPRESENTATION}" == "direct_velocity" ]]; then
                experiment_tag="${DEPTH_FUSION_MODE}_16x16_direct_velocity"
            else
                experiment_tag="${DEPTH_FUSION_MODE}_16x16"
            fi
            ;;
        *)
            echo "Full-dataset evaluation expects exactly one modality: rgb or rgbd" >&2
            exit 1
            ;;
    esac
    checkpoint="${QUADLOCO_ROOT}/outputs/pi05_quadloco_${modality}_all_1500_1epoch_${experiment_tag}/checkpoints/last/pretrained_model"
    output_dir="${QUADLOCO_ROOT}/eval_results/${modality}_all_1500_1epoch_${experiment_tag}"

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
            near_far) train_task "${task}" 18750 ;;
            occluded) train_task "${task}" 25782 ;;
            # 55,836 frames / batch size 2 = 27,918 optimizer steps.
            object_relative) train_task "${task}" 27918 ;;
            all_1500)
                if [[ -z "${STEPS_OVERRIDE}" ]]; then
                    merged_info="${DATASET_BASE}/quadloco-vla-all-rgbd-clean/meta/info.json"
                    if [[ ! -f "${merged_info}" ]]; then
                        echo "Merged dataset metadata not found: ${merged_info}" >&2
                        exit 1
                    fi
                    merged_episodes="$(jq -r '.total_episodes' "${merged_info}")"
                    merged_frames="$(jq -r '.total_frames' "${merged_info}")"
                    if [[ "${merged_episodes}" -ne 1500 ]]; then
                        echo "Merged dataset has ${merged_episodes} episodes; expected 1500" >&2
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
        if [[ "${task}" == "all_1500" ]]; then
            evaluate_all_1500
        else
            evaluate_task "${task}"
        fi
    done
fi
