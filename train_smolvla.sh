#!/usr/bin/env bash
set -euo pipefail

# Low-level SmolVLA trainer. Prefer run_smolvla.sh for complete experiments.
QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-direct-rgbd-1000}"
DATASET_REPO="${DATASET_REPO:-DinhQuangDung/quadloco-vla-direct-rgbd-1000}"
BASE_MODEL="${BASE_MODEL:-${QUADLOCO_ROOT}/checkpoints/smolvla_base_lerobot_0.4.4}"
MODEL_REPO="${MODEL_REPO:-DinhQuangDung/smolvla-quadloco-rgbd-direct-1000-40k-2epoch-pairwise-add-16x16-direct-velocity-30d-state}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUADLOCO_ROOT}/outputs/smolvla_quadloco_rgbd_direct_1000_40k_2epoch_pairwise_add_16x16_direct_velocity_30d-state}"

MODALITY="${MODALITY:-rgbd}"
ACTION_MODE="${ACTION_MODE:-direct_velocity}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-${STEPS}}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
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
DEPTH_MAX="${DEPTH_MAX:-10.0}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-96}"
DEPTH_WIDTH="${DEPTH_WIDTH:-128}"

case "${MODALITY}" in
    rgb)
        INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[45]},"observation.images.camera1":{"type":"VISUAL","shape":[3,480,640]}}'
        DEPTH_ENABLED=false
        ;;
    rgbd)
        INPUT_FEATURES="{\"observation.state\":{\"type\":\"STATE\",\"shape\":[45]},\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,480,640]},\"observation.depth.camera1\":{\"type\":\"VISUAL\",\"shape\":[1,${DEPTH_HEIGHT},${DEPTH_WIDTH}]}}"
        DEPTH_ENABLED=true
        ;;
    *)
        echo "MODALITY must be rgb or rgbd" >&2
        exit 1
        ;;
esac

case "${ACTION_MODE}" in
    waypoint)
        OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[2]}}'
        RENAME_MAP='{}'
        JOB_NAME="smolvla_quadloco_${MODALITY}_waypoint"
        ;;
    direct_velocity)
        OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[3]}}'
        RENAME_MAP='{"observation.velocity_command":"action"}'
        JOB_NAME="smolvla_quadloco_${MODALITY}_direct_velocity"
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
    echo "SmolVLA base checkpoint not found: ${BASE_MODEL}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Training output already exists: ${OUTPUT_DIR}" >&2
    echo "Set OUTPUT_DIR to a new path, or use LeRobot's explicit resume workflow." >&2
    exit 1
fi
if ! "${LEROBOT_PYTHON}" -c 'import num2words' >/dev/null 2>&1; then
    echo "Missing Python dependency: num2words" >&2
    exit 1
fi

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${LEROBOT_PYTHON}" "${LEROBOT_ROOT}/src/lerobot/scripts/lerobot_train.py" \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=smolvla \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.repo_id="${MODEL_REPO}" \
    --policy.push_to_hub="${PUSH_MODEL_TO_HUB}" \
    --policy.input_features="${INPUT_FEATURES}" \
    --policy.output_features="${OUTPUT_FEATURES}" \
    --policy.max_state_dim=32 \
    --policy.state_token_dim=30 \
    --policy.state_feature_indices='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,36,37,38,39,40,41]' \
    --policy.action_mode="${ACTION_MODE}" \
    --rename_map="${RENAME_MAP}" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --policy.vision_lora_enabled="${VISION_LORA_ENABLED}" \
    --policy.vision_lora_rank="${VISION_LORA_RANK}" \
    --policy.vision_lora_alpha="${VISION_LORA_ALPHA}" \
    --policy.vision_lora_dropout="${VISION_LORA_DROPOUT}" \
    --policy.vision_lora_targets="${VISION_LORA_TARGETS}" \
    --policy.load_vlm_weights=true \
    --policy.depth_enabled="${DEPTH_ENABLED}" \
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
    --policy.depth_resize_with_rgb=true \
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
    --log_freq="${LOG_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --save_checkpoint="${SAVE_CHECKPOINT}" \
    --eval_freq=0 \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.disable_artifact=true
