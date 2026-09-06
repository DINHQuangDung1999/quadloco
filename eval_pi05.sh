#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$(command -v python)}"

VLA_CHECKPOINT="${VLA_CHECKPOINT:-${QUADLOCO_ROOT}/outputs/pi05_quadloco_rgbd_near_far_50k_16x16/checkpoints/last/pretrained_model}"
LOCOMOTION_CHECKPOINT="${LOCOMOTION_CHECKPOINT:-${QUADLOCO_ROOT}/ckpt/model_999.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUADLOCO_ROOT}/eval_results/checkpoint_rgbd_near_far_50k_16x16_last}"
EPISODES="${EPISODES:-100}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10.0}"
EVAL_SEED="${EVAL_SEED:-42}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-5555}"
VLA_DEVICE="${VLA_DEVICE:-cuda}"
VLA_PRECISION="${VLA_PRECISION:-fp16}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
TASKS="${TASKS:-near_far}"

if [[ ! -f "${VLA_CHECKPOINT}/config.json" ]]; then
    echo "PI0.5 checkpoint not found: ${VLA_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${LOCOMOTION_CHECKPOINT}" ]]; then
    echo "Locomotion checkpoint not found: ${LOCOMOTION_CHECKPOINT}" >&2
    exit 1
fi

read -r -a task_list <<< "${TASKS}"
if (( ${#task_list[@]} == 0 )); then
    echo "TASKS must contain at least one navigation mode." >&2
    exit 1
fi
for task in "${task_list[@]}"; do
    case "${task}" in
        direct|occluded|relational|near_far|near_far_two_object|object_relative) ;;
        *)
            echo "Unsupported navigation mode in TASKS: ${task}" >&2
            exit 1
            ;;
    esac
done

server_pid=""
stop_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap stop_server EXIT INT TERM

cd "${QUADLOCO_ROOT}"
PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${LEROBOT_PYTHON}" scripts/pi05_velocity_server.py \
    --checkpoint "${VLA_CHECKPOINT}" \
    --host "${VLA_HOST}" \
    --port "${VLA_PORT}" \
    --device "${VLA_DEVICE}" \
    --precision "${VLA_PRECISION}" &
server_pid=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
while ! (echo >"/dev/tcp/${VLA_HOST}/${VLA_PORT}") 2>/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" || true
        echo "PI0.5 server exited before becoming ready." >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for the PI0.5 server." >&2
        exit 1
    fi
    sleep 1
done

for task in "${task_list[@]}"; do
    echo "[INFO] Evaluating ${task} (${EPISODES} episodes, ${EPISODE_LENGTH_S}s limit, seed=${EVAL_SEED})"
    "${ISAACLAB_PYTHON}" scripts/quadloco_rsl_rl/eval_pi05.py \
        --navigation_mode "${task}" \
        --num_episodes "${EPISODES}" \
        --episode_length_s "${EPISODE_LENGTH_S}" \
        --seed "${EVAL_SEED}" \
        --checkpoint "${LOCOMOTION_CHECKPOINT}" \
        --vla_host "${VLA_HOST}" \
        --vla_port "${VLA_PORT}" \
        --output_dir "${OUTPUT_DIR}/${task}" \
        --headless
done
