#!/usr/bin/env bash
set -euo pipefail

# Low-level SmolVLA evaluator. Prefer run_smolvla.sh for named experiments.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${ROOT}/third_party/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$(command -v python)}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:?Set VLA_CHECKPOINT}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
TASKS="${TASKS:-direct}"
LOCOMOTION_CHECKPOINT="${LOCOMOTION_CHECKPOINT:-${ROOT}/ckpt/model_999.pt}"
EPISODES="${EPISODES:-50}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10.0}"
EVAL_SEED="${EVAL_SEED:-42}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-5555}"
VLA_DEVICE="${VLA_DEVICE:-cuda}"
VLA_PRECISION="${VLA_PRECISION:-fp16}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"

[[ -f "${VLA_CHECKPOINT}/config.json" ]] || { echo "SmolVLA checkpoint not found: ${VLA_CHECKPOINT}" >&2; exit 2; }
[[ -f "${LOCOMOTION_CHECKPOINT}" ]] || { echo "Locomotion checkpoint not found: ${LOCOMOTION_CHECKPOINT}" >&2; exit 2; }
for task in ${TASKS}; do
    case "${task}" in direct|near_far|object_relative|occluded|relational|near_far_two_object) ;; *) echo "Unsupported task: ${task}" >&2; exit 2 ;; esac
done

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${LEROBOT_PYTHON}" "${ROOT}/scripts/smolvla_velocity_server.py" \
    --checkpoint "${VLA_CHECKPOINT}" --host "${VLA_HOST}" --port "${VLA_PORT}" \
    --device "${VLA_DEVICE}" --precision "${VLA_PRECISION}" &
server_pid=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
while ! (echo >"/dev/tcp/${VLA_HOST}/${VLA_PORT}") 2>/dev/null; do
    kill -0 "${server_pid}" 2>/dev/null || { wait "${server_pid}" || true; echo "SmolVLA server exited before ready" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "SmolVLA server startup timed out" >&2; exit 1; }
    sleep 1
done

for task in ${TASKS}; do
    "${ISAACLAB_PYTHON}" "${ROOT}/scripts/quadloco_rsl_rl/eval_pi05.py" \
        --navigation_mode "${task}" --num_episodes "${EPISODES}" \
        --episode_length_s "${EPISODE_LENGTH_S}" --seed "${EVAL_SEED}" \
        --checkpoint "${LOCOMOTION_CHECKPOINT}" --vla_host "${VLA_HOST}" --vla_port "${VLA_PORT}" \
        --output_dir "${OUTPUT_DIR}/${task}" --headless
done
