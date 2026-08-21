#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"

# Override these when LeRobot and Isaac Lab use different Python environments.
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$(command -v python)}"

VLA_CHECKPOINT="${VLA_CHECKPOINT:-${QUADLOCO_ROOT}/outputs/pi05_quadloco_rgbd_object_relative_small_1epoch_pairwise_add_16x16/checkpoints/last/pretrained_model}"
LOCOMOTION_CHECKPOINT="${LOCOMOTION_CHECKPOINT:-${QUADLOCO_ROOT}/ckpt/model_999.pt}"
TASK="${TASK:-Unitree-Go2-Quadloco-ManagerBased-Rough-ObjectRelative-DataCollection-v0}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-5555}"
VLA_DEVICE="${VLA_DEVICE:-cuda}"
VLA_PRECISION="${VLA_PRECISION:-fp16}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"

if [[ ! -x "${LEROBOT_PYTHON}" ]]; then
    echo "LeRobot Python executable not found: ${LEROBOT_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "Isaac Lab Python executable not found: ${ISAACLAB_PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${VLA_CHECKPOINT}/config.json" ]]; then
    echo "PI0.5 checkpoint not found: ${VLA_CHECKPOINT}" >&2
    echo "Set VLA_CHECKPOINT to a LeRobot pretrained_model directory." >&2
    exit 1
fi

if [[ ! -f "${LOCOMOTION_CHECKPOINT}" ]]; then
    echo "Locomotion checkpoint not found: ${LOCOMOTION_CHECKPOINT}" >&2
    echo "Set LOCOMOTION_CHECKPOINT to an RSL-RL .pt checkpoint." >&2
    exit 1
fi

server_pid=""

stop_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}

trap stop_server EXIT INT TERM

cd "${QUADLOCO_ROOT}"

echo "[INFO] Starting PI0.5 server with checkpoint: ${VLA_CHECKPOINT}"
PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${LEROBOT_PYTHON}" scripts/pi05_velocity_server.py \
    --checkpoint "${VLA_CHECKPOINT}" \
    --host "${VLA_HOST}" \
    --port "${VLA_PORT}" \
    --device "${VLA_DEVICE}" \
    --precision "${VLA_PRECISION}" &
server_pid=$!

echo "[INFO] Waiting for PI0.5 server at ${VLA_HOST}:${VLA_PORT} ..."
deadline=$((SECONDS + SERVER_START_TIMEOUT))
while ! (echo >"/dev/tcp/${VLA_HOST}/${VLA_PORT}") 2>/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" || true
        echo "PI0.5 server exited before becoming ready." >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for the PI0.5 server." >&2
        exit 1
    fi
    sleep 1
done

echo "[INFO] Starting Quadloco VLA playback."
"${ISAACLAB_PYTHON}" scripts/quadloco_rsl_rl/play_vla.py \
    --task "${TASK}" \
    --checkpoint "${LOCOMOTION_CHECKPOINT}" \
    --num_envs 1 \
    --vla_host "${VLA_HOST}" \
    --vla_port "${VLA_PORT}" \
    --waypoint_vis vla
    "$@"
