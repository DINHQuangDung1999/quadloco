#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
VLA_PYTHON="${VLA_PYTHON:-python}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${QUADLOCO_ROOT}/../pi05_rgb_direct_vel}"

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${VLA_PYTHON}" "${QUADLOCO_ROOT}/scripts/pi05_velocity_server.py" \
    --checkpoint "${VLA_CHECKPOINT}" \
    --host "${VLA_HOST:-127.0.0.1}" \
    --port "${VLA_PORT:-5555}" \
    --device "${VLA_DEVICE:-cuda}" \
    --precision "${VLA_PRECISION:-fp16}" \
    "$@"
