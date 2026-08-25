#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-${QUADLOCO_ROOT}/third_party/lerobot}"
VLA_PYTHON="${VLA_PYTHON:-python}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${QUADLOCO_ROOT}/../pi05_rgb_direct_vel}"

export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec env VLA_PYTHON="${VLA_PYTHON}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    bash "${QUADLOCO_ROOT}/deployment/go2_vla/run.sh" "$@"
