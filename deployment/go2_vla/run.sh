#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
VLA_PYTHON="${VLA_PYTHON:-python}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${QUADLOCO_ROOT}/../pi05_rgb_direct_vel}"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup not found: ${ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -f "${VLA_CHECKPOINT}/config.json" ]]; then
    echo "PI0.5 checkpoint not found: ${VLA_CHECKPOINT}" >&2
    echo "Set VLA_CHECKPOINT to a pretrained_model directory." >&2
    exit 1
fi

# ROS Humble setup scripts may inspect unset AMENT_* variables and are not
# compatible with `set -u`. Restore nounset immediately after sourcing.
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
# Avoid inheriting a stale host-specific interface such as wlp9s0. Set
# QUADLOCO_KEEP_CYCLONEDDS_URI=1 only when CYCLONEDDS_URI is known to be valid.
if [[ "${QUADLOCO_KEEP_CYCLONEDDS_URI:-0}" != "1" ]]; then
    unset CYCLONEDDS_URI
fi
export PYTHONPATH="${QUADLOCO_ROOT}/third_party/lerobot/src:${QUADLOCO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VLA_PYTHON}" -m deployment.go2_vla.deploy \
    --checkpoint "${VLA_CHECKPOINT}" \
    "$@"
