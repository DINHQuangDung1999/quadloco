#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
VLA_TASK_PYTHON="${VLA_TASK_PYTHON:-python3}"

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

# Match the Go2's default ROS 2 Foxy DDS configuration in every fresh shell.
# An inherited CycloneDDS interface configuration can silently hide the
# cross-machine D435i topics, so preserve it only when explicitly requested.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
if [[ "${QUADLOCO_KEEP_CYCLONEDDS_URI:-0}" != "1" ]]; then
    unset CYCLONEDDS_URI
fi
export PYTHONPATH="${QUADLOCO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VLA_TASK_PYTHON}" -m deployment.go2_vla.task \
    --server-host "${VLA_HOST:-127.0.0.1}" \
    --server-port "${VLA_PORT:-5555}" \
    "$@"
