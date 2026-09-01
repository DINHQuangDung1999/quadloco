#!/usr/bin/env bash
set -euo pipefail

# Unified Go2 VLA launcher:
#   ./run_go2_vla.sh deploy --robot-ip ...
#   ./run_go2_vla.sh task --instruction "Navigate to the red cube"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-deploy}"
[[ $# -eq 0 ]] || shift
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
[[ -f "${ROS_SETUP}" ]] || { echo "ROS setup not found: ${ROS_SETUP}" >&2; exit 2; }

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
if [[ "${QUADLOCO_KEEP_CYCLONEDDS_URI:-0}" != 1 ]]; then unset CYCLONEDDS_URI; fi

case "${MODE}" in
    deploy)
        LEROBOT_ROOT="${LEROBOT_ROOT:-${ROOT}/third_party/lerobot}"
        VLA_PYTHON="${VLA_PYTHON:-python}"
        VLA_CHECKPOINT="${VLA_CHECKPOINT:-${ROOT}/../pi05_rgb_direct_vel}"
        [[ -f "${VLA_CHECKPOINT}/config.json" ]] || {
            echo "PI0.5 checkpoint not found: ${VLA_CHECKPOINT}" >&2
            echo "Set VLA_CHECKPOINT to a pretrained_model directory" >&2
            exit 2
        }
        export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
        export PYTHONPATH="${LEROBOT_ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${VLA_PYTHON}" -m deployment.go2_vla.deploy --checkpoint "${VLA_CHECKPOINT}" "$@"
        ;;
    task)
        VLA_TASK_PYTHON="${VLA_TASK_PYTHON:-python3}"
        export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
        export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${VLA_TASK_PYTHON}" -m deployment.go2_vla.task \
            --server-host "${VLA_HOST:-127.0.0.1}" --server-port "${VLA_PORT:-5555}" "$@"
        ;;
    *) echo "Usage: $0 {deploy|task} [arguments...]" >&2; exit 2 ;;
esac
