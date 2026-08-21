#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "train_pi05_velocity.sh is deprecated; launching train_pi05_depth_main.sh" >&2
exec bash "${SCRIPT_DIR}/train_pi05_depth_main.sh" "$@"
