#!/usr/bin/env bash
set -euo pipefail

QUADLOCO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${QUADLOCO_ROOT}/third_party/lerobot"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python)}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-${QUADLOCO_ROOT}/checkpoints/pi05_base_lerobot_0.4.4}"
PI05_REVISION="a538eb273274eb30f126a118f39dbc0ee212c883"

if [[ ! -f "${LEROBOT_ROOT}/pyproject.toml" ]]; then
    echo "Vendored LeRobot source not found: ${LEROBOT_ROOT}" >&2
    exit 1
fi

"${LEROBOT_PYTHON}" -m pip install -e "${LEROBOT_ROOT}[pi]"

if [[ ! -d "${PI05_CHECKPOINT}" ]]; then
    mkdir -p "$(dirname "${PI05_CHECKPOINT}")"
    hf download \
        lerobot/pi05_base \
        --revision "${PI05_REVISION}" \
        --local-dir "${PI05_CHECKPOINT}"
fi

PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${LEROBOT_PYTHON}" -c \
    "import lerobot, lerobot.policies.pi05.modeling_pi05 as m; print(f'LeRobot {lerobot.__version__}: {m.__file__}')"

echo "PI0.5 checkpoint: ${PI05_CHECKPOINT}"
