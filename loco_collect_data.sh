#!/usr/bin/env bash
# set -euo pipefail

# navigation_mode="${1:-near_far}"

# case "${navigation_mode}" in
#     direct) default_episodes=1 ;;
#     occluded) default_episodes=1 ;;
#     relational) default_episodes=1 ;;
#     near_far) default_episodes=1 ;;
#     object_relative) default_episodes=1 ;;
#     *)
#         echo "Unknown navigation mode: ${navigation_mode}" >&2
#         echo "Expected: direct, occluded, relational, near_far, or object_relative" >&2
#         exit 2
#         ;;
# esac

# num_episodes="${2:-${default_episodes}}"
# dataset_root="${3:-/home/summerschool/summerschool_ws/Dataset}"
# dataset_dir="${dataset_root}/quadloco-vla-${navigation_mode}-rgbd"
# dataset_repo_id="DinhQuangDung/quadloco-vla-${navigation_mode}-rgbd"

# python scripts/quadloco_rsl_rl/run_data_collection.py \
#     --navigation_mode "${navigation_mode}" \
#     --num_envs 1 \
#     --checkpoint ckpt/model_999.pt \
#     --dataset_format lerobot \
#     --dataset_repo_id "${dataset_repo_id}" \
#     --num_episodes "${num_episodes}" \
#     --dataset_dir "${dataset_dir}" \
#     --depth_width 128 \
#     --depth_height 96 \
#     --depth_scale 0.001 \
#     # --collect_data

# Collect the two expanded RGB-D reasoning datasets.  The collector only saves
# accepted episodes: goal reached, no timeout, no task change, and no collision.
# Existing dataset directories are never overwritten.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/summerschool/summerschool_ws/Dataset/DinhQuangDung}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
SEED="${SEED:-42}"

for TASK in near_far object_relative; do
    python scripts/quadloco_rsl_rl/run_data_collection.py \
    --navigation_mode "$TASK"  \
    --num_envs 1   \
    --checkpoint ckpt/model_999.pt  \
    --seed "${SEED}" \
    --collect_data    \
    --dataset_format lerobot   \
    --dataset_repo_id "DinhQuangDung/quadloco-vla-${TASK}-rgbd-1000"  \
    --dataset_dir "${DATASET_ROOT}/quadloco-vla-${TASK}-rgbd-1000"   \
    --num_episodes "${NUM_EPISODES}"   \
    --depth_width 128  \
    --depth_height 96   \
    --depth_scale 0.001   \
    --headless;
done
# # Merge the datasets
# export HF_LEROBOT_HOME=/home/summerschool/summerschool_ws/Dataset

# lerobot-edit-dataset \
#     --repo_id DinhQuangDung/quadloco-vla-all-rgbd-small \
#     --operation.type merge \
#     --operation.repo_ids "[
#         'DinhQuangDung/quadloco-vla-direct-rgbd-small',
#         'DinhQuangDung/quadloco-vla-occluded-rgbd-small',
#         'DinhQuangDung/quadloco-vla-relational-rgbd-small',
#         'DinhQuangDung/quadloco-vla-near_far-rgbd-small',
#         'DinhQuangDung/quadloco-vla-object_relative-rgbd-small'
#     ]"
