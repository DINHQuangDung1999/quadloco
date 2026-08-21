# !/usr/bin/env bash
# set -euo pipefail

# navigation_mode="${1:-direct}"

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
#     # --collect_data \
#     # --lerobot_include_depth \
#     # --headless

# declare -A EPISODES=(  [near_far]=150     [object_relative]=150 )
# for TASK in occluded near_far object_relative; do
#     python scripts/quadloco_rsl_rl/run_data_collection.py \
#     --navigation_mode "$TASK"  \
#     --num_envs 1   \
#     --checkpoint ckpt/model_999.pt  \
#     --collect_data    \
#     --dataset_format lerobot   \
#     --dataset_repo_id "DinhQuangDung/quadloco-vla-${TASK}-rgbd"  \
#     --dataset_dir "/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-${TASK}-rgbd-small"   \
#     --num_episodes "${EPISODES[$TASK]}"   \
#     --depth_width 128  \
#     --depth_height 96   \
#     --depth_scale 0.001   \
#     --headless;
# done
# Merge the datasets
export HF_LEROBOT_HOME=/home/summerschool/summerschool_ws/Dataset

lerobot-edit-dataset \
    --repo_id DinhQuangDung/quadloco-vla-all-rgbd-clean \
    --operation.type merge \
    --operation.repo_ids "[
        'DinhQuangDung/quadloco-vla-direct-rgbd-clean',
        'DinhQuangDung/quadloco-vla-occluded-rgbd-clean',
        'DinhQuangDung/quadloco-vla-relational-rgbd-clean',
        'DinhQuangDung/quadloco-vla-near_far-rgbd-clean',
        'DinhQuangDung/quadloco-vla-object_relative-rgbd-clean'
    ]"
