# # !/usr/bin/env bash
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

# Add --headless above for unattended collection. Depth recording is enabled
# by default; pass --no-lerobot_include_depth only when an RGB-only dataset is
# intentionally required.

declare -A EPISODES=([direct]=200 [near_far]=200 [object_relative]=200)
for TASK in direct near_far object_relative; do
    python scripts/quadloco_rsl_rl/run_data_collection.py \
    --navigation_mode "$TASK"  \
    --num_envs 1   \
    --checkpoint ckpt/model_999.pt  \
    --collect_data    \
    --dataset_format lerobot   \
    --dataset_repo_id "DinhQuangDung/quadloco-vla-${TASK}-rgbd"  \
    --dataset_dir "/home/summerschool/summerschool_ws/Dataset/DinhQuangDung/quadloco-vla-${TASK}-rgbd-small"   \
    --num_episodes "${EPISODES[$TASK]}"   \
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
