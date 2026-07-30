python scripts/quadloco_rsl_rl/run_data_collection.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0 \
    --num_envs 1 \
    --checkpoint ckpt/go2_low_rough.pt \
    --collect_data \
    --dataset_format lerobot \
    --dataset_repo_id quadloco/goal_navigation \
    --dataset_task "Navigate to the target" \
    --num_episodes 100 \
    --dataset_dir datasets/quadloco_lerobot_v1
    # --headless \
