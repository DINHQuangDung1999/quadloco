python scripts/quadloco_rsl_rl/run_data_collection.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0 \
    --num_envs 1\
    --checkpoint ckpt/go2_low_rough.pt \
    --num_episodes 100 \
    --dataset_dir datasets/goal_navigation
    # --headless \