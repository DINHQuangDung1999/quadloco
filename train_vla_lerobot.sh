cd /home/summerschool/summerschool_ws/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=DinhQuangDung/quadloco-object-navigation \
    --dataset.root=/home/summerschool/summerschool_ws/Dataset/quadloco-vla-lerobot_v1 \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.input_features='{"observation.state":{"type":"STATE","shape":[3]},"observation.images.camera1":{"type":"VISUAL","shape":[3,480,640]}}' \
    --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.push_to_hub=false \
    --output_dir=/home/summerschool/summerschool_ws/outputs/pi05_quadloco_smoke \
    --job_name=pi05_quadloco_smoke \
    --batch_size=1 \
    --num_workers=0 \
    --steps=2 \
    --log_freq=1 \
    --eval_freq=0 \
    --save_checkpoint=false \
    --wandb.enable=false