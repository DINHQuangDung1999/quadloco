# VLA experiment launchers

The experiment shell scripts are consolidated into two public entry points:

- `run_pi05.sh` for PI0.5
- `run_smolvla.sh` for SmolVLA

Both accept `RUN_PHASE=train`, `RUN_PHASE=eval`, or `RUN_PHASE=all`.

Add `DRY_RUN=true` to print resolved datasets, checkpoints, outputs, and model
repositories without starting training or evaluation.

## PI0.5 template

Edit the configuration block at the top of `run_pi05.sh` for a new experiment.
The main fields are `EXPERIMENT_NAME`, `DATASET_ROOT`, `DATASET_REPO`,
`EVAL_TASKS`, `MODALITIES`, `ACTION_MODE`, `STEPS`, and
`BATCH_SIZE`. The same fields can be overridden for a one-off run:

```bash
EXPERIMENT_NAME=near_far_1epoch \
DATASET_ROOT=/path/to/quadloco-vla-near_far-rgbd-1000 \
DATASET_REPO=DinhQuangDung/quadloco-vla-near_far-rgbd-1000 \
EVAL_TASKS=near_far \
STEPS=18500 \
RUN_PHASE=all \
./run_pi05.sh
```

Both modalities always use proprioception. PI0.5 takes the first 42 values of
the recorded 45D observation, excluding only the final oracle velocity command.
Its run and repository names therefore include `42d-state`. For a local RGB-D
smoke test, set
`MODALITIES=rgbd`, `STEPS=10`,
`BATCH_SIZE=3`, `NUM_WORKERS=0`, `SAVE_CHECKPOINT=false`,
`WANDB_ENABLE=false`, and `HF_HUB_OFFLINE=1`.

## SmolVLA

```bash
RUN_PHASE=all ./run_smolvla.sh
```

To run only one modality, add `MODALITIES=rgb` or `MODALITIES=rgbd`.
SmolVLA selects a 30D state from the recorded 45D input: 12 joint positions,
12 joint velocities, 3D angular velocity, and 3D projected gravity. It excludes
the previous action and oracle velocity command, and uses the `30d-state` suffix.
The orchestrator delegates the actual LeRobot command to
`train_smolvla.sh` and evaluation to `eval_smolvla.sh`, matching the PI0.5
split while allowing each model family to keep different arguments.
Its default dataset, direct-velocity action mode, 40,000 steps, batch size 16,
and RGB-D fusion settings match `run_pi05.sh`.

## Go2 deployment

The deploy and task-client wrappers are combined:

```bash
./run_go2_vla.sh deploy --robot-ip 192.168.123.161
./run_go2_vla.sh task --instruction "Navigate to the red cube"
```
