# Quadloco

Quadloco is an Isaac Lab project for training quadruped locomotion policies and
collecting goal-navigation data for vision-language-action (VLA) training.

## Tested versions

- Isaac Lab `2.3.2` (commit `4df6560e187f2cc66685b41b21b259f4485d0c22`)
- Isaac Sim `5.1.0.0`
- RSL-RL (`rsl-rl-lib`) `5.0.1`
- Python `3.11`

## Setup

Install Isaac Lab by following its installation guide, then place Quadloco next
to the Isaac Lab checkout:

```text
workspace/
├── IsaacLab/
└── quadloco/
```

Quadloco was tested against a specific Isaac Lab commit. In the Isaac Lab
checkout, select that commit and reinstall it:

```bash
cd /path/to/IsaacLab
git status
git fetch origin
git checkout 4df6560e187f2cc66685b41b21b259f4485d0c22
./isaaclab.sh --install
```

Activate the Isaac Lab environment, enter the Quadloco repository, and install
the project:

```bash
conda activate env_isaaclab
cd /path/to/quadloco
python -m pip install -e source/quadloco
```

All commands below assume that the current directory is the Quadloco repository
root. The editable install only needs to be performed once per Python
environment.

Verify the setup by listing the registered environments:

```bash
python scripts/list_envs.py
```

To check the installed versions:

```bash
cat ../IsaacLab/VERSION
git -C ../IsaacLab rev-parse HEAD
python -m pip show isaacsim rsl-rl-lib
```

## 1. Train the locomotion policy

Run the provided training command:

```bash
bash loco_train.sh
```

The equivalent explicit command is:

```bash
python scripts/quadloco_rsl_rl/train.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-v0 \
    --headless
```

Runs and checkpoints are written to:

```text
logs/rsl_rl/unitree_go2_rough_loco/
```

To play a trained checkpoint:

```bash
python scripts/quadloco_rsl_rl/play.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-PLAY-v0 \
    --checkpoint /path/to/model.pt
```

## 2. Collect goal-navigation data

The collection environment combines:

- the trained state-based locomotion policy;
- a scripted goal-to-velocity command; and
- a robot-mounted RGB-D camera.

No separate navigation policy is required for data collection.

### Quick start

Update the checkpoint, dataset path, repository ID, and other settings in
`loco_collect_data.sh`, then run:

```bash
bash loco_collect_data.sh
```

### Explicit command

The following example collects ten trajectories in the default PyTorch format:

```bash
python scripts/quadloco_rsl_rl/run_data_collection.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0 \
    --checkpoint /path/to/model.pt \
    --collect_data \
    --num_episodes 10 \
    --dataset_dir datasets/goal_navigation \
    --headless
```

Important options:

- `--collect_data` enables dataset recording. Without it, the script only runs
  the environment.
- `--dataset_format pt|lerobot` selects per-episode PyTorch files or a LeRobot
  v3 dataset. The default is `pt`.
- `--num_episodes N` controls the number of completed trajectories to save.
- `--dataset_dir PATH` sets the output directory.
- `--num_envs N` controls the number of parallel environments. Keep it at `1`
  for camera collection so neighboring environments do not appear in images.
- `--headless` disables the viewer. Omit it to watch collection.
- `--push_to_hub` uploads a finalized LeRobot dataset. Authenticate first with
  `hf auth login`.

Camera rendering is enabled automatically.

### PyTorch dataset format

Each completed episode is saved immediately:

```text
datasets/goal_navigation/
├── trajectory_000000.pt
├── trajectory_000001.pt
└── ...
```

Load an episode with:

```python
import torch

trajectory = torch.load("datasets/goal_navigation/trajectory_000000.pt")
rgb = trajectory["rgb"]
depth = trajectory["depth"]
joint_actions = trajectory["action"]
velocity_commands = trajectory["velocity_command"]
```

`action` is the low-level locomotion output. `velocity_command` is the scripted
`[vx, vy, wz]` target used as supervision for the high-level navigation policy.

## 3. Collect LeRobot data and train PI0.5

Quadloco includes a patched LeRobot `0.4.4` checkout under
`third_party/lerobot`; no separate LeRobot clone is required.

### Install LeRobot and download PI0.5

Run the setup script once from the Quadloco root:

```bash
bash scripts/setup_vendored_lerobot.sh
```

The script:

1. installs the vendored LeRobot checkout in editable mode; and
2. downloads the compatible PI0.5 checkpoint revision to
   `checkpoints/pi05_base_lerobot_0.4.4`.

The model download is approximately 14.5 GB. The `checkpoints` directory is
ignored by Git.

### Record a LeRobot v3 dataset

The output directory must not already exist:

```bash
python scripts/quadloco_rsl_rl/run_data_collection.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0 \
    --checkpoint /path/to/model.pt \
    --collect_data \
    --dataset_format lerobot \
    --dataset_repo_id YOUR_HF_USERNAME/quadloco-goal-navigation \
    --dataset_dir datasets/quadloco_lerobot_v1 \
    --num_episodes 10 \
    --num_envs 1 \
    --headless
```

LeRobot stores RGB frames as MP4 video and the remaining data and metadata as
Parquet/JSON files. Dataset fields include:

- `action`: high-level navigation command `[vx, vy, wz]`;
- `observation.state`: measured base velocity `[vx, vy, wz]`;
- `observation.policy_state`: full locomotion-policy observation; and
- `observation.low_level_action`: 12-dimensional joint-policy output.

Metric depth is omitted by default because full-resolution float32 depth is
large. Add `--lerobot_include_depth` when depth is required. Add
`--push_to_hub` to upload the finalized dataset using `--dataset_repo_id`.

### Train the high-level velocity policy

Set the dataset, output, and Hugging Face repository values at the top of
`train_pi05_velocity.sh`, or override them with environment variables. Then run:

```bash
bash train_pi05_velocity.sh
```

The script converts the collected dataset to the three-dimensional
`[vx, vy, wz]` training target when necessary, then starts PI0.5 training.

For a two-step, 12-dimensional joint-action smoke test, run:

```bash
bash train_vla_lerobot.sh
```

## PI0.5 compatibility notes

Use the checkpoint downloaded by `scripts/setup_vendored_lerobot.sh`. Do not use
the unpinned `main` revision of `lerobot/pi05_base`: its processor configuration
is newer than LeRobot `0.4.4`.

The vendored checkout also contains a checkpoint-loader compatibility mapping
for PaliGemma's tied token embeddings. No manual patch is required, and strict
checkpoint loading remains enabled.

Confirm that Python imports the vendored checkout:

```bash
python -c \
    "import lerobot.policies.pi05.modeling_pi05 as m; print(m.__file__)"
```

The path should point to `quadloco/third_party/lerobot`, not `site-packages`.
During training, the startup log should show the local checkpoint directory
rather than `lerobot/pi05_base`, and it should not end with:

```text
Warning: Could not remap state dict keys
```
