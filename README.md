# Quadloco

Quadloco is an Isaac Lab project for quadruped locomotion experiments.

## Tested Versions

The current project setup has been tested with:

- Isaac Lab `2.3.2` (workspace commit `4df6560e187f2cc66685b41b21b259f4485d0c22`)
- Isaac Sim `5.1.0.0`
- RSL-RL (`rsl-rl-lib`) `5.0.1`
- Python `3.11`

If you have already cloned IsaacLab, switch to the specific commit with 
```bash 
cd IsaacLab
git status
git fetch origin
git checkout 4df6560e187f2cc66685b41b21b259f4485d0c22
```
then reinstall 
```bash 
./isaaclab.sh --install
```

Verify the checked-out Isaac Lab version with:

```bash
cat ../IsaacLab/VERSION
git -C ../IsaacLab rev-parse HEAD
```
Isaac Sim, RSL-RL, and Python are from the `env_isaaclab` Conda environment used for this
project; the Isaac Lab version is from the sibling workspace checkout. Verify the installed
Python-package versions with:

```bash
conda activate env_isaaclab
python -m pip show isaacsim rsl-rl-lib
```

## Setup

1. Install Isaac Lab and make sure you can run its Python environment.
2. Clone this repository and put on the same level with Isaac Lab.
3. Install `quadloco` in editable mode:

```bash
cd /path/to/quadloco
python -m pip install -e source/quadloco
```

The training scripts import `quadloco` as a Python package. Without the editable install, commands such as:

```bash
python scripts/quadloco_rsl_rl/train.py ...
```

can fail with:

```text
ModuleNotFoundError: No module named 'quadloco'
```

Editable install only needs to be done once per Python environment.

## Verify Installation

List available tasks:

```bash
python scripts/list_envs.py
```

## Training

**Easy run**:
```bash
bash loco_train.sh
```

To train the state-based manager-based Go2 locomotion policy used by the goal-navigation data collector:

```bash
python scripts/quadloco_rsl_rl/train.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-v0 \
    --headless
```

The RSL-RL runs and checkpoints are written under:

```text
logs/rsl_rl/unitree_go2_rough_loco/
```

To play a trained locomotion checkpoint in the standard rough-terrain environment, run:

```bash
python scripts/quadloco_rsl_rl/play.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-PLAY-v0 \
    --checkpoint /path/to/model.pt
```

## Goal-Navigation Data Collection
**Easy run**:
```bash
bash loco_collect_data.sh
```

The data-collection environment uses the trained state-based locomotion policy, a scripted
goal-to-velocity command, and a robot-mounted RGB-D camera. It does not require training a
separate navigation policy before collection.

Collect 10 complete trajectories with:

```bash
python scripts/quadloco_rsl_rl/run_data_collection.py \
    --task Unitree-Go2-Quadloco-ManagerBased-Rough-DataCollection-v0 \
    --checkpoint /path/to/model.pt \
    --collect_data \
    --num_episodes 10 \
    --dataset_dir datasets/goal_navigation \
    --headless
```

Relevant arguments:

- `--collect_data` enables RGB-D trajectory buffering and saving. Without this flag, the
  script behaves as a normal play loop and does not write a dataset.
- `--dataset_format pt|lerobot` selects legacy per-episode PyTorch files or a directly
  loadable LeRobot v3 dataset. The default remains `pt`.
- `--num_episodes N` sets the number of completed trajectories to save. The default is 10.
- `--dataset_dir PATH` sets the output directory. The default is
  `datasets/goal_navigation`.
- `--num_envs N` controls parallel environments. The default is 1 to prevent neighboring
  environments from appearing in the camera images.
- `--checkpoint PATH` selects the trained locomotion checkpoint.
- `--headless` disables the interactive viewer. Omit it to watch the robot-following viewer.

Camera rendering is enabled automatically by the collection script. Each episode is saved
immediately as a separate file:

```text
datasets/goal_navigation/
├── trajectory_000000.pt
├── trajectory_000001.pt
└── ...
```

Load a trajectory with:

```python
import torch

trajectory = torch.load("datasets/goal_navigation/trajectory_000000.pt")
rgb = trajectory["rgb"]
depth = trajectory["depth"]
joint_actions = trajectory["action"]
velocity_commands = trajectory["velocity_command"]
```

`action` contains the low-level locomotion policy output, while `velocity_command` contains
the scripted `[vx, vy, wz]` target suitable as supervision for a future local navigation
module.

### Record directly in LeRobot v3 format

Install the vendored LeRobot checkout in the same Python environment as Isaac Lab:

```bash
pip install -e "./third_party/lerobot[pi]"
```

Then record into a new output directory:

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

The output directory must not already exist. LeRobot writes RGB frames as MP4 video and
state, action, episode, task, and statistics metadata as Parquet/JSON files. The top-level
`action` is the navigation command `[vx, vy, wz]`, and `observation.state` is the measured
base velocity `[vx, vy, wz]`. The full locomotion-policy observation and 12-dimensional
joint-policy output are retained as `observation.policy_state` and
`observation.low_level_action`.
Metric depth is omitted by default because storing full-resolution float32 depth in
Parquet is large; add `--lerobot_include_depth` when it is required.

Add `--push_to_hub` to upload the finalized dataset using `--dataset_repo_id`. Authentication
must already be configured with `hf auth login`.

## PI0.5 Training with LeRobot 0.4.4

This repository includes a patched copy of LeRobot `0.4.4` under
`third_party/lerobot`. It is committed as ordinary Quadloco source and does not require a
separate LeRobot clone.

Do not load
`lerobot/pi05_base` from its unpinned `main` revision: its processor configuration was
updated after LeRobot 0.4.4 and now references `relative_actions_processor`, which 0.4.4
does not provide.

### Fresh installation

After cloning Quadloco, activate the Isaac Lab environment and run the included setup
script:

```bash
conda activate env_isaaclab
cd /path/to/quadloco
bash scripts/setup_vendored_lerobot.sh
```

This installs `third_party/lerobot` in editable mode and downloads the compatible PI0.5
revision `a538eb273274eb30f126a118f39dbc0ee212c883` into
`checkpoints/pi05_base_lerobot_0.4.4`. The `checkpoints` directory is ignored by Git so the
14.5 GB model is not accidentally committed.

### LeRobot 0.4.4 checkpoint-loader compatibility patch

The converted PI0.5 checkpoint stores PaliGemma's tied token embedding under
`lm_head.weight`, while the LeRobot 0.4.4 model also expects
`model.language_model.embed_tokens.weight`. Without the following alias, the loader catches
a strict-loading exception and may continue without completing the pretrained load.
The targeted compatibility mapping is already applied in the vendored
`third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py`. No manual patch is needed.
Strict checkpoint loading remains enabled so unrelated missing or unexpected weights are
not hidden.

Confirm that Python imports the patched editable checkout:

```bash
python -c \
    "import lerobot.policies.pi05.modeling_pi05 as m; print(m.__file__)"
```

The printed path should point inside `/path/to/quadloco/third_party/lerobot`, not
`site-packages`.

### Run training

For the 12-dimensional joint-action smoke test:

```bash
cd /path/to/quadloco
bash train_vla_lerobot.sh
```

For the three-dimensional high-level `[vx, vy, wz]` policy:

```bash
cd /path/to/quadloco
bash train_pi05_velocity.sh
```

The expected startup log should show the local checkpoint directory. If it shows
`lerobot/pi05_base`, the unpinned Hub revision is still being used. A successful strict
load should not end with:

```text
Warning: Could not remap state dict keys
```
