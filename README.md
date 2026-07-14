# Quadloco

Quadloco is an Isaac Lab project for quadruped locomotion experiments.

## Setup

1. Install Isaac Lab and make sure you can run its Python environment.
2. Clone this repository.
3. Install `quadloco` in editable mode:

```bash
cd /path/to/quadloco
python -m pip install -e source/quadloco
```

If you are using Isaac Lab's launcher-managed Python instead of a local venv/conda env, use that interpreter instead of `python`.

## Why Editable Install?

The training scripts import `quadloco` as a Python package. Without the editable install, commands such as:

```bash
python scripts/quadloco_rl_games/train.py ...
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

RL-Games:

```bash
python scripts/quadloco_rl_games/train.py --task Unitree-Quadloco-Direct-Flat-v0
```

RSL-RL:

```bash
python scripts/quadloco_rsl_rl/train.py --task Unitree-Quadloco-Direct-Flat-v0
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

SB3:

```bash
python scripts/quadloco_sb3/train.py --task Unitree-Quadloco-Direct-Flat-v0
```

SKRL:

```bash
python scripts/quadloco_skrl/train.py --task Unitree-Quadloco-Direct-Flat-v0
```

## Goal-Navigation Data Collection

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

## Quick Checks

Zero-action agent:

```bash
python scripts/zero_agent.py --task Unitree-Quadloco-Direct-Flat-v0
```

Random-action agent:

```bash
python scripts/random_agent.py --task Unitree-Quadloco-Direct-Flat-v0
```

## Notes

- For direct environments, actor observations come from `"policy"` and privileged critic observations come from `"critic"`.
- If you change task names or add new tasks, `scripts/list_envs.py` is the quickest way to confirm they are registered correctly.
