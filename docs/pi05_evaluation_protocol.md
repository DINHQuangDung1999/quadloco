# PI0.5 RGB-D Waypoint Evaluation Protocol

This document fixes the evaluation definitions for the QuadLoco RGB-D PI0.5
experiments. Keep it versioned with the code and cite the exact repository
revision used for reported results.

## System under evaluation

PI0.5 consumes RGB-D, the 45-dimensional locomotion observation, and a natural
language instruction. It predicts a robot-frame waypoint
`[x_forward, y_left]`. A deterministic adapter maps the waypoint to
`[vx, 0, wz]`, and the frozen low-level RSL-RL policy produces joint actions.

Ground-truth goals and object identities are available only to the evaluator.
They are not inputs to PI0.5.

## Common binary success definition

An episode is successful when all conditions hold before the 20 s timeout:

1. The correct semantic target is selected.
2. The robot enters the task-specific goal region.
3. It remains there for at least 0.5 s.
4. During that interval, planar speed is below 0.15 m/s and absolute yaw rate
   is below 0.20 rad/s.
5. There is no forbidden collision or fall.

Forbidden collision is contact above 1 N involving the base, thigh, or calf.
Normal foot-ground contact is excluded.

## Task-specific criteria

| Task | Semantic criterion | Goal region | Additional condition |
|---|---|---:|---|
| Direct | Correct instructed object | 1.0 m | Stable stop |
| Occluded | Correct object behind occluder | 1.0 m | No occluder collision |
| Paired relational | Correct identical object qualified by its reference | 1.0 m | Correct target is at least 0.25 m closer than distractors |
| Near/middle/far | Correct initial forward-distance rank | 1.0 m | Report each rank separately |
| Object-relative | Correct generated relative position and metric offset | 0.20 m | Collision-free; any safe behind route is valid |

Do not require exact reproduction of an A* or scripted detour. Any safe route
that satisfies the final task counts as successful.

## Metrics

Report, per task and macro-averaged across tasks:

- task success rate;
- semantic target-selection accuracy;
- geometric success rate;
- collision rate;
- fall rate;
- timeout rate;
- final goal-position error;
- time to success;
- path length;
- SPL (success weighted by path length).

For object-relative positioning, additionally stratify results by relation
(`front`, `behind`, `left`, `right`) and requested surface offset
(`0.50`, `0.75`, `1.00`, `1.25` m). For distance-rank selection, report near,
middle, and far accuracy separately.

## Checkpoint and final evaluation

Checkpoint screening uses at least 20 held-out episodes per task family:

```text
5 tasks × 20 episodes = 100 episodes per checkpoint
```

Final paper results use at least 100 held-out episodes per task family:

```text
5 tasks × 100 episodes = 500 episodes
```

Use unseen random seeds, object positions and sizes, pair spacing, obstacle
sizes and colors, metric offsets, and language variants. Balance discrete
relations and distances rather than relying only on random frequencies.

Select checkpoints primarily by macro-average task success, subject to:

```text
collision rate <= 5%
fall rate <= 1%
```

## RGB-D ablations

For claims about depth, compare the same checkpoint/training budget under:

1. RGB-D input;
2. RGB-only input;
3. depth replaced by the far plane or zeroed, as a diagnostic ablation.

Occluded navigation, obstacle clearance, and object-relative positioning are
the primary depth-sensitive results.

## Running the evaluator

The convenience launcher starts the PI0.5 server once and evaluates all five
task families sequentially:

```bash
VLA_CHECKPOINT=outputs/pi05_quadloco_depth_main_200k/checkpoints/50000/pretrained_model \
EPISODES=20 \
OUTPUT_DIR=eval_results/checkpoint_50000 \
./eval_pi05.sh
```

To run one family manually, start the PI0.5 socket server and invoke:

```bash
python scripts/quadloco_rsl_rl/eval_pi05.py \
    --navigation_mode direct \
    --num_episodes 20 \
    --checkpoint ckpt/model_999.pt \
    --vla_host 127.0.0.1 \
    --vla_port 5555 \
    --output_dir eval_results/checkpoint_50000/direct \
    --headless
```

Run all task families:

```bash
for TASK in direct occluded relational near_far object_relative; do
    python scripts/quadloco_rsl_rl/eval_pi05.py \
        --navigation_mode "$TASK" \
        --num_episodes 20 \
        --checkpoint ckpt/model_999.pt \
        --output_dir "eval_results/checkpoint_50000/$TASK" \
        --headless
done
```

Each run produces `<task>_episodes.jsonl` and `<task>_summary.json`. Preserve
the episode records so confidence intervals and stratified analyses can be
recomputed without rerunning simulation.
