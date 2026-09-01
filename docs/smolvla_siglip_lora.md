# SmolVLA SigLIP LoRA adaptation

## Purpose

The expert-only SmolVLA RGB waypoint policy changed little between 8,000 and
40,000 updates (22% versus 24% success over 50 direct-navigation episodes).
This experiment tests whether limited adaptation of the pretrained visual
representation helps, without updating the language decoder or all SigLIP
weights.

## Hybrid training design

The policy uses the following trainability pattern:

| Component | Training mode |
| --- | --- |
| SigLIP patch embedding, normalization, and MLPs | Frozen |
| SigLIP attention base projections | Frozen |
| SigLIP `q_proj`, `k_proj`, `v_proj`, `out_proj` LoRA residuals | Trainable |
| SmolVLM language decoder | Frozen |
| SmolVLA action expert | Fully trainable |
| State/action/time projections | Fully trainable |
| ConvNeXt depth encoder and fusion (RGB-D only) | Fully trainable |

The LoRA residual for a frozen projection is

```text
y = W x + (alpha / rank) B A dropout(x)
```

where only `A` and `B` are trained. `B` is initialized to zero, so enabling
LoRA does not change the pretrained policy output before the first update.

The installed SigLIP has 12 encoder blocks and hidden width 768. Rank 16 on
all four projections adds

```text
12 blocks * 4 projections * 16 * (768 + 768) = 1,179,648 parameters.
```

## Checkpoint compatibility

This implementation intentionally does not use LeRobot's top-level PEFT
wrapper. That wrapper freezes the complete policy and normally saves an
adapter-only checkpoint, whereas this experiment must fully train and save the
approximately 100M-parameter action expert.

`VisionLoRALinear` subclasses `nn.Linear`, preserving the legacy `weight` and
`bias` state-dict keys. LoRA weights are additional keys in the same normal
`model.safetensors`. Consequently:

- existing non-LoRA checkpoints remain loadable;
- LoRA checkpoints use the existing SmolVLA server and evaluator;
- optimizer and scheduler resume through the normal LeRobot checkpoint path;
- no additional `peft` dependency or adapter-loading step is required.

The saved `config.json` records whether LoRA is enabled and its rank, alpha,
dropout, and target projections. Model construction injects the same LoRA
structure before loading a LoRA-enabled checkpoint.

## Files changed

- `third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`
  adds validated visual-LoRA configuration.
- `third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  defines and injects the residuals and restores their trainability after the
  VLM is frozen.
- `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
  passes the configuration into model construction.
- `run_smolvla.sh` is the unified training/evaluation entry point. It exposes
  the settings to LeRobot and adds `siglip_lora_r<rank>` to LoRA experiment
  names while preserving all existing non-LoRA names.
- `scripts/test_smolvla_vision_lora.py` checks zero-output initialization,
  parameter counts, trainability, and legacy state-dict keys.

## Recommended first run

Run the controlled RGB direct-task waypoint comparison from the original
SmolVLA base checkpoint:

```bash
cd /home/summerschool/summerschool_ws/quadloco

TASKS=direct \
MODALITIES=rgb \
ACTION_MODE=waypoint \
BATCH_SIZE=16 \
STEPS_OVERRIDE=8000 \
SAVE_FREQ_OVERRIDE=2000 \
RUN_TAG=2epoch_siglip_lora \
RUN_PHASE=train \
VISION_LORA_ENABLED=true \
VISION_LORA_RANK=16 \
VISION_LORA_ALPHA=16 \
VISION_LORA_DROPOUT=0.05 \
./run_smolvla.sh
```

The default targets are all four SigLIP attention projections. They may be
overridden with a JSON list in `VISION_LORA_TARGETS`.

Evaluate the final checkpoint with the same experiment variables and:

```bash
RUN_PHASE=eval ./run_smolvla.sh
```

The recent 30K direct-velocity RGB and RGB-D comparison is available as a
preset instead of a separate wrapper script:

```bash
PRESET=direct_30k RUN_PHASE=all ./run_smolvla.sh
```

Keep the first comparison RGB-only. If visual LoRA improves the 50-episode,
10-second direct evaluation, repeat it for near--far and object-relative, then
add the RGB-D branch as a separate ablation.
