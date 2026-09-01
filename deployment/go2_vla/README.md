# Go2 VLA deployment

For the complete laptop/onboard startup, testing, and shutdown procedure, see
[`docs/go2_vla_onboard_deployment.md`](../../docs/go2_vla_onboard_deployment.md).

This package runs the trained RGB-D PI0.5 direct-velocity policy on a real
Unitree Go2. It does not import Isaac Lab, Isaac Sim, or RSL-RL. Motion is
disabled unless `--enable-motion` is passed explicitly.

## Environment

Use Python 3.10 so the process can import the ROS Humble Python extensions.
From the Quadloco root:

```bash
python3.10 -m venv --system-site-packages .venv-deploy
source .venv-deploy/bin/activate
source /opt/ros/humble/setup.bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "third_party/lerobot[pi]"
```

Verify the combined environment:

```bash
python -c "import rclpy, message_filters, torch, lerobot; print(torch.cuda.is_available())"
```

## Dry run

```bash
VLA_PYTHON="$PWD/.venv-deploy/bin/python" \
VLA_CHECKPOINT=/path/to/pretrained_model \
./run_go2_vla.sh deploy \
    --instruction "Navigate to the red cube"
```

The default topics are `/d435i/color/image_raw` and
`/d435i/aligned_depth_to_color/image_raw`. Do not set `CYCLONEDDS_URI` unless
an explicit interface is necessary; use a name reported by `ip -brief address`.
The launcher discards an inherited `CYCLONEDDS_URI` by default. Set
`QUADLOCO_KEEP_CYCLONEDDS_URI=1` to preserve a configuration you have verified.

## Recorded-image test

The deployment automatically supplies a zero vector matching the checkpoint's
recorded state dimension when `--state-source zeros` is used. PI0.5 uses the
first 42 values of the recorded 45D state. SmolVLA selects 30D: joint position,
joint velocity, angular velocity, and projected gravity.

```bash
VLA_PYTHON="$PWD/.venv-deploy/bin/python" \
VLA_CHECKPOINT="$PWD/../pi05_rgb_direct_vel" \
./run_go2_vla.sh deploy \
    --camera-source recorded \
    --recorded-rgb /path/to/frame_rgb.png \
    --instruction "Navigate to the red cube" \
    --state-source zeros \
    --one-shot \
    --max-inference-age 120
```

For a depth-enabled checkpoint, also pass the matching Z16 file with
`--recorded-depth /path/to/frame_depth_z16.png`.

## Onboard standalone rl_sar locomotion output

The standalone Go2 controller has a gated UDP bridge. On the robot, enable it
only for the controller process that already owns `rt/lowcmd`:

```bash
cd /home/unitree/go2_isaac_gazebo
export RL_SAR_ENABLE_VLA_BRIDGE=1
export RL_SAR_VLA_ALLOWED_HOST=192.168.0.88
./cmake_build/bin/rl_real_go2 eth0
```

The bridge listens for bounded velocity commands on UDP 5560 and sends the
first 42 values of rl_sar's exact policy observation to the laptop on UDP 5561.
It is inert outside rl_sar navigation mode. In navigation mode, a command older
than 0.5 seconds becomes zero. The onboard rl_sar process remains the sole
publisher of `rt/lowcmd`.

Start with state return and computation only on the laptop:

```bash
./run_go2_vla.sh task \
    --instruction "Navigate to the red cube" \
    --state-source udp \
    --robot-host 192.168.0.192
```

After checking the printed predictions, explicitly enable UDP motion output:

```bash
./run_go2_vla.sh task \
    --instruction "Navigate to the red cube" \
    --state-source udp \
    --output udp \
    --robot-host 192.168.0.192 \
    --enable-motion
```

The laptop places the received 42D transport state in the first 42 entries of
the recorded 45D shape and pads the unavailable command entries with zeros.
PI0.5 consumes those 42 entries; SmolVLA selects indices 0--23 and 36--41.

## ROS rl_sar locomotion output

The recommended real-robot path publishes the bounded VLA base command as a
ROS 2 `geometry_msgs/Twist`. The `rl_sar` Go2 controller consumes the isolated
`/quadloco/cmd_vel` topic,
runs the locomotion policy, and remains the only process that publishes
`rt/lowcmd`:

```bash
VLA_PYTHON="$PWD/.venv-deploy/bin/python" \
VLA_CHECKPOINT="$PWD/../pi05_rgb_direct_vel" \
./run_go2_vla.sh deploy \
    --instruction "Navigate to the red cube" \
    --output ros \
    --cmd-vel-topic /quadloco/cmd_vel \
    --enable-motion
```

Both `--output ros` and `--enable-motion` are required. Without them the VLA
stays in dry-run mode. Enable navigation mode in `rl_sar` only after its
locomotion controller is standing safely; the VLA publishes three zero commands
when it exits.

This requires the ROS 2 build of `rl_sar`. Its standalone CMake Go2 binary is
built without ROS and therefore does not subscribe to `/cmd_vel`.

The deployment reads the raw dimension and any selected state indices from the
checkpoint. PI0.5 uses a 42D leading slice; SmolVLA selects 30D before padding.

## Separate server and instructed task

Keep the GPU model loaded in a long-lived terminal:

```bash
VLA_PYTHON=/path/to/lerobot/python bash run_vla_server.sh
```

The server loads the default checkpoint at `../pi05_rgb_direct_vel`, listens on
`127.0.0.1:5555`, and has no robot command publisher. It may safely wait without
an active task client.

Start a particular task from another terminal. First use dry-run mode:

```bash
./run_go2_vla.sh task --instruction "Navigate to the red cube"
```

After validating camera input, server inference, and the `rl_sar` navigation
mode, explicitly arm `/quadloco/cmd_vel` output:

```bash
./run_go2_vla.sh task \
    --instruction "Navigate to the red cube" \
    --enable-motion
```

Stopping the task client publishes three zero `Twist` messages. The model
server stays loaded and waits for the next instructed task.

Zero-filled policy state is permitted only for computation-only testing. A
motion-enabled task must receive the exact 45D locomotion observation exported
by the ROS build of `rl_sar`:

```bash
./run_go2_vla.sh task \
    --instruction "Navigate to the red cube" \
    --state-source ros \
    --policy-state-topic /quadloco/policy_state \
    --enable-motion
```
