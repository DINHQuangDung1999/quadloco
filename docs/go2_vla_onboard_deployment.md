# Go2 VLA + onboard rl_sar deployment

This runbook starts the PI0.5 VLA on the laptop and uses the standalone
`rl_sar` controller on the Go2 to convert VLA base-velocity commands into
Unitree LowCmd joint targets.

## Architecture

```text
Go2 D435i --ROS 2 RGB-D--> laptop VLA
Go2 rl_sar --UDP 5561, 42D state--> laptop VLA
laptop VLA --UDP 5560, vx/vy/wz--> Go2 rl_sar --rt/lowcmd--> motors
```

The onboard `rl_sar` process is the only publisher of `rt/lowcmd`. The laptop
never publishes joint commands directly.

The VLA checkpoint records a 45D state. PI0.5 uses its first 42 values, while
SmolVLA selects 30D (q, qdot, angular velocity, and projected gravity). The Go2
sends the first 42 values; the laptop pads the three unavailable command values
with zeros before inference.

## Network assumptions

- Laptop Wi-Fi address: `192.168.0.88`
- Go2 Wi-Fi address: `192.168.0.192` on `wlan0`
- Go2 internal Unitree DDS address: `192.168.123.18` on `eth0`
- VLA command UDP port: `5560`
- Policy-state UDP port: `5561`

Check the addresses after either machine reboots:

```bash
ip -brief address
```

Each fresh laptop task terminal must use ROS domain 0 and pin CycloneDDS to the
robot-facing Wi-Fi interface. `QUADLOCO_KEEP_CYCLONEDDS_URI=1` prevents
`run_go2_vla.sh task` from removing this explicit interface configuration.

The `rl_real_go2` interface argument is `eth0`, because LowState and LowCmd use
the robot's internal Unitree DDS network. The VLA UDP bridge binds all local
interfaces and therefore receives laptop packets through `wlan0`.

## Deployed locomotion policy

The onboard base configuration must contain:

```yaml
go2:
  policy_config_name: "model_999"
```

It selects:

```text
/home/unitree/go2_isaac_gazebo/policy/go2/model_999/config.yaml
/home/unitree/go2_isaac_gazebo/policy/go2/model_999/policy.onnx
```

`model_999` has one 45D input, no observation history, and 12 action outputs:

```yaml
model_forward_mode: "single"
num_observations: 45
observations_history: []
```

Optional offline validation on the Go2 does not start LowCmd:

```bash
cd /home/unitree/go2_isaac_gazebo
./cmake_build/bin/validate_policy_model \
    policy/go2/model_999/policy.onnx single 45 0 12
./cmake_build/bin/validate_policy_config go2 model_999
```

## Start the system

Use three terminals. Keep the robot supported during initial validation and
ensure no other process publishes `rt/lowcmd`.

### Terminal 1 — laptop VLA server

```bash
cd /home/dung-admin/go2_ws/quadloco
conda activate vla
bash run_vla_server.sh
```

Leave the server running. It loads the model once and waits for instructed task
clients; it does not command the robot by itself.

### Terminal 2 — onboard rl_sar

SSH to the Go2 and run:

```bash
cd /home/unitree/go2_isaac_gazebo

export RL_SAR_ENABLE_VLA_BRIDGE=1
export RL_SAR_VLA_ALLOWED_HOST=192.168.0.88

./cmake_build/bin/rl_real_go2 eth0
```

The bridge startup message should report fixed limits:

```text
limits=[0.5, 0.05, 0.1]
```

Activate locomotion using the onboard terminal or gamepad:

1. Press `0` (or gamepad A) to enter GetUp.
2. Wait until GetUp completes.
3. Press `1` (or the configured gamepad locomotion combination) to enter
   `RLFSMStateRLLocomotion`.
4. Confirm that `model_999/policy.onnx` loads and its contract validates.
5. Leave navigation mode OFF until the laptop task is ready.

### Terminal 3 — Go2 camera service

The camera uses the Go2's normal Foxy/Fast DDS environment. Do not apply the
laptop's CycloneDDS interface setting here:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/ros2_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>wlan0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'

ros2 launch go2_depth_camera realsense_d435i.launch.py \
    depth_module.profile:=640,480,15 \
    rgb_camera.profile:=640,480,15 \
    enable_sync:=true \
    align_depth.enable:=true \
    pointcloud.enable:=false
```

### Terminal 4 — laptop instructed task

Configure DDS before starting the instructed task. These exports apply only to
the current terminal and must be repeated in every fresh task terminal:

```bash
source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="wlp0s20f3"/></Interfaces></General></Domain></CycloneDDS>'
export QUADLOCO_KEEP_CYCLONEDDS_URI=1

ros2 daemon stop
ros2 daemon start
```

Verify that actual RGB and aligned-depth samples reach the laptop:

```bash
ros2 topic hz /d435i/color/image_raw
```

Stop it with `Ctrl+C`, then check depth:

```bash
ros2 topic hz /d435i/aligned_depth_to_color/image_raw
```

Both commands run continuously, so use separate terminals or stop each with
`Ctrl+C`. Merely seeing the topic names in `ros2 topic list` does not prove
that image samples are reaching the laptop. Do not copy the laptop-specific
`wlp0s20f3` configuration to the Go2.

Once both streams report a frame rate, verify camera input, state return, and
inference without motion from the configured terminal:

```bash
cd /home/dung-admin/go2_ws/quadloco
conda activate vla

./run_go2_vla.sh task \
    --instruction "Navigate to the red ball" \
    --state-source udp \
    --robot-host 192.168.0.192
```

Expected startup messages include:

```text
[POLICY] Connected ... state=45, state_token_dim=42
[STATE] Waiting for 42D onboard policy state on UDP port 5561
[SAFETY] DRY RUN
```

Stop the dry run with `Ctrl+C`. Then start motion-enabled output:

```bash
./run_go2_vla.sh task \
    --instruction "Navigate to the red ball" \
    --state-source udp \
    --output udp \
    --robot-host 192.168.0.192 \
    --max-vx 0.5 \
    --max-vy 0.05 \
    --max-wz 0.1 \
    --enable-motion
```

Confirm that predictions are finite and plausible. In the onboard terminal,
press `N` once and confirm:

```text
Navigation mode: ON
```

Only then do VLA commands replace joystick velocity commands.

## Command bounds and watchdog

The onboard bridge enforces these fixed absolute limits regardless of laptop
arguments:

```text
|vx| <= 0.50 m/s
|vy| <= 0.05 m/s
|wz| <= 0.10 rad/s
```

Packets older than 0.5 seconds cause the onboard command to become zero.
Commands are accepted only from `RL_SAR_VLA_ALLOWED_HOST`, and only while
navigation mode is ON.

The normal controller line displays joystick values (`control.x/y/yaw`), not
the final VLA-overridden observation. It can show zeros while VLA commands are
active. The policy debug CSV records the actual values in `obs_cmd_x`,
`obs_cmd_y`, and `obs_cmd_yaw`.

## Safe stop sequence

Use this order:

1. Press `N` onboard and confirm `Navigation mode: OFF`.
2. Stop the laptop task with `Ctrl+C` so it sends zero datagrams.
3. Press `9` for GetDown when appropriate.
4. Use `P` for passive mode if behavior is unexpected.
5. Stop `rl_real_go2` with `Ctrl+C` only after the robot is safe.

Do not suspend the laptop task with `Ctrl+Z`; suspension bypasses its normal
zero-command cleanup. Use the physical emergency stop whenever software does
not stop the robot promptly.

## Troubleshooting

### No 42D state on the laptop

The policy state is produced while `RLLocomotion` is active. Complete GetUp and
enter locomotion before starting `--state-source udp`. Check that the onboard
bridge owns its command socket:

```bash
ss -lunp | grep 5560
```

### Predictions print, but the robot does not walk

Confirm all of the following:

- The task says `MOTION ENABLED`.
- `Navigation mode: ON` was printed onboard.
- `model_999` loaded and validated.
- The predicted speed is large enough to trigger this policy's gait.
- The controller has not transitioned to passive/damping mode.

The onboard debug log is written under:

```text
/home/unitree/go2_isaac_gazebo/debug_logs/
```

In its CSV, inspect `obs_cmd_x/y/yaw`. Active locomotion normally uses policy
gains (`kp=25`, `kd=0.5` in the current configuration); `kp=0`, `kd=8` indicates
passive/damping output.

### Laptop prints 0.5 but the controller receives a smaller value

The laptop and onboard controller clamp independently. Keep their limits
aligned with the command shown above. The onboard limit is authoritative.

### `rl_real_go2` reports invalid arguments

Pass the internal DDS interface:

```bash
./cmake_build/bin/rl_real_go2 eth0
```
