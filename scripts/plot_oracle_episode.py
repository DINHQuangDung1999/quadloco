"""Plot a trusted local PT episode recorded with --record_scene_state."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    data = torch.load(args.episode, map_location="cpu", weights_only=False)
    out = args.episode.parent
    t = data["timestamp_s"].numpy()
    robot = data["robot_root_state_w"].numpy()
    objects = data["object_root_state_w"].numpy()
    rgb = data["rgb"].numpy()
    depth = data["depth"].numpy().squeeze(-1)
    assert len(t) == len(robot) == len(objects) == len(rgb) == len(depth)
    assert np.isfinite(robot).all() and np.isfinite(objects).all()
    assert np.all(np.diff(t) > 0) and np.ptp(robot[:, :2], axis=0).max() > 1
    visible = objects[0, :, 2] > -1
    names = np.asarray(data["object_names"])[visible]
    goal = data["goal_position_w"].numpy()[0, :2]
    speed = np.linalg.norm(robot[:, 7:10], axis=1)
    fig, (ax, vel) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    segments = np.stack((robot[:-1, :2], robot[1:, :2]), axis=1)
    line = LineCollection(segments, cmap="viridis", linewidth=3)
    line.set_array(t[:-1])
    ax.add_collection(line)
    fig.colorbar(line, ax=ax, label="Time (s)", shrink=0.8)
    ax.scatter(*robot[0, :2], marker="o", color="black", label="Robot start", zorder=5)
    ax.scatter(*robot[-1, :2], marker="X", color="black", label="Robot end", zorder=5)
    for name, state in zip(names, objects[0, visible]):
        shape, color = name.rsplit("_", 2)[-2:]
        ax.scatter(*state[:2], c=color, marker={"sphere": "o", "cube": "s", "pyramid": "^"}.get(shape, "o"), s=100, edgecolors="black", label=f"{color} {shape}")
    ax.add_patch(plt.Circle(goal, 1.0, fill=False, linestyle="--", color="gray", label="Goal tolerance (1 m)"))
    ax.autoscale()
    ax.margins(0.15)
    ax.set(aspect="equal", xlabel="World x (m)", ylabel="World y (m)", title="Go2 trajectory and scene objects")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.grid(alpha=0.2)
    for index, label in zip(range(7, 10), ("vx", "vy", "vz")):
        vel.plot(t, robot[:, index], label=label, alpha=0.8)
    vel.plot(t, speed, color="black", label="Speed", linewidth=1.5)
    vel.set(xlabel="Time (s)", ylabel="World linear velocity (m/s)", title="Measured robot velocity")
    vel.grid(alpha=0.2)
    vel.legend()
    fig.suptitle(data["task"])
    fig.savefig(out / "trajectory.png", dpi=200)
    fig.savefig(out / "trajectory.pdf")
    plt.close(fig)
    # Shared metric depth scale makes snapshots directly comparable.
    indices = np.linspace(0, len(t) - 1, 3, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for col, index in enumerate(indices):
        frame = rgb[index, ..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        axes[0, col].imshow(frame)
        axes[0, col].set_title(f"Front RGB · t = {t[index]:.2f} s")
        im = axes[1, col].imshow(np.ma.masked_invalid(depth[index]), cmap="magma_r", vmin=0.1, vmax=20)
        axes[1, col].set_title("Front depth (image-plane distance)")
        for row in range(2):
            axes[row, col].axis("off")
        plt.imsave(out / f"rgb_frame_{index:06d}.png", frame)
        np.save(out / f"depth_frame_{index:06d}_metres.npy", depth[index])
    fig.colorbar(im, ax=axes[1, :], label="Depth (m)", shrink=0.65)
    fig.suptitle(data["task"])
    fig.savefig(out / "vision_examples.png", dpi=180)
    fig.savefig(out / "vision_examples.pdf")
    plt.close(fig)
    arrays = {key: value.numpy() for key, value in data.items() if torch.is_tensor(value) and key not in ("rgb", "depth")}
    # np.savez_compressed(out / "states.npz", **arrays, object_names=np.asarray(data["object_names"]))
    summary = {
        "task": data["task"], "accepted": data["accepted"], "seed": data["seed"],
        "frames": len(t), "frequency_hz": 1 / data["dt"], "duration_s": len(t) * data["dt"],
        "path_length_m": float(np.linalg.norm(np.diff(robot[:, :2], axis=0), axis=1).sum()),
        "final_goal_distance_m": float(np.linalg.norm(robot[-1, :2] - goal)),
        "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
        "depth_finite_fraction": float(np.isfinite(depth).mean()),
        "visible_objects": names.tolist(), "all_object_names": data["object_names"],
        "maximum_object_speed_m_s": float(np.linalg.norm(objects[:, visible, 7:10], axis=-1).max()),
        "root_state_layout": data["root_state_layout"], "depth_description": data["depth_description"],
        "sampling_description": data["sampling_description"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
