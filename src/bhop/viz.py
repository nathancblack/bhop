"""Visualization tools for bhop trajectories and policy analysis.

Generates matplotlib figures for speed profiles, trajectories, action distributions,
and learned policy heatmaps.
Figures are saved to the figures/ directory.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from bhop.env import BhopEnv

FIGURES_DIR = "figures"


def _ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_speed_over_time(episode_data: dict[str, Any]) -> Figure:
    """Plot horizontal speed vs tick for a single episode.

    Args:
        episode_data: Dict containing at least 'speeds' (list of float).

    Returns:
        matplotlib Figure.
    """
    speeds = episode_data["speeds"]
    fig, ax = plt.subplots()
    ax.plot(speeds, linewidth=0.8)
    ax.axhline(320, color="r", linestyle="--", linewidth=0.8, label="sv_maxspeed (320)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Horizontal Speed (ups)")
    ax.set_title("Speed Over Time")
    ax.legend()
    _ensure_figures_dir()
    fig.savefig(os.path.join(FIGURES_DIR, "speed_over_time.png"), dpi=150)
    plt.close(fig)
    return fig


def plot_trajectory(episode_data: dict[str, Any]) -> Figure:
    """Plot top-down XY path showing strafing pattern.

    Args:
        episode_data: Dict containing at least 'positions' (list of [x, y]).

    Returns:
        matplotlib Figure.
    """
    positions = np.array(episode_data["positions"])
    fig, ax = plt.subplots()
    ax.plot(positions[:, 0], positions[:, 1], linewidth=0.5)
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=6, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=6, label="End")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    ax.set_title("Top-Down Trajectory")
    ax.set_aspect("equal")
    ax.legend()
    _ensure_figures_dir()
    fig.savefig(os.path.join(FIGURES_DIR, "trajectory.png"), dpi=150)
    plt.close(fig)
    return fig


def plot_action_distribution(episode_data: dict[str, Any]) -> Figure:
    """Plot distribution of continuous actions across an episode.

    Forward/right/jump are shown as bar charts of their thresholded values.
    Yaw delta is shown as a histogram of the raw continuous values.

    Args:
        episode_data: Dict containing at least 'actions' (list of action arrays).

    Returns:
        matplotlib Figure.
    """
    actions = np.array(episode_data["actions"])
    labels = ["Forward", "Right", "Jump", "Yaw Delta (deg)"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Forward/right: threshold at ±0.33, show bar chart of {-1, 0, +1} buckets
    for i, ax in enumerate(axes[:2]):
        values = actions[:, i]
        buckets = np.where(values > 0.33, 1, np.where(values < -0.33, -1, 0))
        unique, counts = np.unique(buckets, return_counts=True)
        ax.bar(unique, counts, width=0.6)
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Count")
        ax.set_xticks([-1, 0, 1])

    # Jump: threshold at 0.0, show bar chart of {no, yes}
    jump_values = actions[:, 2]
    jump_buckets = np.where(jump_values > 0.0, 1, 0)
    unique, counts = np.unique(jump_buckets, return_counts=True)
    axes[2].bar(unique, counts, width=0.4)
    axes[2].set_xlabel(labels[2])
    axes[2].set_ylabel("Count")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["No", "Yes"])

    # Yaw delta: histogram of raw continuous values
    axes[3].hist(actions[:, 3], bins=30, edgecolor="black", linewidth=0.5)
    axes[3].set_xlabel(labels[3])
    axes[3].set_ylabel("Count")

    fig.suptitle("Action Distribution")
    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig(os.path.join(FIGURES_DIR, "action_distribution.png"), dpi=150)
    plt.close(fig)
    return fig


def analyze_policy(model: Any) -> dict[str, Any]:
    """Query the model's deterministic action across a grid of speeds and on_ground states.

    Constructs synthetic observations with vel_x=speed, vel_y=0, vel_z=0 and
    queries the model at each (speed, on_ground) pair.

    Args:
        model: Trained SB3 model with a predict() method.

    Returns:
        Dict with keys:
            speeds: 1D array of speed values queried
            ground_jump_pct: float, % of ground observations where model chooses jump
            ground_actions: dict with forward/right/jump/yaw arrays for on_ground=True
            air_actions: dict with forward/right/jump/yaw arrays for on_ground=False
    """
    speeds = np.linspace(0, 800, 81)  # 0, 10, 20, ..., 800

    results: dict[str, Any] = {"speeds": speeds}

    for label, on_ground_val in [("ground_actions", 1.0), ("air_actions", 0.0)]:
        forwards, rights, jumps, yaw_deltas = [], [], [], []
        for speed in speeds:
            obs = np.array([speed, 0.0, speed, 0.0, on_ground_val], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            # Threshold forward/right at ±0.33 to match env._map_action()
            fwd = float(action[0])
            forwards.append(1 if fwd > 0.33 else (2 if fwd < -0.33 else 0))
            rt = float(action[1])
            rights.append(1 if rt > 0.33 else (2 if rt < -0.33 else 0))
            jumps.append(1 if float(action[2]) > 0.0 else 0)
            # Yaw is directly in degrees (clipped to bounds by env)
            yaw_deg = float(np.clip(action[3], -BhopEnv.MAX_YAW_DEG, BhopEnv.MAX_YAW_DEG))
            yaw_deltas.append(yaw_deg)

        results[label] = {
            "forward": np.array(forwards),
            "right": np.array(rights),
            "jump": np.array(jumps),
            "yaw_deg": np.array(yaw_deltas),
        }

    ground_jumps = results["ground_actions"]["jump"]
    results["ground_jump_pct"] = 100.0 * np.mean(ground_jumps)

    return results


def plot_policy_heatmap(analysis_data: dict[str, Any]) -> Figure:
    """Visualize learned policy: yaw delta vs speed, compared to optimal.

    Produces a 3-panel figure:
      1. Yaw delta (deg) vs speed for air observations, with optimal angle overlay
      2. Jump action vs speed for ground observations
      3. Strafe (right action) vs speed for air observations

    Args:
        analysis_data: Output of analyze_policy().

    Returns:
        matplotlib Figure.
    """
    speeds = analysis_data["speeds"]
    air = analysis_data["air_actions"]
    ground = analysis_data["ground_actions"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    # Panel 1: Yaw delta vs speed (air), with optimal angle
    ax = axes[0]
    ax.plot(speeds, np.abs(air["yaw_deg"]), "b.-", markersize=4, label="Learned |yaw|")
    # Optimal per-tick yaw: arctan(wishspeed / speed), in degrees
    safe_speeds = np.where(speeds > 1, speeds, 1)
    optimal_deg = np.degrees(np.arctan(320.0 / safe_speeds))
    # Clamp to env's max yaw for fair comparison
    optimal_deg = np.clip(optimal_deg, 0, BhopEnv.MAX_YAW_DEG)
    ax.plot(speeds, optimal_deg, "r--", linewidth=1.2, label="Optimal arctan(320/speed)")
    ax.set_xlabel("Horizontal Speed (ups)")
    ax.set_ylabel("|Yaw Delta| (deg/tick)")
    ax.set_title("Air: Yaw Angle vs Speed")
    ax.legend()
    ax.set_xlim(0, 800)

    # Panel 2: Jump action vs speed (ground)
    ax = axes[1]
    ax.plot(speeds, ground["jump"], "g.-", markersize=4)
    ax.set_xlabel("Horizontal Speed (ups)")
    ax.set_ylabel("Jump (0=no, 1=yes)")
    ax.set_title(f"Ground: Jump Action vs Speed  (jumps {analysis_data['ground_jump_pct']:.0f}% of the time)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_xlim(0, 800)

    # Panel 3: Strafe direction vs speed (air)
    ax = axes[2]
    right_labels = {0: "none", 1: "right", 2: "left"}
    ax.plot(speeds, air["right"], "m.-", markersize=4)
    ax.set_xlabel("Horizontal Speed (ups)")
    ax.set_ylabel("Right Action")
    ax.set_title("Air: Strafe Direction vs Speed")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["none", "right", "left"])
    ax.set_xlim(0, 800)

    fig.suptitle("Learned Policy Analysis", fontsize=13)
    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig(os.path.join(FIGURES_DIR, "policy_heatmap.png"), dpi=150)
    plt.close(fig)
    return fig
