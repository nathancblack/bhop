"""Evaluation script: load a trained model, run episodes, print stats.

Usage:
    python scripts/evaluate.py --model-path models/bhop_ppo --n-episodes 10
    python scripts/evaluate.py --model-path models/bhop_ppo --export-demo out.dm_68
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import bhop  # noqa: F401 -- triggers env registration
from bhop.demo import DemoWriter, TickRecord


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        --model-path: Path to saved model (required)
        --n-episodes: Number of evaluation episodes (default: 10)
        --seed: Random seed (default: 42)

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained bhop agent")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-demo", type=str, default=None,
        help="Export first episode as .dm_68 demo file",
    )
    parser.add_argument(
        "--env-id", type=str, default="bhop/BhopFlat-v0",
        help="Gymnasium environment ID",
    )
    return parser.parse_args()


def run_episode(
    model: PPO,
    env: gym.Env,
    deterministic: bool = True,
    collect_ticks: bool = False,
) -> dict:
    """Run a single episode and collect per-tick data.

    Args:
        model: Trained SB3 model.
        env: Gymnasium environment.
        deterministic: Use deterministic (mean) policy if True, stochastic if False.
        collect_ticks: If True, also collect TickRecords for demo export.

    Returns:
        Dict with keys: speeds, positions, actions, on_ground, and
        optionally tick_records (list[TickRecord]).
    """
    obs, _ = env.reset()
    episode_data: dict = {
        "speeds": [],
        "positions": [],
        "actions": [],
        "on_ground": [],
    }
    if collect_ticks:
        episode_data["tick_records"] = []

    done = False
    tick = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        phys = env.unwrapped._physics
        episode_data["speeds"].append(info["speed"])
        pos = phys.position[:2].copy()
        episode_data["positions"].append(pos.tolist())
        episode_data["actions"].append(action.copy())
        episode_data["on_ground"].append(bool(phys.on_ground))

        if collect_ticks:
            episode_data["tick_records"].append(TickRecord(
                server_time=tick * 8,  # 125fps = 8ms per tick
                origin_x=phys.position[0],
                origin_y=phys.position[1],
                origin_z=phys.position[2],
                velocity_x=phys.velocity[0],
                velocity_y=phys.velocity[1],
                velocity_z=phys.velocity[2],
                yaw=np.degrees(phys.yaw),
                on_ground=phys.on_ground,
            ))
        tick += 1

    return episode_data


def main() -> None:
    """Load model, run evaluation episodes, and print statistics."""
    args = parse_args()

    env = gym.make(args.env_id)
    env.reset(seed=args.seed)
    model = PPO.load(args.model_path)

    all_episodes = []
    for i in range(args.n_episodes):
        collect = args.export_demo is not None and i == 0
        episode_data = run_episode(model, env, collect_ticks=collect)
        all_episodes.append(episode_data)

    env.close()

    # Export demo if requested
    if args.export_demo:
        ticks = all_episodes[0]["tick_records"]
        # Extract map name from env_id
        map_name = args.env_id.replace("bhop/", "").replace("-", "_").lower()
        writer = DemoWriter(map_name)
        writer.write(ticks, args.export_demo)
        print(f"Exported demo: {args.export_demo} ({len(ticks)} ticks)")

    # Compute stats
    final_speeds = [ep["speeds"][-1] for ep in all_episodes]
    max_speeds = [max(ep["speeds"]) for ep in all_episodes]
    mean_speeds = [np.mean(ep["speeds"]) for ep in all_episodes]
    airborne_pcts = [
        100.0 * (1.0 - np.mean(ep["on_ground"])) for ep in all_episodes
    ]
    jump_counts = [
        sum(
            1
            for a, b in zip(ep["on_ground"][:-1], ep["on_ground"][1:])
            if a and not b
        )
        for ep in all_episodes
    ]

    print(f"{'':=<50}")
    print(f"  Evaluation: {args.n_episodes} episodes")
    print(f"{'':=<50}")
    print(f"  Final speed  — mean: {np.mean(final_speeds):7.1f}  "
          f"max: {np.max(final_speeds):7.1f}  min: {np.min(final_speeds):7.1f}")
    print(f"  Mean speed   — mean: {np.mean(mean_speeds):7.1f}  "
          f"max: {np.max(mean_speeds):7.1f}  min: {np.min(mean_speeds):7.1f}")
    print(f"  Max speed    — mean: {np.mean(max_speeds):7.1f}  "
          f"max: {np.max(max_speeds):7.1f}  min: {np.min(max_speeds):7.1f}")
    print(f"  Airborne %   — mean: {np.mean(airborne_pcts):7.1f}%")
    print(f"  Jump count   — mean: {np.mean(jump_counts):7.1f}  "
          f"max: {np.max(jump_counts):7.0f}")
    print(f"{'':=<50}")


if __name__ == "__main__":
    main()
