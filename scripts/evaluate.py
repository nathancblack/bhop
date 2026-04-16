"""Evaluation script: load a trained model, run episodes, print stats.

Usage:
    python scripts/evaluate.py --model-path models/bhop_ppo --n-episodes 10
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import bhop  # noqa: F401 -- triggers env registration


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
    return parser.parse_args()


def run_episode(model: PPO, env: gym.Env, deterministic: bool = True) -> dict:
    """Run a single episode and collect per-tick data.

    Args:
        model: Trained SB3 model.
        env: Gymnasium environment.
        deterministic: Use deterministic (mean) policy if True, stochastic if False.

    Returns:
        Dict with keys: speeds, positions, actions, on_ground
    """
    obs, _ = env.reset()
    episode_data: dict = {
        "speeds": [],
        "positions": [],
        "actions": [],
        "on_ground": [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_data["speeds"].append(info["speed"])
        pos = env.unwrapped._physics.position[:2].copy()
        episode_data["positions"].append(pos.tolist())
        episode_data["actions"].append(action.copy())
        episode_data["on_ground"].append(bool(env.unwrapped._physics.on_ground))

    return episode_data


def main() -> None:
    """Load model, run evaluation episodes, and print statistics."""
    args = parse_args()

    env = gym.make("bhop/BhopFlat-v0")
    env.reset(seed=args.seed)
    model = PPO.load(args.model_path)

    all_episodes = []
    for i in range(args.n_episodes):
        episode_data = run_episode(model, env)
        all_episodes.append(episode_data)

    env.close()

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
