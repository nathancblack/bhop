"""PPO training entry point for bhop agent.

Usage:
    python scripts/train.py --timesteps 2000000 --n-envs 8 --seed 42
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import bhop  # noqa: F401 -- triggers env registration


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        --timesteps: Total training timesteps (default: 2_000_000)
        --n-envs: Number of parallel environments (default: 8)
        --seed: Random seed (default: 42)
        --save-path: Model save path (default: models/bhop_ppo)

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train PPO on bhop environments")
    parser.add_argument("--env-id", type=str, default="bhop/BhopFlat-v0")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/bhop_ppo")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    return parser.parse_args()


class SpeedLoggingCallback(BaseCallback):
    """Logs mean/max episode speed to TensorBoard from info dicts.

    Collects speed from each step's info dict and logs aggregates
    every `log_interval` steps.
    """

    def __init__(self, log_interval: int = 2048, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_interval = log_interval
        self._speeds: list[float] = []
        self._max_speeds: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._speeds.append(info.get("speed", 0.0))
            self._max_speeds.append(info.get("max_speed", 0.0))

        if len(self._speeds) >= self._log_interval:
            self.logger.record("bhop/mean_speed", np.mean(self._speeds))
            self.logger.record("bhop/max_speed", np.max(self._max_speeds))
            self._speeds.clear()
            self._max_speeds.clear()
        return True


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create a single bhop environment with the given seed."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def main() -> None:
    """Train a PPO agent on bhop/BhopFlat-v0.

    Steps:
        1. Parse CLI args
        2. Create vectorized environment with SubprocVecEnv
        3. Instantiate PPO with recommended hyperparameters
        4. Train and save model
    """
    args = parse_args()

    # Reproducibility: set numpy seed (torch/env seeds handled by SB3 seed param)
    np.random.seed(args.seed)

    # Save run config
    config = {
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": args.ent_coef,
        "net_arch": args.net_arch,
    }

    env = SubprocVecEnv(
        [lambda i=i: make_env(args.env_id, args.seed + i) for i in range(args.n_envs)]
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=args.net_arch),
        tensorboard_log="runs/",
        seed=args.seed,
        verbose=1,
    )

    save_dir = os.path.dirname(args.save_path) or "models"
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="bhop_ppo",
    )
    speed_cb = SpeedLoggingCallback()

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, speed_cb])
    model.save(args.save_path)

    config_path = args.save_path + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {args.save_path}")
    print(f"Config saved to {config_path}")
    env.close()


if __name__ == "__main__":
    main()
