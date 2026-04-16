"""Hyperparameter sweep: train multiple PPO configs and compare results.

Usage:
    python scripts/sweep.py --timesteps 2000000 --n-envs 8 --seed 42
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import bhop  # noqa: F401 -- triggers env registration

# Sweep grid
ENT_COEFS = [0.001, 0.01, 0.02, 0.05]
NET_ARCHS = [[32, 32], [64, 64], [128, 128]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for bhop PPO")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SpeedLoggingCallback(BaseCallback):
    """Logs mean/max episode speed to TensorBoard."""

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


def make_env(seed: int) -> gym.Env:
    env = gym.make("bhop/BhopFlat-v0")
    env.reset(seed=seed)
    return env


def run_episode(model: PPO, env: gym.Env) -> list[float]:
    """Run one episode, return list of per-tick speeds."""
    obs, _ = env.reset()
    speeds = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        speeds.append(info["speed"])
    return speeds


def evaluate_model(model: PPO, n_episodes: int = 10, seed: int = 42) -> dict:
    """Evaluate a model over N episodes, return stats."""
    env = gym.make("bhop/BhopFlat-v0")
    env.reset(seed=seed)

    mean_speeds = []
    max_speeds = []
    for _ in range(n_episodes):
        speeds = run_episode(model, env)
        mean_speeds.append(np.mean(speeds))
        max_speeds.append(max(speeds))

    env.close()
    return {
        "mean_speed": float(np.mean(mean_speeds)),
        "max_speed": float(np.max(max_speeds)),
    }


def run_config(
    ent_coef: float,
    net_arch: list[int],
    timesteps: int,
    n_envs: int,
    seed: int,
) -> dict:
    """Train one config and evaluate it. Returns results dict."""
    run_name = f"ent{ent_coef}_net{'x'.join(str(n) for n in net_arch)}"
    save_dir = os.path.join("models", run_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  ent_coef={ent_coef}, net_arch={net_arch}, timesteps={timesteps}")
    print(f"{'='*60}")

    np.random.seed(seed)

    env = SubprocVecEnv(
        [lambda i=i: make_env(seed + i) for i in range(n_envs)]
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log="runs/",
        seed=seed,
        verbose=0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="bhop_ppo",
    )
    speed_cb = SpeedLoggingCallback()

    start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, speed_cb],
        tb_log_name=run_name,
    )
    elapsed = time.time() - start

    model_path = os.path.join(save_dir, "final_model")
    model.save(model_path)

    config = {
        "ent_coef": ent_coef,
        "net_arch": net_arch,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    env.close()

    # Evaluate
    print(f"  Evaluating {run_name}...")
    eval_results = evaluate_model(model, n_episodes=10, seed=seed)
    print(f"  Mean speed: {eval_results['mean_speed']:.1f}  "
          f"Max speed: {eval_results['max_speed']:.1f}  "
          f"({elapsed:.0f}s)")

    return {
        "run_name": run_name,
        "ent_coef": ent_coef,
        "net_arch": net_arch,
        "mean_speed": eval_results["mean_speed"],
        "max_speed": eval_results["max_speed"],
        "elapsed_s": elapsed,
        "model_path": model_path,
    }


def main() -> None:
    args = parse_args()

    total = len(ENT_COEFS) * len(NET_ARCHS)
    print(f"Sweep: {total} configurations, {args.timesteps} timesteps each")

    results = []
    for i, ent_coef in enumerate(ENT_COEFS):
        for j, net_arch in enumerate(NET_ARCHS):
            idx = i * len(NET_ARCHS) + j + 1
            print(f"\n[{idx}/{total}]", end="")
            result = run_config(
                ent_coef=ent_coef,
                net_arch=net_arch,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
                seed=args.seed,
            )
            results.append(result)

    # Summary table sorted by mean speed descending
    results.sort(key=lambda r: r["mean_speed"], reverse=True)

    print(f"\n\n{'='*70}")
    print("  SWEEP RESULTS (sorted by mean speed)")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'Mean Spd':>10} {'Max Spd':>10} {'Time':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['run_name']:<25} {r['mean_speed']:>10.1f} {r['max_speed']:>10.1f} {r['elapsed_s']:>7.0f}s")
    print(f"{'='*70}")

    best = results[0]
    print(f"\n  Best: {best['run_name']} — mean {best['mean_speed']:.1f} ups, max {best['max_speed']:.1f} ups")
    print(f"  Model: {best['model_path']}")

    if best["mean_speed"] > 400:
        print("  SUCCESS: Agent exceeds 400 ups mean speed!")
    else:
        print("  Agent did not reach 400 ups. Consider longer training (--timesteps 5000000+).")


if __name__ == "__main__":
    main()
