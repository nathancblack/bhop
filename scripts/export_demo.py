"""Standalone demo export pipeline.

Runs scripted bhop inputs (or a trained model) through Q3Physics,
then exports both a .dm_68 demo and optionally a .map file.

Usage:
    # Scripted bhop (no model needed):
    python scripts/export_demo.py --output demo.dm_68 --ticks 1000

    # With map geometry + export:
    python scripts/export_demo.py --output demo.dm_68 --env-id bhop/BhopCorridor-v0 \
        --export-map corridor.map

    # From trained model:
    python scripts/export_demo.py --output demo.dm_68 --model-path models/bhop_10m_continuous
"""

import argparse

import gymnasium as gym
import numpy as np

import bhop  # noqa: F401 -- triggers env registration
from bhop.demo import DemoWriter, TickRecord
from bhop.physics import Q3Physics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Q3 .dm_68 demo")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .dm_68 file path")
    parser.add_argument("--ticks", type=int, default=1000,
                        help="Number of ticks to simulate (scripted mode)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Trained model path (omit for scripted bhop)")
    parser.add_argument("--env-id", type=str, default="bhop/BhopFlat-v0",
                        help="Gymnasium environment ID")
    parser.add_argument("--export-map", type=str, default=None,
                        help="Also export .map file to this path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-name", type=str, default=None,
                        help="Override map name in demo (e.g. q3dm17)")
    return parser.parse_args()


def run_scripted_bhop(n_ticks: int, geometry=None) -> list[TickRecord]:
    """Run scripted bhop inputs through Q3Physics.

    Uses the same inputs as the bhop verification test:
    strafe right + jump + yaw rotation at 0.4 deg/tick.
    """
    phys = Q3Physics(geometry=geometry)
    ticks = []
    yaw_rate = np.radians(0.4)

    for i in range(n_ticks):
        phys.tick(forward_move=0, right_move=127, jump=True,
                  yaw_delta=yaw_rate)
        ticks.append(TickRecord(
            server_time=i * 8,
            origin_x=phys.position[0],
            origin_y=phys.position[1],
            origin_z=phys.position[2],
            velocity_x=phys.velocity[0],
            velocity_y=phys.velocity[1],
            velocity_z=phys.velocity[2],
            yaw=np.degrees(phys.yaw),
            on_ground=phys.on_ground,
        ))

    return ticks


def run_model(model_path: str, env_id: str, n_ticks: int,
              seed: int) -> list[TickRecord]:
    """Run a trained model and collect TickRecords."""
    from stable_baselines3 import PPO

    env = gym.make(env_id, max_episode_steps=n_ticks)
    env.reset(seed=seed)
    model = PPO.load(model_path)

    ticks = []
    obs, _ = env.reset()
    done = False
    tick = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        phys = env.unwrapped._physics
        ticks.append(TickRecord(
            server_time=tick * 8,
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

    env.close()
    return ticks


def main() -> None:
    args = parse_args()

    # Get geometry from env if needed
    geometry = None
    if args.env_id != "bhop/BhopFlat-v0":
        env = gym.make(args.env_id)
        geometry = getattr(env.unwrapped._physics, "geometry", None)
        env.close()

    # Generate ticks
    if args.model_path:
        print(f"Running model: {args.model_path}")
        ticks = run_model(args.model_path, args.env_id, args.ticks, args.seed)
    else:
        print(f"Running scripted bhop ({args.ticks} ticks)")
        ticks = run_scripted_bhop(args.ticks, geometry=geometry)

    # Print stats
    speeds = [np.sqrt(t.velocity_x**2 + t.velocity_y**2) for t in ticks]
    print(f"  Mean speed: {np.mean(speeds):.1f} ups")
    print(f"  Max speed:  {np.max(speeds):.1f} ups")
    print(f"  Final speed: {speeds[-1]:.1f} ups")

    # Export demo
    map_name = args.map_name or args.env_id.replace("bhop/", "").replace("-", "_").lower()
    writer = DemoWriter(map_name)
    writer.write(ticks, args.output)
    print(f"Exported demo: {args.output} ({len(ticks)} ticks)")

    # Export map if requested
    if args.export_map:
        if geometry is None:
            print("Warning: no geometry to export (flat plane has no brushes)")
        else:
            from bhop.map_export import export_map
            export_map(geometry, args.export_map, map_name=map_name)
            print(f"Exported map: {args.export_map} ({len(geometry.brushes)} brushes)")


if __name__ == "__main__":
    main()
