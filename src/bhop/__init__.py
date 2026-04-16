"""Bhop: RL discovery of bunnyhopping via Quake III Arena movement physics."""

from gymnasium.envs.registration import register

from bhop.geometry import corridor_map

register(
    id="bhop/BhopFlat-v0",
    entry_point="bhop.env:BhopEnv",
    max_episode_steps=1000,
)

register(
    id="bhop/BhopCorridor-v0",
    entry_point="bhop.env:BhopEnv",
    max_episode_steps=1000,
    kwargs={"map_geometry": corridor_map()},
)
