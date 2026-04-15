# Gymnasium Environment Design

## Overview

The environment wraps Q3Physics into a standard Gymnasium interface. Flat infinite plane, no obstacles, no rendering. Pure velocity optimization.

## Observation Space

`Box(5,)` float32:

| Index | Field     | Range           | Notes |
|-------|-----------|-----------------|-------|
| 0     | vel_x     | [-2000, 2000]   | Horizontal velocity X |
| 1     | vel_y     | [-2000, 2000]   | Horizontal velocity Y |
| 2     | speed     | [0, 2000]       | Horizontal speed magnitude |
| 3     | vel_z     | [-2000, 2000]   | Vertical velocity |
| 4     | on_ground | {0.0, 1.0}      | Ground contact |

**Why speed is included**: It's redundant (sqrt(vel_x^2 + vel_y^2)) but saves the network from learning the Pythagorean theorem. Standard trick in physics-based RL.

**Why position is excluded**: No obstacles, so position is irrelevant. The agent only needs velocity state to make movement decisions.

**Bounds at 2000**: Skilled bhoppers reach 800-1200 ups. 2000 gives headroom.

## Action Space

`MultiDiscrete([3, 3, 2, N])`:

| Dim | Values | Meaning | Maps to |
|-----|--------|---------|---------|
| 0   | 0/1/2  | forward: none / forward / backward | 0, +127, -127 |
| 1   | 0/1/2  | right: none / right / left | 0, +127, -127 |
| 2   | 0/1    | jump: no / yes | False, True |
| 3   | 0..N   | yaw delta: discrete steps | e.g., N=11 → [-5°, -4°, ..., 0°, ..., +4°, +5°] per tick |

**Why MultiDiscrete**: In real Q3, players hold digital keys (WASD, space). Movement commands are -127, 0, or +127. A continuous Box action space would let the agent find fractional inputs that don't exist in the real game.

**Why jump is not automatic**: The agent must discover that jumping immediately on every ground contact is part of bhop. Auto-jump would hand it the answer.

**Why yaw delta is included**: The agent controls how fast it rotates each tick. This is the mouse movement. Combined with strafe keys, it determines the wish direction relative to velocity. The agent must learn to coordinate yaw rotation with strafe direction -- this IS the core of bhop technique.

### Yaw Implementation Detail

The environment maintains a `yaw` state variable (the agent's facing direction in radians).
Each tick, the yaw delta action is added to it. The wish direction is then computed from yaw + movement keys:

```python
forward_dir = [cos(yaw), sin(yaw), 0]
right_dir = [sin(yaw), -cos(yaw), 0]  # or however Q3 defines it
wishvel = forward_dir * forward_move + right_dir * right_move
wishdir = normalize(wishvel)
wishspeed = |wishvel| * cmd_scale
```

**Edge case**: At zero velocity, yaw defaults to 0. The first forward input goes in +X direction.

### Alternative: Simplified yaw (facing = velocity direction)

For faster initial development, yaw can be locked to the velocity direction:
- Forward pushes along velocity (projection = |v|, capped at 320)
- Right pushes perpendicular (projection = 0, always gains speed)
- Agent can still discover bhop but can't optimize angles

This could be a stepping stone: implement simplified first, add explicit yaw control after basic bhop is validated.

## Reward

```python
reward = horizontal_speed / 320.0  # normalize by sv_maxspeed
```

Simple, dense (every tick), and the normalizer keeps values reasonable.
Exceeding 1.0 means the agent broke the speed cap -- this IS bhop.

**No speed-delta bonus initially** -- keep it simple. If the agent converges to just walking at 320, add a bonus for exceeding 320.

## Episode Structure

- **Length**: 1000 ticks (8 seconds at 125fps)
- **Start**: Zero velocity, on ground, yaw = 0
- **Termination**: Never (no failure conditions). Truncation at 1000 steps.
- **Reset**: Returns to zero velocity on ground.

8 seconds is enough for a skilled bhopper to reach 800+ ups from standing. The agent experiences the full acceleration curve each episode.

## Registration

In `src/bhop/__init__.py`:
```python
from gymnasium.envs.registration import register

register(
    id="bhop/BhopFlat-v0",
    entry_point="bhop.env:BhopEnv",
    max_episode_steps=1000,
)
```

## Training Configuration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([lambda: gymnasium.make("bhop/BhopFlat-v0") for _ in range(8)])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,           # CRITICAL: keeps exploration alive for bhop discovery
    policy_kwargs=dict(
        net_arch=[64, 64],   # small network -- observation is only 5 dims
        # activation_fn defaults to Tanh, good for locomotion
    ),
    tensorboard_log="runs/",
    verbose=1,
)

model.learn(total_timesteps=2_000_000)
model.save("models/bhop_ppo")
```

### Key Hyperparameter Notes

- **`ent_coef=0.01`**: Most important deviation from defaults. Without entropy regularization, the policy collapses to walking before discovering bhop. May need 0.02-0.05 if agent doesn't explore enough.
- **`n_envs=8`**: Physics are cheap (no rendering). More envs = more diverse experience per rollout.
- **`net_arch=[64, 64]`**: Deliberately small. The optimal policy is not complex.
- **`gamma=0.99`**: Speed gains compound over time. Lower gamma would undervalue future speed.
- **`total_timesteps=2M`**: Starting point. May need 5-10M. Checkpoint frequently.
