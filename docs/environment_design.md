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

`Box(4,)` float32 -- continuous, matching human input fidelity:

| Dim | Range | Threshold | Maps to |
|-----|-------|-----------|---------|
| 0   | [-1, 1] | ±0.33 | forward: >0.33 → +127, <-0.33 → -127, else 0 |
| 1   | [-1, 1] | ±0.33 | right: >0.33 → +127, <-0.33 → -127, else 0 |
| 2   | [-1, 1] | 0.0 | jump: >0.0 → yes, else no |
| 3   | [-5, 5] | none | yaw delta: continuous degrees per tick, converted to radians |

**Why continuous with thresholds**: Matches human input fidelity exactly. A human Q3 player has binary WASD + spacebar (discrete movement keys) and continuous mouse yaw. The thresholding converts the Gaussian policy's continuous output into discrete key states, while yaw remains fully continuous. This was converted from an earlier `MultiDiscrete([3, 3, 2, 11])` space that bottlenecked yaw at 1° resolution.

**Why jump is not automatic**: The agent must discover that jumping immediately on every ground contact is part of bhop. Auto-jump would hand it the answer.

**Why yaw delta is included**: The agent controls how fast it rotates each tick. This is the mouse movement. Combined with strafe keys, it determines the wish direction relative to velocity. The agent must learn to coordinate yaw rotation with strafe direction -- this IS the core of bhop technique.

**Note on deterministic evaluation**: The Gaussian policy's mean can land near threshold boundaries (e.g., right action at -0.37, barely past the -0.33 threshold). Stochastic evaluation (`deterministic=False`) is more representative of learned behavior than deterministic.

### Yaw Implementation Detail

The environment maintains a `yaw` state variable (the agent's facing direction in radians).
Each tick, the yaw delta action is added to it. The wish direction is then computed from yaw + movement keys:

```python
forward_dir = [cos(yaw), sin(yaw), 0]
right_dir = [sin(yaw), -cos(yaw), 0]
wishvel = forward_dir * forward_move + right_dir * right_move
wishdir = normalize(wishvel)
wishspeed = |wishvel| * cmd_scale
```

**Edge case**: At zero velocity, yaw defaults to 0. The first forward input goes in +X direction.

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
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log="runs/",
    seed=42,
    verbose=1,
)

model.learn(total_timesteps=10_000_000)
model.save("models/bhop_10m_continuous")
```

### Key Hyperparameter Notes

- **`ent_coef=0.01`**: Most important deviation from defaults. Without entropy regularization, the policy collapses to walking before discovering bhop.
- **`n_envs=8`**: Physics are cheap (no rendering). More envs = more diverse experience per rollout. ~3300 fps on CPU.
- **`net_arch=[128, 128]`**: Best config from 12-config sweep. Larger than [64,64] default -- helps with continuous action space.
- **`gamma=0.99`**: Speed gains compound over time. Lower gamma would undervalue future speed.
- **`total_timesteps=10M`**: 2M is enough to discover bhop (~620 mean ups), 10M refines it (~660 mean, 926 max).

### Training Results (continuous action space)

| Model | Steps | Mean Speed | Max Speed | Notes |
|-------|-------|-----------|-----------|-------|
| Discrete baseline | 2M | 523.5 | 631.1 | 11-step yaw bottleneck |
| Continuous | 2M | 616.6 | 857.6 | +18% mean, +36% max |
| Continuous | 10M | 656.5 | 925.9 | Still accelerating at tick 1000 |

All continuous results use stochastic evaluation. Agent is 96% airborne with ~2 jumps/episode (1000 ticks).
