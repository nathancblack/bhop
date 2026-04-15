# Bhop: RL Discovery of Bunnyhopping

## Project Purpose

Train a reinforcement learning agent (PPO) against a faithful Python reimplementation of Quake III Arena's movement physics. The agent is rewarded for horizontal velocity and should independently discover bunnyhopping -- the exploit where jumping + air strafing bypasses the normal 320 ups speed cap.

## Architecture

```
bhop/
├── CLAUDE.md                 # This file -- project context and conventions
├── pyproject.toml            # Dependencies: gymnasium, stable-baselines3, numpy, tensorboard, pytest
├── src/bhop/
│   ├── __init__.py           # Gymnasium env registration (bhop/BhopFlat-v0)
│   ├── physics.py            # Q3Physics class -- faithful to bg_pmove.c
│   ├── env.py                # BhopEnv(gymnasium.Env) -- wraps physics + reward
│   └── viz.py                # Trajectory plots, policy analysis
├── scripts/
│   ├── train.py              # PPO training entry point
│   └── evaluate.py           # Load model, run episodes, print stats
├── tests/
│   ├── test_physics.py       # Physics correctness (most important tests)
│   └── test_env.py           # Gymnasium API compliance
└── docs/
    ├── quake3_physics.md     # Full C source reference from bg_pmove.c
    ├── environment_design.md # Gymnasium env design details
    └── issues.md             # All issues with sub-issues
```

## Conventions

- Physics engine method names mirror Q3 C function names (`_pm_accelerate`, `_pm_friction`, etc.)
- Physics uses float64 internally; Gymnasium observations are float32
- All tests: `pytest`
- Training logs: `runs/` (gitignored)
- Models: `models/` (gitignored)
- No YAML configs -- use constants and argparse to keep it simple
- Every implementation must be traceable back to the Q3 source in `docs/quake3_physics.md`

## Physics Reference (Quick)

See `docs/quake3_physics.md` for full C source. Here's the essential math:

### Constants (from bg_pmove.c)
```
pm_accelerate     = 10.0    # ground acceleration
pm_airaccelerate  = 1.0     # air acceleration (10x weaker!)
pm_friction       = 6.0     # ground friction
pm_stopspeed      = 100.0   # friction control threshold
sv_gravity        = 800.0   # gravity (units/sec^2)
JUMP_VELOCITY     = 270.0   # vertical velocity on jump
sv_maxspeed       = 320.0   # max wishspeed
frametime         = 0.008   # 125fps tick rate
```

### PM_Accelerate (THE bhop exploit)
```
currentspeed = dot(velocity, wishdir)     # PROJECTION, not magnitude!
addspeed = wishspeed - currentspeed
if addspeed <= 0: return
accelspeed = min(accel * frametime * wishspeed, addspeed)
velocity += accelspeed * wishdir
```

**Why bhop works**: `currentspeed` is the dot product of velocity onto wishdir, NOT the magnitude of velocity. When strafing at an angle to your velocity, this projection is LESS than your actual speed. So `addspeed` stays positive and you gain speed beyond the 320 cap. Combined with zero air friction and jumping immediately on landing (to skip ground friction), speed accumulates across jumps.

### Movement Dispatch
```
tick():
  ground_trace()          # check if on ground
  if on_ground:
    check_jump()          # may set vel_z=270, clear ground
    friction()            # only applies when walking
    accelerate(wishdir, wishspeed, pm_accelerate=10.0)
  else:
    apply_gravity()       # half-step: vel_z -= gravity * dt * 0.5
    accelerate(wishdir, wishspeed, pm_airaccelerate=1.0)
    apply_gravity()       # half-step
  update_position()
  ground_trace()          # re-check
```

## Environment Design (Quick)

See `docs/environment_design.md` for details.

- **Observation**: `Box(5,)` -- [vel_x, vel_y, speed, vel_z, on_ground]
- **Action**: `MultiDiscrete([3, 3, 2, N])` -- [forward, right, jump, yaw_delta]
- **Reward**: `horizontal_speed / 320.0` per tick
- **Episodes**: 1000 ticks (8s), start from zero velocity on ground

## Issue Breakdown

See `docs/issues.md` for the full breakdown with sub-issues. Summary:

1. Project skeleton + physics engine (4 sub-issues)
2. Physics test suite (3 sub-issues)
3. Gymnasium environment (3 sub-issues)
4. Training script (3 sub-issues)
5. Evaluation + visualization (2 sub-issues)
6. Tuning + bhop verification (2 sub-issues)
7. Analysis + documentation (2 sub-issues)

## Verification Checklist

1. `pytest tests/test_physics.py` -- especially bhop test (scripted inputs -> speed > 400)
2. SB3's `check_env(BhopEnv())` passes
3. `python scripts/train.py` runs and TensorBoard shows increasing velocity
4. Trained agent mean speed > 400 ups (bhop discovered)
5. Learned strafe angles approximate known optimal angles
