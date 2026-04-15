# Issue Breakdown with Sub-Issues

Each top-level issue is a GitHub issue. Sub-issues are checkboxes within the issue or separate linked issues, depending on preference. Each sub-issue should be independently implementable and testable.

---

## Issue 1: Project Skeleton + Physics Engine

**Goal**: Create the project structure and implement the core Quake III movement physics.

### 1a: Project skeleton
- Create `pyproject.toml` with all dependencies
- Create `src/bhop/__init__.py` (empty for now)
- Create empty placeholder files for the module structure
- Add `runs/` and `models/` to `.gitignore`
- **Test**: `pip install -e .` succeeds

### 1b: Physics constants and state
- Create `src/bhop/physics.py` with `Q3Physics` class
- Define all constants as class attributes (see CLAUDE.md)
- Implement `__init__()` with state: velocity (float64 ndarray), position, on_ground, jump_held, yaw
- Implement `reset()` -- zero velocity, position at origin, on ground
- Implement `get_state()` and `horizontal_speed` property
- **Test**: Can instantiate Q3Physics, reset, read state

### 1c: Ground physics (friction + acceleration)
- Implement `_pm_friction()` -- ground friction matching Q3 source exactly
- Implement `_pm_accelerate()` -- THE core function with dot product projection
- Implement `_pm_cmd_scale()` -- diagonal normalization
- Implement `_pm_walk_move()` -- ground movement path (friction → accelerate)
- **Test**: Standing player accelerates to ~320 with forward held. Player decelerates with no input. Diagonal is not faster than straight.

### 1d: Air physics (jump + air acceleration + gravity)
- Implement `_pm_check_jump()` -- sets vel_z=270, clears ground, handles jump_held flag
- Implement `_apply_gravity()` -- half-step gravity application
- Implement `_pm_air_move()` -- air movement path (air_accelerate with accel=1.0)
- Implement `_pm_ground_trace()` -- flat plane ground detection (z <= 0)
- Implement `tick()` -- main dispatch: ground_trace → walk_move or air_move → update position → ground_trace
- **Test**: Jump launches player. Gravity brings player back. Air acceleration is weaker than ground. Air strafe + yaw rotation gains speed beyond 320.

---

## Issue 2: Physics Test Suite

**Goal**: Comprehensive tests validating physics correctness against known Q3 behavior. This is the most important issue -- if the physics are wrong, everything downstream is meaningless.

### 2a: Basic physics tests
- `test_standing_still_no_drift` -- no inputs → no movement
- `test_ground_acceleration_caps` -- forward held → converges to ~320
- `test_friction_stops_player` -- moving player with no input → decelerates to 0
- `test_cmdscale_diagonal` -- forward+right not faster than forward alone on ground
- `test_jump_velocity` -- jump sets vel_z ≈ 270
- `test_gravity_landing` -- player lands after jumping with no input
- `test_no_air_friction` -- airborne player doesn't lose horizontal speed
- `test_air_acceleration_weak` -- air accel much slower than ground accel
- **Test**: `pytest tests/test_physics.py` all pass

### 2b: PM_Accelerate exact value tests
- `test_pm_accelerate_exact_ground` -- known velocity + wishdir → exact expected result
- `test_pm_accelerate_addspeed_clamp` -- verify addspeed clamping when near maxspeed
- `test_pm_accelerate_no_accel_when_above` -- no acceleration when projection exceeds wishspeed
- `test_pm_accelerate_perpendicular` -- perpendicular wishdir gives maximum addspeed
- **Test**: All produce exact expected numerical results (within float tolerance)

### 2c: The bhop test (most important test in the project)
- `test_bhop_exceeds_maxspeed` -- scripted bhop inputs (jump, air strafe with yaw rotation, land, repeat) produce speed > 400 ups
- `test_bhop_speed_increases_over_jumps` -- speed after jump N+1 > speed after jump N
- `test_no_bhop_without_strafe` -- jumping without strafing does NOT exceed 320
- `test_no_bhop_without_jumping` -- strafing on ground does NOT exceed 320
- **Test**: Confirms the exploit exists in the physics and requires both jumping AND strafing

---

## Issue 3: Gymnasium Environment

**Goal**: Wrap Q3Physics into a standard Gymnasium env that PPO can train against.

### 3a: Core environment class
- Create `src/bhop/env.py` with `BhopEnv(gymnasium.Env)`
- Define observation_space (Box(5,)) and action_space (MultiDiscrete)
- Implement `reset()` -- reset physics, return initial observation
- Implement `step()` -- map action → physics inputs, tick, compute reward, return obs/reward/terminated/truncated/info
- Implement action mapping (MultiDiscrete indices → forward_move, right_move, jump, yaw_delta)
- Reward = horizontal_speed / 320.0
- **Test**: Can create env, reset, take a step

### 3b: Environment registration and compliance
- Register as `bhop/BhopFlat-v0` in `__init__.py`
- Create `tests/test_env.py`
- Test Gymnasium API compliance (spaces match, reset/step return correct types)
- Test with SB3's `check_env()`
- Test that random actions don't crash over a full episode
- **Test**: `gymnasium.make("bhop/BhopFlat-v0")` works; `check_env` passes

### 3c: Environment edge cases and info dict
- Handle zero-velocity yaw (default to 0 when speed < 1 if using velocity-aligned yaw)
- Add useful info to the info dict: `{"speed": float, "max_speed": float, "jumps": int}`
- Ensure observation values stay within declared bounds
- **Test**: Edge cases don't crash; info dict contains expected keys

---

## Issue 4: Training Script

**Goal**: End-to-end PPO training pipeline that produces a model and TensorBoard logs.

### 4a: Basic training script
- Create `scripts/train.py`
- Parse CLI args: `--timesteps`, `--n-envs`, `--seed`, `--save-path`
- Create vectorized environment with `SubprocVecEnv`
- Instantiate PPO with recommended hyperparameters
- Train and save model
- **Test**: `python scripts/train.py --timesteps 10000` runs without error

### 4b: Logging and callbacks
- Add TensorBoard logging (`tensorboard_log="runs/"`)
- Create custom SB3 callback that logs: mean episode speed, max speed reached, speed at episode end
- Add model checkpointing every N timesteps
- **Test**: TensorBoard shows custom metrics; checkpoints appear in `models/`

### 4c: Training reproducibility
- Set random seeds (env, numpy, torch)
- Log all hyperparameters to TensorBoard
- Save run config alongside model
- **Test**: Two runs with same seed produce similar results

---

## Issue 5: Evaluation + Visualization

**Goal**: Tools to understand what the trained agent learned.

### 5a: Evaluation script
- Create `scripts/evaluate.py`
- Load trained model, run N episodes deterministically
- Print: mean/max/min speed, speed at episode end, % time airborne, jump count
- **Test**: Runs against a saved model and prints meaningful stats

### 5b: Visualization module
- Create `src/bhop/viz.py`
- `plot_speed_over_time(episode_data)` -- speed vs tick for a single episode
- `plot_trajectory(episode_data)` -- top-down XY path showing strafing pattern
- `plot_action_distribution(episode_data)` -- histogram of chosen actions
- Save plots to `figures/` directory
- **Test**: Functions produce valid matplotlib figures

---

## Issue 6: Tuning + Bhop Verification

**Goal**: Get the agent to actually discover bhop. This may require hyperparameter tuning.

### 6a: Hyperparameter exploration
- Try different `ent_coef` values: 0.001, 0.01, 0.02, 0.05
- Try different network sizes: [32, 32], [64, 64], [128, 128]
- Try longer training: 5M, 10M timesteps
- Document results in a markdown file or notebook
- **Test**: At least one configuration produces an agent that exceeds 400 ups

### 6b: Policy analysis
- Extract learned policy: for a range of speeds and on_ground states, what actions does the agent choose?
- Compare strafe angle choices against known optimal angles
- Identify whether the agent learned: (1) always jump on ground, (2) strafe + yaw in air, (3) angle adjustment with speed
- **Test**: Policy analysis shows bhop-consistent behavior

---

## Issue 7: Analysis + Documentation

**Goal**: Polish and document the results.

### 7a: Analysis notebook
- Create `notebooks/analysis.ipynb`
- Training curves with annotations
- Side-by-side comparison: random agent vs trained agent
- Strafe angle vs speed plot overlaid with theoretical optimal
- **Test**: Notebook runs end-to-end

### 7b: README and documentation
- Project overview and motivation
- Results summary with key plots
- How to reproduce (install, train, evaluate)
- Physics explanation for non-technical readers
- **Test**: Someone can clone the repo and reproduce results following the README

---

## Future Issues (Stretch Goals)

### Issue 8: 3D environment with explicit yaw control
- If not already done: add yaw delta to action space
- Add pitch control for looking up/down
- Still flat plane but full 3D velocity tracking

### Issue 9: Map geometry
- Add simple obstacles (walls, ramps)
- BSP-like collision detection (or simplified box collision)
- Agent must learn to bhop while navigating

### Issue 10: CUDA-accelerated physics
- Rewrite physics simulation as CUDA kernels
- Run thousands of environments in parallel on GPU
- Similar to NVIDIA IsaacGym approach
- Dramatic training speedup

### Issue 11: Comparison with other RL algorithms
- SAC, A2C, DQN comparisons
- Which algorithms discover bhop fastest?
- Which produce the most optimal bhop technique?
