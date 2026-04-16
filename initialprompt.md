# Task: Q3 Demo Export and Pixel-Based Navigation

You are continuing work on an ongoing project. Read CLAUDE.md and all referenced documentation files before doing anything. Follow the conventions, structure, and patterns established in the existing codebase exactly. Do not refactor, rename, or reorganize existing code unless explicitly instructed to. Your sole task is described below.

---

## How to work through this prompt

**Do not start implementing everything at once.** This prompt describes the remaining scope, but you must work through it one piece at a time.

**Your first action: enter plan mode.** Before implementing anything, enter plan mode and break the "What you're implementing" section into a detailed, ordered plan. Each plan step should be a single function or logical unit. Present the plan to the user for approval before writing any code.

Each cycle, you implement **one function or one logical unit**. After finishing it, stop, report what you did, run it to confirm it works, and ask the user before moving on to the next one.

**Rules:**
- Enter plan mode first. Break the work into small steps. Get user approval on the plan.
- Implement only what the user approves. Do not implement the next piece without asking.
- Each cycle should be a single function or logical unit. Do not batch multiple pieces.
- After implementing each piece, run it to confirm it works.

---

## Context: What has already been done

### Issue 1: Project skeleton + physics engine -- COMPLETE
- `src/bhop/physics.py`: Q3Physics class with all 13 methods, fully verified
- `src/bhop/__init__.py`: Gymnasium env registration (bhop/BhopFlat-v0, bhop/BhopCorridor-v0)
- `pyproject.toml`: All dependencies configured, including `notebook` optional deps (jupyter, nbconvert, matplotlib)

### Issue 2: Physics test suite -- COMPLETE (16/16 tests passing)
- `tests/test_physics.py`: All 16 tests implemented and passing
  - TestBasicPhysics (8 tests): standing still, ground accel, friction, diagonal, jump, gravity, air friction, air accel
  - TestPMAccelerate (4 tests): exact ground, addspeed clamp, no accel above, perpendicular
  - TestBhop (4 tests): exceeds maxspeed, speed increases over jumps, no bhop without strafe, no bhop without jumping

### Issue 3: Gymnasium environment -- COMPLETE (9/9 tests passing, continuous action space)
- `src/bhop/env.py`: BhopEnv fully implemented with **continuous action space**
  - Observation: Box(5,) float32 -- [vel_x, vel_y, speed, vel_z, on_ground], clipped to [-2000, 2000]
  - Action: **Box(4,) float32** -- [forward, right, jump, yaw_delta]
    - forward/right: thresholded at ±0.33 → {-127, 0, +127}
    - jump: thresholded at 0.0
    - yaw_delta: continuous degrees in [-5.0, +5.0], converted to radians
  - Reward: horizontal_speed / 320.0
  - Info dict: {speed, max_speed, jumps}
  - Jump counting via ground-to-air transition detection
  - Obs clipping to declared bounds
  - SB3 check_env passes
  - Accepts optional `map_geometry: MapGeometry | None` parameter, passed to Q3Physics
- `tests/test_env.py`: All 9 tests passing with continuous actions

### Issue 4: Training script -- COMPLETE
- `scripts/train.py`: Fully implemented
  - CLI args: --env-id, --timesteps, --n-envs, --seed, --save-path, --ent-coef, --net-arch
  - `--env-id` defaults to "bhop/BhopFlat-v0", also works with "bhop/BhopCorridor-v0"
  - SubprocVecEnv, PPO with configurable hyperparams
  - SpeedLoggingCallback, CheckpointCallback every 50k steps
  - Verified: runs on CUDA with 8 envs, ~3300 fps (flat), ~3150 fps (corridor)

### Issue 5: Evaluation + Visualization -- COMPLETE
- `scripts/evaluate.py`: Fully implemented
  - `run_episode(model, env, deterministic=True)`: accepts `deterministic` param
  - `main()`: Loads model, runs N episodes, prints stats
- `src/bhop/viz.py`: All 5 visualization functions, updated for continuous actions
  - `plot_action_distribution()`: histograms for continuous yaw, thresholded bars for forward/right/jump
  - `analyze_policy()`: thresholds continuous actions to match env._map_action()

### Issue 6: Tuning + Bhop Verification -- COMPLETE
- `scripts/sweep.py`: 12-config grid sweep (ent_coef x net_arch)

### Continuous action space conversion -- COMPLETE
All 8 sub-tasks done:
1. env.py → Box(4,) action space with threshold-based _map_action()
2. test_env.py updated for continuous actions
3. viz.py updated (histograms, direct yaw degrees)
4. evaluate.py updated (deterministic param)
5. train.py + sweep.py verified (no changes needed, PPO auto-selects Gaussian policy)
6. 2M model trained: `models/bhop_2m_continuous`
7. 10M model trained: `models/bhop_10m_continuous`
8. Notebook updated to load continuous model, uses stochastic eval

### Issue 9: Map geometry + collision -- COMPLETE (23/23 geometry tests passing)

Added AABB collision detection to the physics engine. Geometry is dual-purpose: usable in Python training AND designed to map directly to Q3 brush primitives for `.map` export.

- `src/bhop/geometry.py` (new):
  - `Brush` dataclass: `mins`, `maxs` as float64 arrays (accepts tuples or arrays)
  - `TraceResult` dataclass: `fraction`, `normal`, `hit`
  - `MapGeometry` class: holds `list[Brush]`, `add_brush(mins, maxs)` method
  - `trace_ray(start, end, brush)`: slab method ray-AABB intersection, returns earliest entry fraction + outward-facing surface normal. Handles: parallel rays, start-inside-brush (allsolid, fraction=0), ray too short, zero-length rays
  - `trace(start, end, geometry)`: iterates all brushes, returns nearest hit
  - `_DIST_EPSILON = 0.03125` (1/32 unit, matches Q3): used in parallel-axis containment and allsolid checks to prevent false positives when player is on a brush surface
  - Map factories: `corridor_map()`, `turn_map()`, `platform_map()`

- `src/bhop/physics.py` (modified):
  - `Q3Physics.__init__()` accepts optional `geometry: MapGeometry | None = None`
  - `_pm_clip_velocity(velocity, normal, overbounce=1.001)`: mirrors Q3's PM_ClipVelocity
  - `_pm_slide_move()`: replaces raw `position += velocity * FRAMETIME`. Traces movement, clips velocity on hit, retries up to 4 bumps (MAX_CLIP_PLANES). Uses SURFACE_CLIP_EPSILON (1/32 unit) offset. When geometry=None, falls back to raw position update (identical to original)
  - `_pm_ground_trace()`: two paths -- flat-plane (geometry=None, unchanged) and geometry-based (traces downward 0.25 units, ground if normal_z > 0.7 = MIN_WALK_NORMAL). World floor at z=0 always present as fallback
  - `ground_normal` state variable tracks surface normal
  - Full backwards compatibility: `Q3Physics()` with no geometry behaves identically to before

- `src/bhop/env.py` (modified):
  - `BhopEnv.__init__()` accepts optional `map_geometry: MapGeometry | None`
  - Passes geometry through to `Q3Physics(geometry=map_geometry)`

- `src/bhop/__init__.py` (modified):
  - Registered `bhop/BhopCorridor-v0` with `corridor_map()` geometry

- `tests/test_geometry.py` (new, 23 tests):
  - TestTrace (9): empty space, face-on hit, reverse hit, parallel miss, inside brush, on-surface not allsolid, too short, multi-brush nearest, empty geometry
  - TestCollision (3): wall stop, wall slide, corridor containment
  - TestGroundTrace (5): floor brush, raised platform, midair, world floor fallback, wall face not ground
  - TestPhysicsWithGeometry (3): bhop in corridor, no-geometry backwards compat, jump and land
  - TestOriginalPhysicsRegression (3): standing still, ground acceleration, bhop exceeds maxspeed

- `scripts/train.py` (modified):
  - Added `--env-id` CLI arg (default: "bhop/BhopFlat-v0")
  - `make_env()` accepts `env_id` parameter

#### Key implementation notes for future phases:
- **Wall brushes must extend below floor** (e.g., z=-64 to z=128, not z=0 to z=128). Due to DIST_EPSILON in the slab method, a player at z=0 on a floor surface is considered "outside" a wall brush that starts exactly at z=0. This matches standard Q3 mapping practice where brushes overlap at seams.
- **Overbounce from slide-move can leave vel_z slightly positive.** The geometry ground trace does NOT gate on velocity direction (unlike the flat-plane path). This prevents the player from bouncing infinitely on floor surfaces.
- **Point traces, not bbox**: simplified from Q3's CM_BoxTrace. If demo playback shows drift from this simplification, upgrade to bbox traces later.

### Training results summary

| Model | Mean Speed | Max Speed | Airborne % | Notes |
|-------|-----------|-----------|------------|-------|
| Discrete 2M (`models/bhop_2m`) | 523.5 | 631.1 | 96% | Deterministic eval, 10 identical episodes |
| Continuous 2M (`models/bhop_2m_continuous`) | 616.6 | 857.6 | 96% | Stochastic eval (deterministic is poor due to threshold effects) |
| Continuous 10M (`models/bhop_10m_continuous`) | 656.5 | 925.9 | 96% | Best flat-plane model. Still accelerating at tick 1000 |
| Corridor 2M (`models/bhop_corridor_2m`) | 540 | 628 | -- | First corridor model. Bhop discovered with collision physics |

**Important note on continuous models**: The deterministic policy (mean of Gaussian) often lands near threshold boundaries (e.g., right action at -0.37 barely crossing the -0.33 threshold). Stochastic eval (`deterministic=False`) is much more representative of learned behavior. This is a known issue with thresholded continuous actions.

### Issue 7a: Analysis notebook -- COMPLETE
- `notebooks/analysis.ipynb`: All 4 sections implemented, loads `models/bhop_2m_continuous`
- Uses `deterministic=False` for trained agent evaluation
- Verified: `jupyter nbconvert --execute` runs end-to-end

### Issue 7b: README -- NOT STARTED
- Deferred until after demo work.

### Full test suite: 48 passed, 0 skipped
```
.venv/bin/pytest tests/ -v
```

---

## What you're implementing

The end goal is an RL agent that bhops through a Q3 map with obstacles, trained from pixel observations, with its runs playable as Q3 demo files in ioquake3/DeFRaG. Two major phases remain.

### Phase 2: Q3 demo export (Issue 10)

Record the agent's per-tick inputs and write them as a Q3 `.dm_68` demo file that plays back in ioquake3.

#### Sub-tasks:

**2a: Research and document the `.dm_68` format**
- The demo format stores: gamestate messages (server info, config strings, baselines) + snapshots (playerstate, entity deltas) + `usercmd_t` per frame
- Key struct: `usercmd_t { serverTime, angles[3], forwardmove, rightmove, upmove, buttons }`
- Research from ioquake3 source: `cl_main.c` (CL_WriteDemoMessage), `msg.c` (MSG_WriteBits)
- Document the minimum viable demo: single player, flat map, no entities

**2b: Implement demo writer**
- `src/bhop/demo.py`: class that accumulates per-tick `usercmd_t` and writes `.dm_68`
- Input: list of `(forward_move, right_move, jump, yaw, pitch)` per tick + map name
- Output: binary `.dm_68` file playable in ioquake3
- **Test first on flat plane**: record a known bhop sequence, play in ioquake3, verify it looks right

**2c: Q3 map file generation**
- `src/bhop/map_export.py`: convert Python AABB geometry → Q3 `.map` text format
- Q3 `.map` format is human-readable: brushes defined by plane equations
- AABB → 6 planes is straightforward
- Add spawn point entity, light, worldspawn
- The `.map` compiles to `.bsp` via `q3map2` (external tool, user runs separately)
- **Test**: generated `.map` compiles with q3map2; loads in ioquake3

**2d: End-to-end pipeline test**
- Train agent on a simple corridor map in Python
- Export its best run as a `.dm_68` demo
- Export the map as `.map`, compile to `.bsp`
- Play demo in ioquake3 on that map
- Verify the agent's trajectory matches (no major drift/clipping)

### Phase 3: Pixel-based observation (Issue 12)

Replace vector observations with rendered frames. See `docs/pixel_obs_plan.md` for the detailed plan already written.

#### Sub-tasks (summarized, full detail in pixel_obs_plan.md):

**3a: Top-down renderer** -- pygame, 84x84 RGB, shows player/map/velocity
**3b: Frame stacking wrapper** -- stack 4 frames for motion inference
**3c: CNN policy** -- SB3 CnnPolicy, may need custom feature extractor
**3d: Training + comparison** -- 10M+ steps, compare against vector-obs baseline
**3e: Analysis** -- CNN feature visualization, Grad-CAM, training curve comparison

---

## Key implementation notes

1. **The continuous action space is final.** Do not revert to discrete.
2. **Existing models** (`models/bhop_2m`, `models/bhop_2m_continuous`, `models/bhop_10m_continuous`, `models/bhop_corridor_2m`) must not be deleted. They are baselines.
3. **Stochastic eval**: continuous models should be evaluated with `deterministic=False` for representative results.
4. **The Q3 demo format is complex.** Phase 2a (research) should be thorough before writing code. The ioquake3 source is the authoritative reference.
5. **Map geometry is dual-purpose**: the `Brush` AABB data in `geometry.py` maps directly to Q3 brush primitives (6 axis-aligned planes). Phase 2c's map exporter converts these.
6. **Wall brushes must extend below floor** for proper collision (see Issue 9 notes above).
7. **Point traces may cause demo drift.** If Q3 playback desyncs from Python simulation, the likely cause is point-vs-bbox trace difference. Upgrade `trace_ray` to accept a player bbox if needed.

---

## Verification checklist

After each phase:

**Phase 2 (demo export):**
1. Demo file plays in ioquake3 without crashing
2. Flat-plane bhop demo visually matches Python simulation
3. Corridor map compiles to .bsp and loads in ioquake3
4. Agent's corridor run plays back correctly in Q3

**Phase 3 (pixel obs):**
1. Renderer produces correct 84x84 RGB frames
2. Frame-stacked env passes check_env
3. CNN policy trains and shows learning signal
4. Agent discovers bhop from pixels (speed > 400 mean)

---

## Updating this file (loop instructions)

When the user writes **"PREPARE NEXT SESSION"**, you must immediately update this file (`initialprompt.md`) to reflect the current state of the project. This is the handoff procedure — after this update the user will clear the context window and start a fresh session by reading this file.

**When triggered by "PREPARE NEXT SESSION", do the following:**
1. Move all completed items from "What you're implementing" to "What has already been done" with full implementation details (file paths, function names, what was verified).
2. If all current items are done, replace the task description with the next issue from `docs/issues.md`. If the core project is complete, list stretch goals.
3. Preserve the same format and level of detail as the existing "What has already been done" sections.
4. Keep the pacing instructions ("How to work through this prompt" section) and this "Updating this file" section intact — they must survive into the next session.
5. Update the "Full test suite" line if the test count changed.
6. Add any important notes about pending work, known issues, or decisions the next session needs to be aware of.
7. **Update docs/**: Ensure `docs/issues.md`, `docs/environment_design.md`, and any other docs reflect the current state. Mark completed issues, update action space or architecture descriptions if they changed, and add any new issues that were defined during the session.

After updating, confirm to the user that the file is ready and they can safely clear the context.

This file is the handoff document between conversation contexts. Treat it as the single source of truth for what has been done and what remains.
