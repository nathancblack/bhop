# bhop — Reinforcement Learning Discovers Bunnyhopping

> A PPO agent, rewarded only for going fast, independently rediscovers
> **bunnyhopping** — the 1999 Quake III Arena movement exploit that speedrunners
> spent years perfecting by hand.

The agent is never told how to move. It is dropped into a faithful Python
reimplementation of Quake III's movement physics, given a reward of
*"horizontal speed ÷ 320"* every tick, and left to figure out the rest. What it
learns is the same trick the Quake community discovered: jump continuously and
steer in the air to bypass the engine's 320 units-per-second speed cap. The best
agent sustains **~650 ups and peaks near 970 ups — roughly 3× the cap.**

| | |
|---|---|
| **Domain** | Reinforcement learning · game physics · reverse engineering |
| **Stack** | Python · Gymnasium · Stable-Baselines3 (PPO) · NumPy · pytest |
| **Lines that matter** | A from-scratch port of Quake III's `bg_pmove.c`, plus a from-scratch writer for Quake III's binary demo format |
| **Result** | Agent exceeds the speed cap by ~3× and the learned strafe angle tracks the theoretical optimum |
| **Tests** | 91 passing (physics correctness, env API, demo codec round-trips) |

---

## Watch it (no install required)

The headline deliverable for a quick look is a set of **self-contained HTML
players** in [`site/`](site/). Each is a single file — open it in any browser or
embed it with `<iframe src="bhop_run.html">`. It animates the agent's run
top-down: the path is drawn out behind a moving marker, a live gauge shows speed
against the 320 cap (turning orange when the agent breaks it), and jumps are
counted as they happen.

| File | What it shows |
|---|---|
| [`site/bhop_run.html`](site/bhop_run.html) | The trained PPO agent bunnyhopping on a flat plane |
| [`site/bhop_scripted.html`](site/bhop_scripted.html) | The *same exploit driven by hand-written inputs* — proves the trick lives in the physics, not the network |
| [`site/bhop_corridor.html`](site/bhop_corridor.html) | The agent bhopping through a walled corridor with real collision |

Regenerate any of them (see [Visualization](#visualization)) from a model
checkpoint or from scripted inputs.

Static figures live in [`figures/`](figures/):
`speed_over_time.png`, `trajectory.png`, `policy_heatmap.png`,
`random_vs_trained.png`, `training_curves.png`.

---

## Why bunnyhopping works (the one trick to understand)

Quake III caps your ground speed at 320 ups. Bunnyhopping beats the cap because
of one detail in how the engine adds speed. Acceleration is computed against the
**projection** of your current velocity onto the direction you're asking to go —
not your actual speed:

```c
currentspeed = DotProduct(velocity, wishdir);   // projection, NOT magnitude
addspeed     = wishspeed - currentspeed;         // wishspeed is capped at 320
if (addspeed <= 0) return;                        // already fast in that direction
accelspeed   = min(accel * frametime * wishspeed, addspeed);
velocity    += accelspeed * wishdir;              // add speed along wishdir
```

When you strafe at an *angle* to where you're already moving, that projection
(`currentspeed`) is smaller than your real speed. So `addspeed` stays positive
and the engine keeps handing you speed — even past 320. Stack the right
conditions and the speed compounds:

1. **Air acceleration is 10× weaker but unbounded.** `pm_airaccelerate = 1.0`
   vs `pm_accelerate = 10.0`, but the projection rule applies in the air too.
2. **There is no air friction.** Speed gained in the air is kept.
3. **Jumping skips ground friction.** Land and immediately re-jump and the
   friction step never runs — so nothing ever bleeds the speed back down.

The optimal technique is to hold a strafe key and rotate your view at roughly
`arctan(320 / speed)` per tick, keeping `wishdir` just off your velocity so the
projection stays low. The agent discovers this on its own — see
`figures/policy_heatmap.png`, where the learned per-tick turn rate tracks that
theoretical curve.

This is faithfully reproduced in [`src/bhop/physics.py`](src/bhop/physics.py);
every method mirrors a function in Quake III's `bg_pmove.c`
(`_pm_accelerate`, `_pm_friction`, `_pm_air_move`, …) and is traceable to the
C source documented in [`docs/quake3_physics.md`](docs/quake3_physics.md).

---

## Results

Best model: `models/bhop_10m_continuous` (10M timesteps, continuous actions).

| Metric (flat plane, 1000-tick episodes) | Value |
|---|---|
| Mean speed (stochastic policy, 10 eps) | **~650 ups** (±87) |
| Peak speed | **~970 ups** (~3× the 320 cap) |
| Jumps per episode | ~12 |
| Speed cap (`sv_maxspeed`) | 320 ups |

The agent learned the full technique: **jump on landing, strafe in the air, and
adjust its turn rate with speed.** The walled-corridor agent
(`models/bhop_corridor_2m`) reaches ~520 ups while staying inside the geometry,
showing the behavior survives collision and wall-sliding.

> **An honest note on the policy.** Under the *deterministic* policy the agent
> often commits to a single jump and settles near the cap; it bunnyhops reliably
> under the *stochastic* policy, where sampling re-triggers the
> release-and-rejump timing each cycle. The numbers above and the players in
> `site/` use the stochastic policy. This is a genuine and interesting property
> of the learned solution, not a reporting artifact.

---

## Architecture

```
bhop/
├── src/bhop/
│   ├── physics.py      Q3Physics — faithful port of bg_pmove.c (accel, friction,
│   │                   air-strafe, gravity) + AABB collision and wall-sliding
│   ├── geometry.py     AABB brushes + ray/slab tracing; dual-purpose: drives
│   │                   in-sim collision AND exports to Quake .map format
│   ├── env.py          BhopEnv — Gymnasium env; continuous Box(4) action space,
│   │                   Box(5) observation, reward = horizontal_speed / 320
│   ├── demo.py         From-scratch Quake III .dm_68 demo writer (see below)
│   ├── map_export.py   MapGeometry → Quake .map text (compilable to .bsp)
│   └── viz.py          matplotlib figures: speed, trajectory, policy heatmaps
├── scripts/
│   ├── train.py        PPO training (SubprocVecEnv, TensorBoard, checkpoints)
│   ├── evaluate.py     Load a model, run episodes, print stats, optional demo export
│   ├── export_demo.py  Physics/agent run → playable Quake .dm_68 (+ optional .map)
│   └── export_run.py   Physics/agent run → self-contained HTML player  ← portfolio viz
├── tests/              91 tests: physics correctness, env API, demo codec round-trips
├── docs/               Physics reference, env design, .dm_68 format notes, issue log
├── figures/            Rendered analysis plots
└── site/               Generated HTML players (the shareable artifacts)
```

### Standout: a hand-rolled Quake III demo writer

[`src/bhop/demo.py`](src/bhop/demo.py) implements Quake III's binary demo format
(`.dm_68`) from scratch so an agent's run can be **replayed inside a real Quake
III / ioquake3 engine**, not just plotted. This required reverse-engineering and
reimplementing:

- the **adaptive Huffman codec** Quake uses for network messages (seeded from
  Quake's `msg_hData[256]` frequency table, with the key discovery that
  `MSG_WriteBits` uses a *static* tree — no per-symbol updates),
- **LSB-first bit-level I/O** matching `msg.c`,
- the **playerstate delta encoding** with Quake's exact 32-field netfield table
  and its 0 / 13-bit / 32-bit float scheme,
- and full **gamestate + snapshot message framing**.

It round-trips against its own reader in the test suite. Format notes are in
[`docs/dm68_format.md`](docs/dm68_format.md).

---

## Quickstart

```bash
# Install (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"

# Run the tests (the physics tests are the ones that matter)
pytest

# Train (best result was 10M timesteps; this is the default-ish call)
python scripts/train.py --timesteps 2000000 --n-envs 8 --seed 42
tensorboard --logdir runs/          # watch bhop/mean_speed climb past 320

# Evaluate a saved model
python scripts/evaluate.py --model-path models/bhop_10m_continuous --n-episodes 10
```

---

## Visualization

Generate a shareable HTML player from any model checkpoint or from scripted
inputs:

```bash
# Trained agent on the flat plane
python scripts/export_run.py --model-path models/bhop_10m_continuous \
    --output site/bhop_run.html

# The exploit with no learning involved — hand-written bhop inputs
python scripts/export_run.py --scripted --output site/bhop_scripted.html

# Trained agent in the collision corridor (walls are drawn in the player)
python scripts/export_run.py --model-path models/bhop_corridor_2m \
    --env-id bhop/BhopCorridor-v0 --output site/bhop_corridor.html
```

The output is one dependency-free `.html` file with the trajectory inlined as
JSON and a vanilla-JS canvas animation (play/pause, scrub, live speed gauge,
jump counter). Drop it straight into a personal site or embed it in an `<iframe>`.

Static analysis figures are produced by [`src/bhop/viz.py`](src/bhop/viz.py).

### Replay inside an actual Quake engine

For the full effect, export the run as a real Quake III demo and watch it in
ioquake3 / CPMA:

```bash
python scripts/export_demo.py --model-path models/bhop_10m_continuous \
    --output bhop_agent.dm_68
# then in ioquake3:  \demo bhop_agent
```

---

## How it works, end to end

1. **Physics** (`physics.py`) ticks at Quake's 125 fps (8 ms/tick), reproducing
   ground acceleration, friction, jumping, half-step gravity, air-strafe
   acceleration, and optional AABB collision with wall-sliding.
2. **Environment** (`env.py`) wraps the physics as a Gymnasium env. The agent
   sees `[vel_x, vel_y, speed, vel_z, on_ground]` and outputs a continuous
   `[forward, right, jump, yaw_delta]` — binary movement keys plus a continuous
   "mouse turn," mirroring a human's input fidelity. Reward is
   `horizontal_speed / 320` per tick, so faster is always better and beating the
   cap is the only way to score well.
3. **Training** (`train.py`) runs PPO across 8 parallel envs, logging mean/max
   speed to TensorBoard and checkpointing along the way.
4. **Analysis** (`evaluate.py`, `viz.py`) measures the learned policy and
   compares its strafe angles against the theoretical optimum.
5. **Export** (`export_run.py`, `export_demo.py`, `map_export.py`) turns a run
   into a web player, a native Quake demo, and/or a compilable Quake map.

## References

- Quake III Arena movement: `bg_pmove.c` (id Software, GPL) — see
  `docs/quake3_physics.md`.
- Demo/network format: ioquake3 `msg.c`, `cl_main.c`, `bg_public.h`; Python
  reference codec from `jfedor2/quake3-proxy-aimbot`.
