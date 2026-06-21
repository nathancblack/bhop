# Task: Debug Q3 Demo Playback in quake3e

You are continuing work on an ongoing project. Read CLAUDE.md and all referenced documentation files before doing anything. Follow the conventions, structure, and patterns established in the existing codebase exactly. Do not refactor, rename, or reorganize existing code unless explicitly instructed to. Your sole task is described below.

---

## How to work through this prompt

You are **debugging Q3 demo playback**. The demo writer is fully implemented (Steps 2-14 from the original plan are DONE), but the exported `.dm_68` files don't play correctly in quake3e yet.

**Your approach:**
1. Read `src/bhop/demo.py`, `docs/dm68_format.md`, and the bugs-fixed section below
2. Understand what's been fixed already so you don't repeat work
3. Continue debugging from the current state (described below)
4. Each fix: edit code, run tests, regenerate demo, ask user to test in quake3e
5. Reference ioquake3 source (github.com/ioquake/ioq3) as the authoritative spec

**Rules:**
- Run `pytest tests/test_demo.py` after every change
- Regenerate demos with: `.venv/bin/python scripts/export_demo.py --output <path> --ticks 500 --map-name bhop_flat`
- The quake3e binary is at: `/home/nate/code/bhop/CPMA-153-full-pack-v1/Q3/quake3e.x64`
- Launch command: `cd /home/nate/code/bhop/CPMA-153-full-pack-v1/Q3 && ./quake3e.x64 +set fs_game "" +demo bhop_test`
- Demo goes to: `/home/nate/code/bhop/CPMA-153-full-pack-v1/Q3/baseq3/demos/bhop_test.dm_68`
- Custom BSP map at: `/home/nate/code/bhop/CPMA-153-full-pack-v1/Q3/baseq3/maps/bhop_flat.bsp`

---

## Current debugging state

### What's happening now

The demo **loads and plays** in quake3e — the engine parses the gamestate, loads the map, and starts playing snapshots. However:

- **The player falls through the floor and the camera goes into the ground, rotating around on the floor surface**
- The HUD shows "out of ammo" (expected since we don't set weapon stats)
- The map `bhop_flat` loads (custom BSP compiled from `maps/bhop_flat.map`)

### Likely causes (investigate these)

1. **Player origin at (0,0,0) may be below or at the exact floor surface**. The bhop_flat map has floor at z=0. Our physics starts the player at origin (0,0,0). In Q3, the player model has a bbox extending below the origin point (player origin is at eye height, feet are ~24 units below). So origin z=0 means feet at z=-24, which is inside/below the floor.

2. **The map may need rebuilding**. The current `bhop_flat.map` uses `common/caulk` for walls/ceiling (invisible) and `base_floor/techfloor2` for the floor. It was compiled with q3map2 but had a "leaked" warning because the original version wasn't a sealed box. A new version with walls was written to `maps/bhop_flat.map` but **was NOT recompiled** — the BSP in `baseq3/maps/` is still from the old open caulk-only map.
   - q3map2 binary: `/tmp/nrc/install/q3map2` (built from netradiant-custom source at `/tmp/nrc`)
   - Compile command: `export LD_LIBRARY_PATH=/tmp/nrc/install:$LD_LIBRARY_PATH && /tmp/nrc/install/q3map2 -game quake3 -fs_basepath /home/nate/code/bhop/maps /home/nate/code/bhop/maps/bhop_flat.map`
   - Then light: `/tmp/nrc/install/q3map2 -game quake3 -fs_basepath /home/nate/code/bhop/maps -light -fast /home/nate/code/bhop/maps/bhop_flat.bsp`
   - Copy: `cp /home/nate/code/bhop/maps/bhop_flat.bsp /home/nate/code/bhop/CPMA-153-full-pack-v1/Q3/baseq3/maps/bhop_flat.bsp`
   - NOTE: There's a symlink `maps/baseq3/pak0.pk3 -> CPMA pack's pak0.pk3` for texture resolution during BSP compile
   - NOTE: q3map2 needs `LD_LIBRARY_PATH=/tmp/nrc/install` for libassimp_.so. If `/tmp/nrc` doesn't exist, clone and rebuild: `cd /tmp && git clone --depth 1 https://github.com/Garux/netradiant-custom.git nrc && cd nrc && DEPENDENCIES_CHECK=off ASSIMP_INTERNAL=yes make binaries-q3map2 -j$(nproc)`

3. **Player z-origin needs to be raised**. Our scripted bhop starts at the physics origin (0,0,0). For Q3 playback, the player origin should be at standing eye height above the floor. Try offsetting origin_z by +40 or +56 in `_tick_to_playerstate()` or in the export script.

4. **`pm_type` field may need to be non-zero** to prevent the cgame from treating the player as spawning/dying. `pm_type=0` is PM_NORMAL which should be fine, but check if the cgame does anything special when the player hasn't fully "spawned" (no weapon, no armor, etc.).

5. **Stats may need more fields**. Currently we set `stats[STAT_HEALTH]=100`. The cgame may also check `stats[STAT_ARMOR]`, `stats[STAT_WEAPONS]` (bitmask of held weapons), etc. A player with no weapons may trigger special "spectator" or "dead" rendering.

### What to try first

1. **Recompile the sealed map** (the .map file was updated with walls/ceiling/light but BSP wasn't regenerated)
2. **Offset player z-origin** by +40 units so the player is above the floor surface  
3. **Add weapon stats** if the player still appears dead/spectating

---

## Bugs already fixed (DO NOT re-introduce these)

These bugs were found and fixed during this debugging session. Each was verified by re-running tests and regenerating demos.

### Bug 1: Missing CS_GAME_VERSION configstring
- **Symptom**: "CLIENT/SERVER GAME MISMATCH: BASEQ3-1/"
- **Cause**: Q3's cgame checks configstring index 20 (`CS_GAME_VERSION`) against compiled-in `GAME_VERSION = "baseq3-1"`. We weren't writing this configstring.
- **Fix**: Added `buf.write_byte(SVC_CONFIGSTRING); buf.write_short(20); buf.write_string("baseq3-1")` to `_write_gamestate()`
- **Location**: `src/bhop/demo.py`, `_write_gamestate()`, around line 690

### Bug 2: Missing SVC_EOF at end of messages
- **Symptom**: "CL_PARSESERVERMESSAGE: ILLEGIBLE SERVER MESSAGE"  
- **Cause**: `CL_ParseServerMessage` loops reading commands until `SVC_EOF`. Both gamestate and snapshot message payloads need `SVC_EOF` at the end. Gamestate needs TWO — one to end the configstring loop (inner), one to end the server message (outer).
- **Fix**: Added `buf.write_byte(SVC_EOF)` at end of both `_write_gamestate()` and `_write_snapshot()`
- **Location**: `src/bhop/demo.py`, end of `_write_gamestate()` and `_write_snapshot()`

### Bug 3: Wrong stat array encoding
- **Symptom**: "CL_PARSEPACKETENTITIES: END OF MESSAGE"
- **Cause**: We wrote four 16-bit zero bitmasks (`write_short(0)` x4 = 64 bits) for stat arrays. Q3's actual format starts with a single bit: 0 = no arrays changed (return immediately), 1 = then per-array changed bits + bitmasks. With no changes, it's just 1 bit, not 64.
- **Fix**: Changed `_write_stat_arrays()` to write `buf.write_bits(0, 1)` when no stats changed, or a proper hierarchical encoding when stats are set (currently writes health=100).
- **Location**: `src/bhop/demo.py`, `_write_stat_arrays()`

### Bug 4: Wrong float field encoding (3-way vs 2-way)
- **Symptom**: "CL_PARSEPACKETENTITIES: END OF MESSAGE" (bit misalignment in playerstate)
- **Cause**: Our `_write_delta_field` had a 3-way encoding for floats: (1) zero → single 0 bit, (2) small integer → `1,0` + 13 bits, (3) full float → `1,1` + 32 bits. But Q3's actual encoding has only 2 paths: (1) small integer (including zero) → `0` + 13 bits, (2) full float → `1` + 32 bits. Zero is encoded as 13-bit value 4096 (0 + FLOAT_INT_BIAS). Our phantom "zero = 1 bit" path saved 13 bits per zero field, causing every subsequent field to be misaligned.
- **Fix**: Removed the zero special case. All integers that fit in 13-bit biased range (including zero) use path 1: `write_bits(0, 1)` + `write_bits(val + FLOAT_INT_BIAS, 13)`.
- **Location**: `src/bhop/demo.py`, `_write_delta_field()`, around line 562

### Bug 5: get_data() buffer size formula
- **Symptom**: "CL_PARSEPACKETENTITIES: END OF MESSAGE" (intermittent, depends on bit alignment)
- **Cause**: Our `get_data()` used `(self._bit + 7) >> 3` (ceiling division) for buffer size. Q3 uses `(msg->bit >> 3) + 1`. These differ when `_bit` is a multiple of 8: e.g., 64 bits → we return 8 bytes, Q3 expects 9. The reader sets `readcount = (bit>>3)+1` after reading, and checks `readcount > cursize` for overflow. With cursize=8 instead of 9, the check false-positives.
- **Fix**: Changed to `(self._bit >> 3) + 1` with padding, matching Q3's formula exactly.
- **Location**: `src/bhop/demo.py`, `MsgBuffer.get_data()`, around line 300

### Bug 6: Non-delta snapshots
- **Symptom**: Various parse errors on later snapshots
- **Cause**: Delta snapshots reference previous snapshot by sequence number. Q3 only caches ~32 snapshots. Our `delta_num=seq-1` would reference snapshots outside the cache window after snapshot 32. Also, `delta_num` is a byte, so it wraps at 256.
- **Fix**: Changed all snapshots to non-delta (`delta_num=0, from_ps=None`). Larger files but guaranteed to parse correctly.
- **Location**: `src/bhop/demo.py`, `DemoWriter.write()`, around line 804

### Other fixes applied:
- **CS_SYSTEMINFO**: Changed from `\sv_serverid\0` to `\sv_serverid\1234\sv_pure\0\sv_maxclients\8`
- **CS_SERVERINFO**: Added `\version\baseq3-1` (redundant with CS_GAME_VERSION fix but harmless)
- **export_demo.py**: Added `--map-name` CLI flag to override map name in demo
- **Health stats**: `_write_stat_arrays()` now sets `stats[STAT_HEALTH]=100` to prevent player appearing dead

---

## Context: What has already been done

### Issue 10: Q3 demo export -- IMPLEMENTED, DEBUGGING PLAYBACK

All 13 implementation steps (2-14) are complete. The code is fully implemented and passes 91 tests.

**10a: Research .dm_68 format -- COMPLETE**
- `docs/dm68_format.md`: Full format documentation

**10b: Demo writer -- COMPLETE (with bugs fixed above)**
- `src/bhop/demo.py` (~810 lines): Full implementation:
  - `HuffmanCodec`: static Huffman tree from pre-computed structure (Q3's msg_hData)
  - `MsgBuffer` / `MsgReader`: MSG_WriteBits/ReadBits with Huffman encoding
  - `write_delta_playerstate()`: 32-field netfield delta with float 2-way encoding
  - `_write_gamestate()`: configstrings (CS_SERVERINFO, CS_SYSTEMINFO, CS_GAME_VERSION, CS_PLAYERS)
  - `_write_snapshot()`: playerstate delta + entity terminator + SVC_EOF
  - `_write_stat_arrays()`: hierarchical stat encoding with STAT_HEALTH=100
  - `DemoWriter(map_name).write(ticks, path)`: complete .dm_68 with non-delta snapshots + EOF
  - `TickRecord` dataclass: per-tick physics state for demo export
- `tests/test_demo.py`: 29 tests (Huffman round-trip, MsgBuffer round-trip, DemoWriter structure + physics integration)

**10c: Q3 map file generation -- COMPLETE**
- `src/bhop/map_export.py`: `_brush_to_planes()`, `export_map()`
- `tests/test_map_export.py`: 14 tests
- `maps/bhop_flat.map`: Simple sealed flat arena (4096x4096, floor at z=0, walls, ceiling, light entity)
- BSP compiled via q3map2 (from netradiant-custom built at `/tmp/nrc/install/q3map2`)

**10d: End-to-end pipeline**
- `scripts/export_demo.py`: standalone pipeline (scripted bhop or trained model → .dm_68)
  - `--map-name` flag overrides map name in demo (use `bhop_flat` for custom map)
- `scripts/evaluate.py`: added `--export-demo` and `--env-id` flags

### Full test suite: 91 passed, 0 skipped
```
.venv/bin/pytest tests/ -q
```

### Previous issues (1-9) -- ALL COMPLETE
See the sections below (unchanged from previous session).

### Issue 1: Project skeleton + physics engine -- COMPLETE
- `src/bhop/physics.py`: Q3Physics class with all 13 methods, fully verified
- `src/bhop/__init__.py`: Gymnasium env registration (bhop/BhopFlat-v0, bhop/BhopCorridor-v0)
- `pyproject.toml`: All dependencies configured

### Issue 2: Physics test suite -- COMPLETE (16/16 tests passing)
- `tests/test_physics.py`: All 16 tests

### Issue 3: Gymnasium environment -- COMPLETE (9/9 tests passing, continuous action space)
- `src/bhop/env.py`: BhopEnv with continuous Box(4,) action space

### Issue 4: Training script -- COMPLETE
- `scripts/train.py`: PPO training with SubprocVecEnv

### Issue 5: Evaluation + Visualization -- COMPLETE
- `scripts/evaluate.py`, `src/bhop/viz.py`

### Issue 6: Tuning + Bhop Verification -- COMPLETE
- `scripts/sweep.py`

### Issue 8: Continuous action space -- COMPLETE
### Issue 9: Map geometry + collision -- COMPLETE (23/23 tests)

### Training results summary

| Model | Mean Speed | Max Speed | Notes |
|-------|-----------|-----------|-------|
| Continuous 10M (`models/bhop_10m_continuous`) | 656.5 | 925.9 | Best flat-plane model |
| Corridor 2M (`models/bhop_corridor_2m`) | 540 | 628 | First corridor model |

---

## Key implementation notes

1. **The Huffman tree is STATIC** after initialization. Q3's `MSG_WriteBits` does NOT call `Huff_addRef`. Our implementation uses a pre-computed tree structure.
2. **Float encoding is 2-way, not 3-way.** Zero is encoded as 13-bit biased integer (value 4096), not as a special 1-bit case. See Bug 4 above.
3. **Stat arrays use hierarchical encoding.** Single bit for "any changed?", then per-array bits. See Bug 3 above.
4. **Message data size must use Q3's formula: `(bit >> 3) + 1`**, not ceiling division. See Bug 5 above.
5. **Every message payload must end with SVC_EOF.** Gamestate needs two (inner loop + outer). See Bug 2 above.
6. **All snapshots are non-delta** (delta_num=0). Larger but reliable. Can optimize later once playback works.
7. **CS_GAME_VERSION (configstring 20)** must be `"baseq3-1"` to match cgame's compiled GAME_VERSION.
8. **q3map2** is built at `/tmp/nrc/install/q3map2` (needs `LD_LIBRARY_PATH=/tmp/nrc/install`). If `/tmp` was cleaned, rebuild from netradiant-custom source.

---

## Verification checklist

**Phase 2 (demo export) -- IN PROGRESS:**
1. [x] Demo file loads in quake3e without parse errors
2. [ ] Player is visible and moves correctly (CURRENT BLOCKER)
3. [ ] Flat-plane bhop demo visually matches Python simulation
4. [ ] Corridor map compiles to .bsp and loads  
5. [ ] Agent's corridor run plays back correctly

---

## Updating this file (loop instructions)

When the user writes **"PREPARE NEXT SESSION"**, you must immediately update this file (`initialprompt.md`) to reflect the current state of the project.

**When triggered by "PREPARE NEXT SESSION", do the following:**
1. Move all completed items from "What you're implementing" to "What has already been done" with full implementation details.
2. If all current items are done, replace the task description with the next issue from `docs/issues.md`.
3. Preserve the same format and level of detail.
4. Keep the pacing instructions and this section intact.
5. Update the "Full test suite" line if the test count changed.
6. Add any important notes about pending work, known issues, or decisions the next session needs.
7. **Update docs/**: Ensure `docs/issues.md` and other docs reflect current state.

After updating, confirm to the user that the file is ready and they can safely clear the context.
