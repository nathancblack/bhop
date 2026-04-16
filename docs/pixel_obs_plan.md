# Issue 12: Pixel-Based Observation (Visual Bhop Discovery)

**Goal**: Replace the 5-element vector observation with rendered image frames so the agent must discover bunnyhopping from raw pixels. The vector-obs agent serves as the performance baseline.

---

## Motivation

The current agent receives `[vel_x, vel_y, speed, vel_z, on_ground]` directly -- it knows its exact speed and ground state. A pixel-based agent must infer this from visual cues alone, making the bhop discovery significantly harder and more impressive. This also bridges the project into computer vision territory: the policy network becomes a CNN (or vision transformer) that must learn a useful visual representation as a byproduct of maximizing speed.

---

## Architecture Overview

```
Observation pipeline:
  Q3Physics state → Renderer (pygame/OpenGL) → RGB frame(s) → CNN feature extractor → MLP policy head

Two rendering options:
  A) Top-down (2D): bird's-eye XY view showing player position, velocity vector, ground indicator
  B) First-person (3D): simplified FPS view with ground plane, horizon line, motion cues

Option A is much simpler and recommended as the starting point.
```

---

## Sub-tasks

### 12a: Top-down renderer

Add a render method to `BhopEnv` that produces an RGB image of the current state.

- **Canvas**: 84x84 or 128x128 RGB uint8 (standard RL image sizes)
- **Elements to render**:
  - Ground plane (solid color)
  - Player position (dot/marker at center or tracking)
  - Velocity vector (arrow showing direction and magnitude)
  - Ground contact indicator (color change or border when on_ground)
  - Optional: trail of recent positions (last ~50 ticks) to show trajectory
- **Implementation**: Use `pygame.Surface` for CPU rendering (fast enough for RL training, no GPU needed for rendering)
- **Interface**: `render_mode="rgb_array"` returns `np.ndarray` of shape `(H, W, 3)` dtype `uint8`
- **Test**: Render 100 frames of a random episode, verify shapes and no NaN/Inf

Design decisions:
- Camera should track the player (player stays centered, world scrolls)
- Scale the view so that ~2 seconds of trajectory is visible at typical bhop speeds
- Velocity arrow length should be proportional to speed, capped so it doesn't leave the frame
- Use distinct colors: green=on_ground, red=airborne, white=velocity arrow, gray=trail

### 12b: Frame stacking wrapper

A single frame is ambiguous (no velocity/acceleration info). Stack consecutive frames so the agent can infer motion.

- Use `gymnasium.wrappers.FrameStack(env, num_stack=4)` (standard approach from Atari RL)
- Observation becomes `(4, H, W, 3)` or `(4, 3, H, W)` depending on channel convention
- Also apply standard preprocessing:
  - `gymnasium.wrappers.ResizeObservation(env, (84, 84))` if renderer outputs larger frames
  - `gymnasium.wrappers.GrayscaleObservation(env)` -- optional, reduces input size 3x
- **Test**: Wrapped env passes `check_env()`, observation shapes are correct

### 12c: CNN policy configuration

Configure SB3's PPO to use a CNN feature extractor instead of MLP.

- Use `CnnPolicy` instead of `MlpPolicy` in PPO constructor
- SB3's default `NatureCNN` (from the Atari DQN paper) should work as a starting point:
  - Conv(32, 8x8, stride 4) → Conv(64, 4x4, stride 2) → Conv(64, 3x3, stride 1) → Flatten → Linear(512)
- If needed, define a custom feature extractor via `policy_kwargs=dict(features_extractor_class=..., features_extractor_kwargs=...)` for different architectures
- Hyperparameter considerations vs vector-obs baseline:
  - Learning rate: may need to be lower (1e-4 instead of 3e-4) -- CNNs are more sensitive
  - Batch size: larger batches help with visual noise (256 or 512 instead of 64)
  - n_steps: may need more steps per rollout for stable gradients
  - Training timesteps: expect 10-50x more steps needed (20M-100M) compared to vector obs
- **Test**: Model instantiates, can call `model.predict()` on a frame-stacked observation

### 12d: Training + comparison

Train the pixel-based agent and compare against the vector-obs baseline.

- Training script: adapt `scripts/train.py` or create `scripts/train_visual.py`
  - Add `--render-mode rgb_array` flag to enable visual observations
  - Add `--frame-stack 4` flag
  - Add `--policy cnn` flag to select CnnPolicy
- Start with a long run: 10M-20M steps minimum (pixel-based RL is much less sample-efficient)
- Monitor with TensorBoard: compare `bhop/mean_speed` curves between vector and pixel agents
- **Key questions to answer**:
  1. Does the pixel agent discover bhop at all?
  2. How many steps does it take vs the vector agent? (expect 10-50x more)
  3. What top speed does it reach? (expect lower ceiling initially)
  4. Does the CNN learn a useful representation? (visualize conv filters / feature maps)

### 12e: Analysis and visualization

Extend the analysis notebook with pixel-obs results.

- Side-by-side training curves: vector-obs vs pixel-obs
- Render sample episodes as video (save as MP4 or GIF using the renderer)
- CNN feature visualization:
  - Plot first-layer conv filters (what edges/patterns does it look for?)
  - Grad-CAM or saliency maps: what parts of the frame drive the policy's decisions?
  - t-SNE of CNN embeddings colored by speed: does the representation separate slow/fast states?
- Compare learned policies: does the pixel agent find the same strafe angles as the vector agent?

---

## Technical Notes

### Rendering performance
- The renderer must be fast since it runs every tick (125 fps × 8 parallel envs = 1000 renders/sec)
- `pygame.Surface` with `surfarray` is typically fast enough for 84x84
- If rendering becomes the bottleneck, consider:
  - Rendering every Nth frame and repeating (action repeat / frame skip)
  - Using raw numpy array operations instead of pygame
  - GPU rendering with OpenGL (overkill for top-down but needed for first-person)

### Frame skip
- Standard Atari RL uses frame skip of 4 (act every 4th frame, repeat action between)
- For bhop at 125fps, skipping 4 frames = acting at ~31 Hz. This may be too slow for precise strafe control.
- Start without frame skip. If training is too slow, try skip=2 (62.5 Hz).

### What makes this a good CV project
- The CNN must learn to extract velocity (from frame differences), speed (from motion blur or trail length), and ground state (from color) -- all from raw pixels
- Feature visualization (Grad-CAM, filter plots, t-SNE) demonstrates understanding of CNN internals
- The vector-obs baseline gives a clean upper bound to compare against
- The "discovered an exploit from pixels" framing is compelling for presentations

---

## Stretch: First-Person Rendering (12f)

After top-down works, add a first-person 3D renderer for a more authentic Quake perspective.

- Render a ground plane with a grid texture (motion cues from parallax)
- Horizon line shifts based on vertical velocity (visual jump/land feedback)
- Simple skybox or solid color sky
- No walls/geometry needed for the flat-plane env
- HUD overlay: optional crosshair (gives agent a fixed reference point)
- This makes the problem much harder (3D projection, partial observability) but more visually impressive
- Use OpenGL via `moderngl` or `pyglet` for rendering

---

## Dependencies to add

```toml
# In pyproject.toml [project.optional-dependencies]
visual = ["pygame>=2.5", "opencv-python>=4.8"]  # renderer + video export
```
