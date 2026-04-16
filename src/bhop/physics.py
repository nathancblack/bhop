"""Quake III Arena movement physics reimplementation.

Faithful Python port of bg_pmove.c. Method names mirror the original C functions.
See docs/quake3_physics.md for the full C source reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bhop.geometry import MapGeometry


class Q3Physics:
    """Quake III movement physics simulation on a flat infinite plane.

    All internal state uses float64. Method names mirror bg_pmove.c functions.

    Constants (from bg_pmove.c):
        pm_accelerate: Ground acceleration factor (10.0)
        pm_airaccelerate: Air acceleration factor (1.0)
        pm_friction: Ground friction coefficient (6.0)
        pm_stopspeed: Speed threshold for friction clamping (100.0)
        sv_gravity: Gravity in units/sec^2 (800.0)
        JUMP_VELOCITY: Vertical velocity applied on jump (270.0)
        sv_maxspeed: Maximum wishspeed (320.0)
        FRAMETIME: Seconds per tick at 125fps (0.008)
    """

    # Constants from bg_pmove.c
    pm_accelerate: float = 10.0
    pm_airaccelerate: float = 1.0
    pm_friction: float = 6.0
    pm_stopspeed: float = 100.0
    sv_gravity: float = 800.0
    JUMP_VELOCITY: float = 270.0
    sv_maxspeed: float = 320.0
    FRAMETIME: float = 0.008  # 125fps

    def __init__(self, geometry: MapGeometry | None = None) -> None:
        """Initialize physics state. Calls reset() to set initial values.

        Args:
            geometry: Optional map geometry for collision detection.
                When None, uses flat infinite plane (original behavior).
        """
        self.geometry = geometry
        self.velocity: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.position: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.on_ground: bool = True
        self.jump_held: bool = False
        self.yaw: float = 0.0
        self.ground_normal: NDArray[np.float64] | None = np.array([0.0, 0.0, 1.0])
        self.reset()

    def reset(self) -> None:
        """Reset to initial state: zero velocity, origin position, on ground."""
        self.velocity[:] = 0.0
        self.position[:] = 0.0
        self.on_ground = True
        self.jump_held = False
        self.yaw = 0.0
        self.ground_normal = np.array([0.0, 0.0, 1.0])

    @property
    def horizontal_speed(self) -> float:
        """Return horizontal speed magnitude: sqrt(vel_x^2 + vel_y^2)."""
        return float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))

    def get_state(self) -> dict[str, float | bool]:
        """Return current state as a dict for observation building.

        Returns:
            Dict with keys: vel_x, vel_y, vel_z, speed, on_ground
        """
        return {
            "vel_x": self.velocity[0],
            "vel_y": self.velocity[1],
            "vel_z": self.velocity[2],
            "speed": self.horizontal_speed,
            "on_ground": self.on_ground,
        }

    def tick(
        self,
        forward_move: float,
        right_move: float,
        jump: bool,
        yaw_delta: float,
    ) -> None:
        """Execute one physics tick (main movement dispatch).

        Mirrors the Q3 movement dispatch:
            ground_trace() -> walk_move or air_move -> update_position -> ground_trace()

        Args:
            forward_move: Forward/backward input (-127, 0, or 127)
            right_move: Right/left input (-127, 0, or 127)
            jump: Whether jump is pressed
            yaw_delta: Change in yaw (radians) this tick
        """
        self.yaw += yaw_delta

        if not jump:
            self.jump_held = False

        self._pm_ground_trace()

        if self.on_ground:
            self._pm_walk_move(forward_move, right_move, jump)
        else:
            self._pm_air_move(forward_move, right_move)

        self._pm_slide_move()

        self._pm_ground_trace()

    # Maximum number of clip planes / bumps in slide move (matches Q3)
    MAX_CLIP_PLANES: int = 4

    def _pm_slide_move(self) -> None:
        """Move player with collision detection and wall sliding.

        When no geometry is set, falls back to raw position update.
        With geometry, traces movement and slides along surfaces up to
        MAX_CLIP_PLANES times (matching Q3's PM_SlideMove).
        """
        from bhop.geometry import trace

        if self.geometry is None:
            self.position += self.velocity * self.FRAMETIME
            return

        time_left = self.FRAMETIME

        for _ in range(self.MAX_CLIP_PLANES):
            if time_left <= 0:
                break

            end = self.position + self.velocity * time_left
            result = trace(self.position, end, self.geometry)

            if not result.hit:
                # No collision -- move the full remaining distance
                self.position = end
                break

            if result.fraction > 0:
                # Move up to the hit point, offset slightly along normal
                # to avoid getting stuck on the surface
                self.position = (
                    self.position
                    + result.fraction * (end - self.position)
                    + result.normal * 0.03125  # 1/32 unit, Q3's SURFACE_CLIP_EPSILON
                )

            if result.fraction == 0 and result.normal is not None and np.all(result.normal == 0):
                # Started inside a brush (allsolid) -- nudge out
                # This shouldn't happen in normal play but handles edge cases
                break

            # Clip velocity against the surface and try remaining time
            self.velocity = self._pm_clip_velocity(
                self.velocity, result.normal
            )
            time_left *= 1.0 - result.fraction

    def _pm_clip_velocity(
        self,
        velocity: NDArray[np.float64],
        normal: NDArray[np.float64],
        overbounce: float = 1.001,
    ) -> NDArray[np.float64]:
        """Clip velocity against a surface normal (wall sliding).

        Removes the component of velocity going into the surface, with a small
        overbounce factor to push slightly away. Mirrors Q3's PM_ClipVelocity:
            backoff = dot(velocity, normal) * overbounce
            velocity -= backoff * normal

        Args:
            velocity: Velocity vector to clip (not modified in place).
            normal: Surface normal (unit vector, outward from surface).
            overbounce: Overbounce factor (1.001 in Q3, OVERCLIP).

        Returns:
            New clipped velocity vector.
        """
        backoff = np.dot(velocity, normal) * overbounce
        return velocity - backoff * normal

    # Minimum surface normal Z component to count as walkable ground (Q3 MIN_WALK_NORMAL)
    MIN_WALK_NORMAL: float = 0.7
    # Distance to trace downward for ground detection (Q3 uses 0.25)
    GROUND_TRACE_DIST: float = 0.25

    def _pm_ground_trace(self) -> None:
        """Check if player is on ground.

        Without geometry: flat plane at z=0 (original behavior).
        With geometry: traces downward to detect brush surfaces as ground.
        A surface is walkable ground if its normal_z > MIN_WALK_NORMAL (0.7).
        The flat z=0 floor is always kept as a fallback.
        """
        if self.geometry is None:
            # Original flat-plane behavior
            if self.position[2] <= 0 and self.velocity[2] <= 0:
                self.position[2] = 0.0
                self.velocity[2] = 0.0
                self.on_ground = True
            else:
                self.on_ground = False
            return

        from bhop.geometry import trace

        # Trace downward from current position
        start = self.position.copy()
        end = self.position.copy()
        end[2] -= self.GROUND_TRACE_DIST

        result = trace(start, end, self.geometry)

        if result.hit and result.normal is not None and result.normal[2] > self.MIN_WALK_NORMAL:
            # Standing on a walkable surface. No velocity check here --
            # Q3's trace-based ground detection snaps if a surface is found
            # within GROUND_TRACE_DIST, regardless of velocity direction.
            # (Overbounce from slide-move can leave vel_z slightly positive.)
            self.position[2] = start[2] - result.fraction * self.GROUND_TRACE_DIST
            self.velocity[2] = 0.0
            self.on_ground = True
            self.ground_normal = result.normal.copy()
            return

        # Fallback: world floor at z=0 (always present)
        if self.position[2] <= 0 and self.velocity[2] <= 0:
            self.position[2] = 0.0
            self.velocity[2] = 0.0
            self.on_ground = True
            self.ground_normal = np.array([0.0, 0.0, 1.0])
            return

        self.on_ground = False
        self.ground_normal = None

    def _pm_check_jump(self) -> bool:
        """Check and execute jump if conditions are met.

        Sets velocity.z = JUMP_VELOCITY, clears on_ground, sets jump_held.
        Requires jump_held to be False (must release jump between jumps).

        Returns:
            True if jump was executed, False otherwise.
        """
        if self.jump_held:
            return False
        self.jump_held = True
        self.on_ground = False
        self.velocity[2] = self.JUMP_VELOCITY
        return True

    def _pm_friction(self) -> None:
        """Apply ground friction. Only effective when on_ground is True.

        Mirrors PM_Friction from bg_pmove.c:
            - Uses horizontal speed only (ignores vel_z)
            - If speed < 1, zero out horizontal velocity
            - control = max(speed, pm_stopspeed)
            - drop = control * pm_friction * frametime
            - Scale velocity by max(0, (speed - drop) / speed)
        """
        if not self.on_ground:
            return
        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed < 1:
            self.velocity[0] = 0.0
            self.velocity[1] = 0.0
            return
        control = max(speed, self.pm_stopspeed)
        drop = control * self.pm_friction * self.FRAMETIME
        newspeed = max(speed - drop, 0.0) / speed
        self.velocity[0] *= newspeed
        self.velocity[1] *= newspeed
        self.velocity[2] *= newspeed

    def _pm_accelerate(
        self,
        wishdir: NDArray[np.float64],
        wishspeed: float,
        accel: float,
    ) -> None:
        """Apply acceleration (THE core function that enables bunnyhopping).

        Mirrors PM_Accelerate from bg_pmove.c:
            currentspeed = dot(velocity, wishdir)  # projection, NOT magnitude!
            addspeed = wishspeed - currentspeed
            if addspeed <= 0: return
            accelspeed = min(accel * frametime * wishspeed, addspeed)
            velocity += accelspeed * wishdir

        Args:
            wishdir: Normalized wish direction (unit vector).
            wishspeed: Desired speed in wish direction.
            accel: Acceleration factor (pm_accelerate or pm_airaccelerate).
        """
        currentspeed = np.dot(self.velocity, wishdir)
        addspeed = wishspeed - currentspeed
        if addspeed <= 0:
            return
        accelspeed = accel * self.FRAMETIME * wishspeed
        if accelspeed > addspeed:
            accelspeed = addspeed
        self.velocity += accelspeed * wishdir

    def _pm_cmd_scale(
        self, forward_move: float, right_move: float
    ) -> float:
        """Compute command scale for diagonal movement normalization.

        Prevents forward+strafe from being sqrt(2) faster than forward alone.
        Mirrors PM_CmdScale from bg_pmove.c.

        Args:
            forward_move: Forward/backward input value.
            right_move: Right/left input value.

        Returns:
            Scale factor to apply to wishspeed.
        """
        max_val = max(abs(forward_move), abs(right_move))
        if max_val == 0:
            return 0.0
        total = np.sqrt(forward_move**2 + right_move**2)
        scale = self.sv_maxspeed * max_val / (127.0 * total)
        return float(scale)

    def _pm_walk_move(
        self,
        forward_move: float,
        right_move: float,
        jump: bool,
    ) -> None:
        """Ground movement path: check_jump -> friction -> accelerate.

        If jump succeeds, switches to air_move for this tick.
        Mirrors PM_WalkMove from bg_pmove.c.

        Args:
            forward_move: Forward/backward input.
            right_move: Right/left input.
            jump: Whether jump is pressed.
        """
        if jump:
            if self._pm_check_jump():
                self._pm_air_move(forward_move, right_move)
                return
        self._pm_friction()
        wishdir, wishspeed = self._compute_wish_direction(forward_move, right_move)
        if wishspeed > 0:
            self._pm_accelerate(wishdir, wishspeed, self.pm_accelerate)

    def _pm_air_move(
        self, forward_move: float, right_move: float
    ) -> None:
        """Air movement path: half-gravity -> air_accelerate -> half-gravity.

        Uses pm_airaccelerate (1.0) instead of pm_accelerate (10.0).
        Mirrors PM_AirMove from bg_pmove.c.

        Args:
            forward_move: Forward/backward input.
            right_move: Right/left input.
        """
        self._apply_gravity()
        wishdir, wishspeed = self._compute_wish_direction(forward_move, right_move)
        if wishspeed > 0:
            self._pm_accelerate(wishdir, wishspeed, self.pm_airaccelerate)
        self._apply_gravity()

    def _apply_gravity(self) -> None:
        """Apply half-step gravity: velocity.z -= sv_gravity * frametime * 0.5."""
        self.velocity[2] -= self.sv_gravity * self.FRAMETIME * 0.5

    def _compute_wish_direction(
        self, forward_move: float, right_move: float
    ) -> tuple[NDArray[np.float64], float]:
        """Compute wish direction and wish speed from movement inputs and yaw.

        Args:
            forward_move: Forward/backward input (-127, 0, or 127).
            right_move: Right/left input (-127, 0, or 127).

        Returns:
            Tuple of (wishdir as unit vector, wishspeed after cmd_scale).
        """
        forward_dir = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        right_dir = np.array([np.sin(self.yaw), -np.cos(self.yaw), 0.0])
        wishvel = forward_dir * forward_move + right_dir * right_move
        wishvel[2] = 0.0
        speed = np.linalg.norm(wishvel)
        if speed < 1e-10:
            return np.zeros(3, dtype=np.float64), 0.0
        wishdir = wishvel / speed
        wishspeed = speed * self._pm_cmd_scale(forward_move, right_move)
        return wishdir, wishspeed
